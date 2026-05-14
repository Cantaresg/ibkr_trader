"""
Continuous out-of-sample backtest of a DRL trading checkpoint.

Tiles sequential non-overlapping 252-step episodes over a test window,
chains their NAV curves into a single equity series, then reports
annualised metrics vs a SPY buy-and-hold benchmark.

Usage:
    python scripts/backtest.py \
        --checkpoint checkpoints/algo_rppo_best/best/best_model.zip \
        --start 2020-01-01 --end 2023-12-31

    # Save daily equity curve to CSV:
    python scripts/backtest.py ... --out-csv data/processed/backtest_rppo.csv
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import pyarrow.parquet  # noqa: F401  — must precede torch/SB3

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    from sb3_contrib import RecurrentPPO
    _RPPO_AVAILABLE = True
except ImportError:
    _RPPO_AVAILABLE = False

from src.utils.config_loader import load_config
from src.utils.logging_config import setup_logging
from src.environment.data_store import MarketDataStore
from src.environment.trading_env import TradingEnv
from src.environment.wrappers import FlattenDictObservation


# ---------------------------------------------------------------------------
def _load_model(path: str):
    try:
        return PPO.load(path)
    except Exception:
        pass
    try:
        return SAC.load(path)
    except Exception:
        pass
    if _RPPO_AVAILABLE:
        return RecurrentPPO.load(path)
    raise ValueError(f"Could not load checkpoint as PPO, SAC, or RecurrentPPO: {path}")


def _is_recurrent(model) -> bool:
    return _RPPO_AVAILABLE and isinstance(model, RecurrentPPO)


# ---------------------------------------------------------------------------
class _SequentialEnv(TradingEnv):
    """TradingEnv subclass that supports forced episode start dates for backtesting."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._forced_idx: int | None = None

    def force_start(self, date_idx: int) -> None:
        self._forced_idx = date_idx

    def reset(self, *, seed=None, options=None):
        if self._forced_idx is None:
            return super().reset(seed=seed, options=options)

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._date_idx       = self._forced_idx
        self._using_syn      = False
        self._syn_ep         = None
        start_date           = self.ds.dates[self._date_idx]
        self._ticker_indices = self.ds.get_candidates(start_date, 20)
        self.portfolio.reset()
        self._step           = 0
        self._forced_idx     = None  # consume

        obs  = self._build_obs()
        info = {
            "start_date": str(start_date),
            "tickers":    self._selected_tickers(),
            "synthetic":  False,
        }
        return obs, info


# ---------------------------------------------------------------------------
def _find_date_idx(data_store: MarketDataStore, date_str: str) -> int:
    target = pd.Timestamp(date_str)
    dates  = data_store.dates
    pos    = int(np.searchsorted(dates, target))
    return min(pos, len(dates) - 1)


def _episode_start_indices(data_store: MarketDataStore, start: str, end: str,
                            ep_len: int) -> list[int]:
    end_ts    = pd.Timestamp(end)
    idx       = _find_date_idx(data_store, start)
    indices   = []
    while idx + ep_len < len(data_store.dates):
        if data_store.dates[idx] > end_ts:
            break
        indices.append(idx)
        idx += ep_len
    return indices


# ---------------------------------------------------------------------------
def run_backtest(model, data_store: MarketDataStore, cfg: dict,
                 start: str, end: str, seed: int = 42) -> pd.DataFrame:

    ep_len    = cfg["environment"]["episode_length"]
    recurrent = _is_recurrent(model)

    ep_indices = _episode_start_indices(data_store, start, end, ep_len)
    if not ep_indices:
        raise ValueError(f"No valid episode start dates between {start} and {end}.")

    n_ep = len(ep_indices)
    d0   = data_store.dates[ep_indices[0]].date()
    d1   = data_store.dates[ep_indices[-1]].date()
    print(f"  {n_ep} episodes x {ep_len} steps  ({d0} to {d1})")

    records    = []
    cum_factor = 1.0

    for ep_num, start_idx in enumerate(ep_indices):
        ep_start_date = str(data_store.dates[start_idx].date())
        ep_end_idx    = min(start_idx + ep_len + 5, len(data_store.dates) - 1)
        ep_end_date   = str(data_store.dates[ep_end_idx].date())

        def _make_env(sidx=start_idx, sd=ep_start_date, ed=ep_end_date, ep=ep_num):
            env = _SequentialEnv(
                data_store,
                start_date           = sd,
                end_date             = ed,
                lookback             = cfg["features"]["lookback_window"],
                episode_length       = ep_len,
                initial_capital      = cfg["environment"]["initial_capital"],
                transaction_cost_bps = cfg["environment"]["transaction_cost_bps"],
                reward_alpha         = cfg["reward"]["excess_return_weight"],
                reward_beta          = cfg["reward"]["drawdown_penalty_weight"],
                reward_gamma         = cfg["reward"]["transaction_cost_weight"],
                drawdown_threshold   = cfg["reward"]["drawdown_threshold"],
                regime_weights       = None,
                synthetic_store      = None,
                synthetic_ratio      = 0.0,
                seed                 = seed + ep,
            )
            env.force_start(sidx)
            return FlattenDictObservation(env)

        vec_env       = DummyVecEnv([_make_env])
        obs           = vec_env.reset()
        lstm_state    = None
        ep_start_flag = np.ones((1,), dtype=bool)
        ep_start_cum  = cum_factor

        for _ in range(ep_len):
            if recurrent:
                action, lstm_state = model.predict(
                    obs, state=lstm_state,
                    episode_start=ep_start_flag, deterministic=True,
                )
                ep_start_flag = np.zeros((1,), dtype=bool)
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, _reward, done, info = vec_env.step(action)
            inf      = info[0]
            step_ret = float(inf.get("portfolio_return", 0.0))
            date_str = inf.get("date", "")

            cum_factor *= (1 + step_ret)
            records.append({
                "date":        pd.Timestamp(date_str) if date_str else None,
                "step_return": step_ret,
                "cum_return":  cum_factor - 1.0,
                "episode":     ep_num,
            })

            if done[0]:
                break

        vec_env.close()
        ep_ret = cum_factor / ep_start_cum - 1.0
        print(f"  Ep {ep_num + 1}/{n_ep}  start={ep_start_date}  return={ep_ret:+.2%}  "
              f"cum={cum_factor - 1.0:+.2%}")

    df = pd.DataFrame(records).dropna(subset=["date"]).set_index("date")
    return df


# ---------------------------------------------------------------------------
def _metrics(returns: pd.Series, label: str) -> dict:
    n      = len(returns)
    total  = float((1 + returns).prod() - 1)
    cagr   = float((1 + total) ** (252 / n) - 1)
    vol    = float(returns.std() * np.sqrt(252))
    sharpe = float((returns.mean() * 252) / (returns.std() * np.sqrt(252) + 1e-9))
    cum    = (1 + returns).cumprod()
    dd     = float((cum / cum.cummax() - 1).min())
    calmar = cagr / abs(dd) if dd != 0 else float("nan")
    return dict(label=label, total=total, cagr=cagr, vol=vol,
                sharpe=sharpe, max_dd=dd, calmar=calmar, n=n)


def _spy_returns(start: str, end: str) -> pd.Series | None:
    try:
        import yfinance as yf
        spy = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)
        return spy["Close"].pct_change().dropna()
    except Exception as e:
        print(f"  SPY download failed: {e}")
        return None


def _print_table(rows: list[dict]) -> None:
    hdr = f"{'':24}  {'Total Ret':>10}  {'CAGR':>8}  {'Ann Vol':>8}  {'Sharpe':>7}  {'Max DD':>8}  {'Calmar':>7}"
    sep = "-" * len(hdr)
    print(f"\n{sep}")
    print(hdr)
    print(sep)
    for r in rows:
        print(
            f"  {r['label']:<22}  "
            f"{r['total']:>+10.2%}  "
            f"{r['cagr']:>+8.2%}  "
            f"{r['vol']:>8.2%}  "
            f"{r['sharpe']:>+7.2f}  "
            f"{r['max_dd']:>8.2%}  "
            f"{r['calmar']:>7.2f}"
        )
    print(sep)


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",     required=True)
    p.add_argument("--config",         default="config/config.yaml")
    p.add_argument("--start",          default="2020-01-01")
    p.add_argument("--end",            default="2023-12-31")
    p.add_argument("--episode-length", type=int, default=None,
                   help="Override episode length from config")
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--out-csv",        default=None,
                   help="Save daily equity curve to this CSV path")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging()

    cfg = load_config(args.config)
    if args.episode_length is not None:
        cfg["environment"]["episode_length"] = args.episode_length

    print(f"Loading checkpoint: {args.checkpoint}")
    model = _load_model(args.checkpoint)

    print("Loading MarketDataStore...")
    data_store = MarketDataStore(config_path=args.config)

    print(f"\nBacktest window: {args.start} to {args.end}")
    df = run_backtest(model, data_store, cfg, args.start, args.end, seed=args.seed)

    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv)
        print(f"\nEquity curve saved to {args.out_csv}")

    rows = [_metrics(df["step_return"], label=Path(args.checkpoint).parts[-3])]

    print("\nFetching SPY benchmark...")
    spy = _spy_returns(args.start, args.end)
    if spy is not None:
        aligned = spy.reindex(df.index, method="nearest").dropna()
        rows.append(_metrics(aligned, label="SPY buy-and-hold"))

    _print_table(rows)


if __name__ == "__main__":
    main()
