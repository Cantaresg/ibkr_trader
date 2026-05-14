"""
Walk-forward backtest: run the saved PPO checkpoint sequentially through a date
range without lookahead, tracking the cumulative equity curve.

Episodes are chained end-to-end: each episode starts the day after the previous
one ended, capital carries over (compounded). Portfolio state resets each episode
but the cumulative NAV multiplier is preserved.

Usage:
    # Single run with scanner stock selection (default):
    python scripts/walkforward_backtest.py \\
        --checkpoint checkpoints/best_combined_5m/best/best_model.zip \\
        --start 2019-01-01 --end 2022-12-31 --episode-length 126

    # 10 randomised trials (different random 20-stock draws) + scanner baseline:
    python scripts/walkforward_backtest.py \\
        --checkpoint checkpoints/best_combined_5m/best/best_model.zip \\
        --start 2019-01-01 --end 2022-12-31 --episode-length 126 \\
        --n-trials 10
"""
from __future__ import annotations
import sys
import argparse
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pyarrow.parquet  # noqa: F401  — must precede torch/SB3

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.utils.config_loader import load_config
from src.utils.logging_config import setup_logging
from src.environment.data_store import MarketDataStore
from src.environment.trading_env import TradingEnv, N_STOCKS
from src.environment.wrappers import FlattenDictObservation


# ---------------------------------------------------------------------------
# TradingEnv subclass: forced episode start + optional random stock selection
# ---------------------------------------------------------------------------
class SequentialTradingEnv(TradingEnv):
    """
    TradingEnv that:
    1. Starts each episode at a caller-specified date index (no random sampling).
    2. Optionally replaces the scanner's top-20 with a random draw from the
       full ticker universe (used for robustness trials).
    """

    def __init__(self, *args, ticker_rng: np.random.Generator | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._forced_start_idx: int | None = None
        self._ticker_rng = ticker_rng          # None = use scanner; else random draw

    def force_start(self, date_idx: int):
        self._forced_start_idx = date_idx

    def reset(self, *, seed=None, options=None):
        if self._forced_start_idx is not None:
            _orig = self.valid_starts
            self.valid_starts = [self._forced_start_idx]
            self._regime_weights    = None
            self._starts_by_regime  = None
            obs, info = super().reset(seed=seed, options=options)
            self.valid_starts           = _orig
            self._forced_start_idx      = None
        else:
            obs, info = super().reset(seed=seed, options=options)

        # After parent sets self._ticker_indices via scanner, optionally replace
        # with a random draw from the full universe.
        if self._ticker_rng is not None:
            n_tickers = len(self.ds.ticker_list)
            sampled   = self._ticker_rng.choice(n_tickers, size=N_STOCKS, replace=False).tolist()
            self._ticker_indices = sampled
            # Rebuild obs with the new ticker selection
            obs, info = self._rebuild_obs_after_ticker_change(obs, info)

        return obs, info

    def _rebuild_obs_after_ticker_change(self, obs, info):
        """Re-build observation arrays after overriding _ticker_indices."""
        from src.environment.trading_env import _softmax
        obs = self._build_obs()
        info["tickers"] = self._selected_tickers()
        return obs, info


# ---------------------------------------------------------------------------
# SPY buy-and-hold helper
# ---------------------------------------------------------------------------
def load_spy_prices(raw_dir: str, start: str, end: str) -> pd.Series:
    spy_path = Path(raw_dir) / "market" / "SPY.parquet"
    df = pd.read_parquet(spy_path)
    df.index = pd.DatetimeIndex(df.index)
    mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
    return df.loc[mask, "close"]


# ---------------------------------------------------------------------------
# Core walk-forward loop
# ---------------------------------------------------------------------------
def run_walkforward(
    model: PPO,
    data_store: MarketDataStore,
    cfg: dict,
    start: str,
    end: str,
    seed: int = 42,
    ticker_rng: np.random.Generator | None = None,
    label: str = "scanner",
    verbose: bool = True,
) -> list[dict]:
    ep_len   = cfg["environment"]["episode_length"]
    lookback = cfg["features"]["lookback_window"]

    env = SequentialTradingEnv(
        data_store,
        start_date=start,
        end_date=end,
        lookback=lookback,
        episode_length=ep_len,
        initial_capital=cfg["environment"]["initial_capital"],
        transaction_cost_bps=cfg["environment"]["transaction_cost_bps"],
        reward_alpha=cfg["reward"]["excess_return_weight"],
        reward_beta=cfg["reward"]["drawdown_penalty_weight"],
        reward_gamma=cfg["reward"]["transaction_cost_weight"],
        drawdown_threshold=cfg["reward"]["drawdown_threshold"],
        regime_weights=None,
        synthetic_store=None,
        synthetic_ratio=0.0,
        seed=seed,
        ticker_rng=ticker_rng,
    )
    wrapped = FlattenDictObservation(env)

    # Build sequential, non-overlapping start indices
    all_starts_sorted = sorted(set(env.valid_starts))
    if not all_starts_sorted:
        raise ValueError(f"No valid episode starts in [{start}, {end}]")

    cursor = all_starts_sorted[0]
    episode_starts = []
    while True:
        idx = next((i for i in all_starts_sorted if i >= cursor), None)
        if idx is None:
            break
        episode_starts.append(idx)
        cursor = idx + ep_len

    if verbose:
        print(f"\n  [{label}]  {len(episode_starts)} episodes, "
              f"{ep_len} days each, "
              f"{data_store.dates[episode_starts[0]].date()} to "
              f"{data_store.dates[min(episode_starts[-1]+ep_len, len(data_store.dates)-1)].date()}")

    records  = []
    cum_nav  = 1.0
    ep_num   = 0

    for start_idx in episode_starts:
        ep_num += 1
        env.force_start(start_idx)
        obs, info = wrapped.reset()

        ep_start_date = info.get("start_date", str(data_store.dates[start_idx].date()))
        ep_nav_start  = None
        ep_maxdd      = 0.0
        step          = 0
        done          = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = np.squeeze(action)
            obs, reward, terminated, truncated, step_info = wrapped.step(action)
            done = terminated or truncated

            nav_abs = float(step_info.get("nav", cfg["environment"]["initial_capital"]))
            if ep_nav_start is None:
                ep_nav_start = nav_abs

            ep_maxdd = max(ep_maxdd, float(step_info.get("drawdown", 0.0)))
            step    += 1

            ep_nav_norm      = nav_abs / ep_nav_start if ep_nav_start else 1.0
            portfolio_cum_nav = cum_nav * ep_nav_norm

            records.append({
                "trial":         label,
                "date":          step_info.get("date", ""),
                "episode":       ep_num,
                "step":          step,
                "nav_abs":       round(nav_abs, 2),
                "ep_nav_norm":   round(ep_nav_norm, 6),
                "portfolio_nav": round(portfolio_cum_nav, 6),
                "drawdown":      round(float(step_info.get("drawdown", 0.0)), 6),
            })

        ep_return = (nav_abs / ep_nav_start) if ep_nav_start and ep_nav_start > 0 else 1.0
        cum_nav  *= ep_return

        if verbose:
            print(f"    ep {ep_num:2d}  {ep_start_date}  "
                  f"return {(ep_return-1)*100:+.2f}%  "
                  f"maxDD {ep_maxdd*100:.1f}%  "
                  f"cum_nav {cum_nav:.4f}")

    wrapped.close()
    return records


# ---------------------------------------------------------------------------
# Stats extraction
# ---------------------------------------------------------------------------
def compute_stats(records: list[dict]) -> dict:
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    total_return = df["portfolio_nav"].iloc[-1] - 1.0
    peak         = df["portfolio_nav"].cummax()
    drawdowns    = (df["portfolio_nav"] - peak) / peak
    max_dd       = float(drawdowns.min())

    n_years = (df["date"].iloc[-1] - df["date"].iloc[0]).days / 365.25
    ann_ret = (df["portfolio_nav"].iloc[-1] ** (1 / n_years)) - 1.0 if n_years > 0 else 0.0

    df["daily_ret"] = df["portfolio_nav"].pct_change().fillna(0.0)
    sharpe = float((df["daily_ret"].mean() / (df["daily_ret"].std() + 1e-10)) * np.sqrt(252))

    # Win rate: fraction of episodes with positive return
    ep_returns = (df.groupby("episode")["ep_nav_norm"].last() - 1.0)
    win_rate   = float((ep_returns > 0).mean())

    return {
        "total_return": total_return,
        "ann_return":   ann_ret,
        "sharpe":       sharpe,
        "max_dd":       max_dd,
        "win_rate":     win_rate,
        "n_episodes":   int(df["episode"].max()),
    }


def spy_stats(spy_prices: pd.Series, start: str, end: str) -> dict:
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    w = spy_prices[(spy_prices.index >= s) & (spy_prices.index <= e)]
    if w.empty:
        return {}
    eq        = w / w.iloc[0]
    peak      = eq.cummax()
    max_dd    = float(((eq - peak) / peak).min())
    n_years   = (w.index[-1] - w.index[0]).days / 365.25
    total_ret = float(w.iloc[-1] / w.iloc[0]) - 1.0
    ann_ret   = (w.iloc[-1] / w.iloc[0]) ** (1 / n_years) - 1.0 if n_years > 0 else 0.0
    daily     = eq.pct_change().fillna(0.0)
    sharpe    = float((daily.mean() / (daily.std() + 1e-10)) * np.sqrt(252))
    return {"total_return": total_ret, "ann_return": ann_ret, "sharpe": sharpe, "max_dd": max_dd}


# ---------------------------------------------------------------------------
# Print final comparison table
# ---------------------------------------------------------------------------
def print_comparison(trial_stats: list[tuple[str, dict]], spy: dict):
    print("\n" + "=" * 88)
    print(f"  {'Trial':<20} {'Total Ret':>10} {'Ann Ret':>9} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>8}")
    print("  " + "-" * 82)

    for label, s in trial_stats:
        print(f"  {label:<20} "
              f"{s['total_return']*100:>+9.2f}%  "
              f"{s['ann_return']*100:>+8.2f}%  "
              f"{s['sharpe']:>7.2f}  "
              f"{s['max_dd']*100:>+7.2f}%  "
              f"{s['win_rate']*100:>7.0f}%")

    if spy:
        print("  " + "-" * 82)
        print(f"  {'SPY buy-and-hold':<20} "
              f"{spy['total_return']*100:>+9.2f}%  "
              f"{spy['ann_return']*100:>+8.2f}%  "
              f"{spy['sharpe']:>7.2f}  "
              f"{spy['max_dd']*100:>+7.2f}%")

    # Summary across random trials (exclude scanner baseline = first entry)
    random_stats = [s for lbl, s in trial_stats if lbl.startswith("random_")]
    if random_stats:
        arr_total  = np.array([s["total_return"] for s in random_stats])
        arr_sharpe = np.array([s["sharpe"]       for s in random_stats])
        arr_dd     = np.array([s["max_dd"]        for s in random_stats])
        arr_win    = np.array([s["win_rate"]       for s in random_stats])
        print("  " + "-" * 82)
        print(f"  {'Random mean':<20} "
              f"{arr_total.mean()*100:>+9.2f}%  "
              f"{'':9}  "
              f"{arr_sharpe.mean():>7.2f}  "
              f"{arr_dd.mean()*100:>+7.2f}%  "
              f"{arr_win.mean()*100:>7.0f}%")
        print(f"  {'Random std':<20} "
              f"{arr_total.std()*100:>9.2f}%  "
              f"{'':9}  "
              f"{arr_sharpe.std():>7.2f}  "
              f"{arr_dd.std()*100:>7.2f}%  "
              f"{arr_win.std()*100:>7.0f}%")
        beats_spy = (arr_total > spy.get("total_return", 0)).mean() if spy else None
        if beats_spy is not None:
            print(f"\n  Random trials beating SPY total return: "
                  f"{beats_spy*100:.0f}% ({int(beats_spy*len(arr_total))}/{len(arr_total)})")

    print("=" * 88)


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",      required=True)
    p.add_argument("--config",          default="config/config.yaml")
    p.add_argument("--start",           default="2019-01-01")
    p.add_argument("--end",             default="2022-12-31")
    p.add_argument("--episode-length",  type=int,  default=None)
    p.add_argument("--n-trials",        type=int,  default=0,
                   help="Number of randomised stock-selection trials in addition to scanner run")
    p.add_argument("--seed",            type=int,  default=42)
    p.add_argument("--out",             default=None,
                   help="CSV output path for the scanner-baseline equity curve")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging()

    cfg = load_config(args.config)
    if args.episode_length is not None:
        cfg["environment"]["episode_length"] = args.episode_length

    print(f"Loading checkpoint: {args.checkpoint}")
    model = PPO.load(args.checkpoint)

    print("Loading MarketDataStore...")
    data_store = MarketDataStore(config_path=args.config)

    raw_dir    = cfg["data"]["raw_dir"]
    spy_prices = load_spy_prices(raw_dir, args.start, args.end)

    trial_stats: list[tuple[str, dict]] = []
    master_rng  = np.random.default_rng(args.seed)

    # --- Trial 0: scanner baseline ---
    print(f"\n[0/{'scanner'}]  Running scanner baseline...")
    rec0 = run_walkforward(model, data_store, cfg,
                           args.start, args.end,
                           seed=args.seed, ticker_rng=None,
                           label="scanner", verbose=True)
    trial_stats.append(("scanner", compute_stats(rec0)))

    # Optionally save scanner equity curve
    if rec0:
        out_path = args.out or f"results/walkforward_{args.start}_{args.end}.csv"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rec0[0].keys()))
            w.writeheader()
            w.writerows(rec0)
        print(f"  Equity curve saved to {out_path}")

    # --- Randomised trials ---
    for t in range(1, args.n_trials + 1):
        trial_seed  = int(master_rng.integers(0, 2**31))
        ticker_rng  = np.random.default_rng(trial_seed)
        label       = f"random_{t:02d}"
        print(f"\n[{t}/{args.n_trials}]  Running {label} (seed={trial_seed})...")
        rec = run_walkforward(model, data_store, cfg,
                              args.start, args.end,
                              seed=trial_seed, ticker_rng=ticker_rng,
                              label=label, verbose=True)
        trial_stats.append((label, compute_stats(rec)))

    # --- Final comparison table ---
    print_comparison(trial_stats, spy_stats(spy_prices, args.start, args.end))


if __name__ == "__main__":
    main()
