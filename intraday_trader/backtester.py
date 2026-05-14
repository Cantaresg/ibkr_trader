"""
IntradayPolicyRunner: rolls out a trained intraday PPO policy through
IntradayTradingEnv and collects per-bar and per-day performance metrics.

Intraday analogue of src/backtesting/vectorbt_runner.py.

Usage:
    from intraday_trader.backtester import IntradayPolicyRunner
    from intraday_trader.data_store import IntradayDataStore

    ds     = IntradayDataStore()
    runner = IntradayPolicyRunner("checkpoints/intraday_ppo/best/best_model.zip", ds,
                                  start_date="2024-01-01", end_date="2024-06-30")
    result = runner.run(n_episodes=20)
    runner.print_summary(result["mean_metrics"])
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.backtesting.metrics import (
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    calmar_ratio,
    annualised_return,
    win_rate,
    print_summary,
)
from src.environment.wrappers import FlattenDictObservation
from intraday_trader.constants import BARS_PER_DAY
from intraday_trader.data_store import IntradayDataStore
from intraday_trader.env import IntradayTradingEnv
from src.utils.logging_config import get_logger

log = get_logger("intraday.backtester")

# Intraday annualisation: 7 bars/day × 252 trading days
BARS_PER_YEAR = BARS_PER_DAY * 252


class IntradayPolicyRunner:
    """
    Rolls out a trained SB3 PPO policy through IntradayTradingEnv and collects
    per-bar returns, daily PnL, NAV series, and portfolio weights.

    Parameters
    ----------
    model_path    : Path to a .zip SB3 PPO or RecurrentPPO checkpoint.
    data_store    : Pre-loaded IntradayDataStore.
    start_date    : Episode start sampling lower bound (inclusive).
    end_date      : Episode start sampling upper bound (inclusive).
    deterministic : Whether to use deterministic policy (default True).
    seed          : Base random seed.
    algo          : "ppo" or "rppo". If None, auto-detected from the checkpoint.
    """

    def __init__(
        self,
        model_path: str,
        data_store: IntradayDataStore,
        start_date: str,
        end_date: str,
        deterministic: bool = True,
        seed: int = 0,
        algo: str | None = None,
        min_position_weight: float = 0.0,
        eod_force_flat: bool = True,
    ):
        import pyarrow.parquet  # noqa: F401 — must precede torch/SB3 on Windows
        from stable_baselines3 import PPO

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")

        self.data_store    = data_store
        self.start_date    = start_date
        self.end_date      = end_date
        self.deterministic = deterministic
        self.seed          = seed
        self.algo          = (algo or "ppo").lower()

        ModelClass, self.algo = _resolve_model_class(model_path, self.algo)
        self._is_recurrent = (self.algo == "rppo")
        log.info("Loading intraday %s model from %s", self.algo.upper(), model_path)
        self.model = ModelClass.load(model_path)
        # Infer n_stocks from the saved action space: Box shape = (n_stocks + 1,)
        self._n_stocks = int(self.model.action_space.shape[0]) - 1
        self.min_position_weight = float(min_position_weight)
        self.eod_force_flat = eod_force_flat

    # ------------------------------------------------------------------
    def run_episode(self, seed: int) -> dict:
        """
        Run one full episode (n_days × BARS_PER_DAY steps).

        Returns
        -------
        dict with keys:
          nav            : np.ndarray (n_bars + 1,)  — NAV at each bar boundary
          bar_returns    : np.ndarray (n_bars,)       — per-bar portfolio returns
          daily_pnl      : list[float]                — fractional daily PnL
          weights        : np.ndarray (n_bars, N_STOCKS+1)
          bar_timestamps : list[str]                  — ISO timestamps per bar
          start_bar      : int
          start_ts       : str
        """
        base_env = IntradayTradingEnv(
            self.data_store,
            start_date=self.start_date,
            end_date=self.end_date,
            n_stocks=self._n_stocks,
            min_position_weight=self.min_position_weight,
            eod_force_flat=self.eod_force_flat,
            seed=seed,
        )
        env = FlattenDictObservation(base_env)

        obs, info = env.reset(seed=seed)

        nav_series     = [base_env.portfolio.nav]
        bar_returns    = []
        weight_history = []
        timestamps     = [str(self.data_store.bar_timestamps[base_env._start_bar_idx])]

        lstm_state    = None
        is_first_step = True

        done = False
        while not done:
            if self._is_recurrent:
                ep_start = np.array([is_first_step], dtype=bool)
                action, lstm_state = self.model.predict(
                    obs, state=lstm_state, episode_start=ep_start,
                    deterministic=self.deterministic)
                is_first_step = False
            else:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
            obs, _reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            nav_series.append(base_env.portfolio.nav)
            bar_returns.append(step_info.get("net_return", 0.0))
            weight_history.append(base_env.portfolio.weights.copy())

            flat_bar = base_env._current_flat_bar
            if flat_bar < self.data_store.n_bars:
                timestamps.append(str(self.data_store.bar_timestamps[flat_bar]))
            else:
                timestamps.append("")

        return {
            "nav":            np.array(nav_series, dtype=np.float64),
            "bar_returns":    np.array(bar_returns, dtype=np.float64),
            "daily_pnl":      list(base_env.portfolio.daily_pnl_history),
            "weights":        np.array(weight_history, dtype=np.float32),
            "bar_timestamps": timestamps,
            "start_bar":      info["start_bar"],
            "start_ts":       info["start_ts"],
        }

    # ------------------------------------------------------------------
    def run(
        self,
        n_episodes: int = 20,
    ) -> dict:
        """
        Run n_episodes and aggregate bar-level and day-level metrics.

        Returns
        -------
        dict with keys:
          episodes     : list of per-episode dicts from run_episode()
          mean_metrics : dict of aggregated metrics
        """
        log.info(
            "Intraday backtest: %d episodes  [%s → %s]",
            n_episodes, self.start_date, self.end_date,
        )

        episodes: list[dict] = []

        # Accumulators — bar-level
        all_bar_sharpes:  list[float] = []
        all_bar_sortinos: list[float] = []
        # Accumulators — day-level
        all_daily_sharpes:    list[float] = []
        all_daily_sortinos:   list[float] = []
        all_calmars:          list[float] = []
        all_mdd:              list[float] = []
        all_ann_ret:          list[float] = []
        all_daily_win_rates:  list[float] = []
        all_avg_daily_pnl:    list[float] = []

        for i in range(n_episodes):
            ep = self.run_episode(seed=self.seed + i)

            bar_rets   = ep["bar_returns"]
            daily_rets = np.array(ep["daily_pnl"], dtype=np.float64)
            nav        = ep["nav"]

            # --- Bar-level metrics ---
            b_sharpe  = _bar_sharpe(bar_rets)
            b_sortino = _bar_sortino(bar_rets)

            # --- Day-level metrics (reusing src/backtesting/metrics.py) ---
            if len(daily_rets) >= 2:
                d_sharpe  = sharpe_ratio(daily_rets)
                d_sortino = sortino_ratio(daily_rets)
                d_calmar  = calmar_ratio(daily_rets, nav)
                d_mdd     = max_drawdown(nav)
                d_ann_ret = annualised_return(daily_rets)
                d_winrate = win_rate(daily_rets)
                d_avg     = float(daily_rets.mean())
            else:
                d_sharpe  = 0.0
                d_sortino = 0.0
                d_calmar  = 0.0
                d_mdd     = float(max_drawdown(nav))
                d_ann_ret = 0.0
                d_winrate = 0.0
                d_avg     = 0.0

            ep_metrics = {
                "bar_sharpe":      b_sharpe,
                "bar_sortino":     b_sortino,
                "daily_sharpe":    d_sharpe,
                "daily_sortino":   d_sortino,
                "calmar":          d_calmar,
                "max_drawdown":    d_mdd,
                "annualised_return": d_ann_ret,
                "daily_win_rate":  d_winrate,
                "avg_daily_pnl":   d_avg,
                "n_days":          len(daily_rets),
                "n_bars":          len(bar_rets),
            }
            ep["metrics"] = ep_metrics
            episodes.append(ep)

            all_bar_sharpes.append(b_sharpe)
            all_bar_sortinos.append(b_sortino)
            all_daily_sharpes.append(d_sharpe)
            all_daily_sortinos.append(d_sortino)
            all_calmars.append(d_calmar)
            all_mdd.append(d_mdd)
            all_ann_ret.append(d_ann_ret)
            all_daily_win_rates.append(d_winrate)
            all_avg_daily_pnl.append(d_avg)

        mean_metrics = {
            "label":              f"{n_episodes} episodes [{self.start_date} – {self.end_date}]",
            "bar_sharpe":         float(np.mean(all_bar_sharpes)),
            "bar_sharpe_std":     float(np.std(all_bar_sharpes, ddof=1)) if n_episodes > 1 else 0.0,
            "bar_sortino":        float(np.mean(all_bar_sortinos)),
            "daily_sharpe":       float(np.mean(all_daily_sharpes)),
            "daily_sharpe_std":   float(np.std(all_daily_sharpes, ddof=1)) if n_episodes > 1 else 0.0,
            "daily_sortino":      float(np.mean(all_daily_sortinos)),
            "calmar":             float(np.mean(all_calmars)),
            "max_drawdown":       float(np.mean(all_mdd)),
            "annualised_return":  float(np.mean(all_ann_ret)),
            "daily_win_rate":     float(np.mean(all_daily_win_rates)),
            "avg_daily_pnl":      float(np.mean(all_avg_daily_pnl)),
        }

        log.info(
            "Intraday backtest results: "
            "DailySharpe=%.3f±%.3f  BarSharpe=%.3f  MDD=%.1f%%  WinRate=%.1f%%  AnnRet=%.1f%%",
            mean_metrics["daily_sharpe"], mean_metrics["daily_sharpe_std"],
            mean_metrics["bar_sharpe"],
            mean_metrics["max_drawdown"] * 100,
            mean_metrics["daily_win_rate"] * 100,
            mean_metrics["annualised_return"] * 100,
        )

        return {"episodes": episodes, "mean_metrics": mean_metrics}

    # ------------------------------------------------------------------
    @staticmethod
    def print_summary(mean_metrics: dict) -> None:
        """Print a human-readable results table."""
        label = mean_metrics.get("label", "")
        print(f"\n{'=' * 60}")
        print(f"  Intraday Backtest: {label}")
        print(f"{'=' * 60}")
        print(f"  Bar Sharpe (annualised):  {mean_metrics['bar_sharpe']:+.3f}  ± {mean_metrics.get('bar_sharpe_std', 0.0):.3f}")
        print(f"  Bar Sortino:              {mean_metrics['bar_sortino']:+.3f}")
        print(f"  Daily Sharpe:             {mean_metrics['daily_sharpe']:+.3f}  ± {mean_metrics.get('daily_sharpe_std', 0.0):.3f}")
        print(f"  Daily Sortino:            {mean_metrics['daily_sortino']:+.3f}")
        print(f"  Calmar Ratio:             {mean_metrics['calmar']:+.3f}")
        print(f"  Max Drawdown:             {mean_metrics['max_drawdown']:.1%}")
        print(f"  Annualised Return:        {mean_metrics['annualised_return']:+.1%}")
        print(f"  Daily Win Rate:           {mean_metrics['daily_win_rate']:.1%}")
        print(f"  Avg Daily PnL:            {mean_metrics['avg_daily_pnl']:+.4f}")
        print(f"{'=' * 60}\n")

    # ------------------------------------------------------------------
    def to_dataframe(self, run_result: dict) -> pd.DataFrame:
        """
        Convert run results to a DataFrame with one row per episode per bar.

        Columns: episode, bar_timestamp, nav, bar_return, weights (per ticker)
        """
        from intraday_trader.constants import N_STOCKS, INTRADAY_UNIVERSE

        rows = []
        tickers = INTRADAY_UNIVERSE[:N_STOCKS]

        for ep_i, ep in enumerate(run_result["episodes"]):
            bar_rets = ep["bar_returns"]
            navs     = ep["nav"]
            weights  = ep["weights"]
            tss      = ep["bar_timestamps"]

            for t, (br, nav, w) in enumerate(zip(bar_rets, navs[1:], weights)):
                row = {
                    "episode":    ep_i,
                    "bar":        t,
                    "timestamp":  tss[t + 1] if t + 1 < len(tss) else "",
                    "nav":        nav,
                    "bar_return": br,
                }
                for ti, ticker in enumerate(tickers):
                    row[f"w_{ticker}"] = float(w[ti]) if ti < len(w) else 0.0
                rows.append(row)

        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Intraday-specific metric helpers (bar-level)
# ---------------------------------------------------------------------------

_RISK_FREE_PER_BAR = 0.05 / BARS_PER_YEAR


def _bar_sharpe(bar_returns: np.ndarray) -> float:
    """Annualised Sharpe computed from per-bar returns."""
    if len(bar_returns) < 2:
        return 0.0
    excess = bar_returns - _RISK_FREE_PER_BAR
    std = excess.std(ddof=1)
    if std < 1e-12:
        return 0.0
    return float(excess.mean() / std * np.sqrt(BARS_PER_YEAR))


def _bar_sortino(bar_returns: np.ndarray) -> float:
    """Annualised Sortino computed from per-bar returns."""
    if len(bar_returns) < 2:
        return 0.0
    excess   = bar_returns - _RISK_FREE_PER_BAR
    downside = excess[excess < 0]
    if len(downside) < 2:
        return 0.0
    downside_std = np.sqrt((downside ** 2).mean())
    if downside_std < 1e-12:
        return 0.0
    return float(excess.mean() / downside_std * np.sqrt(BARS_PER_YEAR))


# ---------------------------------------------------------------------------
# Model loading helper
# ---------------------------------------------------------------------------

def _resolve_model_class(model_path: str, algo_hint: str):
    """
    Return (ModelClass, algo_str) for the given checkpoint.

    Strategy:
      1. If algo_hint is "rppo" (and sb3_contrib is available), use RecurrentPPO.
      2. Otherwise fall back to PPO.
      3. As a last resort, peek at the zip metadata for the class name.
    """
    from stable_baselines3 import PPO

    # Try to detect from checkpoint metadata first
    try:
        import zipfile, json
        with zipfile.ZipFile(model_path, "r") as zf:
            if "data" in zf.namelist():
                meta = json.loads(zf.read("data"))
                policy_class = str(meta.get("policy_class", {}))
                if "RecurrentActorCriticPolicy" in policy_class or "Lstm" in policy_class:
                    try:
                        from sb3_contrib import RecurrentPPO
                        return RecurrentPPO, "rppo"
                    except ImportError:
                        pass
    except Exception:
        pass

    if algo_hint == "rppo":
        try:
            from sb3_contrib import RecurrentPPO
            return RecurrentPPO, "rppo"
        except ImportError:
            log.warning("sb3_contrib not available — falling back to PPO for loading")

    return PPO, "ppo"
