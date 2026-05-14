"""
Policy rollout → portfolio metrics using the trading environment.

Instead of vectorbt's built-in portfolio simulation (which requires a fixed
weight matrix upfront), we roll out the trained policy step-by-step through
TradingEnv and collect the daily NAV series, then compute metrics from it.
vectorbt is used for any additional analysis that benefits from its
vectorized operations (e.g. benchmark comparison).

Usage:
    runner = PolicyRunner(model_path, data_store, start_date, end_date)
    result = runner.run(n_episodes=20, deterministic=True)
    metrics.print_summary(result["mean_metrics"])
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.environment.data_store import MarketDataStore
from src.environment.trading_env import TradingEnv
from src.environment.wrappers import FlattenDictObservation
from src.backtesting.metrics import summary as compute_metrics, print_summary
from src.utils.logging_config import get_logger

log = get_logger("backtesting.runner")


class PolicyRunner:
    """
    Rolls out a trained SB3 PPO policy through TradingEnv and collects
    per-episode NAV series, daily returns, and portfolio weights.
    """

    def __init__(
        self,
        model_path: str,
        data_store: MarketDataStore,
        start_date: str,
        end_date: str,
        deterministic: bool = True,
        seed: int = 0,
    ):
        # Import here so pyarrow is always loaded first at the call site
        import pyarrow.parquet  # noqa: F401
        from stable_baselines3 import PPO

        self.data_store   = data_store
        self.start_date   = start_date
        self.end_date     = end_date
        self.deterministic = deterministic
        self.seed         = seed

        log.info("Loading model from %s", model_path)
        self.model = PPO.load(model_path)

    # ------------------------------------------------------------------
    def run_episode(self, seed: int) -> dict:
        """
        Run one full episode and return per-step data.
        Returns dict with: nav, daily_returns, weights_history, tickers, dates.
        """
        base_env = TradingEnv(
            self.data_store,
            start_date=self.start_date,
            end_date=self.end_date,
            seed=seed,
        )
        env = FlattenDictObservation(base_env)

        obs, info = env.reset(seed=seed)
        nav_series    = [base_env.portfolio.nav]
        return_series = []
        weight_history = []
        date_series   = [str(self.data_store.dates[base_env._date_idx])]

        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            nav_series.append(base_env.portfolio.nav)
            return_series.append(step_info.get("net_return", 0.0))
            weight_history.append(base_env.portfolio.weights.copy())
            date_series.append(step_info.get("date", ""))

        return {
            "nav":            np.array(nav_series, dtype=np.float64),
            "daily_returns":  np.array(return_series, dtype=np.float64),
            "weights":        np.array(weight_history, dtype=np.float32),
            "tickers":        info["tickers"],
            "dates":          date_series,
            "start_date":     info["start_date"],
        }

    # ------------------------------------------------------------------
    def run(
        self,
        n_episodes: int = 20,
        benchmark_returns: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Run n_episodes and aggregate metrics across all episodes.
        Returns dict with per-episode results and mean_metrics.
        """
        log.info("Running %d episodes [%s → %s]", n_episodes, self.start_date, self.end_date)
        episodes = []
        all_sharpes = []
        all_calmars = []
        all_mdd     = []
        all_ann_ret = []

        for i in range(n_episodes):
            ep = self.run_episode(seed=self.seed + i)
            m  = compute_metrics(
                ep["daily_returns"],
                ep["nav"],
                benchmark_returns=benchmark_returns,
                label=f"Episode {i+1} ({ep['start_date']})",
            )
            ep["metrics"] = m
            episodes.append(ep)
            all_sharpes.append(m["sharpe"])
            all_calmars.append(m["calmar"])
            all_mdd.append(m["max_drawdown"])
            all_ann_ret.append(m["annualised_return"])

        mean_metrics = {
            "label":            f"{n_episodes} episodes [{self.start_date} – {self.end_date}]",
            "sharpe":           float(np.mean(all_sharpes)),
            "sharpe_std":       float(np.std(all_sharpes, ddof=1)),
            "calmar":           float(np.mean(all_calmars)),
            "max_drawdown":     float(np.mean(all_mdd)),
            "annualised_return": float(np.mean(all_ann_ret)),
        }

        log.info(
            "Results: Sharpe=%.3f±%.3f  Calmar=%.3f  MDD=%.1f%%  AnnRet=%.1f%%",
            mean_metrics["sharpe"], mean_metrics["sharpe_std"],
            mean_metrics["calmar"],
            mean_metrics["max_drawdown"] * 100,
            mean_metrics["annualised_return"] * 100,
        )

        return {"episodes": episodes, "mean_metrics": mean_metrics}

    # ------------------------------------------------------------------
    def to_dataframe(self, run_result: dict) -> pd.DataFrame:
        """
        Convert run results to a DataFrame indexed by date for analysis.
        One row per episode per date: episode_id, date, nav, daily_return, weights.
        """
        rows = []
        for ep_i, ep in enumerate(run_result["episodes"]):
            for t, (date, ret) in enumerate(zip(ep["dates"][1:], ep["daily_returns"])):
                rows.append({
                    "episode": ep_i,
                    "start":   ep["start_date"],
                    "date":    date,
                    "nav":     ep["nav"][t + 1],
                    "daily_return": ret,
                })
        return pd.DataFrame(rows)
