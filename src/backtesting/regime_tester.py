"""
Regime stress tests: evaluate the policy on specific historical periods
known for distinct market conditions.

Periods covered:
  - 2015-Q3:      China-driven flash correction (Aug 2015)
  - 2018-Q4:      Fed rate-hike sell-off (Oct–Dec 2018)
  - 2020-crash:   COVID crash (Feb–Mar 2020)
  - 2020-recovery: V-shaped recovery (Apr–Aug 2020)
  - 2022-bear:    Inflation/rate-hike sustained bear (Jan–Dec 2022)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from pathlib import Path

from src.backtesting.vectorbt_runner import PolicyRunner
from src.backtesting.metrics import summary as compute_metrics, print_summary
from src.utils.logging_config import get_logger

log = get_logger("backtesting.regime_tester")


STRESS_PERIODS = [
    {"name": "2015-Q3 Flash",       "start": "2015-07-01", "end": "2015-09-30"},
    {"name": "2018-Q4 Selloff",     "start": "2018-10-01", "end": "2018-12-31"},
    {"name": "2020 COVID Crash",    "start": "2020-02-01", "end": "2020-03-31"},
    {"name": "2020 V-Recovery",     "start": "2020-04-01", "end": "2020-08-31"},
    {"name": "2022 Bear Market",    "start": "2022-01-01", "end": "2022-12-31"},
]


@dataclass
class StressResult:
    name: str
    start: str
    end: str
    sharpe: float
    calmar: float
    max_drawdown: float
    annualised_return: float
    n_episodes: int


class RegimeTester:
    """
    Evaluates a trained policy across pre-defined stress periods.
    """

    def __init__(
        self,
        model_path: str,
        data_store,
        n_episodes: int = 20,
        periods: Optional[list[dict]] = None,
    ):
        self.model_path  = model_path
        self.data_store  = data_store
        self.n_episodes  = n_episodes
        self.periods     = periods or STRESS_PERIODS

    # ------------------------------------------------------------------
    def run_all(self, spy_raw_dir: str = "data/raw") -> list[StressResult]:
        """Run policy on all stress periods. Returns list of StressResult."""
        results = []
        for period in self.periods:
            log.info("Stress test: %s (%s → %s)", period["name"], period["start"], period["end"])
            result = self._run_period(period, spy_raw_dir)
            if result is not None:
                results.append(result)
                log.info(
                    "  Sharpe=%.3f  Calmar=%.3f  MDD=%.1f%%  AnnRet=%.1f%%",
                    result.sharpe, result.calmar,
                    result.max_drawdown * 100, result.annualised_return * 100,
                )
        self.print_table(results)
        return results

    # ------------------------------------------------------------------
    def _run_period(self, period: dict, spy_raw_dir: str) -> Optional[StressResult]:
        try:
            runner = PolicyRunner(
                model_path  = self.model_path,
                data_store  = self.data_store,
                start_date  = period["start"],
                end_date    = period["end"],
                deterministic = True,
            )
            spy_returns = _load_spy_returns(
                spy_raw_dir, period["start"], period["end"]
            )
            result = runner.run(n_episodes=self.n_episodes, benchmark_returns=spy_returns)
            m = result["mean_metrics"]

            return StressResult(
                name              = period["name"],
                start             = period["start"],
                end               = period["end"],
                sharpe            = m["sharpe"],
                calmar            = m["calmar"],
                max_drawdown      = m["max_drawdown"],
                annualised_return = m["annualised_return"],
                n_episodes        = self.n_episodes,
            )
        except Exception as e:
            log.warning("Period %s failed: %s", period["name"], e)
            return None

    # ------------------------------------------------------------------
    @staticmethod
    def print_table(results: list[StressResult]) -> None:
        print("\n" + "=" * 72)
        print("  Regime Stress Test Results")
        print("=" * 72)
        print(f"  {'Period':<22}  {'Sharpe':>7}  {'Calmar':>7}  {'MDD':>6}  {'AnnRet':>7}")
        print("  " + "-" * 68)
        for r in results:
            print(
                f"  {r.name:<22}  {r.sharpe:>7.3f}  {r.calmar:>7.3f}  "
                f"{r.max_drawdown:>5.1%}  {r.annualised_return:>6.1%}"
            )
        print("=" * 72 + "\n")

    # ------------------------------------------------------------------
    def save_results(self, results: list[StressResult], out_path: str) -> None:
        import json
        from dataclasses import asdict
        with open(out_path, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        log.info("Stress results saved to %s", out_path)


# ---------------------------------------------------------------------------
def _load_spy_returns(raw_dir: str, start_date: str, end_date: str) -> Optional[np.ndarray]:
    try:
        spy_path = Path(raw_dir) / "market" / "SPY.parquet"
        spy = pd.read_parquet(spy_path)
        spy = spy.loc[start_date:end_date, "close"]
        return spy.pct_change().dropna().values.astype(np.float64)
    except Exception:
        return None
