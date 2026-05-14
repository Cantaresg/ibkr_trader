"""
Walk-forward validation: 11 overlapping windows across 2010–2025.

Each window: 3 years train / 1 year val / 1 year test, sliding by 1 year.

  Window 1:  train 2010-2012  val 2013  test 2014
  Window 2:  train 2011-2013  val 2014  test 2015
  ...
  Window 11: train 2020-2022  val 2023  test 2024

Per-window sequence (no lookahead):
  1. Train PPO on training split (warm-start from previous window checkpoint)
  2. Evaluate checkpoints on validation split → select best by Sharpe
  3. Report test split metrics (never touched during selection)

Results saved to: data/processed/walk_forward/results.parquet
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.backtesting.metrics import summary as compute_metrics, print_summary
from src.utils.logging_config import get_logger

log = get_logger("backtesting.walk_forward")

RESULTS_DIR = Path("data/processed/walk_forward")


# ---------------------------------------------------------------------------
@dataclass
class WFWindow:
    window_id: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str


@dataclass
class WFResult:
    window_id: int
    train_start: str
    test_end: str
    val_sharpe: float = 0.0
    test_sharpe: float = 0.0
    test_calmar: float = 0.0
    test_max_drawdown: float = 0.0
    test_annualised_return: float = 0.0
    best_checkpoint: str = ""
    n_val_episodes: int = 0
    n_test_episodes: int = 0


def build_windows(
    global_start: str = "2010-01-01",
    global_end:   str = "2024-12-31",
    train_years:  int = 3,
    val_years:    int = 1,
    test_years:   int = 1,
    n_windows:    int = 11,
) -> list[WFWindow]:
    """Generate the list of walk-forward windows."""
    windows = []
    start = pd.Timestamp(global_start)
    for i in range(n_windows):
        yr = i  # slide by 1 year each window
        tr_s = start + pd.DateOffset(years=yr)
        tr_e = tr_s  + pd.DateOffset(years=train_years) - pd.DateOffset(days=1)
        va_s = tr_e  + pd.DateOffset(days=1)
        va_e = va_s  + pd.DateOffset(years=val_years)   - pd.DateOffset(days=1)
        te_s = va_e  + pd.DateOffset(days=1)
        te_e = te_s  + pd.DateOffset(years=test_years)  - pd.DateOffset(days=1)
        if te_e > pd.Timestamp(global_end):
            break
        windows.append(WFWindow(
            window_id   = i + 1,
            train_start = tr_s.strftime("%Y-%m-%d"),
            train_end   = tr_e.strftime("%Y-%m-%d"),
            val_start   = va_s.strftime("%Y-%m-%d"),
            val_end     = va_e.strftime("%Y-%m-%d"),
            test_start  = te_s.strftime("%Y-%m-%d"),
            test_end    = te_e.strftime("%Y-%m-%d"),
        ))
    return windows


# ---------------------------------------------------------------------------
class WalkForwardRunner:
    """
    Runs the full 11-window walk-forward validation.

    Each window:
      1. Trains a PPO agent on the training split
      2. Picks the best checkpoint by validation Sharpe
      3. Evaluates on the test split (held-out)
    """

    def __init__(
        self,
        config: dict,
        data_store,          # MarketDataStore (passed in to avoid re-loading)
        n_eval_episodes: int = 20,
        n_test_episodes: int = 30,
        warm_start: bool = True,
        results_dir: str = str(RESULTS_DIR),
    ):
        self.cfg            = config
        self.data_store     = data_store
        self.n_eval_episodes = n_eval_episodes
        self.n_test_episodes = n_test_episodes
        self.warm_start     = warm_start
        self.results_dir    = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        wf_cfg = config.get("backtesting", {})
        self.windows = build_windows(
            train_years = wf_cfg.get("train_years", 3),
            val_years   = wf_cfg.get("val_years",   1),
            test_years  = wf_cfg.get("test_years",  1),
            n_windows   = wf_cfg.get("walk_forward_windows", 11),
        )
        log.info("Walk-forward: %d windows", len(self.windows))
        for w in self.windows:
            log.info(
                "  Window %2d: train %s–%s | val %s–%s | test %s–%s",
                w.window_id, w.train_start, w.train_end,
                w.val_start, w.val_end,
                w.test_start, w.test_end,
            )

    # ------------------------------------------------------------------
    def run_all(
        self,
        total_timesteps_per_window: int = 2_000_000,
        skip_completed: bool = True,
    ) -> list[WFResult]:
        """
        Run all windows sequentially. Returns list of WFResult.
        Set skip_completed=True to resume an interrupted run.
        """
        results = []
        prev_checkpoint = None

        for window in self.windows:
            result_path = self.results_dir / f"window_{window.window_id:02d}.json"
            if skip_completed and result_path.exists():
                log.info("Window %d: loading cached result", window.window_id)
                with open(result_path) as f:
                    results.append(WFResult(**json.load(f)))
                # Find best checkpoint for warm-starting next window
                prev_checkpoint = results[-1].best_checkpoint or None
                continue

            log.info(
                "=== Window %d/%d: train %s–%s ===",
                window.window_id, len(self.windows),
                window.train_start, window.train_end,
            )
            result = self._run_window(
                window,
                total_timesteps=total_timesteps_per_window,
                warm_start_path=prev_checkpoint if self.warm_start else None,
            )
            results.append(result)

            # Save result
            with open(result_path, "w") as f:
                json.dump(asdict(result), f, indent=2)
            log.info(
                "Window %d done: val_sharpe=%.3f  test_sharpe=%.3f  test_mdd=%.1f%%",
                window.window_id, result.val_sharpe, result.test_sharpe,
                result.test_max_drawdown * 100,
            )

            # Best checkpoint from this window for next window's warm start
            prev_checkpoint = result.best_checkpoint or None

        self._save_summary(results)
        return results

    # ------------------------------------------------------------------
    def _run_window(
        self,
        window: WFWindow,
        total_timesteps: int,
        warm_start_path: Optional[str],
    ) -> WFResult:
        # Lazy import to keep top-level clean
        import pyarrow.parquet  # noqa: F401
        from src.training.trainer import Trainer
        from src.backtesting.vectorbt_runner import PolicyRunner

        run_name = f"wf_window_{window.window_id:02d}"

        # Override dates in config for this window
        cfg = dict(self.cfg)
        cfg["training_start"] = window.train_start
        cfg["training_end"]   = window.train_end
        cfg["eval_start"]     = window.val_start
        cfg["eval_end"]       = window.val_end
        cfg["config_path"]    = self.cfg.get("config_path", "config/config.yaml")

        # --- Train ---
        trainer = Trainer(cfg, run_name=run_name, warm_start_path=warm_start_path)
        trainer.train(total_timesteps=total_timesteps)

        # --- Pick best checkpoint by val Sharpe ---
        ckpt_dir = Path(self.cfg["training"]["checkpoint_dir"]) / run_name
        best_ckpt, best_val_sharpe = self._select_best_checkpoint(
            ckpt_dir, window, n_episodes=self.n_eval_episodes,
        )

        # --- Test ---
        test_runner = PolicyRunner(
            model_path  = best_ckpt,
            data_store  = self.data_store,
            start_date  = window.test_start,
            end_date    = window.test_end,
            deterministic = True,
        )
        # SPY returns as benchmark
        spy_returns = self._get_spy_returns(window.test_start, window.test_end)
        test_result = test_runner.run(
            n_episodes       = self.n_test_episodes,
            benchmark_returns = spy_returns,
        )
        m = test_result["mean_metrics"]

        return WFResult(
            window_id              = window.window_id,
            train_start            = window.train_start,
            test_end               = window.test_end,
            val_sharpe             = best_val_sharpe,
            test_sharpe            = m["sharpe"],
            test_calmar            = m["calmar"],
            test_max_drawdown      = m["max_drawdown"],
            test_annualised_return = m["annualised_return"],
            best_checkpoint        = best_ckpt,
            n_val_episodes         = self.n_eval_episodes,
            n_test_episodes        = self.n_test_episodes,
        )

    # ------------------------------------------------------------------
    def _select_best_checkpoint(
        self,
        ckpt_dir: Path,
        window: WFWindow,
        n_episodes: int,
    ) -> tuple[str, float]:
        """
        Evaluate all saved checkpoints on the validation split and return
        (path_to_best, val_sharpe).
        """
        from src.backtesting.vectorbt_runner import PolicyRunner

        checkpoints = sorted(ckpt_dir.glob("ppo_*_steps.zip"))
        # Always include the best model saved by SharpeEvalCallback
        best_cb_path = ckpt_dir / "best" / "best_model.zip"
        if best_cb_path.exists():
            checkpoints.append(best_cb_path)

        if not checkpoints:
            final = ckpt_dir / "final_model.zip"
            return str(final), 0.0

        best_path   = str(checkpoints[-1])
        best_sharpe = -np.inf

        for ckpt in checkpoints:
            try:
                runner = PolicyRunner(
                    model_path  = str(ckpt),
                    data_store  = self.data_store,
                    start_date  = window.val_start,
                    end_date    = window.val_end,
                    deterministic = True,
                )
                result = runner.run(n_episodes=n_episodes)
                sharpe = result["mean_metrics"]["sharpe"]
                log.info("  Checkpoint %s: val_sharpe=%.3f", ckpt.name, sharpe)
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_path   = str(ckpt)
            except Exception as e:
                log.warning("  Checkpoint %s failed: %s", ckpt.name, e)

        return best_path, float(best_sharpe)

    # ------------------------------------------------------------------
    def _get_spy_returns(self, start_date: str, end_date: str) -> np.ndarray:
        """Load SPY daily returns for the given date range."""
        try:
            spy_path = Path(self.cfg["data"]["raw_dir"]) / "market" / "SPY.parquet"
            spy = pd.read_parquet(spy_path)
            spy = spy.loc[start_date:end_date, "close"]
            returns = spy.pct_change().dropna().values.astype(np.float64)
            return returns
        except Exception:
            return None

    # ------------------------------------------------------------------
    def _save_summary(self, results: list[WFResult]) -> None:
        """Save summary table to parquet and print to console."""
        rows = [asdict(r) for r in results]
        df = pd.DataFrame(rows)
        out = self.results_dir / "results.parquet"
        df.to_parquet(out, index=False)
        log.info("Walk-forward summary saved to %s", out)
        self.print_summary_table(results)

    # ------------------------------------------------------------------
    @staticmethod
    def print_summary_table(results: list[WFResult]) -> None:
        print("\n" + "=" * 80)
        print(f"  Walk-Forward Results ({len(results)} windows)")
        print("=" * 80)
        print(f"  {'Win':>3}  {'Period':>20}  {'ValSharpe':>9}  {'TestSharpe':>10}  {'Calmar':>6}  {'MDD':>6}  {'AnnRet':>7}")
        print("  " + "-" * 76)
        sharpes = []
        for r in results:
            period = f"{r.train_start[:4]}-{r.test_end[:4]}"
            print(
                f"  {r.window_id:>3}  {period:>20}  {r.val_sharpe:>9.3f}  "
                f"{r.test_sharpe:>10.3f}  {r.test_calmar:>6.3f}  "
                f"{r.test_max_drawdown:>5.1%}  {r.test_annualised_return:>6.1%}"
            )
            sharpes.append(r.test_sharpe)
        print("  " + "-" * 76)
        print(f"  {'Mean':>3}  {'':>20}  {'':>9}  {np.mean(sharpes):>10.3f}")
        print("=" * 80 + "\n")
