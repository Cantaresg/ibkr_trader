"""
Run backtesting: walk-forward validation or regime stress tests.

Usage:
  # Walk-forward (11 windows, ~37 min per window × 11 = ~7 hours)
  python scripts/run_backtest.py walk-forward

  # Walk-forward, resume from a partially completed run
  python scripts/run_backtest.py walk-forward --resume

  # Walk-forward with fewer timesteps per window (faster, less accurate)
  python scripts/run_backtest.py walk-forward --timesteps 1000000

  # Regime stress tests on a specific model
  python scripts/run_backtest.py stress --model checkpoints/phase1_baseline/final_model.zip

  # Quick evaluation on a date range
  python scripts/run_backtest.py eval --model checkpoints/phase1_baseline/final_model.zip \\
      --start 2019-01-01 --end 2019-12-31 --episodes 30
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pyarrow.parquet  # noqa: F401 — must precede torch

from src.utils.config_loader import load_config
from src.utils.logging_config import setup_logging
from src.environment.data_store import MarketDataStore
from src.backtesting.metrics import print_summary


def cmd_walk_forward(args, cfg, ds):
    from src.backtesting.walk_forward import WalkForwardRunner
    runner = WalkForwardRunner(
        config          = cfg,
        data_store      = ds,
        n_eval_episodes = args.eval_episodes,
        n_test_episodes = args.test_episodes,
        warm_start      = not args.no_warm_start,
    )
    results = runner.run_all(
        total_timesteps_per_window = args.timesteps,
        skip_completed             = args.resume,
    )
    return results


def cmd_stress(args, cfg, ds):
    from src.backtesting.regime_tester import RegimeTester
    tester = RegimeTester(
        model_path  = args.model,
        data_store  = ds,
        n_episodes  = args.episodes,
    )
    results = tester.run_all(spy_raw_dir=cfg["data"]["raw_dir"])
    out = Path("data/processed/walk_forward/stress_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    tester.save_results(results, str(out))
    return results


def cmd_eval(args, cfg, ds):
    from src.backtesting.vectorbt_runner import PolicyRunner
    import numpy as np, pandas as pd
    runner = PolicyRunner(
        model_path  = args.model,
        data_store  = ds,
        start_date  = args.start,
        end_date    = args.end,
        deterministic = not args.stochastic,
    )
    spy_path = Path(cfg["data"]["raw_dir"]) / "market" / "SPY.parquet"
    spy_returns = None
    if spy_path.exists():
        spy = pd.read_parquet(spy_path).loc[args.start:args.end, "close"]
        spy_returns = spy.pct_change().dropna().values.astype(np.float64)

    result = runner.run(n_episodes=args.episodes, benchmark_returns=spy_returns)
    print_summary(result["mean_metrics"])
    return result


def parse_args():
    p = argparse.ArgumentParser(description="IBKR DRL Backtesting")
    p.add_argument("--config", default="config/config.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    # walk-forward
    wf = sub.add_parser("walk-forward", help="Run 11-window walk-forward validation")
    wf.add_argument("--timesteps",      type=int, default=2_000_000)
    wf.add_argument("--eval-episodes",  type=int, default=20)
    wf.add_argument("--test-episodes",  type=int, default=30)
    wf.add_argument("--resume",         action="store_true", help="Skip already-completed windows")
    wf.add_argument("--no-warm-start",  action="store_true")

    # stress tests
    st = sub.add_parser("stress", help="Regime stress tests on a trained model")
    st.add_argument("--model",    required=True)
    st.add_argument("--episodes", type=int, default=20)

    # single eval
    ev = sub.add_parser("eval", help="Evaluate a model on a date range")
    ev.add_argument("--model",      required=True)
    ev.add_argument("--start",      default="2019-01-01")
    ev.add_argument("--end",        default="2019-12-31")
    ev.add_argument("--episodes",   type=int, default=30)
    ev.add_argument("--stochastic", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    setup_logging()

    cfg = load_config(args.config)
    cfg["config_path"] = args.config

    print("Loading MarketDataStore...", flush=True)
    ds = MarketDataStore(config_path=args.config)

    if args.command == "walk-forward":
        cmd_walk_forward(args, cfg, ds)
    elif args.command == "stress":
        cmd_stress(args, cfg, ds)
    elif args.command == "eval":
        cmd_eval(args, cfg, ds)


if __name__ == "__main__":
    main()
