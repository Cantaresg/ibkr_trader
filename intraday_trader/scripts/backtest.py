"""
Run an intraday backtest against a saved PPO checkpoint.

Usage:
    python intraday_trader/scripts/backtest.py \\
        --checkpoint intraday_trader/checkpoints/intraday_ppo/best/best_model.zip \\
        --start 2024-01-01 --end 2024-06-30 --n-episodes 20

    # Save per-bar results to CSV:
    python intraday_trader/scripts/backtest.py \\
        --checkpoint intraday_trader/checkpoints/intraday_ppo/best/best_model.zip \\
        --start 2024-01-01 --end 2024-06-30 --output-csv results/intraday_backtest.csv

    # Use stochastic policy (non-deterministic):
    python intraday_trader/scripts/backtest.py \\
        --checkpoint ... --stochastic
"""
import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pyarrow.parquet  # noqa: F401 — must precede torch/SB3 on Windows

from src.utils.config_loader import load_config
from src.utils.logging_config import setup_logging, get_logger
from intraday_trader.data_store import IntradayDataStore
from intraday_trader.backtester import IntradayPolicyRunner

log = get_logger("scripts.intraday_backtest")


def parse_args():
    p = argparse.ArgumentParser(description="Intraday DRL policy backtest")
    p.add_argument("--checkpoint", required=True,
                   help="Path to .zip SB3 PPO checkpoint")
    p.add_argument("--start", default="2024-01-01",
                   help="Backtest start date (YYYY-MM-DD, default: 2024-01-01)")
    p.add_argument("--end", default="2024-12-31",
                   help="Backtest end date   (YYYY-MM-DD, default: 2024-12-31)")
    p.add_argument("--n-episodes", type=int, default=20,
                   help="Number of episodes to roll out (default: 20)")
    p.add_argument("--config", default="intraday_trader/config.yaml",
                   help="Config file path")
    p.add_argument("--output-csv", default=None,
                   help="If provided, save per-bar episode data to this CSV path")
    p.add_argument("--output-json", default=None,
                   help="If provided, save mean_metrics dict to this JSON path")
    p.add_argument("--seed", type=int, default=0,
                   help="Base random seed for episode sampling")
    p.add_argument("--algo", default=None, choices=["ppo", "rppo"],
                   help="Force algo class for loading (auto-detected if omitted)")
    p.add_argument("--stochastic", action="store_true",
                   help="Use stochastic policy instead of deterministic (default: deterministic)")
    p.add_argument("--log-file", default=None,
                   help="Optional path to write log output")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(log_file=args.log_file)

    log.info(
        "Intraday backtest: checkpoint=%s  [%s → %s]  n_episodes=%d",
        args.checkpoint, args.start, args.end, args.n_episodes,
    )

    log.info("Loading IntradayDataStore from %s...", args.config)
    data_store = IntradayDataStore(config_path=args.config)

    runner = IntradayPolicyRunner(
        model_path    = args.checkpoint,
        data_store    = data_store,
        start_date    = args.start,
        end_date      = args.end,
        deterministic = not args.stochastic,
        seed          = args.seed,
        algo          = args.algo,
    )

    result = runner.run(n_episodes=args.n_episodes)

    # --- Print summary ---
    runner.print_summary(result["mean_metrics"])

    # --- Optional CSV export ---
    if args.output_csv:
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df = runner.to_dataframe(result)
        df.to_csv(out_path, index=False)
        log.info("Per-bar episode data saved to %s  (%d rows)", out_path, len(df))
        print(f"Per-bar data saved: {out_path}")

    # --- Optional JSON export ---
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result["mean_metrics"], f, indent=2)
        log.info("Mean metrics saved to %s", out_path)
        print(f"Metrics saved:      {out_path}")

    return result["mean_metrics"]


if __name__ == "__main__":
    main()
