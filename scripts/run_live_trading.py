"""
Live trading entry point.

Connects to IBKR (paper or live TWS), runs EOD data refresh after market close,
then executes one rebalance cycle at 9:35am ET each trading day.

Usage:
    # Paper trading (default, port 7497):
    python scripts/run_live_trading.py \\
        --checkpoint checkpoints/rppo_full_syn50/best/best_model.zip

    # Live trading (port 7496):
    python scripts/run_live_trading.py \\
        --checkpoint checkpoints/rppo_full_syn50/best/best_model.zip \\
        --live

    # Single one-shot rebalance (no scheduling loop, for testing):
    python scripts/run_live_trading.py \\
        --checkpoint checkpoints/rppo_full_syn50/best/best_model.zip \\
        --once
"""
import sys
import time
import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pyarrow.parquet  # noqa: F401  — must precede torch/SB3
import pytz
import pandas as pd

from src.utils.config_loader import load_config, ticker_to_sector
from src.utils.logging_config import setup_logging
from src.live_trading.broker import IBBroker
from src.live_trading.inference import LiveInferenceEngine
from src.live_trading.live_feature_builder import LiveFeatureBuilder
from src.live_trading.position_manager import PositionManager
from src.live_trading.risk_guard import RiskGuard
from src.live_trading.executor import DailyExecutor
from src.live_trading.data_updater import DailyDataUpdater

ET = pytz.timezone("America/New_York")


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",      required=True,
                   help="Path to .zip PPO checkpoint")
    p.add_argument("--config",          default="config/config.yaml")
    p.add_argument("--universe",        default=None,
                   help="Comma-separated tickers, or path to a .txt/.csv file "
                        "(one ticker per line). Defaults to training universe.")
    p.add_argument("--live",            action="store_true",
                   help="Use live TWS port (7496). Default: paper port (7497).")
    p.add_argument("--once",            action="store_true",
                   help="Run one rebalance cycle immediately and exit")
    p.add_argument("--capital",          type=float, default=None,
                   help="Capital to deploy in USD (overrides config initial_capital). "
                        "Default: 2000.0")
    p.add_argument("--skip-data-update", action="store_true",
                   help="Skip EOD data refresh (use when data is already current)")
    p.add_argument("--log-file",        default="logs/live_trading.log")
    return p.parse_args()


def load_universe(universe_arg: str | None, cfg: dict) -> list[str]:
    """
    Resolve --universe to a flat ticker list.
    If None, falls back to the training universe from config.
    Accepts: comma-separated string, path to .txt/.csv (one ticker per line).
    """
    from src.utils.config_loader import all_tickers
    if universe_arg is None:
        tickers = all_tickers(cfg["data"]["universe_file"])
        log.info("Using training universe: %d tickers", len(tickers))
        return tickers

    p = Path(universe_arg)
    if p.exists():
        lines = p.read_text().splitlines()
        tickers = [l.strip().upper() for l in lines if l.strip() and not l.startswith("#")]
    else:
        tickers = [t.strip().upper() for t in universe_arg.split(",") if t.strip()]

    log.info("Custom universe: %d tickers", len(tickers))
    return tickers


# ---------------------------------------------------------------------------
def build_executor(args, cfg, universe: list[str]) -> DailyExecutor:
    port = 7496 if args.live else 7497
    mode = "LIVE" if args.live else "PAPER"

    if args.live:
        confirm = input(
            "\n*** WARNING: LIVE trading mode selected (real money) ***\n"
            "Type 'yes' to confirm: "
        ).strip().lower()
        if confirm != "yes":
            print("Aborted.")
            sys.exit(0)

    print(f"Mode: {mode}  |  Port: {port}  |  Checkpoint: {args.checkpoint}")

    ibkr_cfg   = cfg["ibkr"]
    broker     = IBBroker(
        host=ibkr_cfg["host"],
        port=port,
        client_id=ibkr_cfg["client_id"],
    )

    print("Initialising LiveFeatureBuilder (any-universe live inference)...")
    feature_builder = LiveFeatureBuilder(config_path=args.config)

    inference = LiveInferenceEngine(
        checkpoint_path=args.checkpoint,
        feature_builder=feature_builder,
        lookback=cfg["features"]["lookback_window"],
    )

    t2s = ticker_to_sector(cfg["data"]["universe_file"])

    risk_guard = RiskGuard(
        risk_cfg=ibkr_cfg["risk"],
        ticker_to_sector=t2s,
    )

    pos_mgr = PositionManager(
        execution_cfg=ibkr_cfg["execution"],
    )

    executor = DailyExecutor(
        broker=broker,
        inference=inference,
        position_mgr=pos_mgr,
        risk_guard=risk_guard,
        initial_capital=cfg["environment"]["initial_capital"],
        universe=universe,
        order_timeout_s=ibkr_cfg["execution"]["order_timeout_seconds"],
    )

    return executor


# ---------------------------------------------------------------------------
def _wait_until_et(hour: int, minute: int) -> None:
    """Block until the next occurrence of HH:MM Eastern time."""
    while True:
        now_et = datetime.now(ET)
        target = now_et.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if now_et >= target:
            target += timedelta(days=1)
        delta = (target - now_et).total_seconds()
        if delta <= 60:
            time.sleep(delta)
            return
        # Sleep in chunks and log progress every 30 min
        print(f"  Waiting {delta/3600:.1f}h until {hour:02d}:{minute:02d} ET "
              f"({target.strftime('%Y-%m-%d %H:%M %Z')})")
        time.sleep(min(delta - 60, 1800))


def _is_market_day(d: date) -> bool:
    """Very rough check — skips weekends. Use a proper calendar for holidays."""
    return d.weekday() < 5   # Mon-Fri


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    setup_logging(log_file=args.log_file)
    cfg  = load_config(args.config)
    if args.capital is not None:
        cfg["environment"]["initial_capital"] = args.capital

    ex_cfg   = cfg["ibkr"]["execution"]
    run_hour = ex_cfg.get("start_hour_et", 9)
    run_min  = ex_cfg.get("start_minute_et", 35)

    results_dir = Path("results/live")
    results_dir.mkdir(parents=True, exist_ok=True)

    universe = load_universe(args.universe, cfg)
    updater  = DailyDataUpdater(config_path=args.config)
    executor = build_executor(args, cfg, universe)

    # ------------------------------------------------------------------
    if args.once:
        print("Running single rebalance cycle...")
        executor.broker.connect()
        summary = executor.run(trade_date=date.today())
        executor.broker.disconnect()
        print(json.dumps(summary, indent=2))
        return

    # ------------------------------------------------------------------
    # Daily loop
    print(f"\nLive trading loop started. Rebalance at {run_hour:02d}:{run_min:02d} ET each trading day.")
    print("Press Ctrl+C to stop.\n")

    while True:
        today = date.today()

        if not _is_market_day(today):
            # Sleep until next weekday
            next_day = today + timedelta(days=1)
            while not _is_market_day(next_day):
                next_day += timedelta(days=1)
            print(f"Weekend/holiday — next trading day: {next_day}")
            time.sleep(3600)
            continue

        # EOD data refresh (runs after previous day's close)
        if not args.skip_data_update:
            print(f"\n[{datetime.now(ET).strftime('%H:%M ET')}] Starting EOD data refresh...")
            try:
                updater.run(as_of=today)
            except Exception as e:
                print(f"  Data refresh failed: {e} — continuing with existing data")

        # Wait for 9:35am ET
        _wait_until_et(run_hour, run_min)
        actual_date = date.today()

        if not _is_market_day(actual_date):
            continue

        print(f"\n[{datetime.now(ET).strftime('%H:%M ET')}] Running rebalance for {actual_date}...")
        try:
            executor.broker.connect()
            summary = executor.run(trade_date=actual_date)
            executor.broker.disconnect()
        except Exception as e:
            print(f"  Rebalance failed: {e}")
            summary = {"date": str(actual_date), "status": "error", "error": str(e)}
            try:
                executor.broker.disconnect()
            except Exception:
                pass

        # Persist daily result
        result_file = results_dir / f"{actual_date}.json"
        result_file.write_text(json.dumps(summary, indent=2))
        print(f"  Done. Summary: {summary}")

        # Sleep until tomorrow before starting the next EOD update
        time.sleep(3600)


if __name__ == "__main__":
    main()
