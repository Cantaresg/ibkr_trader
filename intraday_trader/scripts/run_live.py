"""
Intraday live trading entry point.

Connects to IBKR TWS (paper or live), runs a data refresh before market open,
then executes hourly intraday decisions throughout the trading session.

Usage:
    # Paper trading (default, port 7497):
    python intraday_trader/scripts/run_live.py \\
        --checkpoint intraday_trader/checkpoints/intraday_ppo/best/best_model.zip

    # Live trading (port 7496):
    python intraday_trader/scripts/run_live.py \\
        --checkpoint intraday_trader/checkpoints/intraday_ppo/best/best_model.zip \\
        --live

    # Single session (today only, for testing):
    python intraday_trader/scripts/run_live.py \\
        --checkpoint intraday_trader/checkpoints/intraday_ppo/best/best_model.zip \\
        --once
"""
import sys
import time
import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pyarrow.parquet  # noqa: F401  — must precede torch/SB3

import pytz

from src.utils.config_loader import load_config
from src.utils.logging_config import setup_logging, get_logger
from src.live_trading.broker import IBBroker
from src.live_trading.position_manager import PositionManager
from intraday_trader.constants import INTRADAY_UNIVERSE
from intraday_trader.executor import IntradayExecutor, IntradayRiskGuard
from intraday_trader.inference import IntradayInferenceEngine
from intraday_trader.data_updater import IntradayDataUpdater

log = get_logger("scripts.intraday_live")
ET  = pytz.timezone("America/New_York")

# Static sector map for the fixed intraday universe
_INTRADAY_SECTOR_MAP = {
    "AAPL":  "technology",
    "MSFT":  "technology",
    "NVDA":  "technology",
    "AMD":   "technology",
    "AMZN":  "consumer_discretionary",
    "META":  "communication_services",
    "GOOGL": "communication_services",
    "TSLA":  "consumer_discretionary",
    "JPM":   "financials",
    "BAC":   "financials",
    "SPY":   "etf",
    "QQQ":   "etf",
}


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Run intraday DRL live trading")
    p.add_argument("--checkpoint", required=True, help="Path to .zip intraday PPO checkpoint")
    p.add_argument("--config",     default="intraday_trader/config.yaml")
    p.add_argument("--live",       action="store_true",
                   help="Use live TWS port (7496). Default: paper port (7497).")
    p.add_argument("--once",       action="store_true",
                   help="Run one session today and exit (no daily loop)")
    p.add_argument("--capital",    type=float, default=None,
                   help="Capital to deploy in USD (overrides config environment.initial_capital)")
    p.add_argument("--skip-data-update", action="store_true",
                   help="Skip morning data refresh (use when data is already current)")
    p.add_argument("--log-file",   default="intraday_trader/logs/intraday_live.log")
    return p.parse_args()


# ---------------------------------------------------------------------------
def build_executor(args, cfg) -> IntradayExecutor:
    port = 7496 if args.live else 7497
    mode = "LIVE" if args.live else "PAPER"

    if args.live:
        confirm = input(
            "\n*** WARNING: LIVE intraday trading (real money) ***\n"
            "Type 'yes' to confirm: "
        ).strip().lower()
        if confirm != "yes":
            print("Aborted.")
            sys.exit(0)

    print(f"Mode: {mode}  |  Port: {port}  |  Checkpoint: {args.checkpoint}")

    ibkr_cfg = cfg.get("ibkr", {})
    env_cfg  = cfg.get("environment", {})

    initial_capital = args.capital or env_cfg.get("initial_capital", 5_000.0)
    universe        = cfg.get("universe", {}).get("tickers", INTRADAY_UNIVERSE)

    broker = IBBroker(
        host      = "127.0.0.1",
        port      = port,
        client_id = ibkr_cfg.get("client_id", 2),
    )

    inference = IntradayInferenceEngine(
        checkpoint_path = args.checkpoint,
        universe        = universe,
        initial_capital = initial_capital,
    )

    risk_cfg = ibkr_cfg.get("risk", {
        "max_position_weight":       0.20,
        "max_sector_weight":         0.60,
        "daily_loss_limit":         -0.015,
        "drawdown_halt_threshold":  -0.05,
        "drawdown_resume_threshold":-0.02,
    })
    risk_guard = IntradayRiskGuard(risk_cfg=risk_cfg, ticker_to_sector=_INTRADAY_SECTOR_MAP)

    exec_cfg = ibkr_cfg.get("execution", {})
    pos_mgr  = PositionManager(execution_cfg=exec_cfg)

    bar_times_raw = exec_cfg.get("bar_times_et", [(9, 35), (10, 35), (11, 35), (12, 35), (13, 35), (14, 35)])
    bar_times     = [tuple(bt) for bt in bar_times_raw]
    eod_raw       = (exec_cfg.get("eod_liquidate_hour_et", 15), exec_cfg.get("eod_liquidate_minute_et", 45))

    executor = IntradayExecutor(
        broker           = broker,
        inference        = inference,
        position_mgr     = pos_mgr,
        risk_guard       = risk_guard,
        initial_capital  = initial_capital,
        universe         = universe,
        order_timeout_s  = exec_cfg.get("order_timeout_seconds", 45),
        bar_times_et     = bar_times,
        eod_liquidate_et = eod_raw,
    )

    return executor


# ---------------------------------------------------------------------------
def _is_market_day(d: date) -> bool:
    return d.weekday() < 5


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
        print(f"  Waiting {delta/3600:.1f}h until {hour:02d}:{minute:02d} ET")
        time.sleep(min(delta - 60, 1800))


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    setup_logging(log_file=args.log_file)

    cfg     = load_config(args.config)
    updater = IntradayDataUpdater(config_path=args.config)
    executor = build_executor(args, cfg)

    results_dir = Path("intraday_trader/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    if args.once:
        print("Running single intraday session...")
        if not args.skip_data_update:
            print("Refreshing intraday data...")
            updater.run()
        executor.broker.connect()
        summary = executor.run_session(trade_date=date.today())
        print(json.dumps(summary, indent=2, default=str))
        return

    # ------------------------------------------------------------------
    print("\nIntraday live trading loop started. Press Ctrl+C to stop.\n")

    while True:
        today = date.today()

        if not _is_market_day(today):
            next_day = today + timedelta(days=1)
            while not _is_market_day(next_day):
                next_day += timedelta(days=1)
            print(f"Weekend — next trading day: {next_day}")
            time.sleep(3600)
            continue

        _wait_until_et(6, 30)
        if not args.skip_data_update:
            print(f"\n[{datetime.now(ET).strftime('%H:%M ET')}] Morning data refresh...")
            try:
                updater.run(as_of=today)
            except Exception as e:
                log.error("Data refresh failed: %s — continuing with existing data", e)

        _wait_until_et(9, 30)
        actual_date = date.today()
        if not _is_market_day(actual_date):
            continue

        print(f"\n[{datetime.now(ET).strftime('%H:%M ET')}] Starting intraday session for {actual_date}...")
        try:
            summary = executor.run_session(trade_date=actual_date)
        except Exception as e:
            log.error("Session failed: %s", e)
            summary = {"date": str(actual_date), "status": "error", "error": str(e)}
            try:
                executor.broker.disconnect()
            except Exception:
                pass

        result_file = results_dir / f"{actual_date}.json"
        result_file.write_text(json.dumps(summary, indent=2, default=str))
        print(f"  Session done. PnL: {summary.get('day_pnl_pct', 'N/A'):.2f}%")

        time.sleep(3600)


if __name__ == "__main__":
    main()
