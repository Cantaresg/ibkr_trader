"""
Download fresh 1h OHLCV data and rebuild intraday features.

Run once daily before market open (e.g. 6 AM ET) to ensure the DataStore
is up-to-date before training or live trading begins.

Usage:
    # Default config:
    python intraday_trader/scripts/update_data.py

    # Force full rebuild of all cached features:
    python intraday_trader/scripts/update_data.py --force-rebuild

    # Custom config:
    python intraday_trader/scripts/update_data.py --config path/to/config.yaml
"""
import sys
import argparse
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logging_config import setup_logging, get_logger
from intraday_trader.data_updater import IntradayDataUpdater

log = get_logger("scripts.update_intraday_data")


def parse_args():
    p = argparse.ArgumentParser(description="Download and rebuild intraday data + features")
    p.add_argument("--config", default="intraday_trader/config.yaml",
                   help="Config file path (default: intraday_trader/config.yaml)")
    p.add_argument("--force-rebuild", action="store_true",
                   help="Rebuild all feature parquets even if they already exist")
    p.add_argument("--as-of", default=None,
                   help="Override 'today' for the update (YYYY-MM-DD). Mainly for testing.")
    p.add_argument("--log-file", default=None,
                   help="Optional path to write log output")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(log_file=args.log_file)

    as_of: date | None = None
    if args.as_of:
        as_of = date.fromisoformat(args.as_of)

    log.info("Starting intraday data update (config=%s, force_rebuild=%s, as_of=%s)",
             args.config, args.force_rebuild, as_of or "today")

    updater = IntradayDataUpdater(config_path=args.config)

    if args.force_rebuild:
        # Full rebuild: clear processed feature cache then run normally
        import shutil
        from src.utils.config_loader import load_config
        cfg      = load_config(args.config)
        proc_dir = cfg.get("data", {}).get("processed_dir", "intraday_trader/data/processed")
        feat_dir = Path(proc_dir) / "features"
        mkt_file = Path(proc_dir) / "market_features_1h.parquet"

        if feat_dir.exists():
            shutil.rmtree(feat_dir)
            log.info("Cleared feature cache at %s", feat_dir)
        if mkt_file.exists():
            mkt_file.unlink()
            log.info("Cleared market feature cache at %s", mkt_file)

    updater.run(as_of=as_of)
    log.info("Data update complete")
    print("Intraday data update complete.")


if __name__ == "__main__":
    main()
