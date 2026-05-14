"""
Convert GDELT raw tone timelines into sentiment features, then rebuild
the full feature store so the agent sees real historical news signal.

Steps:
  1. Per-ticker: load GDELT tone → sentiment_store (data/raw/sentiment/)
  2. Global indices: pool 16 macro topics → data/raw/global_sentiment.parquet
  3. Rebuild market features cache (includes global sentiment)
  4. Rebuild per-ticker feature parquets (overwrite=True)

Run this after download_gdelt.py has finished.

Usage:
    python scripts/build_gdelt_features.py
    python scripts/build_gdelt_features.py --tickers AAPL MSFT NVDA
    python scripts/build_gdelt_features.py --no-rebuild   # sentiment only, skip pipeline
"""
import pyarrow.parquet  # Windows DLL fix: must precede torch/transformers

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import ohlcv_store
from src.data.gdelt_store import build_global_sentiment, build_ticker_sentiment
from src.data.market_data import load_all as load_market
from src.features import market_features
from src.features.pipeline import build_all
from src.utils.config_loader import all_tickers, load_config
from src.utils.logging_config import get_logger, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build sentiment features from GDELT data, then rebuild feature store"
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--tickers", nargs="+", help="Override ticker list")
    parser.add_argument(
        "--gdelt-dir", default=None,
        help="Override GDELT raw dir (default: gdelt_raw_dir from config)",
    )
    parser.add_argument(
        "--no-rebuild", action="store_true",
        help="Build sentiment store only; skip feature pipeline rebuild",
    )
    args = parser.parse_args()

    setup_logging("INFO", "logs/build_gdelt_features.log")
    log = get_logger("build_gdelt_features")

    cfg = load_config(args.config)
    raw_dir = cfg["data"]["raw_dir"]
    processed_dir = cfg["data"]["processed_dir"]
    gdelt_dir = args.gdelt_dir or cfg["data"].get("gdelt_raw_dir", "G:/My Drive/ibkr_gdelt_raw")
    norm_window = cfg["features"]["normalization_window"]

    tickers = args.tickers or all_tickers(cfg["data"]["universe_file"])
    log.info("GDELT feature build: %d tickers | gdelt_dir: %s", len(tickers), gdelt_dir)

    # --- Step 1: per-stock sentiment ---
    log.info("=== Building per-stock GDELT sentiment ===")
    built = skipped = missing = 0
    for i, ticker in enumerate(tickers, 1):
        ohlcv = ohlcv_store.load(raw_dir, ticker)
        if ohlcv is None or len(ohlcv) == 0:
            skipped += 1
            continue
        result = build_ticker_sentiment(ticker, gdelt_dir, raw_dir, ohlcv.index)
        if result is not None:
            built += 1
        else:
            missing += 1
        if i % 25 == 0:
            log.info("  Progress: %d/%d tickers", i, len(tickers))
    log.info(
        "Per-stock: %d built, %d no GDELT data yet, %d skipped (no OHLCV)",
        built, missing, skipped,
    )

    # --- Step 2: global index sentiment ---
    log.info("=== Building global index sentiment ===")
    mkt = load_market(raw_dir)
    spy_mkt = mkt.get("SPY")
    if spy_mkt is not None:
        build_global_sentiment(gdelt_dir, spy_mkt.index)
    else:
        log.warning("SPY market data not found — skipping global sentiment")

    if args.no_rebuild:
        log.info("--no-rebuild set. Done.")
        return

    # --- Step 3: rebuild market features (includes global sentiment columns 7-9) ---
    log.info("=== Rebuilding market features cache ===")
    market_features.build(tickers=tickers, raw_dir=raw_dir, cache=True, overwrite=True)

    # --- Step 4: rebuild per-ticker feature parquets ---
    log.info("=== Rebuilding per-ticker feature store (overwrite=True) ===")
    build_all(
        tickers=tickers,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        norm_window=norm_window,
        overwrite=True,
    )

    log.info("=== Done. Features now include GDELT sentiment. ===")
    log.info("Next: re-run the syn_01/02/03 experiments with the new features, or")
    log.info("      resume training: python scripts/train_agent.py")


if __name__ == "__main__":
    main()
