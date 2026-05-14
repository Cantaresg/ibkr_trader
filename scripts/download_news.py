"""
Daily news download + FinBERT sentiment scoring.

Usage:
    python scripts/download_news.py
    python scripts/download_news.py --start-date 2015-01-01
    python scripts/download_news.py --tickers AAPL MSFT NVDA   # subset for testing
    python scripts/download_news.py --no-score                  # download only, skip FinBERT
"""
import pyarrow.parquet  # Windows DLL fix: must precede transformers/torch

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import finbert_scorer, news_downloader, ohlcv_store
from src.utils.config_loader import all_tickers, load_config
from src.utils.logging_config import get_logger, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Download news and score sentiment with FinBERT")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--start-date", default="2015-01-01",
                        help="Earliest date to fetch on first download (default: 2015-01-01). "
                             "Ignored for tickers that already have downloaded data.")
    parser.add_argument("--tickers", nargs="+",
                        help="Override ticker list (useful for quick testing)")
    parser.add_argument("--no-score", action="store_true",
                        help="Download articles only; skip FinBERT inference")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Device for FinBERT inference (default: cuda)")
    args = parser.parse_args()

    setup_logging("INFO")
    log = get_logger("download_news")

    cfg = load_config(args.config)
    raw_dir = cfg["data"]["raw_dir"]
    news_raw_dir = cfg["data"].get("news_raw_dir", f"{raw_dir}/news")

    api_key = cfg["data"].get("finnhub_key", "")
    if api_key.startswith("${"):
        api_key = ""
    if not api_key:
        log.error("FINNHUB_KEY not set in .env — cannot download news. Exiting.")
        sys.exit(1)

    tickers = args.tickers or all_tickers(cfg["data"]["universe_file"])
    log.info("Universe: %d tickers | news_raw_dir: %s", len(tickers), news_raw_dir)

    start_date = date.fromisoformat(args.start_date)

    # 1a — Download per-stock articles
    log.info("=== Downloading per-stock news articles (from %s) ===", start_date)
    news_downloader.download_all(
        tickers,
        news_raw_dir,
        api_key,
        start_date=start_date,
    )

    # 1b — Download global macro ETF news
    log.info("=== Downloading global macro news ===")
    news_downloader.download_global_news(news_raw_dir, api_key, start_date=start_date)

    if args.no_score:
        log.info("--no-score set. Skipping FinBERT. Done.")
        return

    # 2 — FinBERT inference + daily aggregation
    log.info("=== Loading FinBERT model ===")
    model = finbert_scorer.load_model(device=args.device)

    # 2a — Per-stock sentiment
    log.info("=== Scoring per-stock articles ===")
    scored = skipped = 0
    for i, ticker in enumerate(tickers, 1):
        ohlcv = ohlcv_store.load(raw_dir, ticker)
        if ohlcv is None or len(ohlcv) == 0:
            skipped += 1
            continue
        result = finbert_scorer.build_ticker_sentiment(
            ticker, news_raw_dir, raw_dir, ohlcv.index, model
        )
        if result is not None:
            scored += 1
        if i % 25 == 0:
            log.info("  Progress: %d/%d tickers processed", i, len(tickers))
    log.info("Per-stock: %d scored, %d skipped (no OHLCV)", scored, skipped)

    # 2b — Global sentiment (uses SPY calendar as index anchor)
    log.info("=== Scoring global macro articles ===")
    spy_ohlcv = ohlcv_store.load(raw_dir, "SPY")
    if spy_ohlcv is not None:
        finbert_scorer.build_global_sentiment(news_raw_dir, spy_ohlcv.index, model)
    else:
        log.warning("SPY OHLCV not found — skipping global sentiment")

    log.info("=== Done ===")
    log.info("Next step: rebuild feature store with  python scripts/build_features.py")


if __name__ == "__main__":
    main()
