"""
First-time bulk download of 1h OHLCV for the full 125-stock intraday universe.

Downloads in batches of 20 tickers to avoid yfinance rate limits.
Existing files are skipped unless --force is passed.

Usage:
    python intraday_trader/scripts/download_initial_1h.py
    python intraday_trader/scripts/download_initial_1h.py --force   # re-download all
    python intraday_trader/scripts/download_initial_1h.py --dry-run # list tickers only
    python intraday_trader/scripts/download_initial_1h.py --batch-size 10  # slower but safer
"""
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pyarrow.parquet  # noqa: F401 — must precede torch/SB3 on Windows

import pandas as pd
import yfinance as yf

from intraday_trader.constants import INTRADAY_UNIVERSE, UNIVERSE_FILE
from intraday_trader.data_updater import _filter_market_hours
from src.utils.config_loader import load_config, all_tickers
from src.utils.logging_config import setup_logging, get_logger

log = get_logger("scripts.download_initial_1h")

_YF_PERIOD   = "730d"
_SLEEP_BATCH = 2.0   # seconds between batches


def parse_args():
    p = argparse.ArgumentParser(description="Initial bulk 1h OHLCV download for intraday universe")
    p.add_argument("--config",      default="intraday_trader/config.yaml")
    p.add_argument("--batch-size",  type=int, default=20, help="Tickers per yfinance batch")
    p.add_argument("--force",       action="store_true", help="Re-download even if file exists")
    p.add_argument("--dry-run",     action="store_true", help="Print tickers and exit")
    p.add_argument("--log-file",    default=None)
    return p.parse_args()


def _load_universe(cfg: dict) -> list[str]:
    universe_file = cfg.get("universe", {}).get("file", UNIVERSE_FILE)
    try:
        tickers = all_tickers(universe_file)
        log.info("Loaded %d tickers from %s", len(tickers), universe_file)
        return tickers
    except Exception as e:
        log.warning("Could not load %s (%s) — using fallback list", universe_file, e)
        return list(INTRADAY_UNIVERSE)


def download_batch(
    tickers: list[str],
    ohlcv_dir: Path,
    force: bool,
) -> tuple[int, int]:
    """Download one batch. Returns (n_ok, n_skip)."""
    to_download = []
    for t in tickers:
        p = ohlcv_dir / f"{t}.parquet"
        if p.exists() and not force:
            log.debug("  Skipping %s (exists)", t)
        else:
            to_download.append(t)

    if not to_download:
        return 0, len(tickers)

    try:
        raw = yf.download(
            to_download,
            period=_YF_PERIOD,
            interval="1h",
            auto_adjust=True,
            progress=False,
            multi_level_index=True,
            group_by="ticker",
        )
    except Exception as e:
        log.error("  Batch download failed: %s", e)
        return 0, len(tickers)

    n_ok = 0
    for ticker in to_download:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if ticker in raw.columns.get_level_values(0):
                    df = raw[ticker].copy()
                else:
                    log.warning("  %s missing from batch result", ticker)
                    continue
            else:
                df = raw.copy()

            df.columns = [c.lower() for c in df.columns]
            df = _filter_market_hours(df)
            df.dropna(how="all", inplace=True)

            if df.empty:
                log.warning("  %s: empty after filtering", ticker)
                continue

            out_path = ohlcv_dir / f"{ticker}.parquet"
            df.to_parquet(out_path)
            log.info("  Saved %d bars for %s", len(df), ticker)
            n_ok += 1
        except Exception as e:
            log.error("  Failed saving %s: %s", ticker, e)

    n_skip = len(tickers) - len(to_download)
    return n_ok, n_skip


def main():
    args = parse_args()
    setup_logging(log_file=args.log_file)

    cfg      = load_config(args.config)
    raw_dir  = cfg.get("data", {}).get("raw_dir", "intraday_trader/data/raw")
    tickers  = _load_universe(cfg)

    if args.dry_run:
        print(f"Universe ({len(tickers)} tickers):")
        for i, t in enumerate(tickers, 1):
            print(f"  {i:3d}. {t}")
        return

    ohlcv_dir = Path(raw_dir) / "ohlcv"
    ohlcv_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output dir: %s", ohlcv_dir)

    batch_size = args.batch_size
    batches = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]
    total_ok = 0
    total_skip = 0

    for batch_i, batch in enumerate(batches, 1):
        log.info("=== Batch %d/%d: %s ===", batch_i, len(batches), batch)
        ok, skip = download_batch(batch, ohlcv_dir, force=args.force)
        total_ok   += ok
        total_skip += skip

        if batch_i < len(batches):
            log.info("  Sleeping %.1fs before next batch...", _SLEEP_BATCH)
            time.sleep(_SLEEP_BATCH)

    log.info(
        "=== Download complete: %d downloaded, %d skipped, %d total tickers ===",
        total_ok, total_skip, len(tickers),
    )
    print(f"\nDone: {total_ok} tickers downloaded, {total_skip} already existed.")


if __name__ == "__main__":
    main()
