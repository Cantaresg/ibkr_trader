"""
Backfill 1h OHLCV history for the intraday universe using IBKR historical data.

IBKR provides up to ~10 years of 1h bars — far beyond yfinance's 730-day limit.
This script fetches the gap between 2015-01-01 and the start of each ticker's
existing yfinance data, then prepends it to the parquet file.

Existing data is never overwritten — only the missing historical range is added.

Prerequisites:
    - TWS or IB Gateway running on this machine (paper or live, port 7497 / 7496)
    - API access enabled in TWS: Configure → API → Settings → Enable Socket Clients
    - 127.0.0.1 in the trusted IP list

Usage:
    python scripts/backfill_1h_ibkr.py
    python scripts/backfill_1h_ibkr.py --port 7496          # live TWS
    python scripts/backfill_1h_ibkr.py --dry-run            # show date gaps only
    python scripts/backfill_1h_ibkr.py --tickers AAPL MSFT  # subset
    python scripts/backfill_1h_ibkr.py --start-date 2015-01-01
"""
from __future__ import annotations
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

import pyarrow.parquet  # noqa: F401 — must precede torch/SB3 on Windows

import pandas as pd

from ib_async import IB, Stock

from intraday_trader.data_updater import _filter_market_hours
from src.utils.config_loader import load_config, all_tickers
from src.utils.logging_config import setup_logging, get_logger

log = get_logger("scripts.backfill_1h_ibkr")

_DEFAULT_START   = "2015-01-01"
_DEFAULT_HOST    = "127.0.0.1"
_DEFAULT_PORT    = 7497
_CLIENT_ID       = 10              # distinct from live trading (1) and intraday (2)
_REQUEST_PAUSE   = 12.0            # seconds between tickers — IBKR pacing: 60 req/10 min
_COLS            = ["open", "high", "low", "close", "volume"]


def parse_args():
    p = argparse.ArgumentParser(description="Backfill 1h OHLCV from IBKR")
    p.add_argument("--config",      default="intraday_trader/config.yaml")
    p.add_argument("--host",        default=_DEFAULT_HOST)
    p.add_argument("--port",        type=int, default=_DEFAULT_PORT,
                   help="7497=TWS paper, 7496=TWS live, 4002=Gateway paper, 4001=Gateway live")
    p.add_argument("--start-date",  default=_DEFAULT_START,
                   help="Earliest date to request (IBKR limit ~10 years back)")
    p.add_argument("--tickers",     nargs="+", default=None,
                   help="Subset of tickers (default: full universe)")
    p.add_argument("--dry-run",     action="store_true",
                   help="Print date gaps without downloading")
    p.add_argument("--log-file",    default=None)
    return p.parse_args()


def _bars_to_df(bars) -> pd.DataFrame:
    """Convert ib_async BarDataList to a clean UTC-indexed OHLCV DataFrame."""
    rows = []
    for b in bars:
        dt = b.date
        if isinstance(dt, str):
            dt = pd.Timestamp(dt)
        elif not isinstance(dt, pd.Timestamp):
            dt = pd.Timestamp(dt)
        rows.append({
            "timestamp": dt,
            "open":   float(b.open),
            "high":   float(b.high),
            "low":    float(b.low),
            "close":  float(b.close),
            "volume": float(b.volume),
        })
    df = pd.DataFrame(rows).set_index("timestamp")
    # Normalise to UTC
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("America/New_York").tz_convert("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df = df[_COLS]
    return df


def _load_existing(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        df.columns = [c.lower() for c in df.columns]
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize("UTC")
        return df[_COLS]
    except Exception as e:
        log.warning("Could not read %s: %s", path, e)
        return None


def backfill_ticker(
    ib:         IB,
    ticker:     str,
    ohlcv_dir:  Path,
    start_date: str,
    dry_run:    bool = False,
) -> str:
    """
    Fetch and prepend missing historical bars for one ticker.
    Returns a status string for the summary log.
    """
    out_path = ohlcv_dir / f"{ticker}.parquet"
    existing = _load_existing(out_path)

    if existing is not None and len(existing) > 0:
        existing_start = existing.index[0]
        target_start   = pd.Timestamp(start_date, tz="UTC")
        if existing_start <= target_start:
            log.info("  %s: already starts at %s — skip", ticker, existing_start.date())
            return "skip"
        # Request data up to just before the existing data starts
        end_dt = existing_start
    else:
        end_dt = pd.Timestamp.now(tz="UTC")

    end_dt_str = end_dt.strftime("%Y%m%d %H:%M:%S UTC")
    gap_days   = (end_dt - pd.Timestamp(start_date, tz="UTC")).days
    log.info("  %s: requesting %d days of 1h history up to %s", ticker, gap_days, end_dt.date())

    if dry_run:
        existing_str = existing.index[0].date() if existing is not None else "none"
        print(f"  {ticker:<6}  gap: {start_date} -> {end_dt.date()}  ({gap_days}d)  existing from: {existing_str}")
        return "dry-run"

    contract = Stock(ticker, "SMART", "USD")
    try:
        ib.qualifyContracts(contract)
    except Exception as e:
        log.warning("  %s: could not qualify contract: %s", ticker, e)
        return "error"

    # Chunk into 1-year requests working backwards from end_dt.
    # IBKR times out on multi-year single requests for 1h bars.
    target_start = pd.Timestamp(start_date, tz="UTC")
    chunk_end    = end_dt
    all_chunks: list[pd.DataFrame] = []

    while chunk_end > target_start:
        chunk_end_str = chunk_end.strftime("%Y%m%d %H:%M:%S UTC")
        try:
            bars = ib.reqHistoricalData(
                contract,
                endDateTime    = chunk_end_str,
                durationStr    = "1 Y",
                barSizeSetting = "1 hour",
                whatToShow     = "TRADES",
                useRTH         = True,
                formatDate     = 2,
                keepUpToDate   = False,
                timeout        = 120,
            )
        except Exception as e:
            log.error("  %s: reqHistoricalData failed at %s: %s", ticker, chunk_end.date(), e)
            break

        if not bars:
            log.debug("  %s: no bars for chunk ending %s — stopping", ticker, chunk_end.date())
            break

        chunk_df = _bars_to_df(bars)
        chunk_df = _filter_market_hours(chunk_df)
        chunk_df = chunk_df[chunk_df.index >= target_start]

        if not chunk_df.empty:
            all_chunks.append(chunk_df)
            log.info("  %s: chunk ending %s -> %d bars", ticker, chunk_end.date(), len(chunk_df))

        # Step back 1 year for the next chunk
        chunk_end = chunk_df.index[0] if not chunk_df.empty else chunk_end - pd.DateOffset(years=1)
        if chunk_end <= target_start:
            break

        time.sleep(2)  # brief pause between chunks for the same ticker

    if not all_chunks:
        log.warning("  %s: no bars returned in any chunk", ticker)
        return "empty"

    new_df = pd.concat(all_chunks)
    new_df = new_df[~new_df.index.duplicated(keep="last")]
    new_df.sort_index(inplace=True)
    new_df = new_df[new_df.index >= target_start]

    if new_df.empty:
        log.warning("  %s: empty after filtering", ticker)
        return "empty"

    # Merge with existing yfinance data
    if existing is not None and len(existing) > 0:
        combined = pd.concat([new_df, existing])
    else:
        combined = new_df

    combined = combined[~combined.index.duplicated(keep="last")]
    combined.sort_index(inplace=True)

    combined.to_parquet(out_path)
    log.info(
        "  %s: saved %d bars total (%d new, %d existing)  range: %s to %s",
        ticker,
        len(combined),
        len(new_df),
        len(existing) if existing is not None else 0,
        combined.index[0].date(),
        combined.index[-1].date(),
    )
    return f"+{len(new_df)}"


def main():
    args = parse_args()
    setup_logging(log_file=args.log_file)

    cfg     = load_config(args.config)
    raw_dir = cfg.get("data", {}).get("raw_dir", "intraday_trader/data/raw")
    ohlcv_dir = Path(raw_dir) / "ohlcv"
    ohlcv_dir.mkdir(parents=True, exist_ok=True)

    # Universe
    if args.tickers:
        tickers = args.tickers
    else:
        universe_file = cfg.get("universe", {}).get("file", "config/universe.yaml")
        tickers = all_tickers(universe_file)

    log.info("Backfill plan: %d tickers, start=%s, port=%d", len(tickers), args.start_date, args.port)

    if args.dry_run:
        print(f"\nDry run — showing date gaps for {len(tickers)} tickers:\n")
        for ticker in tickers:
            backfill_ticker(IB(), ticker, ohlcv_dir, args.start_date, dry_run=True)
        return

    # Connect
    ib = IB()
    log.info("Connecting to IBKR %s:%d...", args.host, args.port)
    ib.connect(args.host, args.port, clientId=_CLIENT_ID, timeout=15)
    log.info("Connected. Account: %s", ib.managedAccounts())

    results = {}
    try:
        for i, ticker in enumerate(tickers, 1):
            log.info("[%d/%d] %s", i, len(tickers), ticker)
            status = backfill_ticker(ib, ticker, ohlcv_dir, args.start_date)
            results[ticker] = status

            if status not in ("skip",) and i < len(tickers):
                log.info("  Pausing %.1fs (IBKR pacing)...", _REQUEST_PAUSE)
                time.sleep(_REQUEST_PAUSE)

    except KeyboardInterrupt:
        log.warning("Interrupted by user.")
    finally:
        ib.disconnect()
        log.info("Disconnected.")

    # Summary
    skipped = sum(1 for s in results.values() if s == "skip")
    errors  = sum(1 for s in results.values() if s in ("error", "empty"))
    filled  = sum(1 for s in results.values() if s.startswith("+"))
    print(f"\n{'='*50}")
    print(f"  Backfill complete: {filled} filled, {skipped} skipped, {errors} errors")
    print(f"  Next step: rebuild features and scanner:")
    print(f"    python intraday_trader/scripts/build_scanner.py")
    print(f"    python intraday_trader/scripts/generate_synthetic.py --force")
    print(f"    python intraday_trader/scripts/train.py --algo rppo --run-name intraday_rppo_v2")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
