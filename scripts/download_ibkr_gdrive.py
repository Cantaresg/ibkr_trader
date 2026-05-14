"""
Download IBKR historical data and fundamentals to Google Drive.

Saves everything to G:\\My Drive\\ibkr_data\\ so the dataset persists across
machines and is backed up automatically.

Directory layout:
    G:/My Drive/ibkr_data/
        ohlcv/          {TICKER}.parquet    — 1h OHLCV bars (2015-today)
        fundamentals/   {TICKER}.parquet    — parsed key ratios
        fundamentals/   {TICKER}_raw.xml    — raw IBKR ReportSnapshot XML

Prerequisites:
    - TWS or IB Gateway running locally
    - API access enabled: Configure → API → Settings → Enable Socket Clients
    - 127.0.0.1 in trusted IPs

Usage:
    python scripts/download_ibkr_gdrive.py
    python scripts/download_ibkr_gdrive.py --port 7496           # live TWS
    python scripts/download_ibkr_gdrive.py --tickers AAPL MSFT   # subset
    python scripts/download_ibkr_gdrive.py --dry-run
    python scripts/download_ibkr_gdrive.py --skip-ohlcv          # only fundamentals
    python scripts/download_ibkr_gdrive.py --skip-fundamentals    # only OHLCV
    python scripts/download_ibkr_gdrive.py --start-date 2010-01-01
"""
from __future__ import annotations
import sys
import time
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime, timezone, date

sys.path.insert(0, str(Path(__file__).parent.parent))

import pyarrow.parquet  # noqa: F401 — must precede torch/SB3 on Windows
import pandas as pd
import numpy as np

from ib_async import IB, Stock

from intraday_trader.data_updater import _filter_market_hours
from src.utils.config_loader import load_config, all_tickers
from src.utils.logging_config import setup_logging, get_logger

log = get_logger("scripts.download_ibkr_gdrive")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_OUT_DIR  = Path("G:/My Drive/ibkr_data")
_DEFAULT_START    = "2015-01-01"
_DEFAULT_HOST     = "127.0.0.1"
_DEFAULT_PORT     = 7497            # TWS paper; use 7496 for live TWS
_CLIENT_ID        = 15              # distinct from live (1), intraday (2), backfill (10)
_OHLCV_PAUSE      = 16.0           # seconds between tickers (IBKR: max 60 req / 10 min)
_FUND_PAUSE       = 6.0            # fundamentals requests are lighter
_OHLCV_COLS       = ["open", "high", "low", "close", "volume"]

# Fundamental ratio fields we extract from the IBKR ReportSnapshot XML
_RATIO_TAGS = {
    "PEEXCLXOR":    "pe_ratio",          # P/E excluding extraordinary
    "PRICE2BK":     "pb_ratio",          # Price-to-book
    "TTMROEPCT":    "roe",               # TTM return on equity (%)
    "QCURRATIO":    "current_ratio",     # Current ratio (quarterly)
    "QTOTD2EQ":     "debt_to_equity",    # Total debt / equity (quarterly)
    "TTMREVCHG":    "revenue_growth",    # TTM revenue change (%)
    "TTMEPSCHG":    "eps_growth",        # TTM EPS change (%)
    "MKTCAP":       "market_cap",        # Market capitalisation ($M)
    "TTMREVPS":     "revenue_per_share", # Revenue per share
    "TTMNPMGN":     "net_profit_margin", # Net profit margin (%)
    "TTMGROSMGN":   "gross_margin",      # Gross margin (%)
    "TTMROAPCT":    "roa",               # Return on assets (%)
}


# ===========================================================================
# OHLCV helpers
# ===========================================================================

def _bars_to_df(bars) -> pd.DataFrame:
    """Convert ib_async BarDataList → clean UTC-indexed OHLCV DataFrame."""
    rows = []
    for b in bars:
        dt = b.date
        if not isinstance(dt, pd.Timestamp):
            dt = pd.Timestamp(dt)
        rows.append({
            "timestamp": dt,
            "open":   float(b.open),
            "high":   float(b.high),
            "low":    float(b.low),
            "close":  float(b.close),
            "volume": float(b.volume),
        })
    if not rows:
        return pd.DataFrame(columns=_OHLCV_COLS)
    df = pd.DataFrame(rows).set_index("timestamp")
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("America/New_York").tz_convert("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df[_OHLCV_COLS]


def _load_existing_ohlcv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        df.columns = [c.lower() for c in df.columns]
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize("UTC")
        return df[_OHLCV_COLS]
    except Exception as e:
        log.warning("Could not read %s: %s", path, e)
        return None


def download_ohlcv(
    ib:         IB,
    ticker:     str,
    ohlcv_dir:  Path,
    start_date: str,
    dry_run:    bool = False,
) -> str:
    """
    Fetch 1h OHLCV for *ticker*, prepend any historical gap, append any
    recent gap, and save to *ohlcv_dir*/{ticker}.parquet.

    Strategy:
    - If no existing file: download full history from start_date to today.
    - If file exists:
        * If file starts after start_date: back-fill missing history.
        * Forward-fill any gap between last stored bar and today.
    """
    out_path  = ohlcv_dir / f"{ticker}.parquet"
    existing  = _load_existing_ohlcv(out_path)
    target_start = pd.Timestamp(start_date, tz="UTC")
    now_utc      = pd.Timestamp.now(tz="UTC")

    if dry_run:
        if existing is not None and len(existing) > 0:
            gap_back  = max(0, (existing.index[0]  - target_start).days)
            gap_fwd   = max(0, (now_utc - existing.index[-1]).days - 1)
            print(f"  {ticker:<6}  existing: {existing.index[0].date()}→{existing.index[-1].date()}"
                  f"  back-gap: {gap_back}d  fwd-gap: {gap_fwd}d")
        else:
            gap = (now_utc - target_start).days
            print(f"  {ticker:<6}  no file — would download ~{gap} days from {start_date}")
        return "dry-run"

    contract = Stock(ticker, "SMART", "USD")
    try:
        ib.qualifyContracts(contract)
    except Exception as e:
        log.warning("  %s: could not qualify contract: %s", ticker, e)
        return "error"

    all_chunks: list[pd.DataFrame] = []

    # ---- Back-fill historical gap ----------------------------------------
    if existing is None or existing.empty:
        back_end = now_utc
    elif existing.index[0] > target_start:
        back_end = existing.index[0]
    else:
        back_end = None  # no back-fill needed

    if back_end is not None:
        chunk_end = back_end
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
                log.debug("  %s: no bars in chunk ending %s", ticker, chunk_end.date())
                break

            chunk_df = _bars_to_df(bars)
            chunk_df = _filter_market_hours(chunk_df)
            chunk_df = chunk_df[chunk_df.index >= target_start]

            if not chunk_df.empty:
                all_chunks.append(chunk_df)
                log.info("  %s: back-fill chunk ending %s -> %d bars",
                         ticker, chunk_end.date(), len(chunk_df))
                chunk_end = chunk_df.index[0]
            else:
                chunk_end = chunk_end - pd.DateOffset(years=1)

            if chunk_end <= target_start:
                break
            time.sleep(2)

    # ---- Forward-fill recent gap -----------------------------------------
    if existing is not None and not existing.empty:
        last_bar = existing.index[-1]
        fwd_gap_days = (now_utc - last_bar).days
        if fwd_gap_days > 1:
            fwd_start = last_bar - pd.Timedelta(days=7)  # 7-day buffer
            fwd_end   = now_utc
            fwd_end_str = fwd_end.strftime("%Y%m%d %H:%M:%S UTC")
            log.info("  %s: forward-fill from %s (%d days)", ticker, fwd_start.date(), fwd_gap_days)
            try:
                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime    = fwd_end_str,
                    durationStr    = f"{min(fwd_gap_days + 10, 30)} D",
                    barSizeSetting = "1 hour",
                    whatToShow     = "TRADES",
                    useRTH         = True,
                    formatDate     = 2,
                    keepUpToDate   = False,
                    timeout        = 60,
                )
                if bars:
                    fwd_df = _bars_to_df(bars)
                    fwd_df = _filter_market_hours(fwd_df)
                    if not fwd_df.empty:
                        all_chunks.append(fwd_df)
                        log.info("  %s: forward %d new bars", ticker, len(fwd_df))
            except Exception as e:
                log.warning("  %s: forward-fill failed: %s", ticker, e)

    # ---- Merge and save ---------------------------------------------------
    if not all_chunks and (existing is None or existing.empty):
        log.warning("  %s: no bars returned", ticker)
        return "empty"

    if all_chunks:
        new_df = pd.concat(all_chunks)
        new_df = new_df[~new_df.index.duplicated(keep="last")]
        new_df.sort_index(inplace=True)

        if existing is not None and not existing.empty:
            combined = pd.concat([existing, new_df])
        else:
            combined = new_df

        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)
        combined = combined[combined.index >= target_start]
        combined.to_parquet(out_path)
        n_new = len(combined) - (len(existing) if existing is not None else 0)
        log.info(
            "  %s: saved %d bars total (+%d new)  range: %s to %s",
            ticker, len(combined), max(n_new, 0),
            combined.index[0].date(), combined.index[-1].date(),
        )
        return f"+{max(n_new, 0)}"
    else:
        # Nothing to add — up to date
        log.info("  %s: already up to date (%d bars)", ticker, len(existing))
        return "skip"


# ===========================================================================
# Fundamentals helpers
# ===========================================================================

def _parse_snapshot_xml(xml_text: str) -> dict[str, float]:
    """Extract key fundamental ratios from an IBKR ReportSnapshot XML string."""
    ratios: dict[str, float] = {}
    try:
        root = ET.fromstring(xml_text)
        # Ratios live inside <Ratios> / <Group> / <Ratio field="..." value="..."/>
        for ratio_el in root.iter("Ratio"):
            field = ratio_el.get("FieldName") or ratio_el.get("field") or ""
            if field in _RATIO_TAGS:
                raw = ratio_el.get("Value") or ratio_el.get("value") or ""
                try:
                    ratios[_RATIO_TAGS[field]] = float(raw)
                except (ValueError, TypeError):
                    pass
    except ET.ParseError as e:
        log.warning("XML parse error: %s", e)
    return ratios


def download_fundamentals(
    ib:            IB,
    ticker:        str,
    fund_dir:      Path,
    force_refresh: bool = False,
) -> str:
    """
    Fetch IBKR ReportSnapshot fundamentals for *ticker*, save parsed parquet
    and raw XML.  Skips if a parquet already exists and is < 7 days old
    (unless force_refresh=True).
    """
    parquet_path = fund_dir / f"{ticker}.parquet"
    xml_path     = fund_dir / f"{ticker}_raw.xml"

    # Age check — fundamentals rarely change more than weekly
    if not force_refresh and parquet_path.exists():
        age_days = (datetime.now() - datetime.fromtimestamp(parquet_path.stat().st_mtime)).days
        if age_days < 7:
            log.info("  %s fundamentals: cached (%d days old) — skip", ticker, age_days)
            return "skip"

    contract = Stock(ticker, "SMART", "USD")
    try:
        ib.qualifyContracts(contract)
    except Exception as e:
        log.warning("  %s: could not qualify contract: %s", ticker, e)
        return "error"

    # IBKR supports: ReportSnapshot, ReportsFinSummary, ReportsFinStatements
    # ReportSnapshot is the quickest and covers the most useful ratios.
    try:
        xml_text = ib.reqFundamentalData(contract, "ReportSnapshot")
    except Exception as e:
        log.warning("  %s: reqFundamentalData failed: %s", ticker, e)
        return "error"

    if not xml_text:
        log.warning("  %s: empty fundamentals response", ticker)
        return "empty"

    # Save raw XML for inspection / reprocessing
    xml_path.write_text(xml_text, encoding="utf-8")

    ratios = _parse_snapshot_xml(xml_text)
    if not ratios:
        log.warning("  %s: no ratios extracted from snapshot", ticker)
        return "empty"

    df = pd.DataFrame([ratios], index=[pd.Timestamp.now().normalize()])
    df.index.name = "as_of"
    df.to_parquet(parquet_path)
    log.info("  %s fundamentals: %d ratios saved  (%s)",
             ticker, len(ratios), ", ".join(ratios.keys()))
    return "ok"


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Download IBKR 1h OHLCV + fundamentals to G:/My Drive/ibkr_data"
    )
    p.add_argument("--config",      default="intraday_trader/config.yaml")
    p.add_argument("--out-dir",     default=str(_DEFAULT_OUT_DIR),
                   help="Root output directory (default: G:/My Drive/ibkr_data)")
    p.add_argument("--host",        default=_DEFAULT_HOST)
    p.add_argument("--port",        type=int, default=_DEFAULT_PORT,
                   help="7497=TWS paper, 7496=TWS live, 4002=Gateway paper, 4001=Gateway live")
    p.add_argument("--start-date",  default=_DEFAULT_START,
                   help="Earliest bar date for OHLCV backfill (default: 2015-01-01)")
    p.add_argument("--tickers",     nargs="+", default=None,
                   help="Subset of tickers (default: full universe)")
    p.add_argument("--skip-ohlcv",         action="store_true",
                   help="Skip 1h OHLCV download, only run fundamentals")
    p.add_argument("--skip-fundamentals",  action="store_true",
                   help="Skip fundamentals, only run OHLCV")
    p.add_argument("--force-fundamentals", action="store_true",
                   help="Re-download fundamentals even if recent cache exists")
    p.add_argument("--dry-run",     action="store_true",
                   help="Show what would be downloaded without making requests")
    p.add_argument("--log-file",    default=None)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(log_file=args.log_file)

    # ---- Resolve output directories --------------------------------------
    out_dir   = Path(args.out_dir)
    ohlcv_dir = out_dir / "ohlcv"
    fund_dir  = out_dir / "fundamentals"
    ohlcv_dir.mkdir(parents=True, exist_ok=True)
    fund_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output root: %s", out_dir)

    # ---- Universe --------------------------------------------------------
    cfg = load_config(args.config)
    if args.tickers:
        tickers = args.tickers
    else:
        universe_file = cfg.get("universe", {}).get("file", "config/universe.yaml")
        tickers = all_tickers(universe_file)
    log.info("Universe: %d tickers", len(tickers))

    if args.dry_run:
        print(f"\nDry-run mode — no IBKR requests will be made.\n")
        print(f"Output dir : {out_dir}")
        print(f"Start date : {args.start_date}")
        print(f"Tickers    : {len(tickers)}\n")
        for t in tickers:
            ohlcv_path = ohlcv_dir / f"{t}.parquet"
            existing   = _load_existing_ohlcv(ohlcv_path)
            target_start = pd.Timestamp(args.start_date, tz="UTC")
            now_utc      = pd.Timestamp.now(tz="UTC")
            if existing is not None and len(existing) > 0:
                gap_back = max(0, (existing.index[0] - target_start).days)
                gap_fwd  = max(0, (now_utc - existing.index[-1]).days - 1)
                print(f"  {t:<6}  {existing.index[0].date()} to {existing.index[-1].date()}"
                      f"  back-gap: {gap_back}d  fwd-gap: {gap_fwd}d")
            else:
                gap = (now_utc - target_start).days
                print(f"  {t:<6}  (no file)  ~{gap}d to download from {args.start_date}")
        return

    # ---- Connect to IBKR -------------------------------------------------
    ib = IB()
    log.info("Connecting to IBKR %s:%d (clientId=%d)...", args.host, args.port, _CLIENT_ID)
    ib.connect(args.host, args.port, clientId=_CLIENT_ID, timeout=15)
    log.info("Connected. Managed accounts: %s", ib.managedAccounts())

    ohlcv_results: dict[str, str] = {}
    fund_results:  dict[str, str] = {}

    def _ensure_connected() -> bool:
        """Reconnect if the socket dropped. Returns True when connected."""
        if ib.isConnected():
            return True
        log.warning("IBKR connection lost -- reconnecting...")
        try:
            ib.disconnect()
        except Exception:
            pass
        time.sleep(10)
        for attempt in range(3):
            try:
                ib.connect(args.host, args.port, clientId=_CLIENT_ID, timeout=20)
                log.info("Reconnected (attempt %d). Accounts: %s",
                         attempt + 1, ib.managedAccounts())
                return True
            except Exception as e:
                log.warning("Reconnect attempt %d failed: %s", attempt + 1, e)
                time.sleep(15)
        log.error("All reconnect attempts failed.")
        return False

    try:
        for i, ticker in enumerate(tickers, 1):
            log.info("[%d/%d] %s", i, len(tickers), ticker)

            # Ensure connection is alive before each ticker
            if not _ensure_connected():
                log.error("Cannot reconnect -- aborting remaining tickers.")
                for t in tickers[i - 1:]:
                    ohlcv_results.setdefault(t, "error")
                    fund_results.setdefault(t, "error")
                break

            # -- OHLCV --
            if not args.skip_ohlcv:
                status = download_ohlcv(ib, ticker, ohlcv_dir, args.start_date)
                ohlcv_results[ticker] = status
                if status not in ("skip", "dry-run") and i < len(tickers):
                    log.info("  Pacing pause %.1fs...", _OHLCV_PAUSE)
                    time.sleep(_OHLCV_PAUSE)

            # -- Fundamentals --
            if not args.skip_fundamentals:
                if not _ensure_connected():
                    fund_results[ticker] = "error"
                else:
                    fstatus = download_fundamentals(
                        ib, ticker, fund_dir,
                        force_refresh=args.force_fundamentals,
                    )
                    fund_results[ticker] = fstatus
                    if fstatus not in ("skip",) and not args.skip_ohlcv:
                        time.sleep(1)
                    elif fstatus not in ("skip",):
                        time.sleep(_FUND_PAUSE)

    except KeyboardInterrupt:
        log.warning("Interrupted by user -- saving results so far.")
    finally:
        ib.disconnect()
        log.info("Disconnected from IBKR.")

    # ---- Summary ---------------------------------------------------------
    def _count(d: dict, pred) -> int:
        return sum(1 for v in d.values() if pred(v))

    print(f"\n{'='*60}")
    print(f"  OHLCV results ({len(ohlcv_results)} tickers processed):")
    print(f"    Updated : {_count(ohlcv_results, lambda s: s.startswith('+'))}")
    print(f"    Skipped : {_count(ohlcv_results, lambda s: s == 'skip')}")
    print(f"    Errors  : {_count(ohlcv_results, lambda s: s in ('error', 'empty'))}")
    if fund_results:
        print(f"  Fundamentals results ({len(fund_results)} tickers processed):")
        print(f"    Saved   : {_count(fund_results, lambda s: s == 'ok')}")
        print(f"    Skipped : {_count(fund_results, lambda s: s == 'skip')}")
        print(f"    Errors  : {_count(fund_results, lambda s: s in ('error', 'empty'))}")
    print(f"\n  Data saved to: {out_dir}")
    print(f"{'='*60}\n")

    errors = [t for t, s in {**ohlcv_results, **fund_results}.items()
              if s in ("error", "empty")]
    if errors:
        print(f"  Tickers with errors: {errors}\n")


if __name__ == "__main__":
    main()
