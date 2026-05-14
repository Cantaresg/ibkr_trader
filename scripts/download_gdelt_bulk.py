"""
Download GDELT sentiment via bulk GKG files — no API key, no rate limiting.

Instead of the (rate-limited) GDELT DOC API, this script downloads the
daily GKG v1 bulk files from data.gdeltproject.org, filters for company
mentions, and writes one parquet per ticker to gdelt_raw_dir.

Output format is identical to download_gdelt.py so gdelt_store.py and
build_gdelt_features.py work unchanged.

Output columns per ticker parquet: date, tone, article_count
  tone          : GDELT average tone for the day (-100 to +100)
  article_count : number of GKG records mentioning this company that day

GKG v1 files are one per calendar day, ~52 MB compressed / 170 MB raw.
Processing: ~0.3 s/file. Download: ~50 MB/file → ~144 GB for 11 years
(streaming — raw files are NOT stored; only the tiny aggregated parquets).

Usage:
    python scripts/download_gdelt_bulk.py
    python scripts/download_gdelt_bulk.py --start-date 2015-01-01
    python scripts/download_gdelt_bulk.py --tickers AAPL MSFT NVDA
    python scripts/download_gdelt_bulk.py --no-global   # skip index queries
"""
import pyarrow.parquet  # Windows DLL fix

import argparse
import io
import sys
import time
import zipfile
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.market_data import load_all as load_market
from src.utils.config_loader import all_tickers, load_config
from src.utils.logging_config import get_logger, setup_logging

log = get_logger("download_gdelt_bulk")

_GKG_URL = "http://data.gdeltproject.org/gkg/{date}.gkg.csv.zip"
_DOWNLOAD_TIMEOUT = 120   # seconds
_REQ_INTERVAL = 2.0       # seconds between files (bulk server is generous)
_MAX_RETRIES = 3

# Lowercase GKG organization name fragments for each ticker.
# We use substring matching, so "apple" matches "apple inc.", "apple computer", etc.
# Multiple entries → article counted if ANY name matches.
TICKER_NAMES: dict[str, list[str]] = {
    # Technology & Communication
    "AAPL":  ["apple"],
    "MSFT":  ["microsoft"],
    "NVDA":  ["nvidia"],
    "GOOGL": ["alphabet", "google"],
    "META":  ["meta platforms", "meta", "facebook"],
    "AVGO":  ["broadcom"],
    "ORCL":  ["oracle"],
    "CRM":   ["salesforce"],
    "AMD":   ["advanced micro devices"],
    "ADBE":  ["adobe"],
    "QCOM":  ["qualcomm"],
    "TXN":   ["texas instruments"],
    "INTC":  ["intel"],
    "NOW":   ["servicenow"],
    "AMAT":  ["applied materials"],
    "MU":    ["micron"],
    "LRCX":  ["lam research"],
    "KLAC":  ["kla"],
    "CSCO":  ["cisco"],
    "VZ":    ["verizon"],
    "T":     ["at&t"],
    # Healthcare
    "LLY":   ["eli lilly", "lilly"],
    "UNH":   ["unitedhealth"],
    "JNJ":   ["johnson & johnson", "j&j"],
    "ABBV":  ["abbvie"],
    "MRK":   ["merck"],
    "TMO":   ["thermo fisher"],
    "ABT":   ["abbott"],
    "DHR":   ["danaher"],
    "AMGN":  ["amgen"],
    "PFE":   ["pfizer"],
    "ISRG":  ["intuitive surgical"],
    "MDT":   ["medtronic"],
    "SYK":   ["stryker"],
    "GILD":  ["gilead"],
    "CVS":   ["cvs health", "cvs"],
    "HCA":   ["hca healthcare"],
    "CI":    ["cigna"],
    "ELV":   ["elevance health"],
    "ZTS":   ["zoetis"],
    "BSX":   ["boston scientific"],
    "BDX":   ["becton dickinson"],
    # Financials
    "BRK-B": ["berkshire hathaway", "berkshire"],
    "JPM":   ["jpmorgan", "j.p. morgan"],
    "V":     ["visa"],
    "MA":    ["mastercard"],
    "BAC":   ["bank of america"],
    "WFC":   ["wells fargo"],
    "GS":    ["goldman sachs"],
    "MS":    ["morgan stanley"],
    "SPGI":  ["s&p global"],
    "BLK":   ["blackrock"],
    "CB":    ["chubb"],
    "AXP":   ["american express"],
    "USB":   ["us bancorp"],
    "PGR":   ["progressive"],
    "TFC":   ["truist financial", "truist"],
    "COF":   ["capital one"],
    "MCO":   ["moody's", "moodys"],
    "ICE":   ["intercontinental exchange"],
    "CME":   ["cme group"],
    "AON":   ["aon"],
    "TRV":   ["travelers companies", "travelers"],
    # Consumer
    "AMZN":  ["amazon"],
    "TSLA":  ["tesla"],
    "HD":    ["home depot"],
    "MCD":   ["mcdonald's", "mcdonalds"],
    "NKE":   ["nike"],
    "SBUX":  ["starbucks"],
    "TJX":   ["tjx"],
    "LOW":   ["lowe's", "lowes"],
    "BKNG":  ["booking holdings", "booking.com"],
    "CMG":   ["chipotle"],
    "PG":    ["procter & gamble", "procter gamble"],
    "KO":    ["coca-cola", "coca cola"],
    "PEP":   ["pepsico"],
    "PM":    ["philip morris"],
    "COST":  ["costco"],
    "WMT":   ["walmart"],
    "MO":    ["altria"],
    "CL":    ["colgate"],
    "GIS":   ["general mills"],
    "KMB":   ["kimberly-clark"],
    "STZ":   ["constellation brands"],
    # Industrials & Energy
    "XOM":   ["exxonmobil", "exxon mobil", "exxon"],
    "CVX":   ["chevron"],
    "COP":   ["conocophillips"],
    "SLB":   ["schlumberger"],
    "EOG":   ["eog resources"],
    "OXY":   ["occidental petroleum", "occidental"],
    "GE":    ["general electric", "ge aerospace"],
    "CAT":   ["caterpillar"],
    "HON":   ["honeywell"],
    "UPS":   ["united parcel service"],
    "RTX":   ["raytheon", "rtx"],
    "LMT":   ["lockheed martin"],
    "DE":    ["deere", "john deere"],
    "BA":    ["boeing"],
    "MMM":   ["3m"],
    "FDX":   ["fedex"],
    "EMR":   ["emerson electric"],
    "ETN":   ["eaton"],
    "PH":    ["parker hannifin"],
    "NOC":   ["northrop grumman"],
    "GD":    ["general dynamics"],
    # Utilities, REITs, Materials
    "NEE":   ["nextera energy"],
    "DUK":   ["duke energy"],
    "SO":    ["southern company"],
    "D":     ["dominion energy"],
    "AEP":   ["american electric power"],
    "EXC":   ["exelon"],
    "SRE":   ["sempra"],
    "PLD":   ["prologis"],
    "AMT":   ["american tower"],
    "EQIX":  ["equinix"],
    "CCI":   ["crown castle"],
    "SPG":   ["simon property"],
    "LIN":   ["linde"],
    "APD":   ["air products"],
    "SHW":   ["sherwin-williams", "sherwin williams"],
    "ECL":   ["ecolab"],
    "NEM":   ["newmont"],
    "FCX":   ["freeport-mcmoran", "freeport"],
    "VMC":   ["vulcan materials"],
    "MLM":   ["martin marietta"],
}

# Index/macro topics: same labels as download_gdelt.py GDELT_INDEX_QUERIES
INDEX_NAMES: dict[str, list[str]] = {
    "SP500":       ["s&p 500", "s&p500", "spx"],
    "NASDAQ":      ["nasdaq composite", "nasdaq 100", "nasdaq"],
    "DOW_JONES":   ["dow jones", "djia"],
    "RUSSELL2000": ["russell 2000"],
    "VIX":         ["vix", "cboe volatility"],
    "FED":         ["federal reserve", "fomc"],
    "ECB":         ["european central bank", "ecb"],
    "BOJ":         ["bank of japan", "boj"],
    "TREASURY":    ["treasury yield", "10-year yield"],
    "CREDIT":      ["credit spread", "high yield"],
    "EUROPE":      ["stoxx 600", "euro stoxx", "ftse 100"],
    "JAPAN":       ["nikkei", "topix"],
    "CHINA":       ["hang seng", "shanghai composite"],
    "EMERGING":    ["emerging markets", "msci emerging"],
    "OIL":         ["crude oil", "wti", "brent"],
    "GOLD":        ["gold price"],
}


def _safe(label: str) -> str:
    return label.replace("-", "_").replace(".", "_")


def _out_path(gdelt_dir: str, label: str) -> Path:
    return Path(gdelt_dir) / f"{_safe(label)}.parquet"


def _progress_file(gdelt_dir: str) -> Path:
    return Path(gdelt_dir) / ".last_processed_date"


def _read_progress(gdelt_dir: str) -> date | None:
    p = _progress_file(gdelt_dir)
    if not p.exists():
        return None
    try:
        return date.fromisoformat(p.read_text().strip())
    except Exception:
        return None


def _write_progress(gdelt_dir: str, last_date: date) -> None:
    p = _progress_file(gdelt_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(last_date.isoformat())


def _flush_accumulator(
    acc: dict[str, list[tuple[date, float, int]]],
    gdelt_dir: str,
) -> None:
    """Write accumulated rows to parquets, merging with any existing data."""
    for label, rows in acc.items():
        if not rows:
            continue
        p = _out_path(gdelt_dir, label)
        new_df = pd.DataFrame(rows, columns=["date", "tone", "article_count"])
        new_df["date"] = pd.to_datetime(new_df["date"])
        if p.exists():
            existing = pd.read_parquet(p)
            existing["date"] = pd.to_datetime(existing["date"])
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["date"]).sort_values("date")
        else:
            combined = new_df.sort_values("date")
        p.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(p, index=False)
    acc.clear()


def _download_gkg(day: date) -> bytes | None:
    """Download and return compressed GKG bytes for one calendar day."""
    url = _GKG_URL.format(date=day.strftime("%Y%m%d"))
    for attempt in range(_MAX_RETRIES + 1):
        try:
            r = requests.get(url, timeout=_DOWNLOAD_TIMEOUT)
            if r.status_code == 404:
                return None  # date not in GDELT (weekend / holiday)
            r.raise_for_status()
            return r.content
        except requests.RequestException as exc:
            if attempt == _MAX_RETRIES:
                log.warning("GKG %s: download failed — %s", day, exc)
                return None
            time.sleep(5 * (2 ** attempt))
    return None


def _parse_gkg(data: bytes, name_sets: dict[str, list[str]]) -> dict[str, tuple[float, int]]:
    """
    Parse compressed GKG bytes, filter for company/index mentions.

    Returns {label: (sum_tone, article_count)} for the day.
    """
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
        with z.open(z.namelist()[0]) as f:
            raw = f.read().decode("latin-1", errors="replace")
    except Exception as exc:
        log.warning("GKG parse error: %s", exc)
        return {}

    tone_acc: dict[str, list[float]] = defaultdict(list)

    for line in raw.split("\n"):
        cols = line.split("\t")
        if len(cols) < 8:
            continue
        orgs = cols[6].lower()
        tone_raw = cols[7].split(",")[0]
        try:
            tone = float(tone_raw)
        except ValueError:
            continue

        for label, names in name_sets.items():
            for name in names:
                if name in orgs:
                    tone_acc[label].append(tone)
                    break  # count article once per label

    return {
        label: (sum(vals) / len(vals), len(vals))
        for label, vals in tone_acc.items()
    }



_CHECKPOINT_EVERY = 100  # flush to disk every N days (crash recovery)


def download_bulk(
    tickers: list[str],
    gdelt_dir: str,
    index_dir: str,
    trading_dates: list[date],
    include_indices: bool = True,
) -> None:
    """
    For each trading date: download the GKG file, filter for all companies
    and indices in one pass. Accumulates results in memory and flushes to
    parquets every CHECKPOINT_EVERY days (avoids per-day read-write churn).
    """
    stock_names = {t: TICKER_NAMES.get(t, [t.lower()]) for t in tickers}
    idx_names = INDEX_NAMES if include_indices else {}

    # Resume from saved progress marker
    last_done = _read_progress(gdelt_dir)
    start_day = (last_done + timedelta(days=1)) if last_done else trading_dates[0]
    pending = [d for d in trading_dates if d >= start_day]

    log.info(
        "Bulk GKG: %d tickers + %d indices | %d days to process (from %s)",
        len(tickers), len(idx_names), len(pending), start_day,
    )

    # In-memory accumulators: {label: [(date, tone, count), ...]}
    stock_acc: dict[str, list] = defaultdict(list)
    idx_acc: dict[str, list] = defaultdict(list)

    combined_names = {**stock_names, **idx_names}

    for i, day in enumerate(pending, 1):
        t0 = time.monotonic()

        data = _download_gkg(day)
        if data is None:
            log.debug("%s: no GKG file (weekend/holiday or download error)", day)
            _write_progress(gdelt_dir, day)
            continue

        results = _parse_gkg(data, combined_names)

        for ticker in tickers:
            if ticker in results:
                tone, count = results[ticker]
                stock_acc[ticker].append((day, tone, count))

        if include_indices:
            for label in idx_names:
                if label in results:
                    tone, count = results[label]
                    idx_acc[label].append((day, tone, count))

        elapsed = time.monotonic() - t0

        # Periodic checkpoint
        if i % _CHECKPOINT_EVERY == 0:
            log.info("[%d/%d] Checkpointing to disk...", i, len(pending))
            _flush_accumulator(stock_acc, gdelt_dir)
            _flush_accumulator(idx_acc, index_dir)
            _write_progress(gdelt_dir, day)
            log.info("[%d/%d] Checkpoint done. Last date: %s", i, len(pending), day)
        elif i % 25 == 0 or i <= 3:
            n_hits = len(results)
            log.info(
                "[%d/%d] %s: %d labels with mentions | %.1fs",
                i, len(pending), day, n_hits, elapsed,
            )

        wait = _REQ_INTERVAL - elapsed
        if wait > 0:
            time.sleep(wait)

    # Final flush
    log.info("Final flush to disk...")
    _flush_accumulator(stock_acc, gdelt_dir)
    _flush_accumulator(idx_acc, index_dir)
    if pending:
        _write_progress(gdelt_dir, pending[-1])

    log.info("Bulk GKG download complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download GDELT sentiment via bulk GKG files (no rate limits)"
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--start-date", default="2015-01-01",
        help="Earliest trading date to process (default: 2015-01-01)",
    )
    parser.add_argument(
        "--end-date", default=None,
        help="Latest date to process (default: today)",
    )
    parser.add_argument("--tickers", nargs="+", help="Override ticker list")
    parser.add_argument("--gdelt-dir", default=None)
    parser.add_argument("--no-global", action="store_true", help="Skip index sentiment")
    args = parser.parse_args()

    setup_logging("INFO", "logs/gdelt_bulk.log")

    cfg = load_config(args.config)
    raw_dir = cfg["data"]["raw_dir"]
    gdelt_dir = args.gdelt_dir or cfg["data"].get("gdelt_raw_dir", "G:/My Drive/ibkr_gdelt_raw")
    index_dir = str(Path(gdelt_dir) / "indices")

    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date) if args.end_date else date.today()

    tickers = args.tickers or all_tickers(cfg["data"]["universe_file"])

    # Use SPY market data as the trading calendar
    mkt = load_market(raw_dir)
    spy = mkt.get("SPY")
    if spy is None:
        log.error("SPY market data not found — run download_data.py first")
        sys.exit(1)

    trading_dates = [
        d.date() for d in spy.index
        if start_date <= d.date() <= end_date
    ]
    log.info(
        "Trading dates: %d | %s to %s",
        len(trading_dates), trading_dates[0], trading_dates[-1],
    )

    n_files = len(trading_dates)
    est_gb = n_files * 52 / 1024
    est_min = round(n_files * (_REQ_INTERVAL + 0.5) / 60)
    log.info(
        "Estimated download: ~%.0f GB over ~%d min (streaming, not stored)",
        est_gb, est_min,
    )

    download_bulk(
        tickers=tickers,
        gdelt_dir=gdelt_dir,
        index_dir=index_dir,
        trading_dates=trading_dates,
        include_indices=not args.no_global,
    )

    log.info("=== Done. Next step: python scripts/build_gdelt_features.py ===")


if __name__ == "__main__":
    main()
