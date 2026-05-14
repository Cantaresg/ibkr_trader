"""
Download GDELT daily tone timelines per ticker into a separate folder.

Uses GDELT's timelinetone mode: one request covers a full year, returning
pre-aggregated daily tone scores and article counts — no FinBERT needed.
Saves raw tone/count parquets to gdelt_raw_dir for later integration.

Output columns per ticker: date, tone, article_count
  tone          : GDELT average tone for the day (-100 to +100; positive = bullish)
  article_count : number of matching articles that day

Usage:
    python scripts/download_gdelt.py
    python scripts/download_gdelt.py --start-date 2017-01-01
    python scripts/download_gdelt.py --tickers AAPL MSFT NVDA
    python scripts/download_gdelt.py --no-global        # skip index/macro news

GDELT API: no key required. timelinetone returns 1 year per request.
Requests are paced at ~6 sec apart with exponential backoff on errors.
"""
import pyarrow.parquet  # Windows DLL fix: must precede torch imports

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import requests
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import all_tickers, load_config
from src.utils.logging_config import get_logger, setup_logging

log = get_logger("download_gdelt")

_GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
_REQ_INTERVAL = 8.0   # seconds between requests — GDELT allows ~60/min but be conservative
_CHUNK_YEARS = 1      # one year per API call (timelinetone handles this well)
_MAX_RETRIES = 3      # retry attempts on 429 or network errors only
_RETRY_BASE = 30.0    # seconds for first retry; doubles each attempt

# Company-name queries for each stock ticker.
# Searching by company name captures far more articles than the ticker symbol alone
# (e.g., "Apple" vs "AAPL"), since financial journalists write the name, not the ticker.
TICKER_QUERIES: dict[str, str] = {
    # Technology & Communication
    "AAPL":  '"Apple"',
    "MSFT":  '"Microsoft"',
    "NVDA":  '"Nvidia" OR "NVDA"',
    "GOOGL": '"Alphabet" OR "Google"',
    "META":  '"Meta Platforms" OR "Meta" OR "Facebook"',
    "AVGO":  '"Broadcom"',
    "ORCL":  '"Oracle"',
    "CRM":   '"Salesforce"',
    "AMD":   '"Advanced Micro Devices" OR "AMD"',
    "ADBE":  '"Adobe"',
    "QCOM":  '"Qualcomm"',
    "TXN":   '"Texas Instruments"',
    "INTC":  '"Intel"',
    "NOW":   '"ServiceNow"',
    "AMAT":  '"Applied Materials"',
    "MU":    '"Micron Technology" OR "Micron"',
    "LRCX":  '"Lam Research"',
    "KLAC":  '"KLA Corporation" OR "KLA"',
    "CSCO":  '"Cisco"',
    "VZ":    '"Verizon"',
    "T":     '"AT&T"',
    # Healthcare
    "LLY":   '"Eli Lilly"',
    "UNH":   '"UnitedHealth"',
    "JNJ":   '"Johnson Johnson" OR "J&J"',
    "ABBV":  '"AbbVie"',
    "MRK":   '"Merck"',
    "TMO":   '"Thermo Fisher"',
    "ABT":   '"Abbott Laboratories" OR "Abbott"',
    "DHR":   '"Danaher"',
    "AMGN":  '"Amgen"',
    "PFE":   '"Pfizer"',
    "ISRG":  '"Intuitive Surgical"',
    "MDT":   '"Medtronic"',
    "SYK":   '"Stryker"',
    "GILD":  '"Gilead Sciences" OR "Gilead"',
    "CVS":   '"CVS Health" OR "CVS"',
    "HCA":   '"HCA Healthcare"',
    "CI":    '"Cigna"',
    "ELV":   '"Elevance Health"',
    "ZTS":   '"Zoetis"',
    "BSX":   '"Boston Scientific"',
    "BDX":   '"Becton Dickinson"',
    # Financials
    "BRK-B": '"Berkshire Hathaway"',
    "JPM":   '"JPMorgan"',
    "V":     '"Visa"',
    "MA":    '"Mastercard"',
    "BAC":   '"Bank of America"',
    "WFC":   '"Wells Fargo"',
    "GS":    '"Goldman Sachs"',
    "MS":    '"Morgan Stanley"',
    "SPGI":  '"S&P Global"',
    "BLK":   '"BlackRock"',
    "CB":    '"Chubb"',
    "AXP":   '"American Express"',
    "USB":   '"US Bancorp"',
    "PGR":   '"Progressive" insurance',
    "TFC":   '"Truist Financial"',
    "COF":   '"Capital One"',
    "MCO":   '"Moody\'s"',
    "ICE":   '"Intercontinental Exchange"',
    "CME":   '"CME Group"',
    "AON":   '"Aon"',
    "TRV":   '"Travelers Companies" OR "Travelers"',
    # Consumer
    "AMZN":  '"Amazon"',
    "TSLA":  '"Tesla"',
    "HD":    '"Home Depot"',
    "MCD":   '"McDonald\'s"',
    "NKE":   '"Nike"',
    "SBUX":  '"Starbucks"',
    "TJX":   '"TJX Companies" OR "TJX"',
    "LOW":   '"Lowe\'s"',
    "BKNG":  '"Booking Holdings" OR "Booking.com"',
    "CMG":   '"Chipotle"',
    "PG":    '"Procter Gamble"',
    "KO":    '"Coca-Cola"',
    "PEP":   '"PepsiCo"',
    "PM":    '"Philip Morris"',
    "COST":  '"Costco"',
    "WMT":   '"Walmart"',
    "MO":    '"Altria"',
    "CL":    '"Colgate"',
    "GIS":   '"General Mills"',
    "KMB":   '"Kimberly-Clark"',
    "STZ":   '"Constellation Brands"',
    # Industrials & Energy
    "XOM":   '"ExxonMobil" OR "Exxon"',
    "CVX":   '"Chevron"',
    "COP":   '"ConocoPhillips"',
    "SLB":   '"Schlumberger" OR "SLB"',
    "EOG":   '"EOG Resources"',
    "OXY":   '"Occidental Petroleum" OR "Occidental"',
    "GE":    '"GE Aerospace" OR "General Electric"',
    "CAT":   '"Caterpillar"',
    "HON":   '"Honeywell"',
    "UPS":   '"United Parcel Service" OR "UPS"',
    "RTX":   '"Raytheon" OR "RTX"',
    "LMT":   '"Lockheed Martin"',
    "DE":    '"Deere" OR "John Deere"',
    "BA":    '"Boeing"',
    "MMM":   '"3M"',
    "FDX":   '"FedEx"',
    "EMR":   '"Emerson Electric"',
    "ETN":   '"Eaton"',
    "PH":    '"Parker Hannifin"',
    "NOC":   '"Northrop Grumman"',
    "GD":    '"General Dynamics"',
    # Utilities, REITs, Materials
    "NEE":   '"NextEra Energy"',
    "DUK":   '"Duke Energy"',
    "SO":    '"Southern Company"',
    "D":     '"Dominion Energy"',
    "AEP":   '"American Electric Power"',
    "EXC":   '"Exelon"',
    "SRE":   '"Sempra Energy" OR "Sempra"',
    "PLD":   '"Prologis"',
    "AMT":   '"American Tower"',
    "EQIX":  '"Equinix"',
    "CCI":   '"Crown Castle"',
    "SPG":   '"Simon Property"',
    "LIN":   '"Linde"',
    "APD":   '"Air Products"',
    "SHW":   '"Sherwin-Williams"',
    "ECL":   '"Ecolab"',
    "NEM":   '"Newmont"',
    "FCX":   '"Freeport-McMoRan" OR "Freeport"',
    "VMC":   '"Vulcan Materials"',
    "MLM":   '"Martin Marietta"',
}


def _ticker_query(ticker: str) -> str:
    """Return the GDELT search query for a stock ticker (company name, not symbol)."""
    return TICKER_QUERIES.get(ticker, f'"{ticker}"')


# Major indices and macro topics searched by name, not ETF ticker.
# Keys become filenames (label); values are GDELT query strings.
GDELT_INDEX_QUERIES: dict[str, str] = {
    # US equity indices
    "SP500":        '"S&P 500" OR "S&P500" OR "SPX"',
    "NASDAQ":       '"Nasdaq Composite" OR "Nasdaq 100" OR "Nasdaq index"',
    "DOW_JONES":    '"Dow Jones" OR "DJIA" OR "Dow 30"',
    "RUSSELL2000":  '"Russell 2000" OR "small-cap index"',
    # Volatility / fear gauge
    "VIX":          '"VIX" OR "CBOE volatility" OR "fear index"',
    # Central banks & rates
    "FED":          '"Federal Reserve" OR "FOMC" OR "Fed rate" OR "Powell"',
    "ECB":          '"European Central Bank" OR "ECB rate" OR "Lagarde"',
    "BOJ":          '"Bank of Japan" OR "BOJ" OR "Ueda"',
    # Rates & credit
    "TREASURY":     '"Treasury yield" OR "10-year yield" OR "US bonds"',
    "CREDIT":       '"credit spread" OR "high yield" OR "investment grade spread"',
    # International equity
    "EUROPE":       '"Stoxx 600" OR "Euro Stoxx" OR "DAX" OR "FTSE 100"',
    "JAPAN":        '"Nikkei" OR "Topix" OR "Japan stocks"',
    "CHINA":        '"Shanghai Composite" OR "Hang Seng" OR "CSI 300"',
    "EMERGING":     '"emerging markets" OR "MSCI Emerging" OR "EM equities"',
    # Commodities (affect equity risk appetite)
    "OIL":          '"crude oil" OR "WTI" OR "Brent crude"',
    "GOLD":         '"gold price" OR "gold rally" OR "gold selloff"',
}


def _safe(ticker: str) -> str:
    return ticker.replace("-", "_").replace(".", "_")


def _path(gdelt_dir: str, ticker: str) -> Path:
    return Path(gdelt_dir) / f"{_safe(ticker)}.parquet"


def _last_downloaded_year(gdelt_dir: str, label: str) -> int | None:
    """Return the last fully-downloaded year for label, or None."""
    p = _path(gdelt_dir, label)
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if df.empty:
        return None
    return pd.to_datetime(df["date"]).dt.year.max()


def _parse_gdelt_date(raw: str) -> date | None:
    """Parse GDELT timeline date (20170101T000000Z or 20170101000000)."""
    s = raw.replace("T", "").replace("Z", "")
    if len(s) >= 8:
        try:
            return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
        except ValueError:
            return None
    return None


def _fetch_with_retry(params: dict, label: str, year: int) -> list:
    """
    GET the GDELT API with exponential backoff on 429 and network errors.

    Empty body = no articles for that query/period — returned immediately as [].
    Only genuine HTTP errors (429, 5xx) and connection problems trigger retries.
    Returns list of timeline dicts, or empty list if no data / retries exhausted.
    """
    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = requests.get(_GDELT_URL, params=params, timeout=30)

            # 429: rate-limited — worth retrying after a longer wait
            if resp.status_code == 429:
                raise requests.HTTPError("429 Too Many Requests", response=resp)

            resp.raise_for_status()

            # Empty body = legitimate "no results" from GDELT, not an error
            text = resp.text.strip()
            if not text:
                return []

            data = resp.json()

            # timelinetone response: {"timeline": [{"series": "Average Tone", "data": [...]}]}
            # Unpack to the inner [{date, value}, ...] list.
            tl = data.get("timeline") if isinstance(data, dict) else None
            if tl and isinstance(tl, list):
                inner = tl[0].get("data", []) if isinstance(tl[0], dict) else []
                return inner

            # Fallback for other response shapes
            for key in ("data",):
                if isinstance(data, dict) and key in data and data[key]:
                    return data[key]
            if isinstance(data, list) and data:
                first = data[0]
                if isinstance(first, dict) and "data" in first:
                    return first["data"]
                return data

            return []  # valid JSON but no recognisable timeline

        except requests.HTTPError as exc:
            # Retry on 429 and 5xx; give up on other 4xx
            status = exc.response.status_code if exc.response is not None else 0
            if status not in (429,) and not (500 <= status < 600):
                log.debug("%s [%d]: HTTP %s — skipping", label, year, status)
                return []
            if attempt == _MAX_RETRIES:
                log.warning("%s [%d]: giving up after %d retries — %s",
                            label, year, _MAX_RETRIES, exc)
                return []
            wait = _RETRY_BASE * (2 ** attempt)
            log.debug("%s [%d]: attempt %d (%s) — retrying in %.0fs",
                      label, year, attempt + 1, exc, wait)
            time.sleep(wait)

        except (requests.ConnectionError, requests.Timeout) as exc:
            if attempt == _MAX_RETRIES:
                log.warning("%s [%d]: giving up after %d retries — %s",
                            label, year, _MAX_RETRIES, exc)
                return []
            wait = _RETRY_BASE * (2 ** attempt)
            log.debug("%s [%d]: attempt %d (%s) — retrying in %.0fs",
                      label, year, attempt + 1, exc, wait)
            time.sleep(wait)

        except Exception as exc:
            # JSON decode error or other unexpected issue — skip this year
            log.debug("%s [%d]: unexpected error — %s", label, year, exc)
            return []

    return []


def download_year(
    label: str,
    query: str,
    gdelt_dir: str,
    year: int,
) -> int:
    """
    Fetch GDELT timelinetone for one calendar year, append to parquet.
    Returns count of new rows written.
    """
    params = {
        "query": f"{query} sourcelang:english",
        "mode": "timelinetone",
        "startdatetime": f"{year}0101000000",
        "enddatetime": f"{year}1231235959",
        "format": "json",
    }

    entries = _fetch_with_retry(params, label, year)
    if not entries:
        return 0

    rows = []
    for e in entries:
        d = _parse_gdelt_date(e.get("date", ""))
        if d is None:
            continue
        rows.append({
            "date": d,
            "tone": float(e.get("value", 0.0)),
            # timelinetone doesn't return counts; use 1 as a presence flag so
            # article_count_zscore captures "news active today vs baseline"
            "article_count": int(e.get("cnt", 1)),
        })

    if not rows:
        return 0

    new_df = pd.DataFrame(rows)
    p = _path(gdelt_dir, label)
    p.parent.mkdir(parents=True, exist_ok=True)

    if p.exists():
        existing = pd.read_parquet(p)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date"]).reset_index(drop=True)
    else:
        combined = new_df

    combined.sort_values("date", inplace=True)
    combined.to_parquet(p)
    return len(new_df)


def download_label_years(
    label: str,
    query: str,
    gdelt_dir: str,
    start_year: int,
    end_year: int,
) -> tuple[int, int]:
    """Download yearly tone timelines for one label. Returns (total_rows, years_fetched)."""
    total, fetched = 0, 0

    for year in range(start_year, end_year + 1):
        t0 = time.monotonic()
        n = download_year(label, query, gdelt_dir, year)
        total += n
        fetched += 1

        elapsed = time.monotonic() - t0
        wait = _REQ_INTERVAL - elapsed
        if wait > 0:
            time.sleep(wait)

    return total, fetched


def download_all(
    items: list[tuple[str, str]],
    gdelt_dir: str,
    start_date: date,
    end_date: date,
    section: str = "",
) -> None:
    """Download yearly tone timelines for a list of (label, query) pairs."""
    prefix = f"[{section}] " if section else ""
    start_year = start_date.year
    end_year = end_date.year

    for i, (label, query) in enumerate(items, 1):
        last_year = _last_downloaded_year(gdelt_dir, label)
        from_year = (last_year + 1) if last_year is not None else start_year

        if from_year > end_year:
            log.debug("%s%s already up-to-date", prefix, label)
            continue

        total, fetched = download_label_years(label, query, gdelt_dir, from_year, end_year)
        log.info(
            "%s[%d/%d] %s: %d daily rows across %d year(s) from %d",
            prefix, i, len(items), label, total, fetched, from_year,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download raw GDELT articles per ticker (no pipeline integration)"
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--start-date", default="2017-01-01",
        help="Earliest date to fetch for new tickers (default: 2017-01-01)"
    )
    parser.add_argument(
        "--end-date", default=None,
        help="Latest date to fetch (default: today)"
    )
    parser.add_argument("--tickers", nargs="+", help="Override ticker list")
    parser.add_argument(
        "--gdelt-dir", default=None,
        help="Override output directory (default: gdelt_raw_dir from config)"
    )
    parser.add_argument(
        "--no-global", action="store_true",
        help="Skip global macro ETF download"
    )
    args = parser.parse_args()

    setup_logging("INFO")

    cfg = load_config(args.config)
    gdelt_dir = args.gdelt_dir or cfg["data"].get("gdelt_raw_dir", "G:/My Drive/ibkr_gdelt_raw")
    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date) if args.end_date else date.today()

    tickers = args.tickers or all_tickers(cfg["data"]["universe_file"])

    # Per-stock: query by company name — captures far more articles than ticker symbol
    stock_items = [(t, _ticker_query(t)) for t in tickers]

    # Indices/macro: query by name — ETF tickers won't match index news
    index_items = list(GDELT_INDEX_QUERIES.items())

    n_years = end_date.year - start_date.year + 1
    n_requests = (len(stock_items) + len(index_items)) * n_years
    est_minutes = round(n_requests * _REQ_INTERVAL / 60)

    log.info("GDELT download: %d stocks + %d indices | %s → %s | dir: %s",
             len(stock_items), len(index_items), start_date, end_date, gdelt_dir)
    log.info("Estimated time: ~%d min (%d requests at %.0fs each — 1 request per ticker per year)",
             est_minutes, n_requests, _REQ_INTERVAL)

    log.info("=== Per-stock articles (%d tickers) ===", len(stock_items))
    download_all(stock_items, gdelt_dir, start_date, end_date)

    if not args.no_global:
        index_dir = str(Path(gdelt_dir) / "indices")
        log.info("=== Major index / macro articles (%d topics) ===", len(index_items))
        download_all(index_items, index_dir, start_date, end_date, section="indices")

    log.info("=== Done. Raw data saved to: %s ===", gdelt_dir)
    log.info("Next: inspect data quality, then run FinBERT scoring when ready.")


if __name__ == "__main__":
    main()
