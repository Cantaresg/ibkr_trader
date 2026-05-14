"""
backfill_market_proxy.py — Extend SPY 1h proxy to 2015 and refresh VIX.

Problem: SPY 1h OHLCV only exists from June 2023.  For every bar before that,
the market features (spy_bar_return, spy_intraday_return, vix_level,
spy_rel_volume, market_breadth_intraday) are all zero.  This means 80% of the
training data has no market context — the model trains blind to whether the
day is risk-on or risk-off.

Fix:
  1. Load all universe 1h OHLCV (which goes back to 2015 via IBKR backfill).
  2. Build an equal-weighted market proxy: OHLC from top N stocks, volume sum.
  3. Merge: proxy covers the pre-2023 gap; real SPY takes precedence for 2023+.
  4. Save merged file to intraday_trader/data/raw/ohlcv/SPY.parquet.
  5. Refresh VIX daily via yfinance so it covers 2026.
  6. Delete the stale market_features_1h.parquet cache so it rebuilds on next
     training run.

Run once:
    python intraday_trader/scripts/backfill_market_proxy.py

Then force-rebuild intraday features:
    python intraday_trader/scripts/update_data.py --force-rebuild
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd

from src.utils.config_loader import load_config, all_tickers
from src.utils.logging_config import setup_logging, get_logger

log = get_logger("scripts.backfill_market_proxy")

# Top-30 highest-weight S&P 500 names — close to actual SPY index weights
# Equal-weight of these tracks SPY much better than equal-weight of all 124.
_TOP_PROXY_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO",
    "BRK-B", "JPM", "LLY", "V", "UNH", "XOM", "MA", "HD", "PG",
    "JNJ", "COST", "BAC", "ABBV", "WMT", "MRK", "NFLX", "CRM",
    "ORCL", "AMD", "CVX", "KO", "PEP",
]


def _load_ohlcv(ohlcv_dir: Path, ticker: str) -> pd.DataFrame | None:
    safe = ticker.replace("-", "_")
    for fname in [f"{ticker}.parquet", f"{safe}.parquet"]:
        p = ohlcv_dir / fname
        if p.exists():
            df = pd.read_parquet(p)
            df.columns = [c.lower() for c in df.columns]
            if df.index.tzinfo is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
            return df[["open", "high", "low", "close", "volume"]]
    return None


def build_proxy(
    ohlcv_dir: Path,
    tickers: list[str],
    start: str = "2015-01-01",
) -> pd.DataFrame:
    """
    Equal-weighted OHLCV proxy from a list of tickers.
    Returns DataFrame with columns [open, high, low, close, volume].
    """
    closes: list[pd.Series] = []
    opens:  list[pd.Series] = []
    highs:  list[pd.Series] = []
    lows:   list[pd.Series] = []
    vols:   list[pd.Series] = []

    for ticker in tickers:
        df = _load_ohlcv(ohlcv_dir, ticker)
        if df is None or df.empty:
            log.debug("Skipping %s (no file)", ticker)
            continue
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
        closes.append(df["close"].rename(ticker))
        opens.append(df["open"].rename(ticker))
        highs.append(df["high"].rename(ticker))
        lows.append(df["low"].rename(ticker))
        vols.append(df["volume"].rename(ticker))
        log.debug("  %s: %d bars (%s to %s)", ticker, len(df),
                  df.index.min().date(), df.index.max().date())

    if not closes:
        raise RuntimeError("No universe OHLCV files found — cannot build proxy")

    n = len(closes)
    log.info("Building equal-weighted proxy from %d tickers", n)

    idx = closes[0].index
    for s in closes[1:]:
        idx = idx.union(s.index)

    def _ew(series_list: list[pd.Series]) -> pd.Series:
        mat = pd.concat([s.reindex(idx) for s in series_list], axis=1)
        return mat.mean(axis=1).astype("float32")

    proxy = pd.DataFrame({
        "open":   _ew(opens),
        "high":   _ew(highs),
        "low":    _ew(lows),
        "close":  _ew(closes),
        "volume": pd.concat([s.reindex(idx).fillna(0) for s in vols], axis=1).sum(axis=1).astype("float32"),
    }, index=idx)

    # Drop bars with all-zero or NaN close (non-trading hours)
    proxy = proxy[proxy["close"] > 0].dropna(subset=["close"])
    proxy.sort_index(inplace=True)
    log.info("Proxy shape: %s  range: %s to %s",
             proxy.shape, proxy.index.min().date(), proxy.index.max().date())
    return proxy


def refresh_vix(raw_dir: Path) -> None:
    """Update VIX daily to today via yfinance, extending existing parquet."""
    try:
        import yfinance as yf
    except ImportError:
        log.warning("yfinance not installed — skipping VIX refresh")
        return

    vix_path = raw_dir / "market" / "VIX.parquet"
    if not vix_path.exists():
        vix_path = raw_dir.parent / "data" / "raw" / "market" / "VIX.parquet"

    existing: pd.DataFrame | None = None
    start_date = "2015-01-01"
    if vix_path.exists():
        existing = pd.read_parquet(vix_path)
        existing.index = pd.to_datetime(existing.index).tz_localize(None)
        last_date = existing.index.max()
        start_date = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        log.info("Existing VIX up to %s — fetching from %s", last_date.date(), start_date)

    new_vix = yf.download("^VIX", start=start_date, progress=False, auto_adjust=True)
    if new_vix.empty:
        log.info("VIX already up to date")
        return

    # yfinance returns multi-level columns in recent versions
    if isinstance(new_vix.columns, pd.MultiIndex):
        new_vix.columns = [c[0].lower() for c in new_vix.columns]
    else:
        new_vix.columns = [c.lower() for c in new_vix.columns]

    new_vix.index = pd.to_datetime(new_vix.index).tz_localize(None)

    if existing is not None:
        combined = pd.concat([existing, new_vix])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)
    else:
        combined = new_vix

    vix_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(vix_path)
    log.info("VIX refreshed: %d rows  range: %s to %s",
             len(combined), combined.index.min().date(), combined.index.max().date())


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill SPY proxy and refresh VIX for full intraday market features")
    parser.add_argument("--config",       default="intraday_trader/config.yaml")
    parser.add_argument("--start",        default="2015-01-01",
                        help="Earliest date for proxy (default: 2015-01-01)")
    parser.add_argument("--n-proxy",      type=int, default=30,
                        help="Number of top stocks to use in equal-weighted proxy (default: 30)")
    parser.add_argument("--skip-vix",     action="store_true")
    parser.add_argument("--skip-rebuild", action="store_true",
                        help="Don't delete market_features cache (rebuild happens on next training run)")
    args = parser.parse_args()

    setup_logging()

    cfg      = load_config(args.config)
    raw_dir  = Path(cfg["data"]["raw_dir"])           # intraday_trader/data/raw
    proc_dir = Path(cfg["data"]["processed_dir"])     # intraday_trader/data/processed
    ohlcv_dir = raw_dir / "ohlcv"

    # --- 1. Determine proxy tickers -------------------------------------------
    all_uni = all_tickers(cfg.get("universe", {}).get("file", "config/universe.yaml"))
    # Prefer our curated top-30 list; fill with universe remainder if needed
    proxy_tickers = [t for t in _TOP_PROXY_TICKERS if t in all_uni]
    if len(proxy_tickers) < args.n_proxy:
        remaining = [t for t in all_uni if t not in proxy_tickers]
        proxy_tickers += remaining[: args.n_proxy - len(proxy_tickers)]
    proxy_tickers = proxy_tickers[: args.n_proxy]
    log.info("Proxy tickers (%d): %s", len(proxy_tickers), proxy_tickers[:10])

    # --- 2. Build proxy -------------------------------------------------------
    proxy = build_proxy(ohlcv_dir, proxy_tickers, start=args.start)

    # --- 3. Merge with real SPY (real takes precedence) ----------------------
    spy_path  = ohlcv_dir / "SPY.parquet"
    real_spy: pd.DataFrame | None = None
    if spy_path.exists():
        real_spy = pd.read_parquet(spy_path)
        real_spy.columns = [c.lower() for c in real_spy.columns]
        if real_spy.index.tzinfo is None:
            real_spy.index = real_spy.index.tz_localize("UTC")
        real_spy = real_spy[["open", "high", "low", "close", "volume"]]
        log.info("Real SPY: %d bars (%s to %s)",
                 len(real_spy), real_spy.index.min().date(), real_spy.index.max().date())

    if real_spy is not None and not real_spy.empty:
        # Proxy fills dates NOT covered by real SPY
        proxy_only = proxy[~proxy.index.isin(real_spy.index)]
        merged = pd.concat([proxy_only, real_spy]).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
    else:
        merged = proxy

    log.info("Merged SPY proxy: %d bars  %s to %s",
             len(merged), merged.index.min().date(), merged.index.max().date())

    # Normalise column dtypes
    for col in ["open", "high", "low", "close"]:
        merged[col] = merged[col].astype("float32")
    merged["volume"] = merged["volume"].astype("float64")

    merged.to_parquet(spy_path)
    log.info("Saved extended SPY proxy to %s", spy_path)

    # --- 4. Refresh VIX -------------------------------------------------------
    if not args.skip_vix:
        # Try the EOD system path first (has data since 2008)
        eod_vix = Path("data/raw/market/VIX.parquet")
        refresh_vix(eod_vix.parent.parent if eod_vix.exists() else raw_dir)

    # --- 5. Invalidate cached market features --------------------------------
    if not args.skip_rebuild:
        cache = proc_dir / "market_features_1h.parquet"
        if cache.exists():
            cache.unlink()
            log.info("Deleted stale market features cache — will rebuild on next data load")

    print("\nDone. Next step:")
    print("  python intraday_trader/scripts/update_data.py --force-rebuild")
    print("This rebuilds intraday features with full market context (2015-present).")


if __name__ == "__main__":
    main()
