"""
IntradayDataUpdater: downloads fresh 1h OHLCV bars and rebuilds intraday features.

Run once daily before market open (e.g., 6am ET) to ensure the DataStore has
up-to-date data including yesterday's full session.

Incremental stitching: instead of re-downloading 730d on every run, we detect
the last stored bar timestamp and only download from there forward. This
allows the database to grow beyond the 730-day yfinance limit over time.
"""
from __future__ import annotations
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from intraday_trader.constants import INTRADAY_UNIVERSE, UNIVERSE_FILE
from intraday_trader.features import build_all as build_features_all
from intraday_trader.market_features import build as build_market, _CACHE_FILENAME
from src.utils.config_loader import load_config, all_tickers
from src.utils.logging_config import get_logger

log = get_logger("intraday.updater")

_YF_1H_PERIOD      = "730d"   # initial download period (yfinance max for 1h)
_YF_INCREMENTAL_DAYS = 7      # look back a few extra days on incremental runs to fill gaps


def _load_universe(cfg: dict) -> list[str]:
    """Load the full 123-stock universe from config, falling back to INTRADAY_UNIVERSE."""
    universe_file = cfg.get("universe", {}).get("file", UNIVERSE_FILE)
    try:
        return all_tickers(universe_file)
    except Exception as e:
        log.warning("Could not load universe from %s (%s) — using fallback list", universe_file, e)
        return list(INTRADAY_UNIVERSE)


class IntradayDataUpdater:
    """Downloads and rebuilds all intraday data artifacts."""

    def __init__(self, config_path: str = "intraday_trader/config.yaml"):
        cfg           = load_config(config_path)
        self.raw_dir  = cfg.get("data", {}).get("raw_dir",       "intraday_trader/data/raw")
        self.proc_dir = cfg.get("data", {}).get("processed_dir", "intraday_trader/data/processed")
        self.tickers  = _load_universe(cfg)

    # ------------------------------------------------------------------
    def run(self, as_of: date | None = None) -> None:
        log.info("IntradayDataUpdater: starting update as of %s", as_of or "today")
        self._update_hourly_ohlcv_incremental(as_of)
        self._rebuild_features()
        self._rebuild_market_features()
        log.info("IntradayDataUpdater: complete")

    # ------------------------------------------------------------------
    def _update_hourly_ohlcv_incremental(self, as_of: date | None = None) -> None:
        """
        Incremental update: only download bars newer than the last stored timestamp.
        On first run (no existing file), downloads the full 730-day yfinance history.
        Appends new bars, deduplicates, and saves back.
        """
        ohlcv_dir = Path(self.raw_dir) / "ohlcv"
        ohlcv_dir.mkdir(parents=True, exist_ok=True)
        today = as_of or date.today()

        for ticker in self.tickers:
            try:
                out_path = ohlcv_dir / f"{ticker}.parquet"
                existing: pd.DataFrame | None = None

                if out_path.exists():
                    try:
                        existing = pd.read_parquet(out_path)
                        if isinstance(existing.columns, pd.MultiIndex):
                            existing.columns = [c[0].lower() for c in existing.columns]
                        else:
                            existing.columns = [c.lower() for c in existing.columns]
                        existing = _filter_market_hours(existing)
                    except Exception as e:
                        log.warning("  Could not read existing %s parquet (%s) — full re-download", ticker, e)
                        existing = None

                if existing is not None and len(existing) > 0:
                    # Incremental: download from (last stored bar − buffer) to today
                    last_ts = existing.index[-1]
                    if hasattr(last_ts, 'date'):
                        last_date = last_ts.date()
                    else:
                        last_date = pd.Timestamp(last_ts).date()
                    from_date = last_date - timedelta(days=_YF_INCREMENTAL_DAYS)
                    end_date  = today + timedelta(days=1)  # yfinance end is exclusive

                    log.info("  Incremental 1h download: %s  from=%s", ticker, from_date)
                    new_df = yf.download(
                        ticker,
                        start=str(from_date),
                        end=str(end_date),
                        interval="1h",
                        auto_adjust=True,
                        progress=False,
                        multi_level_index=False,
                    )
                    if new_df.empty:
                        log.debug("  No new bars for %s", ticker)
                        continue

                    if isinstance(new_df.columns, pd.MultiIndex):
                        new_df.columns = [c[0].lower() for c in new_df.columns]
                    else:
                        new_df.columns = [c.lower() for c in new_df.columns]

                    new_df = _filter_market_hours(new_df)
                    combined = pd.concat([existing, new_df])
                    combined = combined[~combined.index.duplicated(keep="last")]
                    combined.sort_index(inplace=True)
                    combined.to_parquet(out_path)
                    new_rows = len(combined) - len(existing)
                    log.info("  %s: added %d bars (total %d)", ticker, max(new_rows, 0), len(combined))
                else:
                    # First run: download full 730-day history
                    log.info("  Initial 1h download: %s (period=%s)", ticker, _YF_1H_PERIOD)
                    df = yf.download(
                        ticker,
                        period=_YF_1H_PERIOD,
                        interval="1h",
                        auto_adjust=True,
                        progress=False,
                        multi_level_index=False,
                    )
                    if df.empty:
                        log.warning("  No data returned for %s", ticker)
                        continue

                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [c[0].lower() for c in df.columns]
                    else:
                        df.columns = [c.lower() for c in df.columns]

                    df = _filter_market_hours(df)
                    df.to_parquet(out_path)
                    log.info("  Saved %d bars for %s", len(df), ticker)

            except Exception as e:
                log.error("  Failed updating %s: %s", ticker, e)

    # ------------------------------------------------------------------
    def _rebuild_features(self) -> None:
        log.info("Rebuilding intraday per-ticker features (%d tickers)...", len(self.tickers))
        built = build_features_all(self.tickers, self.raw_dir, self.proc_dir, overwrite=True)
        log.info("Built features for %d tickers", len(built))

    # ------------------------------------------------------------------
    def _rebuild_market_features(self) -> None:
        from intraday_trader.features import load_features
        log.info("Rebuilding intraday market features...")
        intraday_returns: dict[str, pd.Series] = {}
        for t in self.tickers:
            df = load_features(self.proc_dir, t)
            if df is not None and "intraday_return" in df.columns:
                intraday_returns[t] = df["intraday_return"]

        cache_path = Path(self.proc_dir) / _CACHE_FILENAME
        if cache_path.exists():
            cache_path.unlink()

        build_market(self.raw_dir, self.proc_dir, intraday_returns, cache=True)
        log.info("Market features rebuilt")


# ------------------------------------------------------------------
def _filter_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only bars within 9:30–15:30 ET."""
    if df.empty:
        return df
    import pytz
    ET = pytz.timezone("America/New_York")
    if df.index.tzinfo is None:
        idx_et = df.index.tz_localize("UTC").tz_convert(ET)
    else:
        idx_et = df.index.tz_convert(ET)
    hours = idx_et.hour
    mask  = (hours >= 9) & (hours <= 15)
    return df[mask]
