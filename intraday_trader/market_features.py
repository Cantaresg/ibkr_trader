"""
Intraday market features (5 columns per 1h bar).

  1  spy_bar_return          — SPY 1h bar return, z-scored (20-bar window)
  2  spy_intraday_return     — SPY return from today's open to current bar
  3  vix_level               — Daily VIX at session open, z-scored (252-day window)
  4  spy_rel_volume          — SPY bar volume / 14-day same-bar average
  5  market_breadth_intraday — fraction of universe stocks with intraday_return > 0
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.normalizer import rolling_zscore, rolling_zscore_series
from intraday_trader.constants import N_MARKET
from src.utils.logging_config import get_logger

log = get_logger("intraday.market_features")

MARKET_FEATURE_COLS: list[str] = [
    "spy_bar_return",
    "spy_intraday_return",
    "vix_level",
    "spy_rel_volume",
    "market_breadth_intraday",
]
assert len(MARKET_FEATURE_COLS) == N_MARKET

_CACHE_FILENAME = "market_features_1h.parquet"


def _load_raw_ohlcv(raw_dir: str, ticker: str) -> pd.DataFrame | None:
    p = Path(raw_dir) / "ohlcv" / f"{ticker}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def _load_daily_vix(raw_dir: str) -> pd.Series | None:
    """Try intraday raw dir first, then EOD market dir as fallback."""
    for candidate in [
        Path(raw_dir) / "vix_daily.parquet",
        Path("data/raw/market/VIX.parquet"),   # EOD system fallback
    ]:
        if candidate.exists():
            df  = pd.read_parquet(candidate)
            col = "close" if "close" in df.columns else df.columns[0]
            return df[col].astype("float32")
    log.warning("Daily VIX not found — using 0 for vix_level feature")
    return None


def _same_bar_avg_volume_spy(spy_1h: pd.DataFrame, lookback_days: int = 14) -> pd.Series:
    """Vectorised: mean SPY volume at same bar-of-day over prior lookback_days."""
    vol   = spy_1h["volume"]
    dates = spy_1h.index.normalize()
    hours = spy_1h.index.hour
    pivot = pd.DataFrame({"date": dates, "hour": hours, "vol": vol.values}) \
              .pivot_table(index="date", columns="hour", values="vol", aggfunc="first")
    avg   = pivot.shift(1).rolling(lookback_days, min_periods=1).mean()
    keys        = pd.MultiIndex.from_arrays([dates, hours])
    avg_stacked = avg.stack(future_stack=True)
    avg_stacked.index = pd.MultiIndex.from_arrays([
        avg_stacked.index.get_level_values(0),
        avg_stacked.index.get_level_values(1),
    ])
    out = pd.Series(avg_stacked.reindex(keys).values, index=spy_1h.index, dtype="float32")
    return out.fillna(vol).clip(lower=0).astype("float32")


def build(
    raw_dir: str,
    processed_dir: str,
    universe_intraday_returns: dict[str, pd.Series] | None = None,
    cache: bool = True,
) -> pd.DataFrame:
    cache_path = Path(processed_dir) / _CACHE_FILENAME
    if cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    spy_1h = _load_raw_ohlcv(raw_dir, "SPY")
    if spy_1h is None:
        raise RuntimeError("SPY 1h OHLCV not found — run IntradayDataUpdater first")

    spy_1h.columns = [c.lower() for c in spy_1h.columns]
    out   = pd.DataFrame(index=spy_1h.index)
    dates = spy_1h.index.normalize()

    out["spy_bar_return"]      = spy_1h["close"].pct_change().clip(-0.1, 0.1).fillna(0).astype("float32")
    day_open                   = spy_1h["open"].groupby(dates).transform("first")
    out["spy_intraday_return"] = ((spy_1h["close"] - day_open) / day_open.replace(0, np.nan)).fillna(0).astype("float32")

    vix_daily   = _load_daily_vix(raw_dir)
    vix_zscore  = rolling_zscore_series(vix_daily, window=252) if vix_daily is not None else None
    vix_series  = pd.Series(0.0, index=spy_1h.index, dtype="float32")
    if vix_zscore is not None:
        # Use date-object keys to avoid tz-aware vs tz-naive mismatch between SPY (UTC) and VIX (tz-naive)
        vix_dict: dict = {pd.Timestamp(k).date(): float(v) for k, v in vix_zscore.items() if not np.isnan(v)}
        for d in sorted(dates.unique()):
            d_date = pd.Timestamp(d).date()
            if d_date in vix_dict:
                vix_series.loc[spy_1h.index[dates == d]] = vix_dict[d_date]
    out["vix_level"] = vix_series

    avg_vol            = _same_bar_avg_volume_spy(spy_1h, lookback_days=14)
    out["spy_rel_volume"] = (spy_1h["volume"] / avg_vol.replace(0, np.nan)).fillna(1).clip(0, 10).astype("float32")

    if universe_intraday_returns:
        aligned = pd.DataFrame({t: s.reindex(spy_1h.index).fillna(0) for t, s in universe_intraday_returns.items()})
        out["market_breadth_intraday"] = (aligned > 0).mean(axis=1).reindex(spy_1h.index).fillna(0.5).astype("float32")
    else:
        out["market_breadth_intraday"] = 0.5

    out[["spy_bar_return", "spy_rel_volume"]] = rolling_zscore(out[["spy_bar_return", "spy_rel_volume"]], window=20)
    out = out[MARKET_FEATURE_COLS].fillna(0).astype("float32")

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache_path)
        log.info("Saved market features (%d bars)", len(out))

    return out


def load(processed_dir: str) -> pd.DataFrame | None:
    p = Path(processed_dir) / _CACHE_FILENAME
    return pd.read_parquet(p) if p.exists() else None
