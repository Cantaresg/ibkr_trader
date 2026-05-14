"""
Intraday per-stock feature engineering from 1h OHLCV data.

20 features per bar:
  1  vwap_dev          — (close - session VWAP) / VWAP; resets each calendar day
  2  intraday_return   — (close - open_today) / open_today
  3  bar_return        — close / prev_close - 1
  4  overnight_gap     — (open_today - prev_day_close) / prev_day_close
  5  rel_volume        — bar_volume / avg same-bar volume (14 trading days)
  6  vol_trend         — rolling 3-bar mean of bar_return
  7  high_low_range    — (high - low) / close
  8  rsi_7bar          — RSI(7) on hourly closes
  9  cumvol_fraction   — cumulative day volume up to this bar / yesterday total volume
 10  price_vs_prev_close — close / yesterday_close - 1
 11  tod_sin           — sin(2π × bar_position_in_day / BARS_PER_DAY)
 12  tod_cos           — cos(2π × bar_position_in_day / BARS_PER_DAY)
 13  bb_position       — (close - BB_lower) / (BB_upper - BB_lower), 14-bar Bollinger
 14  bar_body_ratio    — (close - open) / (high - low + 1e-8)
 15  momentum_open_rank — cross-sectional rank of intraday_return (filled in by DataStore)
 16  atr_ratio         — ATR(3) / 14-day rolling mean ATR(3)
 17  momentum_5d       — 5-trading-day return (close / close[35 bars ago] - 1)
 18  momentum_20d      — 20-trading-day return (close / close[140 bars ago] - 1)
 19  macd_hist         — MACD(12,26,9) histogram on hourly closes
 20  ma20d_dev         — (close - 20-day rolling mean close) / 20-day rolling mean

All except tod_sin, tod_cos, bar_body_ratio, momentum_open_rank are
z-score normalized over a 98-bar (14-day) rolling window, clipped ±3.
"""
from __future__ import annotations
from pathlib import Path
import math

import numpy as np
import pandas as pd

from src.features.normalizer import rolling_zscore
from intraday_trader.constants import BARS_PER_DAY, N_FEATURES
from src.utils.logging_config import get_logger

log = get_logger("intraday.features")

FEATURE_COLS: list[str] = [
    "vwap_dev",
    "intraday_return",
    "bar_return",
    "overnight_gap",
    "rel_volume",
    "vol_trend",
    "high_low_range",
    "rsi_7bar",
    "cumvol_fraction",
    "price_vs_prev_close",
    "tod_sin",
    "tod_cos",
    "bb_position",
    "bar_body_ratio",
    "momentum_open_rank",  # cross-sectional — filled by DataStore as 0.0 initially
    "atr_ratio",
    "momentum_5d",         # 5-trading-day return; critical for overnight/multi-day holding
    "momentum_20d",        # 20-trading-day return; monthly trend context
    "macd_hist",           # MACD(12,26,9) histogram; momentum acceleration
    "ma20d_dev",           # deviation from 20-day rolling mean; mean-reversion signal
]
assert len(FEATURE_COLS) == N_FEATURES


# ---------------------------------------------------------------------------
# Single-ticker feature computation
# ---------------------------------------------------------------------------

def _rsi(closes: pd.Series, window: int = 7) -> pd.Series:
    delta = closes.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).astype("float32")


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 3) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=1).mean().astype("float32")


def _bollinger(close: pd.Series, window: int = 14) -> tuple[pd.Series, pd.Series]:
    ma  = close.rolling(window, min_periods=1).mean()
    std = close.rolling(window, min_periods=1).std(ddof=1).fillna(0)
    return ma - 2 * std, ma + 2 * std   # lower, upper


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD histogram = (EMA_fast - EMA_slow) - EMA_signal(EMA_fast - EMA_slow)."""
    ema_fast = close.ewm(span=fast, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
    return (macd_line - signal_line).astype("float32")


def _session_vwap(df: pd.DataFrame) -> pd.Series:
    """Cumulative VWAP resetting at the start of each calendar date."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    pv      = typical * df["volume"]
    dates   = df.index.normalize()
    cum_pv  = pv.groupby(dates).cumsum()
    cum_vol = df["volume"].groupby(dates).cumsum()
    vwap    = cum_pv / cum_vol.replace(0, np.nan)
    return vwap.astype("float32")


def _same_bar_avg_volume(df: pd.DataFrame, lookback_days: int = 14) -> pd.Series:
    """Mean volume at the same bar-of-day over the prior lookback_days trading days.

    Vectorised via pivot: O(n_bars) instead of the O(n_days^2 * n_bars) loop.
    """
    vol   = df["volume"]
    dates = df.index.normalize()
    hours = df.index.hour

    # Pivot to (date × hour) matrix, one volume value per cell
    pivot = pd.DataFrame({"date": dates, "hour": hours, "vol": vol.values}) \
              .pivot_table(index="date", columns="hour", values="vol", aggfunc="first")

    # shift(1): exclude current day; rolling(lookback_days): mean of prior days
    avg = pivot.shift(1).rolling(lookback_days, min_periods=1).mean()

    # Reindex back to the original flat bar index via (date, hour) MultiIndex
    keys          = pd.MultiIndex.from_arrays([dates, hours])
    avg_stacked   = avg.stack(future_stack=True)                 # (date, hour) → avg_vol
    avg_stacked.index = pd.MultiIndex.from_arrays([
        avg_stacked.index.get_level_values(0),
        avg_stacked.index.get_level_values(1),
    ])
    result = avg_stacked.reindex(keys).values

    out = pd.Series(result, index=df.index, dtype="float32").fillna(vol)
    return out.clip(lower=0).astype("float32")


def compute(hourly_ohlcv: pd.DataFrame, norm_window: int = 98) -> pd.DataFrame:
    """
    Compute all 16 intraday features for a single ticker from 1h OHLCV.

    Input:  pd.DataFrame with columns open/high/low/close/volume, market-hours only.
    Output: pd.DataFrame with 16 columns, same index.
            momentum_open_rank is 0.0 (filled cross-sectionally by DataStore).
    """
    df = hourly_ohlcv.copy()
    df.columns = [c.lower() for c in df.columns]
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    out          = pd.DataFrame(index=df.index)
    dates        = df.index.normalize()
    unique_dates = sorted(dates.unique())

    bar_positions = pd.Series(np.nan, index=df.index, dtype="float32")
    for d in unique_dates:
        for pos, ts in enumerate(df.index[dates == d]):
            bar_positions.loc[ts] = pos

    session_vwap = _session_vwap(df)
    out["vwap_dev"] = ((df["close"] - session_vwap) / session_vwap.replace(0, np.nan)).fillna(0).astype("float32")

    day_open = df["open"].groupby(dates).transform("first")
    out["intraday_return"] = ((df["close"] - day_open) / day_open.replace(0, np.nan)).fillna(0).astype("float32")

    out["bar_return"] = df["close"].pct_change().clip(-0.5, 0.5).fillna(0).astype("float32")

    date_to_last_close: dict = {d: df.loc[dates == d, "close"].iloc[-1] for d in unique_dates}
    prev_close_series = pd.Series(np.nan, index=df.index, dtype="float64")
    for i, d in enumerate(unique_dates):
        if i == 0:
            continue
        prev_c = date_to_last_close[unique_dates[i - 1]]
        prev_close_series.loc[df.index[dates == d]] = prev_c

    out["overnight_gap"] = ((day_open - prev_close_series) / prev_close_series.replace(0, np.nan)).fillna(0).astype("float32")

    avg_vol = _same_bar_avg_volume(df, lookback_days=14)
    out["rel_volume"] = (df["volume"] / avg_vol.replace(0, np.nan)).fillna(1).clip(0, 10).astype("float32")

    out["vol_trend"]      = out["bar_return"].rolling(3, min_periods=1).mean().astype("float32")
    out["high_low_range"] = ((df["high"] - df["low"]) / df["close"].replace(0, np.nan)).fillna(0).clip(0, 0.2).astype("float32")
    out["rsi_7bar"]       = _rsi(df["close"], window=7)

    cum_vol_day    = df["volume"].groupby(dates).cumsum()
    date_to_total  = {d: df.loc[dates == d, "volume"].sum() for d in unique_dates}
    yest_total_vol = pd.Series(np.nan, index=df.index, dtype="float64")
    for i, d in enumerate(unique_dates):
        if i == 0:
            continue
        yest_total_vol.loc[df.index[dates == d]] = date_to_total[unique_dates[i - 1]]
    out["cumvol_fraction"] = (cum_vol_day / yest_total_vol.replace(0, np.nan)).fillna(0).clip(0, 5).astype("float32")

    out["price_vs_prev_close"] = ((df["close"] - prev_close_series) / prev_close_series.replace(0, np.nan)).fillna(0).astype("float32")

    out["tod_sin"] = bar_positions.apply(lambda p: math.sin(2 * math.pi * p / BARS_PER_DAY) if not math.isnan(p) else 0.0).astype("float32")
    out["tod_cos"] = bar_positions.apply(lambda p: math.cos(2 * math.pi * p / BARS_PER_DAY) if not math.isnan(p) else 1.0).astype("float32")

    bb_lower, bb_upper = _bollinger(df["close"], window=14)
    out["bb_position"] = ((df["close"] - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)).fillna(0.5).clip(0, 1).astype("float32")

    out["bar_body_ratio"] = ((df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-8)).clip(-1, 1).astype("float32")
    out["momentum_open_rank"] = 0.0

    atr_now  = _atr(df["high"], df["low"], df["close"], window=3)
    atr_mean = atr_now.rolling(norm_window, min_periods=14).mean().replace(0, np.nan)
    out["atr_ratio"] = (atr_now / atr_mean).fillna(1).clip(0, 5).astype("float32")

    # Multi-day momentum (35 bars = 5 trading days × 7 bars/day)
    bars_5d  = 5  * BARS_PER_DAY   # 35
    bars_20d = 20 * BARS_PER_DAY   # 140
    prev_5d  = df["close"].shift(bars_5d)
    prev_20d = df["close"].shift(bars_20d)
    out["momentum_5d"]  = ((df["close"] - prev_5d)  / prev_5d.replace(0, np.nan)).fillna(0).clip(-0.5, 0.5).astype("float32")
    out["momentum_20d"] = ((df["close"] - prev_20d) / prev_20d.replace(0, np.nan)).fillna(0).clip(-0.5, 0.5).astype("float32")

    out["macd_hist"] = _macd(df["close"], fast=12, slow=26, signal=9)

    ma_20d = df["close"].rolling(bars_20d, min_periods=bars_5d).mean()
    out["ma20d_dev"] = ((df["close"] - ma_20d) / ma_20d.replace(0, np.nan)).fillna(0).clip(-0.5, 0.5).astype("float32")

    to_normalize = [
        "vwap_dev", "intraday_return", "bar_return", "overnight_gap",
        "rel_volume", "vol_trend", "high_low_range", "rsi_7bar",
        "cumvol_fraction", "price_vs_prev_close", "bb_position", "atr_ratio",
        "momentum_5d", "momentum_20d", "macd_hist", "ma20d_dev",
    ]
    out[to_normalize] = rolling_zscore(out[to_normalize], window=norm_window)

    return out[FEATURE_COLS].fillna(0).astype("float32")


# ---------------------------------------------------------------------------
# Build / load helpers
# ---------------------------------------------------------------------------

def _features_path(processed_dir: str, ticker: str) -> Path:
    return Path(processed_dir) / "features" / f"{ticker}.parquet"


def build_ticker(ticker: str, raw_dir: str, processed_dir: str, overwrite: bool = False) -> pd.DataFrame | None:
    out_path = _features_path(processed_dir, ticker)
    if out_path.exists() and not overwrite:
        return pd.read_parquet(out_path)
    raw_path = Path(raw_dir) / "ohlcv" / f"{ticker}.parquet"
    if not raw_path.exists():
        log.warning("No raw 1h OHLCV for %s at %s", ticker, raw_path)
        return None
    try:
        feat = compute(pd.read_parquet(raw_path))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        feat.to_parquet(out_path)
        log.info("Built features for %s (%d bars)", ticker, len(feat))
        return feat
    except Exception as e:
        log.error("Failed building features for %s: %s", ticker, e)
        return None


def build_all(tickers: list[str], raw_dir: str, processed_dir: str, overwrite: bool = False) -> dict[str, pd.DataFrame]:
    return {t: df for t in tickers if (df := build_ticker(t, raw_dir, processed_dir, overwrite)) is not None}


def load_features(processed_dir: str, ticker: str) -> pd.DataFrame | None:
    p = _features_path(processed_dir, ticker)
    return pd.read_parquet(p) if p.exists() else None
