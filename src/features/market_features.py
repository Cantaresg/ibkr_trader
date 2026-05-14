"""
Build the 7-feature daily market matrix used by the HMM and the environment.

Features (in column order):
  0  vix_zscore          — VIX rolling z-score (252d)
  1  vix_term_structure  — VIX3M / VIX ratio, centred at 1.0
  2  spy_trend_20d       — SPY 20-day return, z-scored
  3  market_breadth      — fraction of universe above 200d SMA
  4  yield_spread        — TNX - IRX (10yr minus 3m yield, in pp)
  5  credit_spread       — HYG daily return - IEI daily return
  6  put_call_ratio      — equity P/C ratio (zero-filled; CBOE data unavailable)
"""
from pathlib import Path
import numpy as np
import pandas as pd

from src.data.market_data import load_all as load_market
from src.data.ohlcv_store import load as load_ohlcv
from src.features.normalizer import rolling_zscore_series
from src.utils.logging_config import get_logger

log = get_logger("market_features")

MARKET_FEATURE_COLS = [
    "vix_zscore",
    "vix_term_structure",
    "spy_trend_20d",
    "market_breadth",
    "yield_spread",
    "credit_spread",
    "put_call_ratio",
]
N_MARKET_FEATURES = len(MARKET_FEATURE_COLS)  # 7

_CACHE_PATH = "data/processed/market_features.parquet"
_BREADTH_PATH = "data/processed/market_breadth.parquet"


def build_market_breadth(tickers: list[str], raw_dir: str) -> pd.Series:
    """
    For each trading date, compute the fraction of `tickers` where
    close > SMA(200). Returns a Series indexed by date.
    """
    log.info("Computing market breadth for %d tickers...", len(tickers))
    closes: dict[str, pd.Series] = {}
    for t in tickers:
        df = load_ohlcv(raw_dir, t)
        if df is not None and len(df) >= 200:
            closes[t] = df["close"]

    # Wide DataFrame: date x ticker
    panel = pd.DataFrame(closes).sort_index()
    sma200 = panel.rolling(200, min_periods=150).mean()
    above = (panel > sma200).astype("float32")
    breadth = above.mean(axis=1)  # fraction 0..1
    breadth.name = "market_breadth"
    log.info("Market breadth: %d dates, %.3f mean", len(breadth), breadth.mean())
    return breadth


def build(
    tickers: list[str],
    raw_dir: str,
    cache: bool = True,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Build (or load cached) 10-feature daily market DataFrame."""
    cache_path = Path(_CACHE_PATH)
    if cache and cache_path.exists() and not overwrite:
        log.debug("Loading cached market features from %s", cache_path)
        df = pd.read_parquet(cache_path)
        df.index = pd.to_datetime(df.index)
        return df

    mkt = load_market(raw_dir)
    spy = mkt.get("SPY")
    vix = mkt.get("VIX")
    vix3m = mkt.get("VIX3M")
    tnx = mkt.get("TNX")
    irx = mkt.get("IRX")
    hyg = mkt.get("HYG")
    iei = mkt.get("IEI")

    if spy is None:
        raise RuntimeError("SPY market data not found — run download_data.py first")

    idx = spy.index  # canonical trading calendar

    out = pd.DataFrame(index=idx)

    # 0. VIX rolling z-score
    if vix is not None:
        v = vix["close"].reindex(idx).ffill()
        out["vix_zscore"] = rolling_zscore_series(v, 252)
    else:
        out["vix_zscore"] = 0.0

    # 1. VIX3M / VIX term structure (higher = market expects vol to ease = bullish)
    if vix is not None and vix3m is not None:
        v_ = vix["close"].reindex(idx).ffill().replace(0, np.nan)
        v3_ = vix3m["close"].reindex(idx).ffill()
        ratio = (v3_ / v_).fillna(1.0)
        out["vix_term_structure"] = rolling_zscore_series(ratio, 252)
    else:
        out["vix_term_structure"] = 0.0

    # 2. SPY 20-day trend
    spy_close = spy["close"].reindex(idx).ffill()
    spy_trend = spy_close.pct_change(20)
    out["spy_trend_20d"] = rolling_zscore_series(spy_trend, 252)

    # 3. Market breadth
    breadth_path = Path(_BREADTH_PATH)
    if breadth_path.exists() and not overwrite:
        breadth = pd.read_parquet(breadth_path).iloc[:, 0]
        breadth.index = pd.to_datetime(breadth.index)
    else:
        breadth = build_market_breadth(tickers, raw_dir)
        breadth_path.parent.mkdir(parents=True, exist_ok=True)
        breadth.to_frame().to_parquet(breadth_path)
    out["market_breadth"] = breadth.reindex(idx).ffill().fillna(0.5)

    # 4. Yield spread (10yr - 3m, in percentage points)
    if tnx is not None and irx is not None:
        t10 = tnx["close"].reindex(idx).ffill()
        t3  = irx["close"].reindex(idx).ffill()
        spread = t10 - t3
        out["yield_spread"] = rolling_zscore_series(spread, 252)
    else:
        out["yield_spread"] = 0.0

    # 5. Credit spread (HYG daily return - IEI daily return)
    if hyg is not None and iei is not None:
        hyg_ret = hyg["close"].reindex(idx).ffill().pct_change()
        iei_ret = iei["close"].reindex(idx).ffill().pct_change()
        cs = hyg_ret - iei_ret
        out["credit_spread"] = rolling_zscore_series(cs, 252)
    else:
        out["credit_spread"] = 0.0

    # 6. Put/call ratio (zero-filled: CBOE data unavailable)
    out["put_call_ratio"] = 0.0

    out = out.ffill().fillna(0.0).astype("float32")

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache_path)
        log.info("Market features saved: %d rows x %d cols", len(out), len(out.columns))

    return out
