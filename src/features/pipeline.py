"""
Feature pipeline: assembles all feature groups into a single per-ticker Parquet.

Output columns (30 per-stock features):
  Technical   (19): ema_12, ema_26, sma_50, sma_200, macd_line, macd_signal, macd_hist,
                    adx, rsi_14, rsi_2, cci_20, atr_14, bb_upper, bb_lower, bb_width,
                    hist_vol_20, obv, mfi_14, rsi_divergence
  Order flow   (2): large_trade_proxy, institutional_accumulation
  Fundamentals (5): pe_ratio, pb_ratio, debt_to_equity, roe, revenue_growth
  Price        (4): close_norm, daily_return, return_5d, return_20d

All values are RollingZScore-normalized (252-day window) except price returns
(already scale-free) and the divergence flag (+1/-1).
"""
from pathlib import Path
import numpy as np
import pandas as pd

from src.data import ohlcv_store, fundamentals_store
from src.features import technical, orderflow, fundamentals
from src.features.normalizer import rolling_zscore
from src.utils.logging_config import get_logger

log = get_logger("pipeline")

FEATURE_COLS: list[str] = [
    # Technical (19)
    "ema_12", "ema_26", "sma_50", "sma_200",
    "macd_line", "macd_signal", "macd_hist",
    "adx", "rsi_14", "rsi_2", "cci_20",
    "atr_14", "bb_upper", "bb_lower", "bb_width",
    "hist_vol_20", "obv", "mfi_14", "rsi_divergence",
    # Order flow (2)
    "large_trade_proxy", "institutional_accumulation",
    # Fundamentals (5)
    "pe_ratio", "pb_ratio", "debt_to_equity", "roe", "revenue_growth",
    # Price (4)
    "close_norm", "daily_return", "return_5d", "return_20d",
]
assert len(FEATURE_COLS) == 30


def _out_path(processed_dir: str, ticker: str) -> Path:
    safe = ticker.replace("-", "_")
    return Path(processed_dir) / "features" / f"{safe}.parquet"


def load_features(processed_dir: str, ticker: str) -> pd.DataFrame | None:
    p = _out_path(processed_dir, ticker)
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index)
    return df


def build_ticker(
    ticker: str,
    raw_dir: str,
    processed_dir: str,
    norm_window: int = 252,
    overwrite: bool = False,
) -> pd.DataFrame | None:
    """Build and save the full feature DataFrame for one ticker."""
    out_path = _out_path(processed_dir, ticker)
    if not overwrite and out_path.exists():
        log.debug("%s features already built", ticker)
        return load_features(processed_dir, ticker)

    ohlcv = ohlcv_store.load(raw_dir, ticker)
    if ohlcv is None or len(ohlcv) < 250:
        log.warning("%s: insufficient OHLCV data (%s rows)", ticker, len(ohlcv) if ohlcv is not None else 0)
        return None

    idx = ohlcv.index

    # --- Technical ---
    tech = technical.compute(ohlcv)

    # --- Order flow ---
    of = orderflow.compute(ohlcv)

    # --- Fundamentals ---
    raw_fund = fundamentals_store.load(raw_dir, ticker)
    fund = fundamentals.compute(raw_fund, idx)

    # --- Price features (scale-free, no z-score needed) ---
    close = ohlcv["close"]
    price = pd.DataFrame(index=idx)
    price["close_norm"]   = close / close.rolling(252, min_periods=30).mean() - 1.0
    price["daily_return"] = close.pct_change().clip(-0.5, 0.5)
    price["return_5d"]    = close.pct_change(5).clip(-0.6, 0.6)
    price["return_20d"]   = close.pct_change(20).clip(-0.8, 0.8)

    # --- Concatenate ---
    combined = pd.concat([tech, of, fund, price], axis=1)

    # --- Normalize technical and order flow with rolling z-score ---
    # (price returns and divergence flag are already scale-free; skip them)
    cols_to_normalize = [c for c in FEATURE_COLS
                         if c not in ("rsi_divergence", "close_norm",
                                      "daily_return", "return_5d", "return_20d",
                                      "large_trade_proxy", "institutional_accumulation")]
    if cols_to_normalize:
        normalized = rolling_zscore(combined[cols_to_normalize], window=norm_window)
        combined[cols_to_normalize] = normalized

    # Ensure exact column order and all 33 features present
    for col in FEATURE_COLS:
        if col not in combined.columns:
            combined[col] = 0.0
    combined = combined[FEATURE_COLS].astype("float32")

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path)
    log.debug("%s features saved: %d rows × %d cols", ticker, len(combined), len(combined.columns))
    return combined


def build_all(
    tickers: list[str],
    raw_dir: str,
    processed_dir: str,
    norm_window: int = 252,
    overwrite: bool = False,
) -> dict[str, pd.DataFrame]:
    """Build feature store for all tickers."""
    results = {}
    failed = []
    for i, ticker in enumerate(tickers, 1):
        df = build_ticker(ticker, raw_dir, processed_dir, norm_window, overwrite)
        if df is not None:
            results[ticker] = df
        else:
            failed.append(ticker)
        if i % 25 == 0:
            log.info("Feature pipeline: %d/%d done", i, len(tickers))
    if failed:
        log.warning("Feature build failed for: %s", failed)
    log.info("Feature pipeline complete: %d/%d tickers", len(results), len(tickers))
    return results
