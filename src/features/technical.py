"""
Technical indicator computation via pandas-ta.
All indicators use only prior-close data (no lookahead).
Returns a DataFrame of raw (unnormalized) indicator values.
"""
import numpy as np
import pandas as pd

from src.utils.logging_config import get_logger

log = get_logger("technical")

# Patch numpy for pandas-ta compatibility with NumPy 2.x
if not hasattr(np, "float"):
    np.float = float
    np.int = int
    np.bool = bool
    np.complex = complex


def compute(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical features for one ticker.

    Input: DataFrame with columns [open, high, low, close, volume], DatetimeIndex.
    Output: DataFrame of 19 unnormalized feature columns on the same index.
    """
    import pandas_ta as ta

    df = ohlcv.copy()
    out = pd.DataFrame(index=df.index)

    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]

    # --- Trend ---
    out["ema_12"]   = ta.ema(close, length=12)
    out["ema_26"]   = ta.ema(close, length=26)
    out["sma_50"]   = ta.sma(close, length=50)
    out["sma_200"]  = ta.sma(close, length=200)

    macd = ta.macd(close, fast=12, slow=26, signal=9)
    if macd is not None and len(macd.columns) >= 3:
        cols = macd.columns.tolist()
        out["macd_line"]   = macd.iloc[:, 0]
        out["macd_signal"] = macd.iloc[:, 2] if len(cols) >= 3 else macd.iloc[:, 1]
        out["macd_hist"]   = macd.iloc[:, 1] if len(cols) >= 3 else np.nan
    else:
        out["macd_line"] = out["macd_signal"] = out["macd_hist"] = np.nan

    adx = ta.adx(high, low, close, length=14)
    if adx is not None and len(adx.columns) >= 1:
        out["adx"] = adx.iloc[:, 0]
    else:
        out["adx"] = np.nan

    # --- Momentum ---
    out["rsi_14"] = ta.rsi(close, length=14)
    out["rsi_2"]  = ta.rsi(close, length=2)
    out["cci_20"] = ta.cci(high, low, close, length=20)

    # --- Volatility ---
    out["atr_14"] = ta.atr(high, low, close, length=14)

    bb = ta.bbands(close, length=20, std=2)
    if bb is not None and len(bb.columns) >= 3:
        out["bb_upper"] = bb.iloc[:, 0]   # BBU
        out["bb_lower"] = bb.iloc[:, 1]   # BBL
        out["bb_width"] = (bb.iloc[:, 0] - bb.iloc[:, 1]) / bb.iloc[:, 2].replace(0, np.nan)
    else:
        out["bb_upper"] = out["bb_lower"] = out["bb_width"] = np.nan

    # 20-day historical volatility (annualised)
    log_ret = np.log(close / close.shift(1))
    out["hist_vol_20"] = log_ret.rolling(20).std() * np.sqrt(252)

    # --- Volume ---
    out["obv"] = ta.obv(close, vol)
    out["mfi_14"] = ta.mfi(high, low, close, vol, length=14)

    # RSI divergence flag: +1 confirming, -1 diverging, 0 flat
    rsi_slope   = out["rsi_14"].diff(5).apply(np.sign)
    price_slope = close.diff(5).apply(np.sign)
    out["rsi_divergence"] = (rsi_slope == price_slope).map({True: 1.0, False: -1.0}).astype("float32")

    return out.astype("float32")
