"""
Order flow / participant behavior proxies.
All signals are computable from daily OHLCV alone.
"""
import numpy as np
import pandas as pd


def compute(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Compute order flow features for one ticker.

    Input: DataFrame with columns [open, high, low, close, volume], DatetimeIndex.
    Output: DataFrame with 2 feature columns.
    """
    close  = ohlcv["close"]
    high   = ohlcv["high"]
    low    = ohlcv["low"]
    volume = ohlcv["volume"]

    out = pd.DataFrame(index=ohlcv.index)

    # large_trade_proxy: volume z-score × sign(daily return)
    # Positive = high volume + price up = likely institutional accumulation
    # Negative = high volume + price down = likely distribution
    vol_mean = volume.rolling(20, min_periods=10).mean().replace(0, np.nan)
    vol_std  = volume.rolling(20, min_periods=10).std(ddof=1).replace(0, np.nan)
    vol_z    = (volume - vol_mean) / vol_std
    daily_ret_sign = close.pct_change().apply(np.sign)
    out["large_trade_proxy"] = (vol_z * daily_ret_sign).clip(-3, 3)

    # institutional_accumulation: Chaikin Money Flow (20d)
    # Positive = sustained buying pressure, Negative = selling
    hl_range = (high - low).replace(0, np.nan)
    money_flow_mult = ((close - low) - (high - close)) / hl_range
    money_flow_vol  = money_flow_mult * volume
    cmf_num = money_flow_vol.rolling(20, min_periods=10).sum()
    cmf_den = volume.rolling(20, min_periods=10).sum().replace(0, np.nan)
    out["institutional_accumulation"] = (cmf_num / cmf_den).clip(-1, 1)

    return out.astype("float32")
