"""
RollingZScore: normalize each feature at time t using only the prior 252 trading days.
No global fit — no lookahead bias.
"""
import numpy as np
import pandas as pd


def rolling_zscore(df: pd.DataFrame, window: int = 252, clip: float = 3.0) -> pd.DataFrame:
    """
    Compute rolling z-score for every column in df.
    At time t: z = (x_t - mean(x_{t-window:t})) / std(x_{t-window:t})
    Requires at least `window` prior rows; rows with insufficient history are NaN.
    Clipped to [-clip, clip] to handle outliers.
    """
    # Ensure min_periods never exceeds window for small windows (e.g. 20)
    minp = min(window, max(30, window // 4))
    roll = df.rolling(window=window, min_periods=minp)
    mu = roll.mean()
    sigma = roll.std(ddof=1).replace(0, np.nan)
    z = (df - mu) / sigma
    return z.clip(-clip, clip).astype("float32")


def rolling_zscore_series(s: pd.Series, window: int = 252, clip: float = 3.0) -> pd.Series:
    minp = min(window, max(30, window // 4))
    roll = s.rolling(window=window, min_periods=minp)
    mu = roll.mean()
    sigma = roll.std(ddof=1).replace(0, np.nan)
    z = (s - mu) / sigma
    return z.clip(-clip, clip).astype("float32")


def rank_normalize(s: pd.Series, window: int = 252) -> pd.Series:
    """
    Cross-sectional rank normalization to [-1, 1].
    Used by scanner's momentum_12_1 to rank across the universe each day.
    Applied within a single date's cross-section, not over time.
    """
    rank = s.rank(pct=True)  # 0 to 1
    return ((rank * 2) - 1).astype("float32")  # -1 to 1
