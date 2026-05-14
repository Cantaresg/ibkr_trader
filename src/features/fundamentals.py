"""
Fundamental feature preparation: sector-relative normalization, daily reindex.
"""
import numpy as np
import pandas as pd

from src.data.fundamentals_store import FUNDAMENTAL_COLS, forward_fill_to_daily
from src.utils.logging_config import get_logger

log = get_logger("fundamentals_features")


def compute(
    raw_fundamentals: pd.DataFrame | None,
    ohlcv_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Convert raw quarterly fundamentals to daily feature values.
    Forward-fills with 1-day lag. Returns zeros if no data available.
    """
    return forward_fill_to_daily(raw_fundamentals, ohlcv_index, lag_days=1)


def sector_normalize(
    all_fundamentals: dict[str, pd.DataFrame],
    ticker_to_sector: dict[str, str],
) -> dict[str, pd.DataFrame]:
    """
    Normalize each fundamental ratio relative to sector peers at each date.
    Prevents cross-sector comparisons (tech P/E vs utility P/E).

    Returns the same dict with values replaced by sector-relative z-scores.
    """
    # Group tickers by sector
    sector_groups: dict[str, list[str]] = {}
    for ticker, sector in ticker_to_sector.items():
        sector_groups.setdefault(sector, []).append(ticker)

    result = {}
    for sector, sector_tickers in sector_groups.items():
        # Collect all tickers in this sector that have data
        available = [t for t in sector_tickers if t in all_fundamentals]
        if not available:
            continue

        # Build cross-sectional panel: date × ticker
        panel = pd.concat(
            {t: all_fundamentals[t] for t in available},
            axis=1,
        )
        # panel has MultiIndex columns: (ticker, feature)
        # Normalize each feature across tickers at each date
        for col in FUNDAMENTAL_COLS:
            try:
                cross = panel.xs(col, level=1, axis=1)  # date × ticker
            except KeyError:
                continue
            mu    = cross.mean(axis=1)
            sigma = cross.std(axis=1, ddof=1).replace(0, np.nan)
            normalized = cross.sub(mu, axis=0).div(sigma, axis=0).clip(-3, 3)
            for ticker in available:
                if ticker not in result:
                    result[ticker] = all_fundamentals[ticker].copy()
                if col in normalized.columns:
                    result[ticker][col] = normalized[ticker] if ticker in normalized.columns else np.nan

    # Pass through tickers not in any sector group
    for ticker, df in all_fundamentals.items():
        if ticker not in result:
            result[ticker] = df

    return result
