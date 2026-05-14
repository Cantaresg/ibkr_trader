"""
Sentiment feature: Phase 1 returns neutral baseline aligned to trading calendar.
Phase 2 (GDELT + FinBERT) will populate real scores.
"""
import pandas as pd

from src.data.sentiment_store import get_or_neutral, SENTIMENT_COLS  # noqa: F401


def compute(raw_dir: str, ticker: str, ohlcv_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Return sentiment features aligned to ohlcv_index."""
    return get_or_neutral(raw_dir, ticker, ohlcv_index)
