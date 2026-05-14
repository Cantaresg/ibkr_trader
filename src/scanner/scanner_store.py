"""Persist and load precomputed daily scanner rankings."""
from pathlib import Path
import pandas as pd

_PATH = "data/processed/scanner/rankings.parquet"


def save(rankings: pd.DataFrame) -> None:
    p = Path(_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    rankings.to_parquet(p)


def load() -> pd.DataFrame | None:
    p = Path(_PATH)
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index)
    return df


def get_candidates(rankings: pd.DataFrame, date: pd.Timestamp, n: int = 20) -> list[str]:
    """Return top-n tickers for a given date, dropping empty padding strings."""
    if date not in rankings.index:
        # Fall back to nearest prior date
        prior = rankings.index[rankings.index <= date]
        if len(prior) == 0:
            return []
        date = prior[-1]
    row = rankings.loc[date]
    return [t for t in row.values if t != ''][:n]
