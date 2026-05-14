"""
Rule-based stock scanner.

Daily composite score (uses only data available at close of day t-1):
  score = 0.40 * momentum_12_1  (rank-normalised cross-sectionally)
        + 0.30 * volume_activity (volume z-score vs 20-day average, clipped [-2,2])
        + 0.30 * news_activity   (article-count z-score 5-day, zero in Phase 1)

Selects top N stocks by score each trading day.
No lookahead: all inputs are prior-close prices.
"""
import numpy as np
import pandas as pd

from src.data.ohlcv_store import load as load_ohlcv
from src.data.sentiment_store import load as load_sentiment
from src.features.normalizer import rank_normalize
from src.utils.config_loader import load_config
from src.utils.logging_config import get_logger

log = get_logger("scanner.rule_based")

W_MOM   = 0.40
W_VOL   = 0.30
W_NEWS  = 0.30
MIN_HIST = 300   # minimum rows before a stock enters rankings


def _load_article_count_panel(tickers: list[str], raw_dir: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Load article_count_zscore from sentiment parquets into a wide date×ticker panel."""
    series = {}
    for t in tickers:
        df = load_sentiment(raw_dir, t)
        if df is not None and "article_count_zscore" in df.columns:
            series[t] = df["article_count_zscore"].reindex(index, fill_value=0.0)
    if not series:
        return pd.DataFrame(0.0, index=index, columns=tickers)
    panel = pd.DataFrame(series).reindex(columns=tickers, fill_value=0.0)
    return panel.fillna(0.0)


def build_rankings(
    tickers: list[str],
    raw_dir: str,
    n_candidates: int = 20,
) -> pd.DataFrame:
    """
    Compute daily top-N rankings for all tickers.
    Returns DataFrame: rows=dates, cols=['rank_01'..'rank_20'], values=ticker strings.
    Dates with fewer than n_candidates eligible stocks are padded with ''.
    """
    log.info("Loading OHLCV for scanner (%d tickers)...", len(tickers))

    closes: dict[str, pd.Series] = {}
    volumes: dict[str, pd.Series] = {}

    for t in tickers:
        df = load_ohlcv(raw_dir, t)
        if df is not None and len(df) >= MIN_HIST:
            closes[t] = df["close"].rename(t)
            volumes[t] = df["volume"].rename(t)

    if not closes:
        raise RuntimeError("No OHLCV data found — run download_data.py first")

    # Wide panels: date x ticker
    close_panel  = pd.DataFrame(closes).sort_index()
    volume_panel = pd.DataFrame(volumes).sort_index().reindex(close_panel.index)

    log.info("Computing cross-sectional scanner scores (%d dates)...", len(close_panel))

    # --- News activity (article count z-score, 5-day smoothed) ---
    news_panel = _load_article_count_panel(list(closes.keys()), raw_dir, close_panel.index)
    news_panel_5d = news_panel.rolling(5, min_periods=1).mean()

    # --- Momentum 12_1 ---
    # 12-month return: shift by 252 trading days
    # 1-month return:  shift by 21 trading days
    # Use shift(1) everywhere so today's close is NOT yet known
    ret_12m = close_panel.shift(1).pct_change(252)
    ret_1m  = close_panel.shift(1).pct_change(21)
    mom = ret_12m - ret_1m  # raw momentum signal

    # --- Volume activity z-score ---
    vol_mean = volume_panel.shift(1).rolling(20, min_periods=10).mean()
    vol_std  = volume_panel.shift(1).rolling(20, min_periods=10).std(ddof=1).replace(0, np.nan)
    vol_z    = ((volume_panel.shift(1) - vol_mean) / vol_std).clip(-2, 2)

    # --- Build per-date rankings ---
    rank_cols = [f"rank_{i:02d}" for i in range(1, n_candidates + 1)]
    rows = []
    dates = close_panel.index

    for date in dates:
        # Cross-sectional rank of momentum (uses only data at date t, but computed
        # from shift(1) so it's actually data through close of t-1)
        mom_row  = mom.loc[date].dropna()
        vol_row  = vol_z.loc[date].reindex(mom_row.index).fillna(0)
        news_row = news_panel_5d.loc[date].reindex(mom_row.index).fillna(0).clip(-2, 2)

        # Only include tickers with sufficient history at this date
        valid = mom_row.index
        if len(valid) == 0:
            rows.append([''] * n_candidates)
            continue

        # Rank-normalise momentum cross-sectionally to [-1, 1]
        mom_rank = rank_normalize(mom_row)

        score = W_MOM * mom_rank + W_VOL * vol_row + W_NEWS * news_row

        top_n = score.nlargest(n_candidates).index.tolist()
        # Pad if fewer than n_candidates available
        top_n += [''] * (n_candidates - len(top_n))
        rows.append(top_n[:n_candidates])

    rankings = pd.DataFrame(rows, index=dates, columns=rank_cols)
    log.info("Scanner complete: %d dates x top-%d", len(rankings), n_candidates)
    return rankings
