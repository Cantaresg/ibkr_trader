"""
Intraday daily stock scanner.

Selects the top-N stocks per trading day from the full universe using four signals
computed from prior-day 1h OHLCV only (no lookahead bias):

  score = w_momentum      * momentum_rank        [-1, 1]   rank-normalised cross-sectional
        + w_volume        * volume_spike_z        [-2, 2]   prior-day total volume z-score
        + w_recovery      * recovery_score        [0, 1]    rank-normalised relu(ret + threshold)
        + w_proximity     * proximity_to_52w_low  [0, 1]    rank-normalised 52-week low proximity

Momentum rank:    prior-day return rank-normalised to [-1, 1] cross-sectionally.
Volume spike:     (prior_total_volume / 20-day avg daily volume - 1), z-scored, clipped [-2, 2].
Recovery:         relu(prior_day_return - (-threshold)) — stocks down >threshold% score positively.
                  Rank-normalised to [0, 1] after relu so scores are comparable.
Proximity to low: 1 - (close - 52w_low) / (52w_high - 52w_low).
                  1.0 = at the annual low, 0.0 = at the annual high.
                  Allows growth stocks that have pulled back to their year-low to surface.

Output: DataFrame rows=dates, cols=[rank_01…rank_N], values=ticker strings.
        Dates with fewer than N eligible tickers are padded with ''.

Reference: src/scanner/rule_based.py::build_rankings()
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from intraday_trader.constants import INTRADAY_UNIVERSE, UNIVERSE_FILE
from src.utils.config_loader import load_config, all_tickers
from src.utils.logging_config import get_logger

log = get_logger("intraday.scanner")

_RANKINGS_PATH = "intraday_trader/data/processed/scanner/intraday_rankings.parquet"
_INTRADAY_RANKINGS_PATH = "intraday_trader/data/processed/scanner/intraday_live_rankings.parquet"
_MIN_HISTORY   = 200   # minimum 1h bars required before a ticker enters rankings


def build_rankings(
    tickers: list[str],
    raw_dir: str,
    n_candidates: int = 20,
    w_momentum:  float = 0.35,
    w_volume:    float = 0.35,
    w_recovery:  float = 0.30,
    w_proximity: float = 0.0,
    recovery_threshold: float = 0.03,
) -> pd.DataFrame:
    """
    Build daily top-N rankings from prior-close 1h OHLCV data.

    Parameters
    ----------
    tickers      : Full universe list.
    raw_dir      : Directory with per-ticker `ohlcv/<ticker>.parquet` files.
    n_candidates : Stocks to select per day (default 20).
    w_momentum / w_volume / w_recovery / w_proximity : Signal weights (should sum to 1.0).
    recovery_threshold : Prior-day return below which recovery signal activates
                         (e.g. 0.03 = stocks down >3%).

    Returns
    -------
    DataFrame  rows=trading dates, cols=[rank_01…rank_N], values=ticker strings.
    """
    log.info("Building intraday scanner rankings (%d tickers, top-%d)...", len(tickers), n_candidates)

    # --- Load per-ticker daily summaries from 1h OHLCV ---
    # We need: per-day total volume, per-day close (last bar of day)
    daily_close:  dict[str, pd.Series] = {}
    daily_volume: dict[str, pd.Series] = {}

    for ticker in tickers:
        raw_path = Path(raw_dir) / "ohlcv" / f"{ticker}.parquet"
        if not raw_path.exists():
            continue
        try:
            df = pd.read_parquet(raw_path)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]

            if "close" not in df.columns or "volume" not in df.columns:
                continue
            if len(df) < _MIN_HISTORY:
                continue

            # Normalise index to tz-naive date (yfinance returns tz-aware, Google Drive tz-naive)
            idx = df.index
            if idx.tzinfo is not None:
                idx = idx.tz_localize(None)
            dates = idx.normalize()

            # Per-day close = last bar close of the day
            close_daily = df.groupby(dates)["close"].last()
            # Per-day volume = sum of all bar volumes
            vol_daily   = df.groupby(dates)["volume"].sum()

            daily_close[ticker]  = close_daily
            daily_volume[ticker] = vol_daily
        except Exception as e:
            log.debug("Skipping %s: %s", ticker, e)

    if not daily_close:
        raise RuntimeError(
            "No 1h OHLCV data found. Run download_initial_1h.py first."
        )

    # Wide panels: date x ticker
    close_panel  = pd.DataFrame(daily_close).sort_index()
    volume_panel = pd.DataFrame(daily_volume).sort_index().reindex(close_panel.index)

    log.info("Scanner input: %d dates × %d tickers", len(close_panel), len(close_panel.columns))

    # --- Signal 1: Prior-day return (momentum) ---
    # shift(1) so at date t we only know close through t-1
    prior_ret = close_panel.shift(1).pct_change(1, fill_method=None)   # prior-day return

    # --- Signal 2: Volume spike ---
    # 20-day rolling mean of prior-day volume
    prior_vol      = volume_panel.shift(1)
    vol_mean_20    = prior_vol.rolling(20, min_periods=10).mean()
    vol_std_20     = prior_vol.rolling(20, min_periods=10).std(ddof=1).replace(0, np.nan)
    volume_z       = ((prior_vol - vol_mean_20) / vol_std_20).clip(-2.0, 2.0)

    # --- Signal 4: 52-week low proximity ---
    # 1.0 = stock is AT its 52-week low; 0.0 = at its 52-week high.
    # Uses prior-day close (shift(1)) to avoid lookahead bias.
    prior_close   = close_panel.shift(1)
    low_52w       = prior_close.rolling(252, min_periods=50).min()
    high_52w      = prior_close.rolling(252, min_periods=50).max()
    range_52w     = (high_52w - low_52w).replace(0, np.nan)
    proximity_52w = (1.0 - (prior_close - low_52w) / range_52w).fillna(0.5).clip(0, 1)

    # --- Build per-date rankings ---
    rank_cols = [f"rank_{i:02d}" for i in range(1, n_candidates + 1)]
    rows  = []
    dates = close_panel.index

    for date in dates:
        ret_row = prior_ret.loc[date].dropna()
        vol_row = volume_z.loc[date].reindex(ret_row.index).fillna(0.0)

        valid = ret_row.index
        if len(valid) == 0:
            rows.append([""] * n_candidates)
            continue

        # Signal 1: momentum — rank-normalise to [-1, 1]
        mom_rank = _rank_normalize(ret_row)

        # Signal 2: volume spike (already z-scored, clipped)
        vol_spike = vol_row.reindex(valid).fillna(0.0)

        # Signal 3: recovery — relu(-prior_ret - threshold), rank-normalised [0, 1]
        #   Stocks with prior_ret < -threshold get a positive recovery score.
        recovery_raw  = np.maximum(0.0, -ret_row - recovery_threshold)
        recovery_rank = _rank_normalize_01(recovery_raw)

        # Signal 4: 52-week low proximity, rank-normalised [0, 1]
        prox_row  = proximity_52w.loc[date].reindex(valid).fillna(0.5)
        prox_rank = _rank_normalize_01(prox_row)

        score = (w_momentum  * mom_rank
               + w_volume    * vol_spike
               + w_recovery  * recovery_rank
               + w_proximity * prox_rank)

        top_n = score.nlargest(n_candidates).index.tolist()
        top_n += [""] * (n_candidates - len(top_n))
        rows.append(top_n[:n_candidates])

    rankings = pd.DataFrame(rows, index=dates, columns=rank_cols)
    log.info("Scanner complete: %d dates × top-%d", len(rankings), n_candidates)
    return rankings


def build_intraday_rankings(
    tickers: list[str],
    raw_dir: str,
    n_candidates: int = 20,
    lookback_hours: int = 3,
    w_momentum: float = 0.60,
    w_volume: float = 0.25,
    w_stability: float = 0.15,
    min_history_bars: int = _MIN_HISTORY,
) -> pd.DataFrame:
    """
    Build per-bar intraday rankings from current-day signals using only past bars.

    No lookahead policy:
      - all inputs are shifted by 1 bar before scoring at timestamp t
      - ranking at t is therefore based on information available at t-1

    Signals
    -------
    momentum : lookback-hour return (lagged 1 bar), rank-normalized to [-1, 1]
    volume   : lagged volume z/relative spike, rank-normalized to [-1, 1]
    stability: negative absolute 1h return (lagged), rank-normalized to [-1, 1]
               (penalizes vertical spikes / potential exhaustion bars)

    Returns
    -------
    DataFrame rows=timestamps, cols=[rank_01..rank_N], values=tickers.
    """
    log.info(
        "Building intraday live rankings (%d tickers, top-%d, lookback=%dh)...",
        len(tickers), n_candidates, lookback_hours,
    )

    close_map: dict[str, pd.Series] = {}
    vol_map: dict[str, pd.Series] = {}

    for ticker in tickers:
        raw_path = Path(raw_dir) / "ohlcv" / f"{ticker}.parquet"
        if not raw_path.exists():
            continue
        try:
            df = pd.read_parquet(raw_path)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]

            if "close" not in df.columns or "volume" not in df.columns:
                continue
            if len(df) < min_history_bars:
                continue

            # Strip timezone so all tickers share a consistent tz-naive index
            if df.index.tzinfo is not None:
                df.index = df.index.tz_localize(None)
            close_map[ticker] = df["close"].astype(float)
            vol_map[ticker] = df["volume"].astype(float)
        except Exception as e:
            log.debug("Skipping %s for intraday rankings: %s", ticker, e)

    if not close_map:
        raise RuntimeError("No OHLCV data available to build intraday rankings")

    close_panel = pd.DataFrame(close_map).sort_index()
    vol_panel = pd.DataFrame(vol_map).sort_index().reindex(close_panel.index)

    # Lag all bar-level signals by one bar to avoid lookahead.
    mom_raw = close_panel.pct_change(lookback_hours, fill_method=None).shift(1)
    ret_1h = close_panel.pct_change(1, fill_method=None).shift(1)

    vol_lag = vol_panel.shift(1)
    vol_mu = vol_lag.rolling(20, min_periods=5).mean().replace(0, np.nan)
    vol_rel = (vol_lag / vol_mu - 1.0).clip(-2.0, 5.0)

    stability_raw = -ret_1h.abs()

    rank_cols = [f"rank_{i:02d}" for i in range(1, n_candidates + 1)]
    rows = []
    for ts in close_panel.index:
        mom_row = mom_raw.loc[ts].dropna()
        if mom_row.empty:
            rows.append([""] * n_candidates)
            continue

        valid = mom_row.index
        vol_row = vol_rel.loc[ts].reindex(valid).fillna(0.0)
        stab_row = stability_raw.loc[ts].reindex(valid).fillna(0.0)

        mom_rank = _rank_normalize(mom_row)
        vol_rank = _rank_normalize(vol_row)
        stab_rank = _rank_normalize(stab_row)

        score = w_momentum * mom_rank + w_volume * vol_rank + w_stability * stab_rank
        top_n = score.nlargest(n_candidates).index.tolist()
        top_n += [""] * (n_candidates - len(top_n))
        rows.append(top_n[:n_candidates])

    rankings = pd.DataFrame(rows, index=close_panel.index, columns=rank_cols)
    log.info("Intraday live scanner complete: %d bars × top-%d", len(rankings), n_candidates)
    return rankings


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_rankings(rankings: pd.DataFrame, path: str = _RANKINGS_PATH) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rankings.to_parquet(p)
    log.info("Saved intraday rankings: %s  (%d dates)", p, len(rankings))


def load_rankings(path: str = _RANKINGS_PATH) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index)
    return df


def get_candidates(
    rankings: pd.DataFrame,
    date: pd.Timestamp,
    n: int = 20,
) -> list[str]:
    """
    Return top-n tickers for a given date, dropping empty padding.
    Falls back to nearest prior date if exact date not in index.
    """
    # Normalise to tz-naive date-only so comparison works regardless of
    # whether the caller passes a tz-aware timestamp (e.g. from yfinance data).
    date = pd.Timestamp(date.date())
    if date not in rankings.index:
        prior = rankings.index[rankings.index <= date]
        if len(prior) == 0:
            return []
        date = prior[-1]
    row = rankings.loc[date]
    return [t for t in row.values if t != ""][:n]


# ---------------------------------------------------------------------------
# Normalisation helpers (pure functions)
# ---------------------------------------------------------------------------

def _rank_normalize(series: pd.Series) -> pd.Series:
    """Rank-normalise to [-1, 1] cross-sectionally. NaN → 0."""
    n = len(series)
    if n == 0:
        return series
    ranks = series.rank(method="average").fillna((n + 1) / 2)
    return (2 * ranks / (n + 1) - 1).astype(float)


def _rank_normalize_01(series: pd.Series) -> pd.Series:
    """Rank-normalise to [0, 1]. NaN/all-zero → 0."""
    if series.sum() == 0:
        return pd.Series(0.0, index=series.index)
    n = len(series)
    ranks = series.rank(method="average").fillna(1.0)
    return ((ranks - 1) / max(n - 1, 1)).astype(float)
