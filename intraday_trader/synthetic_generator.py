"""
IntradaySyntheticGenerator: generates synthetic bear market episodes for intraday data.

Method: Return Negation + Vol Scaling (same approach as src/data/synthetic_generator.py)

  1. Sample a bull-market window from the 1h OHLCV history (prior-day SPY return > +0.5%)
  2. Pick N_STOCKS tickers from the universe at random
  3. Negate log-returns of the close series, scale by vol_scale (bear/bull vol ratio)
  4. Reconstruct prices and recompute all 16 intraday features via the existing pipeline
  5. Pack as .npz: stock_features, close_prices, stock_mask

The synthetic episodes are saved to disk and served by IntradaySyntheticStore at train time.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from intraday_trader.constants import (
    BARS_PER_DAY,
    INTRADAY_UNIVERSE,
    LOOKBACK,
    N_FEATURES,
    N_STOCKS,
    UNIVERSE_FILE,
)
from intraday_trader.features import FEATURE_COLS, compute as compute_features
from src.utils.config_loader import load_config, all_tickers
from src.utils.logging_config import get_logger

log = get_logger("intraday.synthetic_generator")

# Bull bar: SPY prior-1h return > +0.05%  (50bp per bar is a strong move for 1h)
_BULL_BAR_RETURN_THRESHOLD = 0.0005
# Episode length = n_days * BARS_PER_DAY
_DEFAULT_EP_DAYS = 21
# Extra warm-up bars so features are stable at episode start
_WARMUP = 100


class IntradaySyntheticGenerator:
    """
    Loads 1h OHLCV at construction, generates synthetic bear episodes on demand.

    Episode arrays returned / saved per episode:
        stock_features  (n_stocks, lookback+ep_bars, N_FEATURES)   float32
        close_prices    (n_stocks, ep_bars+1)                       float32
        stock_mask      (n_stocks,)                                  float32
    """

    def __init__(
        self,
        config_path: str = "intraday_trader/config.yaml",
        n_stocks: int = N_STOCKS,
        n_days_per_episode: int = _DEFAULT_EP_DAYS,
    ):
        cfg           = load_config(config_path)
        self.raw_dir  = cfg.get("data", {}).get("raw_dir",       "intraday_trader/data/raw")
        self.n_stocks = n_stocks
        self.lookback = cfg.get("features", {}).get("lookback_bars", LOOKBACK)
        self.ep_bars  = n_days_per_episode * BARS_PER_DAY
        self._window  = _WARMUP + self.lookback + self.ep_bars + 1

        # Load full universe
        universe_file = cfg.get("universe", {}).get("file", UNIVERSE_FILE)
        try:
            tickers = all_tickers(universe_file)
        except Exception:
            tickers = list(INTRADAY_UNIVERSE)

        log.info("Loading 1h OHLCV for %d tickers...", len(tickers))
        self._ohlcv: dict[str, pd.DataFrame] = {}
        for t in tickers:
            raw_path = Path(self.raw_dir) / "ohlcv" / f"{t}.parquet"
            if not raw_path.exists():
                continue
            try:
                df = pd.read_parquet(raw_path)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0].lower() for c in df.columns]
                else:
                    df.columns = [c.lower() for c in df.columns]
                if len(df) >= self._window:
                    self._ohlcv[t] = df
            except Exception as e:
                log.debug("Skipping %s: %s", t, e)

        self.tickers = list(self._ohlcv.keys())
        log.info("  %d tickers with sufficient history", len(self.tickers))

        # Build bull bar list from SPY (every bar where SPY had a positive prior-bar return)
        self._bull_bars = self._find_bull_bars()
        log.info("  %d bull bar windows available", len(self._bull_bars))

        # Vol scale: estimated once from data
        self._vol_scale_cache: float | None = None

    # ------------------------------------------------------------------
    def generate_negation_episode(
        self,
        vol_scale: float | None = None,
        seed: int | None = None,
    ) -> dict | None:
        """
        Generate one synthetic bear episode by negating 1h log-returns from a bull window.

        Returns dict with keys: stock_features, close_prices, stock_mask
        Returns None if insufficient data for the sampled window.
        """
        rng = np.random.default_rng(seed)

        if vol_scale is None:
            if self._vol_scale_cache is None:
                self._vol_scale_cache = self._estimate_vol_scale()
            vol_scale = self._vol_scale_cache

        if not self._bull_bars:
            log.warning("No bull bar windows — cannot generate negation episode")
            return None

        # Sample a window start from bull bars
        start_bar_i = int(rng.choice(self._bull_bars))

        # Determine the needed calendar timestamps from SPY index
        spy = self._ohlcv.get("SPY") or next(iter(self._ohlcv.values()))
        idx = spy.index
        end_bar_i = start_bar_i + self._window
        if end_bar_i > len(idx):
            return None
        t_start, t_end = idx[start_bar_i], idx[end_bar_i - 1]

        # Pick eligible tickers covering the window
        eligible = [
            t for t in self.tickers
            if t_start in self._ohlcv[t].index and t_end in self._ohlcv[t].index
        ]
        if len(eligible) < self.n_stocks:
            log.debug(
                "Only %d eligible tickers for window %s–%s",
                len(eligible), t_start, t_end,
            )
            return None

        chosen = list(rng.choice(eligible, size=self.n_stocks, replace=False))

        stock_features = np.zeros(
            (self.n_stocks, self.lookback + self.ep_bars, N_FEATURES), dtype=np.float32
        )
        close_prices = np.zeros((self.n_stocks, self.ep_bars + 1), dtype=np.float32)
        stock_mask   = np.zeros(self.n_stocks, dtype=np.float32)

        for j, ticker in enumerate(chosen):
            raw = self._ohlcv[ticker]
            try:
                window = raw.loc[t_start:t_end]
            except Exception:
                continue
            if len(window) < self._window - 20:
                continue

            neg = _negate_ohlcv_1h(window, vol_scale)
            feat_df = _compute_features_from_ohlcv(neg)
            if feat_df is None or len(feat_df) < _WARMUP + self.lookback + self.ep_bars:
                continue

            feat_arr = feat_df[FEATURE_COLS].values.astype(np.float32)
            offset   = _WARMUP
            needed_close = offset + self.lookback + self.ep_bars + 1
            if len(neg) < needed_close:
                continue
            stock_features[j] = feat_arr[offset: offset + self.lookback + self.ep_bars]
            close_prices[j]   = neg["close"].values[
                offset + self.lookback: offset + self.lookback + self.ep_bars + 1
            ]
            stock_mask[j] = 1.0

        if stock_mask.sum() < self.n_stocks * 0.5:
            log.debug(
                "Too many masked slots in negation episode (%d/%d)",
                int(stock_mask.sum()), self.n_stocks,
            )
            return None

        return {
            "stock_features": stock_features,
            "close_prices":   close_prices,
            "stock_mask":     stock_mask,
        }

    # ------------------------------------------------------------------
    def _find_bull_bars(self) -> list[int]:
        """
        Find bar indices in SPY (or first available ticker) where a full
        window of _window bars fits and the prior-bar return is positive.
        We use this as a proxy for 'bull windows'.
        """
        ref = self._ohlcv.get("SPY") or (next(iter(self._ohlcv.values())) if self._ohlcv else None)
        if ref is None:
            return []

        close = ref["close"].values
        n = len(close)
        bull_bars = []
        for i in range(1, n - self._window):
            if close[i - 1] > 0 and close[i] > 0:
                bar_ret = (close[i] - close[i - 1]) / close[i - 1]
                if bar_ret > _BULL_BAR_RETURN_THRESHOLD:
                    bull_bars.append(i)
        return bull_bars

    # ------------------------------------------------------------------
    def _estimate_vol_scale(self) -> float:
        """
        Estimate the bear/bull realized-volatility ratio from SPY 1h returns.
        Clips the result to [1.0, 3.0].
        """
        ref = self._ohlcv.get("SPY") or (next(iter(self._ohlcv.values())) if self._ohlcv else None)
        if ref is None:
            return 1.5

        log_rets = np.diff(np.log(ref["close"].clip(lower=1e-6).values))
        # Split into positive and negative bars
        pos_rets = log_rets[log_rets > 0]
        neg_rets = log_rets[log_rets < 0]

        if len(pos_rets) < 10 or len(neg_rets) < 10:
            return 1.5

        vol_bull = float(np.std(pos_rets))
        vol_bear = float(np.std(neg_rets))
        if vol_bull < 1e-10:
            return 1.5
        return float(np.clip(vol_bear / vol_bull, 1.0, 3.0))


# ---------------------------------------------------------------------------
# OHLCV manipulation helpers
# ---------------------------------------------------------------------------

def _negate_ohlcv_1h(df: pd.DataFrame, vol_scale: float) -> pd.DataFrame:
    """
    Negate log-returns of the close series and reconstruct OHLCV.

    Steps (same logic as src/data/synthetic_generator.py::_negate_ohlcv()):
      1. Compute log-returns on close
      2. Negate and scale by vol_scale
      3. Reconstruct close prices from new log-returns
      4. Scale open/high/low by the close ratio (preserving intra-bar shape)
      5. Scale volume by vol_scale (bear markets have higher volume)
    """
    df = df.copy()
    close = df["close"].clip(lower=1e-6).values

    # Log-returns
    log_rets = np.diff(np.log(close))

    # Negate + scale
    neg_log_rets = -log_rets * vol_scale

    # Reconstruct close
    new_close = np.empty_like(close)
    new_close[0] = close[0]
    for i, lr in enumerate(neg_log_rets):
        new_close[i + 1] = new_close[i] * np.exp(lr)
    new_close = np.maximum(new_close, 1e-6)

    # Scale ratios for open/high/low
    ratio = new_close / np.maximum(close, 1e-6)
    df["close"] = new_close
    df["open"]  = np.maximum(df["open"].values  * ratio, 1e-6)
    df["high"]  = np.maximum(df["high"].values  * ratio, 1e-6)
    df["low"]   = np.maximum(df["low"].values   * ratio, 1e-6)
    df["volume"] = np.maximum(df["volume"].values * vol_scale, 0)

    return df


def _compute_features_from_ohlcv(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Re-run the intraday feature pipeline on a (possibly synthetic) OHLCV DataFrame.
    Returns a DataFrame with FEATURE_COLS, or None on failure.
    """
    try:
        return compute_features(df)
    except Exception as e:
        log.debug("Feature computation failed: %s", e)
        return None
