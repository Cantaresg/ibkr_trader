"""
Live feature builder: computes observation arrays for ANY set of tickers.

Unlike MarketDataStore (which requires pre-built parquet files for a fixed
universe), this class downloads OHLCV fresh via yfinance and computes all
33 features in-memory. The model sees the same feature space regardless of
which stocks are selected — no retraining needed for a different universe.

Typical daily flow:
    builder = LiveFeatureBuilder(config_path="config/config.yaml")
    top20   = builder.scan_universe(my_universe, n=20)
    arrays  = builder.build_obs_arrays(top20)
    # arrays → (stock_feat, stock_mask, market_feat, regime_prob)
    # pass directly to LiveInferenceEngine.predict_from_arrays()
"""
from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from src.data.fundamentals_store import forward_fill_to_daily
from src.data.sentiment_store import neutral_baseline
from src.features import technical, orderflow
from src.features.market_features import N_MARKET_FEATURES
from src.features.normalizer import rolling_zscore
from src.features.pipeline import FEATURE_COLS
from src.regime.hmm_detector import load_model, predict_proba, load_proba
from src.utils.config_loader import load_config
from src.utils.logging_config import get_logger

log = get_logger("live.feature_builder")

N_STOCK_FEATURES  = len(FEATURE_COLS)    # 33
HISTORY_DAYS      = 520    # ~2 calendar years; covers SMA-200 warm-up + 252-day z-score
MIN_HISTORY_ROWS  = 260    # minimum trading-day rows to consider a ticker usable

# Columns to skip z-score normalisation (already scale-free)
_NO_ZSCORE = frozenset({
    "rsi_divergence", "close_norm", "daily_return", "return_5d", "return_20d",
    "sentiment_score", "article_count_zscore", "sentiment_dispersion",
    "large_trade_proxy", "institutional_accumulation",
})

# Market tickers needed for the 7 market features
_MARKET_TICKERS = {
    "SPY":  "spy",
    "^VIX": "vix",
    "^TNX": "tnx",   # 10-yr yield
    "^IRX": "irx",   # 3-month yield
    "LQD":  "lqd",   # investment-grade bonds (credit proxy)
    "HYG":  "hyg",   # high-yield bonds
}


class LiveFeatureBuilder:
    """
    Downloads OHLCV and computes observation arrays for any ticker list.
    Call scan_universe() to pick candidates, then build_obs_arrays() to get
    the arrays needed for model.predict().
    """

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        lookback:    int | None = None,
    ):
        cfg = load_config(config_path)
        self.lookback  = lookback or cfg["features"]["lookback_window"]
        self._hmm      = None
        self._labels   = None
        self._load_hmm()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan_universe(self, universe: list[str], n: int = 20) -> list[str]:
        """
        Score all tickers in `universe` by momentum + volume and return top-n.
        Downloads the minimum OHLCV needed for scoring (~280 trading days).
        Tickers with insufficient history are silently dropped.
        """
        log.info("Scanning %d tickers for top-%d candidates...", len(universe), n)
        scores: dict[str, float] = {}

        closes: dict[str, pd.Series] = {}
        volumes: dict[str, pd.Series] = {}

        raw = self._download_batch(universe, period="2y", fields=["Close", "Volume"])
        for ticker in universe:
            if ticker not in raw or raw[ticker] is None:
                continue
            df = raw[ticker]
            if len(df) < MIN_HISTORY_ROWS:
                continue
            closes[ticker]  = df["Close"]
            volumes[ticker] = df["Volume"]

        if not closes:
            log.warning("No usable tickers after history filter")
            return []

        # Cross-sectional momentum score: 12-month return minus 1-month (momentum 12-1)
        mom_scores: dict[str, float] = {}
        for t, c in closes.items():
            ret_12m = c.pct_change(252).iloc[-2] if len(c) > 253 else 0.0   # skip last day
            ret_1m  = c.pct_change(21).iloc[-2]  if len(c) > 22  else 0.0
            mom_scores[t] = (ret_12m or 0.0) - (ret_1m or 0.0)

        # Cross-sectional rank-normalise momentum to [-1, 1]
        tks = list(mom_scores.keys())
        vals = np.array([mom_scores[t] for t in tks], dtype=float)
        if vals.std() > 0:
            ranks = vals.argsort().argsort().astype(float)
            mom_norm = 2 * ranks / max(len(ranks) - 1, 1) - 1
        else:
            mom_norm = np.zeros(len(tks))
        mom_map = dict(zip(tks, mom_norm))

        # Volume activity: (today's volume / 20-day avg volume) z-scored, clipped
        vol_scores: dict[str, float] = {}
        for t, v in volumes.items():
            avg20 = v.rolling(20, min_periods=5).mean().iloc[-1]
            vol_scores[t] = float(np.clip((v.iloc[-1] / avg20 - 1) if avg20 > 0 else 0.0, -2, 2))

        # Combined score (match scanner weights: 0.40 mom + 0.30 vol + 0.30 news=0)
        for t in tks:
            scores[t] = 0.40 * mom_map.get(t, 0.0) + 0.30 * vol_scores.get(t, 0.0)

        top_n = sorted(scores, key=lambda t: scores[t], reverse=True)[:n]
        log.info("Top-%d selected: %s", n, top_n)
        return top_n

    def build_obs_arrays(
        self,
        tickers: list[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Download OHLCV for `tickers`, compute 33 features, and return arrays
        in the same format as MarketDataStore.get_obs_arrays():

          stock_feat  (n_stocks, lookback, 33)   float32
          stock_mask  (n_stocks,)                 float32  — 1 if valid, 0 if padded
          market_feat (lookback, 7)               float32
          regime_prob (3,)                        float32  — [p_bull, p_bear, p_trans]
        """
        n = len(tickers)
        stock_feat  = np.zeros((n, self.lookback, N_STOCK_FEATURES), dtype=np.float32)
        stock_mask  = np.zeros(n, dtype=np.float32)
        market_feat = np.zeros((self.lookback, N_MARKET_FEATURES), dtype=np.float32)
        regime_prob = np.array([1/3, 1/3, 1/3], dtype=np.float32)   # fallback: uniform

        # 1. Download stock OHLCV
        raw = self._download_batch(tickers, period="3y",
                                   fields=["Open", "High", "Low", "Close", "Volume"])

        # 2. Compute features for each ticker
        for i, ticker in enumerate(tickers):
            if ticker not in raw or raw[ticker] is None:
                continue
            df = raw[ticker]
            if len(df) < MIN_HISTORY_ROWS:
                log.debug("Skipping %s — insufficient history (%d rows)", ticker, len(df))
                continue

            ohlcv = df[["Open", "High", "Low", "Close", "Volume"]].rename(columns=str.lower)
            feat_df = self._features_from_ohlcv(ohlcv)
            if feat_df is None or len(feat_df) < self.lookback:
                continue

            window = feat_df.iloc[-self.lookback:].values    # (lookback, 33)
            stock_feat[i] = window
            stock_mask[i] = 1.0

        # 3. Market features
        mkt = self._build_market_features()
        if mkt is not None and len(mkt) >= self.lookback:
            market_feat = mkt.iloc[-self.lookback:].values.astype(np.float32)

        # 4. Regime
        regime_prob = self._current_regime_prob(mkt)

        return stock_feat, stock_mask, market_feat, regime_prob

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _download_batch(
        self,
        tickers:  list[str],
        period:   str,
        fields:   list[str],
    ) -> dict[str, pd.DataFrame | None]:
        """Download OHLCV for all tickers in one yfinance call."""
        result: dict[str, pd.DataFrame | None] = {t: None for t in tickers}
        if not tickers:
            return result
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = yf.download(
                    tickers,
                    period=period,
                    auto_adjust=True,
                    progress=False,
                    group_by="ticker" if len(tickers) > 1 else None,
                    threads=True,
                )
        except Exception as e:
            log.warning("yfinance batch download failed: %s", e)
            return result

        if len(tickers) == 1:
            t = tickers[0]
            df = raw[[f for f in fields if f in raw.columns]]
            if not df.empty:
                result[t] = df
        else:
            for t in tickers:
                try:
                    if t in raw.columns.get_level_values(0):
                        df = raw[t][[f for f in fields if f in raw[t].columns]]
                        if not df.empty:
                            result[t] = df
                except Exception:
                    pass

        return result

    def _features_from_ohlcv(self, ohlcv: pd.DataFrame) -> pd.DataFrame | None:
        """
        Compute all 33 features in-memory from raw OHLCV.
        Fundamentals and sentiment are zero/neutral (same as synthetic path).
        """
        try:
            idx   = ohlcv.index
            tech  = technical.compute(ohlcv)
            of    = orderflow.compute(ohlcv)
            fund  = forward_fill_to_daily(None, idx)   # returns NaN → filled below
            sent  = neutral_baseline(idx)               # zeros

            close = ohlcv["close"]
            price = pd.DataFrame(index=idx)
            price["close_norm"]   = close / close.rolling(252, min_periods=30).mean() - 1.0
            price["daily_return"] = close.pct_change().clip(-0.5, 0.5)
            price["return_5d"]    = close.pct_change(5).clip(-0.6, 0.6)
            price["return_20d"]   = close.pct_change(20).clip(-0.8, 0.8)

            combined = pd.concat([tech, of, fund, sent, price], axis=1)

            cols_norm = [c for c in FEATURE_COLS if c not in _NO_ZSCORE]
            if cols_norm:
                combined[cols_norm] = rolling_zscore(combined[cols_norm], window=252)

            for col in FEATURE_COLS:
                if col not in combined.columns:
                    combined[col] = 0.0

            return combined[FEATURE_COLS].astype("float32").fillna(0.0)
        except Exception as e:
            log.debug("Feature build failed: %s", e)
            return None

    def _build_market_features(self) -> pd.DataFrame | None:
        """
        Compute the 7 market features from live yfinance data.
        Returns DataFrame (dates, 7) aligned to trading calendar.
        """
        try:
            raw_tks = list(_MARKET_TICKERS.keys())
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = yf.download(
                    raw_tks, period="3y",
                    auto_adjust=True, progress=False,
                    group_by="ticker", threads=True,
                )

            spy  = raw["SPY"]["Close"]
            vix  = raw["^VIX"]["Close"]
            tnx  = raw["^TNX"]["Close"]
            irx  = raw["^IRX"]["Close"]
            lqd  = raw["LQD"]["Close"]
            hyg  = raw["HYG"]["Close"]

            idx = spy.index

            # 1. VIX z-score (252-day)
            vix_z = (vix - vix.rolling(252, min_periods=30).mean()) / (
                vix.rolling(252, min_periods=30).std() + 1e-8)

            # 2. VIX term structure (VIX vs 3-month rolling mean of VIX)
            vix_term = vix / (vix.rolling(63, min_periods=10).mean() + 1e-8) - 1.0

            # 3. SPY trend (price / SMA-20 - 1)
            spy_trend = spy / spy.rolling(20, min_periods=5).mean() - 1.0

            # 4. Market breadth proxy (SPY / SMA-200)
            spy_breadth = (spy > spy.rolling(200, min_periods=50).mean()).astype(float)

            # 5. Yield spread (10yr - 3mo, in pct)
            yield_spread = (tnx - irx) / 100.0

            # 6. Credit spread proxy (LQD return vs SPY return, 20-day)
            credit_spread = (lqd.pct_change(20) - spy.pct_change(20)).clip(-0.3, 0.3)

            # 7. Put-call ratio proxy (zero — same as training)
            put_call = pd.Series(0.0, index=idx)

            mkt = pd.DataFrame({
                "vix_zscore":       vix_z,
                "vix_term_structure": vix_term,
                "spy_trend_20d":    spy_trend,
                "market_breadth":   spy_breadth,
                "yield_spread":     yield_spread,
                "credit_spread":    credit_spread,
                "put_call_ratio":   put_call,
            }, index=idx).ffill().fillna(0.0).astype("float32")

            return mkt
        except Exception as e:
            log.warning("Market feature build failed: %s", e)
            return None

    def _current_regime_prob(self, market_df: pd.DataFrame | None) -> np.ndarray:
        """
        Use the fitted HMM to predict current regime from live market features.
        Falls back to uniform [1/3, 1/3, 1/3] if HMM not available.
        """
        if self._hmm is None or market_df is None:
            return np.array([1/3, 1/3, 1/3], dtype=np.float32)
        try:
            proba = predict_proba(self._hmm, market_df, self._labels)
            row   = proba.iloc[-1].values.astype(np.float32)
            return row
        except Exception as e:
            log.debug("HMM predict failed: %s", e)
            return np.array([1/3, 1/3, 1/3], dtype=np.float32)

    def _load_hmm(self) -> None:
        result = load_model(window_id="global")
        if result is not None:
            self._hmm, self._labels = result
            log.info("HMM loaded (global).")
        else:
            log.warning("No fitted HMM found — regime will be uniform.")
