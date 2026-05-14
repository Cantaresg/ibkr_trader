"""
IntradayInferenceEngine: loads a trained intraday PPO checkpoint and produces
one set of portfolio weights per hourly bar decision.

Builds the 2785-dim flat observation from live 1h OHLCV data and the current
portfolio state, then runs model.predict() to get target weights.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from src.environment.wrappers import FlattenDictObservation
from intraday_trader.constants import (
    BARS_PER_DAY,
    INTRADAY_UNIVERSE,
    LOOKBACK,
    N_FEATURES,
    N_MARKET,
    N_STOCKS,
    PORTFOLIO_DIM,
)
from intraday_trader.env import _apply_min_weight
from intraday_trader.features import FEATURE_COLS, compute as compute_features
from intraday_trader.market_features import MARKET_FEATURE_COLS
from intraday_trader.portfolio import IntradayPortfolioState
from src.utils.logging_config import get_logger

log = get_logger("intraday.inference")


class IntradayInferenceEngine:
    """
    Loads trained PPO checkpoint and runs live hourly inference.

    Maintains an IntradayPortfolioState that the caller must update after
    each bar via update_portfolio_state().
    reset_for_new_day() must be called at the start of each trading session.
    """

    def __init__(
        self,
        checkpoint_path: str,
        universe: list[str] | None = None,
        lookback: int = LOOKBACK,
        initial_capital: float = 5_000.0,
        min_position_weight: float = 0.0,
    ):
        import pyarrow.parquet  # noqa: F401 — must precede torch on Windows
        from intraday_trader.backtester import _resolve_model_class

        self.checkpoint_path = checkpoint_path
        self.universe        = universe or INTRADAY_UNIVERSE
        self.lookback        = lookback

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Intraday checkpoint not found: {checkpoint_path}")

        ModelClass, algo = _resolve_model_class(checkpoint_path, "ppo")
        self._is_recurrent = (algo == "rppo")
        log.info("Loading intraday %s checkpoint from %s", algo.upper(), checkpoint_path)
        self.model = ModelClass.load(checkpoint_path)
        self.model.set_env(None)

        self.portfolio = IntradayPortfolioState(
            n_stocks=N_STOCKS,
            initial_capital=initial_capital,
        )

        self.min_pos_weight = float(min_position_weight)
        self._feat_cache:   dict[str, pd.DataFrame] = {}
        self._market_cache: pd.DataFrame | None = None
        self._bar_history:  list[pd.Timestamp] = []
        self._lstm_state    = None
        self._is_first_step = True

    # ------------------------------------------------------------------
    def reset_for_new_day(self) -> None:
        """Call at 9:30 ET before the first bar of a new session."""
        self.portfolio.reset()
        self._feat_cache    = {}
        self._market_cache  = None
        self._bar_history   = []
        self._is_first_step = True
        log.info("IntradayInferenceEngine: new day reset")

    # ------------------------------------------------------------------
    def update_portfolio_state(
        self,
        stock_weights: np.ndarray,
        cash_weight:   float,
        nav_norm:      float,
        drawdown:      float,
    ) -> None:
        """Sync portfolio state to match actual broker positions after each bar fills."""
        w = np.zeros(N_STOCKS + 1, dtype=np.float32)
        w[:len(stock_weights)] = stock_weights[:N_STOCKS]
        w[-1] = cash_weight
        self.portfolio.weights  = w
        self.portfolio.nav      = self.portfolio.initial_capital * (1.0 + nav_norm)
        self.portfolio.peak_nav = max(self.portfolio.peak_nav, self.portfolio.nav)

    # ------------------------------------------------------------------
    def predict(self, bar_ts: pd.Timestamp) -> tuple[np.ndarray, list[str]]:
        """
        Run model inference for the given bar timestamp.
        Downloads up-to-date 1h OHLCV if needed.

        Returns:
            weights  : np.ndarray (N_STOCKS + 1,) — softmax portfolio weights
            tickers  : list[str] — the fixed intraday universe
        """
        self._refresh_live_data(bar_ts)
        obs = self._build_obs_flat(bar_ts)
        if self._is_recurrent:
            ep_start = np.array([self._is_first_step], dtype=bool)
            action, self._lstm_state = self.model.predict(
                obs, state=self._lstm_state,
                episode_start=ep_start, deterministic=True)
            self._is_first_step = False
        else:
            action, _ = self.model.predict(obs, deterministic=True)
        weights = _softmax(action)
        weights = _apply_min_weight(weights, self.min_pos_weight, N_STOCKS)
        return weights, list(self.universe[:N_STOCKS])

    # ------------------------------------------------------------------
    def _refresh_live_data(self, bar_ts: pd.Timestamp) -> None:
        """Download or update in-memory 1h OHLCV for all universe tickers."""
        for ticker in self.universe:
            try:
                raw = yf.download(
                    ticker,
                    period="5d",
                    interval="1h",
                    auto_adjust=True,
                    progress=False,
                )
                if raw.empty:
                    continue
                raw.columns = [c.lower() if isinstance(c, str) else c[0].lower()
                               for c in raw.columns]
                raw = _filter_market_hours(raw)
                self._feat_cache[ticker] = compute_features(raw)
            except Exception as e:
                log.warning("Failed refreshing %s: %s", ticker, e)

        try:
            spy_raw = yf.download("SPY", period="5d", interval="1h",
                                  auto_adjust=True, progress=False)
            if not spy_raw.empty:
                spy_raw.columns = [c.lower() if isinstance(c, str) else c[0].lower()
                                   for c in spy_raw.columns]
                spy_raw = _filter_market_hours(spy_raw)
                self._market_cache = self._build_market_live(spy_raw)
        except Exception as e:
            log.warning("Failed refreshing SPY market data: %s", e)

    # ------------------------------------------------------------------
    def _build_market_live(self, spy_1h: pd.DataFrame) -> pd.DataFrame:
        """Construct market feature DataFrame from live SPY 1h data."""
        out = pd.DataFrame(index=spy_1h.index)
        dates    = spy_1h.index.normalize()
        day_open = spy_1h["open"].groupby(dates).transform("first")

        out["spy_bar_return"]      = spy_1h["close"].pct_change().clip(-0.1, 0.1).fillna(0).astype("float32")
        out["spy_intraday_return"] = ((spy_1h["close"] - day_open) / day_open.replace(0, np.nan)).fillna(0).astype("float32")
        out["vix_level"]           = 0.0
        out["spy_rel_volume"]      = 1.0
        out["market_breadth_intraday"] = 0.5

        if self._feat_cache:
            breadth = []
            for t in self.universe:
                df = self._feat_cache.get(t)
                if df is not None and "intraday_return" in df.columns:
                    aligned = df["intraday_return"].reindex(spy_1h.index).fillna(0)
                    breadth.append((aligned > 0).astype(float))
            if breadth:
                bd = pd.concat(breadth, axis=1).mean(axis=1)
                out["market_breadth_intraday"] = bd.reindex(spy_1h.index).fillna(0.5).astype("float32")

        return out[MARKET_FEATURE_COLS].fillna(0).astype("float32")

    # ------------------------------------------------------------------
    def _build_obs_flat(self, bar_ts: pd.Timestamp) -> np.ndarray:
        """Build the 2785-dim flat observation for model.predict()."""
        tickers = list(self.universe[:N_STOCKS])

        stock_feat = np.zeros((N_STOCKS, self.lookback, N_FEATURES), dtype=np.float32)
        stock_mask = np.zeros(N_STOCKS, dtype=np.float32)

        for ti, ticker in enumerate(tickers):
            df = self._feat_cache.get(ticker)
            if df is None:
                continue
            rows = df[df.index <= bar_ts].tail(self.lookback)
            if rows.empty:
                continue
            wlen = len(rows)
            stock_feat[ti, self.lookback - wlen:, :] = rows[FEATURE_COLS].values
            stock_mask[ti] = 1.0

        market_feat = np.zeros((self.lookback, N_MARKET), dtype=np.float32)
        if self._market_cache is not None:
            rows = self._market_cache[self._market_cache.index <= bar_ts].tail(self.lookback)
            if not rows.empty:
                wlen = len(rows)
                market_feat[self.lookback - wlen:, :] = rows[MARKET_FEATURE_COLS].values

        portfolio_vec = self.portfolio.state_vector()

        obs = np.concatenate([
            stock_feat.flatten(),
            stock_mask,
            market_feat.flatten(),
            portfolio_vec,
        ]).astype(np.float32)

        return obs


# ------------------------------------------------------------------
def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max()
    exp_x = np.exp(x)
    return (exp_x / exp_x.sum()).astype(np.float32)


def _filter_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only bars within regular market hours (9:30–15:30 ET inclusive)."""
    if df.empty:
        return df
    import pytz
    ET = pytz.timezone("America/New_York")
    if df.index.tzinfo is None:
        idx_et = df.index.tz_localize("UTC").tz_convert(ET)
    else:
        idx_et = df.index.tz_convert(ET)
    hours = idx_et.hour
    mask  = (hours >= 9) & (hours <= 15)
    return df[mask]
