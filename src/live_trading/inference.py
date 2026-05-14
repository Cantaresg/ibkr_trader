"""
Live inference engine: builds the model observation and returns target weights.

Two operating modes:
  - LIVE mode  (default): uses LiveFeatureBuilder — downloads OHLCV from
    yfinance and computes features in-memory for ANY ticker universe.
  - REPLAY mode:          uses MarketDataStore — reads pre-built feature
    parquet files; used by walk-forward backtest scripts.

Observation construction mirrors TradingEnv._build_obs() exactly, then
applies the same FlattenDictObservation flattening used at train time:
    order: stocks, stock_mask, market, portfolio

Supports both PPO (stateless) and RecurrentPPO (LSTM) checkpoints.
For RPPO the LSTM hidden state is carried across consecutive trading days so
the model accumulates regime context — the whole point of using RPPO.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO

try:
    from sb3_contrib import RecurrentPPO
    _RPPO_AVAILABLE = True
except ImportError:
    _RPPO_AVAILABLE = False

from src.environment.trading_env import N_STOCKS, N_FEATURES, N_MARKET_FEATURES, N_REGIME
from src.utils.logging_config import get_logger

log = get_logger("live.inference")

PORTFOLIO_DIM = N_STOCKS + 1 + 1 + 1 + N_REGIME   # 26


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _load_model(path: str):
    """Load checkpoint, auto-detecting PPO vs RecurrentPPO. Returns (model, is_recurrent)."""
    if _RPPO_AVAILABLE:
        try:
            return RecurrentPPO.load(path), True
        except Exception:
            pass
    try:
        return PPO.load(path), False
    except Exception:
        pass
    raise ValueError(f"Cannot load checkpoint as PPO or RecurrentPPO: {path}")


class LiveInferenceEngine:
    """
    Loads a trained PPO or RecurrentPPO checkpoint and exposes predict() for live trading.

    For RecurrentPPO the LSTM hidden state is carried forward across daily predict()
    calls so the model accumulates multi-day regime context between rebalances.
    Call reset_lstm() to discard accumulated state (e.g. after a long data gap).

    Args:
        checkpoint_path: path to .zip model file (PPO or RecurrentPPO)
        feature_builder: LiveFeatureBuilder instance (live mode)
        data_store:      MarketDataStore instance (replay/backtest mode)
        lookback:        observation lookback window (default from config)

    Exactly one of feature_builder or data_store must be provided.
    """

    def __init__(
        self,
        checkpoint_path: str,
        feature_builder=None,   # LiveFeatureBuilder — for any universe
        data_store=None,         # MarketDataStore — for fixed-universe replay
        lookback: int = 30,
    ):
        if feature_builder is None and data_store is None:
            raise ValueError("Provide either feature_builder (live) or data_store (replay)")

        log.info("Loading model from %s", checkpoint_path)
        self.model, self._is_recurrent = _load_model(checkpoint_path)
        log.info("Model type: %s", "RecurrentPPO (LSTM)" if self._is_recurrent else "PPO")

        self._builder        = feature_builder
        self._ds             = data_store
        self.lookback        = lookback
        self._live_mode      = feature_builder is not None

        # Current portfolio state — updated after each rebalance
        self._stock_weights  = np.zeros(N_STOCKS,  dtype=np.float32)
        self._cash_weight    = 1.0
        self._nav_norm       = 1.0
        self._drawdown       = 0.0

        # RPPO LSTM state — persists across daily predict() calls
        # None means "start of episode" (episode_start flag = True)
        self._lstm_state     = None
        self._is_first_step  = True

    # ------------------------------------------------------------------
    def reset_lstm(self) -> None:
        """Discard accumulated LSTM state. Call after restarts or long data gaps."""
        self._lstm_state    = None
        self._is_first_step = True
        log.info("LSTM state reset.")

    # ------------------------------------------------------------------
    def update_portfolio_state(
        self,
        stock_weights: np.ndarray,
        cash_weight:   float,
        nav_norm:      float,
        drawdown:      float,
    ) -> None:
        """Inject real live portfolio state before calling predict()."""
        self._stock_weights = stock_weights.astype(np.float32)
        self._cash_weight   = float(cash_weight)
        self._nav_norm      = float(nav_norm)
        self._drawdown      = float(drawdown)

    # ------------------------------------------------------------------
    def predict(
        self,
        date: pd.Timestamp,
        universe: list[str] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Build observation for `date` and return (target_weights, tickers).

        In live mode:  runs scan_universe(universe) → top-20 → features.
        In replay mode: uses data_store.get_candidates() + get_obs_arrays().

        target_weights: shape (N_STOCKS + 1,) — softmax, sums to 1.0.
                        Last element is cash weight.
        tickers:        list[str] of N_STOCKS selected ticker symbols.
        """
        if self._live_mode:
            return self._predict_live(universe or [])
        else:
            return self._predict_replay(date)

    # ------------------------------------------------------------------
    # Live path (any universe, real-time features)
    # ------------------------------------------------------------------
    def _predict_live(self, universe: list[str]) -> tuple[np.ndarray, list[str]]:
        if not universe:
            log.warning("Empty universe — returning all-cash weights")
            return self._all_cash(), []

        top20 = self._builder.scan_universe(universe, n=N_STOCKS)
        if not top20:
            return self._all_cash(), []

        # Pad to exactly N_STOCKS slots
        tickers = (top20 + [""] * N_STOCKS)[:N_STOCKS]

        stock_feat, stock_mask, market_feat, regime_prob = \
            self._builder.build_obs_arrays(tickers)

        return self._run_model(stock_feat, stock_mask, market_feat, regime_prob, tickers)

    # ------------------------------------------------------------------
    # Replay path (fixed universe via MarketDataStore)
    # ------------------------------------------------------------------
    def _predict_replay(self, date: pd.Timestamp) -> tuple[np.ndarray, list[str]]:
        date_idx = self._resolve_date_idx(date)
        if date_idx is None:
            log.warning("Date %s not in data store", date)
            return self._all_cash(), []

        ticker_indices = self._ds.get_candidates(date, N_STOCKS)
        tickers = [
            self._ds.ticker_list[i] if i >= 0 else "" for i in ticker_indices
        ]
        stock_feat, stock_mask, market_feat, regime_prob = \
            self._ds.get_obs_arrays(date_idx, ticker_indices, self.lookback)

        return self._run_model(stock_feat, stock_mask, market_feat, regime_prob, tickers)

    # ------------------------------------------------------------------
    # Shared model call
    # ------------------------------------------------------------------
    def _run_model(
        self,
        stock_feat:  np.ndarray,   # (N_STOCKS, lookback, 33)
        stock_mask:  np.ndarray,   # (N_STOCKS,)
        market_feat: np.ndarray,   # (lookback, 7)
        regime_prob: np.ndarray,   # (3,)
        tickers:     list[str],
    ) -> tuple[np.ndarray, list[str]]:
        portfolio_vec = np.concatenate([
            self._stock_weights,
            [self._cash_weight],
            [self._nav_norm],
            [self._drawdown],
            regime_prob,
        ]).astype(np.float32)

        # Flatten in training order: stocks, stock_mask, market, portfolio
        flat_obs = np.concatenate([
            stock_feat.flatten(),
            stock_mask.flatten(),
            market_feat.flatten(),
            portfolio_vec.flatten(),
        ]).astype(np.float32)

        if self._is_recurrent:
            # Pass and update LSTM state so regime context accumulates across days
            ep_start = np.array([self._is_first_step], dtype=bool)
            action, new_lstm_state = self.model.predict(
                flat_obs,
                state=self._lstm_state,
                episode_start=ep_start,
                deterministic=True,
            )
            self._lstm_state    = new_lstm_state
            self._is_first_step = False
        else:
            action, _ = self.model.predict(flat_obs, deterministic=True)

        weights = _softmax(np.squeeze(action))

        log.info("Predict: top-5 allocs %s  cash=%.1f%%",
                 [(tickers[i], f"{weights[i]*100:.1f}%")
                  for i in np.argsort(weights[:-1])[-5:][::-1]],
                 weights[-1] * 100)

        return weights, tickers

    # ------------------------------------------------------------------
    def _all_cash(self) -> np.ndarray:
        w = np.zeros(N_STOCKS + 1, dtype=np.float32)
        w[-1] = 1.0
        return w

    def _resolve_date_idx(self, date: pd.Timestamp) -> int | None:
        for delta in range(6):
            d = date - pd.Timedelta(days=delta)
            if d in self._ds.date_to_idx:
                return self._ds.date_to_idx[d]
        return None
