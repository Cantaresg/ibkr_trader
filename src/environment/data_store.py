"""
MarketDataStore: preloads all data once and exposes fast numpy slices.
Shared across multiple TradingEnv instances (read-only after init).

Memory footprint (125 tickers, 4023 dates):
  features_arr:  125 × 4023 × 33 × 4 bytes ≈  66 MB
  close_arr:     125 × 4023 × 4  bytes       ≈   2 MB
  market_arr:          4023 × 7  × 4 bytes   ≈ 0.1 MB
  regime_arr:          4023 × 3  × 4 bytes   ≈ 0.05 MB
  Total:  ~68 MB
"""
from pathlib import Path
import numpy as np
import pandas as pd

from src.data.ohlcv_store import load as load_ohlcv
from src.features.pipeline import load_features, FEATURE_COLS
from src.features.market_features import build as build_market_features, N_MARKET_FEATURES
from src.scanner.scanner_store import load as load_scanner, get_candidates
from src.regime.hmm_detector import load_proba
from src.utils.config_loader import all_tickers, load_config
from src.utils.logging_config import get_logger

log = get_logger("env.data_store")


class MarketDataStore:
    """
    Preloads all market data, features, scanner rankings, and regime
    probabilities into numpy arrays for fast environment step() calls.
    """

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        window_id: str = "global",
    ):
        cfg = load_config(config_path)
        raw_dir = cfg["data"]["raw_dir"]
        proc_dir = cfg["data"]["processed_dir"]
        tickers = all_tickers(cfg["data"]["universe_file"])

        # --- Canonical date index (SPY trading calendar) ---
        spy_path = Path(raw_dir) / "market" / "SPY.parquet"
        spy_df = pd.read_parquet(spy_path)
        self.dates: np.ndarray = pd.DatetimeIndex(spy_df.index)
        self.n_dates: int = len(self.dates)
        self.date_to_idx: dict = {d: i for i, d in enumerate(self.dates)}

        # --- Ticker universe ---
        self.ticker_list: list[str] = tickers
        self.n_tickers: int = len(tickers)
        self.ticker_to_idx: dict[str, int] = {t: i for i, t in enumerate(tickers)}

        n_feat = len(FEATURE_COLS)
        self.features_arr = np.zeros((self.n_dates, self.n_tickers, n_feat), dtype=np.float32)
        self.close_arr    = np.zeros((self.n_dates, self.n_tickers), dtype=np.float32)

        log.info("Loading feature store for %d tickers...", self.n_tickers)
        for i, ticker in enumerate(tickers):
            feat = load_features(proc_dir, ticker)
            ohlcv = load_ohlcv(raw_dir, ticker)
            if feat is None or ohlcv is None:
                continue
            for date, di in self.date_to_idx.items():
                if date in feat.index:
                    self.features_arr[di, i, :] = feat.loc[date].values
                if date in ohlcv.index:
                    self.close_arr[di, i] = ohlcv.loc[date, "close"]
        np.nan_to_num(self.features_arr, nan=0.0, copy=False)
        log.info("Feature store loaded.")

        # --- Market features ---
        mkt_df = build_market_features(tickers, raw_dir, cache=True)
        self.market_arr = np.zeros((self.n_dates, N_MARKET_FEATURES), dtype=np.float32)
        for date, di in self.date_to_idx.items():
            if date in mkt_df.index:
                self.market_arr[di, :] = mkt_df.loc[date].values

        # --- Regime probabilities ---
        regime_df = load_proba(window_id)
        self.regime_arr = np.full((self.n_dates, 3), 1/3, dtype=np.float32)  # uniform default
        if regime_df is not None:
            for date, di in self.date_to_idx.items():
                if date in regime_df.index:
                    self.regime_arr[di, :] = regime_df.loc[date].values

        # --- Scanner rankings ---
        self.scanner_rankings = load_scanner()
        if self.scanner_rankings is None:
            raise RuntimeError("Scanner rankings not found — run build_scanner.py first")

        log.info("MarketDataStore ready: %d dates, %d tickers", self.n_dates, self.n_tickers)

    def get_obs_arrays(
        self,
        date_idx: int,
        ticker_indices: list[int],
        lookback: int = 30,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return raw observation arrays for the given date and ticker indices.
        date_idx   — index of the CURRENT date in self.dates
        Returns:
          stock_feat  (n_stocks, lookback, n_feat)  float32
          stock_mask  (n_stocks,)                   float32  1.0=real, 0.0=padded
          market_feat (lookback, 7)                 float32
          regime_prob (3,)                          float32
        """
        n_stocks = len(ticker_indices)
        n_feat   = self.features_arr.shape[2]

        start = max(0, date_idx - lookback + 1)
        end   = date_idx + 1  # inclusive

        window_len = end - start  # may be < lookback near the beginning

        stock_feat = np.zeros((n_stocks, lookback, n_feat), dtype=np.float32)
        stock_mask = np.zeros(n_stocks, dtype=np.float32)

        for j, ti in enumerate(ticker_indices):
            if ti < 0:  # padded slot
                continue
            f = self.features_arr[start:end, ti, :]  # (window_len, n_feat)
            stock_feat[j, lookback - window_len:, :] = f
            # mark as real only if the stock has non-zero close price at current date
            if self.close_arr[date_idx, ti] > 0:
                stock_mask[j] = 1.0

        mkt_raw = self.market_arr[start:end, :]  # (window_len, N_MARKET_FEATURES)
        market_feat = np.zeros((lookback, N_MARKET_FEATURES), dtype=np.float32)
        market_feat[lookback - window_len:, :] = mkt_raw

        regime_prob = self.regime_arr[date_idx, :].copy()

        return stock_feat, stock_mask, market_feat, regime_prob

    def get_prices(self, date_idx: int, ticker_indices: list[int]) -> np.ndarray:
        """Close prices for the given tickers at date_idx. Shape: (n_stocks,)."""
        return self.close_arr[date_idx, ticker_indices].copy()

    def get_candidates(self, date: pd.Timestamp, n: int = 20) -> list[int]:
        """
        Return ticker indices of the top-n scanner candidates for a given date.
        Pads with -1 for missing slots.
        """
        tickers = get_candidates(self.scanner_rankings, date, n)
        indices = []
        for t in tickers:
            if t in self.ticker_to_idx:
                idx = self.ticker_to_idx[t]
                # Only include if price data available
                date_i = self.date_to_idx.get(date)
                if date_i is not None and self.close_arr[date_i, idx] > 0:
                    indices.append(idx)
        # Pad to n with -1
        while len(indices) < n:
            indices.append(-1)
        return indices[:n]

    def valid_episode_start_indices(
        self,
        start_date: str,
        end_date: str,
        lookback: int = 30,
        episode_length: int = 252,
    ) -> list[int]:
        """
        Return date indices that can serve as valid episode start points.
        Requires lookback days of history before AND episode_length+1 days ahead.
        """
        s = pd.Timestamp(start_date)
        e = pd.Timestamp(end_date)
        valid = []
        for i, d in enumerate(self.dates):
            if d < s or d > e:
                continue
            if i < lookback:
                continue
            if i + episode_length + 1 >= self.n_dates:
                continue
            valid.append(i)
        return valid

    def group_starts_by_regime(self, valid_starts: list[int]) -> dict[int, list[int]]:
        """
        Bucket valid start indices by dominant regime (argmax of regime_arr).
        Returns {0: [bull indices], 1: [bear indices], 2: [trans indices]}.
        Falls back to uniform assignment if regime_arr is all 1/3 (no HMM fitted).
        """
        groups: dict[int, list[int]] = {0: [], 1: [], 2: []}
        for idx in valid_starts:
            regime = int(np.argmax(self.regime_arr[idx]))
            groups[regime].append(idx)
        return groups
