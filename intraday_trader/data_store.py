"""
IntradayDataStore: preloads 1h OHLCV features into numpy arrays for fast env step() calls.

Key design: a flat bar index runs across all dates and all bars within each day.
Days are separated by overnight gaps but the array is contiguous.

Memory footprint (12 tickers, ~2000 bars):
  features_arr:  2000 × 12 × 16 × 4 bytes ≈  1.5 MB
  close_arr:     2000 × 12 × 4  bytes      ≈  0.1 MB
  market_arr:    2000 × 5  × 4  bytes      ≈  0.04 MB
  Total: ~2 MB
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
    N_MARKET,
    N_STOCKS,
    UNIVERSE_FILE,
)
from intraday_trader.features import FEATURE_COLS, build_all, load_features
from intraday_trader.market_features import MARKET_FEATURE_COLS, build as build_market, load as load_market
from src.utils.config_loader import load_config, all_tickers
from src.utils.logging_config import get_logger

log = get_logger("intraday.data_store")


class IntradayDataStore:
    """
    Preloads all intraday data once; exposes fast numpy slices for env step() calls.

    Attributes
    ----------
    tickers         : list[str]        — the fixed intraday universe
    bar_timestamps  : list[pd.Timestamp]  — flat bar index → datetime
    bar_to_day_idx  : np.ndarray (n_bars,) — flat bar → trading day index
    bar_within_day  : np.ndarray (n_bars,) — flat bar → position 0..BARS_PER_DAY-1
    features_arr    : np.ndarray (n_bars, n_tickers, N_FEATURES)
    close_arr       : np.ndarray (n_bars, n_tickers)
    market_arr      : np.ndarray (n_bars, N_MARKET)
    """

    def __init__(
        self,
        config_path: str = "intraday_trader/config.yaml",
        universe: list[str] | None = None,
        force_rebuild: bool = False,
    ):
        cfg      = load_config(config_path)
        raw_dir  = cfg.get("data", {}).get("raw_dir",       "intraday_trader/data/raw")
        proc_dir = cfg.get("data", {}).get("processed_dir", "intraday_trader/data/processed")
        self.raw_dir  = raw_dir
        self.proc_dir = proc_dir

        if universe is None:
            universe_file = cfg.get("universe", {}).get("file", UNIVERSE_FILE)
            try:
                universe = all_tickers(universe_file)
            except Exception:
                universe = cfg.get("universe", {}).get("tickers", INTRADAY_UNIVERSE)
        self.tickers: list[str] = list(universe)
        self.n_tickers = len(self.tickers)
        self.ticker_to_idx: dict[str, int] = {t: i for i, t in enumerate(self.tickers)}

        # --- Load (or build) per-ticker features ---
        log.info("Loading intraday features for %d tickers...", self.n_tickers)
        feat_dfs: dict[str, pd.DataFrame] = {}
        for t in self.tickers:
            df = load_features(proc_dir, t)
            if df is None or force_rebuild:
                df = None
            if df is not None:
                feat_dfs[t] = df

        missing = [t for t in self.tickers if t not in feat_dfs]
        if missing:
            log.info("Building features for missing tickers: %s", missing)
            built = build_all(missing, raw_dir, proc_dir)
            feat_dfs.update(built)

        # --- Build canonical bar index from SPY (or first available ticker) ---
        spy_df = feat_dfs.get("SPY") or (next(iter(feat_dfs.values())) if feat_dfs else None)
        if spy_df is None:
            raise RuntimeError(
                "No intraday feature data found. Run IntradayDataUpdater first."
            )

        self.bar_timestamps: list[pd.Timestamp] = list(spy_df.index)
        self.n_bars = len(self.bar_timestamps)
        self.bar_to_ts: dict[pd.Timestamp, int] = {ts: i for i, ts in enumerate(self.bar_timestamps)}

        # Compute day/bar-within-day indices
        dates = pd.DatetimeIndex(self.bar_timestamps).normalize()
        unique_dates = sorted(dates.unique())
        self.trading_dates: list[pd.Timestamp] = unique_dates
        self.n_days = len(unique_dates)
        date_to_day_idx = {d: i for i, d in enumerate(unique_dates)}

        self.bar_to_day_idx = np.zeros(self.n_bars, dtype=np.int32)
        self.bar_within_day = np.zeros(self.n_bars, dtype=np.int32)

        day_bar_counters: dict[int, int] = {}
        for flat_i, ts in enumerate(self.bar_timestamps):
            d  = ts.normalize()
            di = date_to_day_idx[d]
            self.bar_to_day_idx[flat_i] = di
            cnt = day_bar_counters.get(di, 0)
            self.bar_within_day[flat_i] = cnt
            day_bar_counters[di] = cnt + 1

        # --- Load market features ---
        market_df = load_market(proc_dir)
        if market_df is None or force_rebuild:
            log.info("Building intraday market features...")
            intraday_returns = {
                t: df["intraday_return"] for t, df in feat_dfs.items() if "intraday_return" in df.columns
            }
            market_df = build_market(raw_dir, proc_dir, intraday_returns, cache=True)

        # --- Build numpy arrays aligned to bar_timestamps ---
        self.features_arr = np.zeros((self.n_bars, self.n_tickers, N_FEATURES), dtype=np.float32)
        self.close_arr    = np.zeros((self.n_bars, self.n_tickers), dtype=np.float32)
        self.market_arr   = np.zeros((self.n_bars, N_MARKET), dtype=np.float32)

        for ti, ticker in enumerate(self.tickers):
            df = feat_dfs.get(ticker)
            if df is None:
                continue
            df_aligned = df.reindex(self.bar_timestamps)
            for feat_i, col in enumerate(FEATURE_COLS):
                if col in df_aligned.columns:
                    vals = df_aligned[col].values
                    self.features_arr[:, ti, feat_i] = np.nan_to_num(vals, nan=0.0)

            # Close prices from raw OHLCV
            raw_path = Path(raw_dir) / "ohlcv" / f"{ticker}.parquet"
            if raw_path.exists():
                raw_df = pd.read_parquet(raw_path)
                raw_df.columns = [c.lower() for c in raw_df.columns]
                if "close" in raw_df.columns:
                    close_aligned = raw_df["close"].reindex(self.bar_timestamps).values
                    self.close_arr[:, ti] = np.nan_to_num(close_aligned, nan=0.0)

        # Market features
        mkt_aligned = market_df.reindex(self.bar_timestamps)
        for feat_i, col in enumerate(MARKET_FEATURE_COLS):
            if col in mkt_aligned.columns:
                self.market_arr[:, feat_i] = np.nan_to_num(mkt_aligned[col].values, nan=0.0)

        # --- Cross-sectional: fill momentum_open_rank ---
        rank_col_idx = FEATURE_COLS.index("momentum_open_rank")
        self._fill_momentum_open_rank(rank_col_idx)

        log.info(
            "IntradayDataStore ready: %d bars, %d trading days, %d tickers",
            self.n_bars, self.n_days, self.n_tickers,
        )

        # --- Load scanner rankings (optional) ---
        scanner_cfg      = cfg.get("scanner", {})
        rankings_path    = scanner_cfg.get(
            "rankings_path",
            "intraday_trader/data/processed/scanner/intraday_rankings.parquet",
        )
        intraday_cfg = scanner_cfg.get("intraday", {})
        intraday_rankings_path = intraday_cfg.get(
            "rankings_path",
            "intraday_trader/data/processed/scanner/intraday_live_rankings.parquet",
        )
        self._n_candidates: int = scanner_cfg.get("n_candidates", N_STOCKS)
        self._rankings: pd.DataFrame | None = None
        self._intraday_rankings: pd.DataFrame | None = None
        try:
            from intraday_trader.scanner import load_rankings
            self._rankings = load_rankings(rankings_path)
            if self._rankings is not None:
                log.info("Loaded scanner rankings: %d dates", len(self._rankings))
            else:
                log.warning(
                    "Scanner rankings not found at %s. "
                    "Falling back to first-N tickers. Run build_scanner.py first.",
                    rankings_path,
                )
        except Exception as e:
            log.warning("Could not load scanner rankings (%s). Using first-N fallback.", e)

        try:
            from intraday_trader.scanner import load_rankings
            self._intraday_rankings = load_rankings(intraday_rankings_path)
            if self._intraday_rankings is not None:
                log.info("Loaded intraday scanner rankings: %d bars", len(self._intraday_rankings))
            else:
                log.warning(
                    "Intraday scanner rankings not found at %s. "
                    "Run build_scanner.py --mode intraday first.",
                    intraday_rankings_path,
                )
        except Exception as e:
            log.warning("Could not load intraday scanner rankings (%s).", e)

    # ------------------------------------------------------------------
    def _fill_momentum_open_rank(self, rank_col_idx: int) -> None:
        """
        For each bar, compute cross-sectional rank of intraday_return
        (feature index 1) among all tickers, scaled [-1, 1].
        """
        ir_col = FEATURE_COLS.index("intraday_return")
        for bar_i in range(self.n_bars):
            ir_vals = self.features_arr[bar_i, :, ir_col]
            if ir_vals.sum() == 0:
                continue
            ranks = pd.Series(ir_vals).rank(pct=True).values
            normalized = (ranks * 2 - 1).astype("float32")
            self.features_arr[bar_i, :, rank_col_idx] = normalized

    # ------------------------------------------------------------------
    def get_obs_arrays(
        self,
        flat_bar_idx: int,
        lookback: int = LOOKBACK,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return observation arrays for the first N_STOCKS tickers (legacy / backtest use).
        Prefer get_obs_arrays_for_tickers() in the env.
        """
        default_indices = list(range(min(N_STOCKS, self.n_tickers)))
        while len(default_indices) < N_STOCKS:
            default_indices.append(-1)
        return self.get_obs_arrays_for_tickers(flat_bar_idx, default_indices, lookback)

    # ------------------------------------------------------------------
    def get_obs_arrays_for_tickers(
        self,
        flat_bar_idx: int,
        ticker_indices: list[int],
        lookback: int = LOOKBACK,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return observation arrays for the specified ticker index slots.

        Parameters
        ----------
        ticker_indices : list of length N_STOCKS; each element is an index into
                         self.tickers, or -1 for a masked (empty) slot.

        Returns
        -------
        stock_feat  : (N_STOCKS, lookback, N_FEATURES)  float32
        stock_mask  : (N_STOCKS,)                        float32  1.0=has price, 0=zero
        market_feat : (lookback, N_MARKET)               float32
        """
        start = max(0, flat_bar_idx - lookback + 1)
        end   = flat_bar_idx + 1
        wlen  = end - start

        n = len(ticker_indices)
        stock_feat = np.zeros((n, lookback, N_FEATURES), dtype=np.float32)
        stock_mask = np.zeros(n, dtype=np.float32)

        for slot, ti in enumerate(ticker_indices):
            if ti < 0 or ti >= self.n_tickers:
                continue
            f = self.features_arr[start:end, ti, :]
            stock_feat[slot, lookback - wlen:, :] = f
            if self.close_arr[flat_bar_idx, ti] > 0:
                stock_mask[slot] = 1.0

        market_feat = np.zeros((lookback, N_MARKET), dtype=np.float32)
        market_feat[lookback - wlen:, :] = self.market_arr[start:end, :]

        return stock_feat, stock_mask, market_feat

    # ------------------------------------------------------------------
    def get_bar_returns(self, flat_bar_idx: int) -> np.ndarray:
        """
        Bar-over-bar returns for all tickers at this bar step.
        Uses close[flat_bar_idx + 1] / close[flat_bar_idx] - 1.
        """
        next_idx = flat_bar_idx + 1
        if next_idx >= self.n_bars:
            return np.zeros(N_STOCKS, dtype=np.float32)

        p_now  = self.close_arr[flat_bar_idx,  :N_STOCKS]
        p_next = self.close_arr[next_idx,       :N_STOCKS]

        rets = np.where(
            (p_now > 0) & (p_next > 0),
            (p_next - p_now) / p_now,
            0.0,
        ).astype(np.float32)
        return rets

    # ------------------------------------------------------------------
    def valid_episode_start_bars(
        self,
        start_date: str,
        end_date: str,
        lookback: int = LOOKBACK,
        n_days: int = 21,
    ) -> list[int]:
        """
        Return flat bar indices that can serve as episode starts.
        Each valid start is the FIRST bar of a trading day such that:
          - there are >= lookback bars of history before it
          - there are >= n_days * BARS_PER_DAY + 1 bars after it
        """
        s = pd.Timestamp(start_date)
        e = pd.Timestamp(end_date)
        # Strip timezone from bar_timestamps if present so comparison is tz-naive
        bar_ts = self.bar_timestamps
        if bar_ts[0].tzinfo is not None:
            bar_ts = [t.tz_localize(None) for t in bar_ts]
        required_after = n_days * BARS_PER_DAY + 1
        valid = []
        for flat_i, ts in enumerate(bar_ts):
            if ts < s or ts > e:
                continue
            if self.bar_within_day[flat_i] != 0:
                continue
            if flat_i < lookback:
                continue
            if flat_i + required_after >= self.n_bars:
                continue
            valid.append(flat_i)
        return valid

    # ------------------------------------------------------------------
    def bar_idx_for_timestamp(self, ts: pd.Timestamp) -> int | None:
        """Look up the flat bar index for a given timestamp."""
        return self.bar_to_ts.get(ts)

    # ------------------------------------------------------------------
    def get_candidates(self, date: pd.Timestamp, n: int | None = None) -> list[int]:
        """
        Return ticker indices (into self.tickers) for the top-N scanner picks
        on the given date.  Falls back to first-N tickers when no rankings are
        available or the date is before scanner history begins.

        Returns
        -------
        List of length n (or fewer if universe is small), each element being
        a valid index into self.tickers.  Padded with -1 to indicate mask=0.
        """
        n = n or self._n_candidates
        if self._rankings is not None:
            from intraday_trader.scanner import get_candidates as _sc_get
            ticker_names = _sc_get(self._rankings, date, n=n)
            return self._ticker_names_to_indices(ticker_names, n=n)

        indices = list(range(min(n, self.n_tickers)))
        while len(indices) < n:
            indices.append(-1)
        return indices

    # ------------------------------------------------------------------
    def get_intraday_candidates(self, ts: pd.Timestamp, n: int | None = None) -> list[int]:
        """
        Return intraday-scanner candidate indices for the given timestamp.
        Falls back to daily scanner candidates if intraday rankings are unavailable.
        """
        n = n or self._n_candidates
        if self._intraday_rankings is not None:
            from intraday_trader.scanner import get_candidates as _sc_get
            ticker_names = _sc_get(self._intraday_rankings, ts, n=n)
            return self._ticker_names_to_indices(ticker_names, n=n)
        return self.get_candidates(ts.normalize(), n=n)

    # ------------------------------------------------------------------
    def get_candidates_for_bar(
        self,
        flat_bar_idx: int,
        n: int | None = None,
        use_intraday: bool = False,
    ) -> list[int]:
        """
        Convenience wrapper to fetch candidates for a flat bar index.
        If use_intraday is True, bar-timestamp rankings are used.
        """
        bar = max(0, min(flat_bar_idx, self.n_bars - 1))
        ts = pd.Timestamp(self.bar_timestamps[bar])
        if use_intraday:
            return self.get_intraday_candidates(ts, n=n)
        return self.get_candidates(ts.normalize(), n=n)

    # ------------------------------------------------------------------
    def _ticker_names_to_indices(self, ticker_names: list[str], n: int) -> list[int]:
        indices: list[int] = []
        for ticker in ticker_names:
            idx = self.ticker_to_idx.get(ticker)
            if idx is not None:
                indices.append(idx)
        while len(indices) < n:
            indices.append(-1)
        return indices[:n]
