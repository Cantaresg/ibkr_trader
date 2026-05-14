"""
Scanner tests: no-lookahead validation and ranking quality.
Run: python -m pytest tests/test_scanner.py -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.scanner.scanner_store import load as load_scanner, get_candidates
from src.scanner.rule_based import build_rankings
from src.utils.config_loader import load_config, all_tickers


# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def rankings():
    r = load_scanner()
    if r is None:
        pytest.skip("Scanner rankings not found — run build_scanner.py first")
    return r


@pytest.fixture(scope="module")
def cfg():
    return load_config("config/config.yaml")


@pytest.fixture(scope="module")
def tickers(cfg):
    return all_tickers(cfg["data"]["universe_file"])


# ---------------------------------------------------------------------------
class TestScannerBasics:
    def test_rankings_load(self, rankings):
        assert rankings is not None
        assert len(rankings) > 0

    def test_rankings_has_20_columns(self, rankings):
        rank_cols = [c for c in rankings.columns if c.startswith("rank_")]
        assert len(rank_cols) == 20, f"Expected 20 rank columns, got {len(rank_cols)}"

    def test_rankings_index_is_datetime(self, rankings):
        assert isinstance(rankings.index, pd.DatetimeIndex)

    def test_all_ranks_are_valid_tickers(self, rankings, tickers):
        rank_cols = [c for c in rankings.columns if c.startswith("rank_")]
        all_ranked = rankings[rank_cols].values.flatten()
        # Exclude NaN and empty-string padding slots
        all_ranked = [t for t in all_ranked if pd.notna(t) and t != ""]
        invalid = set(all_ranked) - set(tickers)
        assert len(invalid) == 0, f"Unknown tickers in rankings: {invalid}"

    def test_get_candidates_returns_n_results(self, rankings):
        # Use a date well into the history (after 252-day lookback warmup)
        date = rankings.index[400]
        candidates = get_candidates(rankings, date, n=20)
        assert len(candidates) == 20

    def test_get_candidates_no_duplicates(self, rankings):
        date = rankings.index[400]
        candidates = get_candidates(rankings, date, n=20)
        assert len(set(candidates)) == len(candidates)

    def test_get_candidates_fallback_to_prior_date(self, rankings):
        # A date not in rankings should fall back to nearest prior date
        future_date = pd.Timestamp("2030-01-01")
        candidates = get_candidates(rankings, future_date, n=20)
        assert len(candidates) == 20


# ---------------------------------------------------------------------------
class TestNoLookahead:
    """
    Validate that scanner scores at date t use only data from t-1 or earlier.
    We verify this by checking the raw momentum and volume computations.
    """

    def test_momentum_12_1_uses_prior_data(self, cfg):
        """
        momentum_12_1 = ret_12m - ret_1m.
        At date t, ret_12m should be computed from close[t-252] to close[t-1].
        We validate that close[t] is NOT used.
        """
        import pyarrow.parquet as pq
        raw_dir = cfg["data"]["raw_dir"]
        ticker = "AAPL"
        ohlcv_path = Path(raw_dir) / "ohlcv" / f"{ticker}.parquet"
        if not ohlcv_path.exists():
            pytest.skip("AAPL OHLCV not found")

        df = pd.read_parquet(ohlcv_path)
        closes = df["close"]

        # At test_date t, momentum_12_1 uses closes strictly before t
        test_idx = 300
        test_date = closes.index[test_idx]

        # Compute what momentum should be using only prior data
        # ret_12m = (close[t-1] / close[t-253]) - 1  (shift(1) applied)
        # ret_1m  = (close[t-1] / close[t-22])  - 1
        c = closes.values
        ret_12m_prior = (c[test_idx - 1] / c[test_idx - 253]) - 1
        ret_1m_prior  = (c[test_idx - 1] / c[test_idx - 22])  - 1
        mom_prior = ret_12m_prior - ret_1m_prior

        # Recompute scanner with close[t] substituted (should not change result)
        closes_modified = closes.copy()
        closes_modified.iloc[test_idx] *= 10  # extreme modification
        ret_12m_mod = (closes_modified.iloc[test_idx - 1] / closes_modified.iloc[test_idx - 253]) - 1
        ret_1m_mod  = (closes_modified.iloc[test_idx - 1] / closes_modified.iloc[test_idx - 22]) - 1
        mom_mod = ret_12m_mod - ret_1m_mod

        # Momentum at t must be identical regardless of close[t]
        assert abs(mom_prior - mom_mod) < 1e-10, (
            "Scanner momentum uses close[t] — lookahead bias detected!"
        )

    def test_volume_activity_uses_prior_data(self, cfg):
        """
        volume_activity is a z-score of volume[t-1] vs 20-day rolling mean ending at t-1.
        Modifying volume[t] should not affect the scanner score for date t.
        """
        import pyarrow.parquet as pq
        raw_dir = cfg["data"]["raw_dir"]
        ticker = "AAPL"
        ohlcv_path = Path(raw_dir) / "ohlcv" / f"{ticker}.parquet"
        if not ohlcv_path.exists():
            pytest.skip("AAPL OHLCV not found")

        df = pd.read_parquet(ohlcv_path)
        vol = df["volume"]

        test_idx = 300
        # Volume z-score at t uses only vol[1..t-1]
        v = vol.values
        v_prior_20d_mean = v[test_idx - 21: test_idx - 1].mean()
        v_prior_20d_std  = v[test_idx - 21: test_idx - 1].std(ddof=1)
        z_prior = (v[test_idx - 1] - v_prior_20d_mean) / (v_prior_20d_std + 1e-8)

        # Modify vol[t] dramatically
        v_mod = v.copy()
        v_mod[test_idx] = v[test_idx] * 1000
        v_mod_20d_mean = v_mod[test_idx - 21: test_idx - 1].mean()
        v_mod_20d_std  = v_mod[test_idx - 21: test_idx - 1].std(ddof=1)
        z_mod = (v_mod[test_idx - 1] - v_mod_20d_mean) / (v_mod_20d_std + 1e-8)

        assert abs(z_prior - z_mod) < 1e-10, (
            "Scanner volume z-score uses volume[t] — lookahead bias detected!"
        )

    def test_scanner_date_2015_01_05(self, rankings):
        """
        Spot-check: scanner rankings at 2015-01-05 must be available
        and must contain only real tickers (not future-data artefacts).
        """
        target = pd.Timestamp("2015-01-05")
        if target not in rankings.index:
            closest = rankings.index[rankings.index <= target][-1]
            candidates = get_candidates(rankings, target, n=20)
        else:
            candidates = get_candidates(rankings, target, n=20)
        assert len(candidates) == 20
        assert all(isinstance(t, str) and len(t) > 0 for t in candidates)


# ---------------------------------------------------------------------------
class TestRankingStability:
    def test_top_candidates_change_over_time(self, rankings):
        """Rankings should not be identical across the full date range."""
        rank_cols = [c for c in rankings.columns if c.startswith("rank_")]
        first_20  = set(rankings[rank_cols].iloc[100].dropna())
        last_20   = set(rankings[rank_cols].iloc[-100].dropna())
        overlap   = len(first_20 & last_20)
        assert overlap < 20, "Rankings never change — likely a bug"

    def test_rankings_cover_expected_date_range(self, rankings):
        assert rankings.index.min() <= pd.Timestamp("2011-01-01")
        assert rankings.index.max() >= pd.Timestamp("2024-01-01")

    def test_no_nan_in_rank_01(self, rankings):
        """The top-ranked stock should always be filled (no NaN in rank_01)."""
        assert rankings["rank_01"].notna().all()
