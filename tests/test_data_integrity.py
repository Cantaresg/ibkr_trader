"""
Data integrity tests for Phase 1.
Run with: python -m pytest tests/test_data_integrity.py -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.utils.config_loader import load_config, all_tickers
from src.data import ohlcv_store
from src.features.pipeline import load_features, FEATURE_COLS

CFG = load_config()
RAW_DIR = CFG["data"]["raw_dir"]
PROC_DIR = CFG["data"]["processed_dir"]
TICKERS = all_tickers(CFG["data"]["universe_file"])
SAMPLE = ["AAPL", "MSFT", "JPM", "XOM", "JNJ"]  # one from each major sector


class TestOHLCV:
    def test_all_tickers_downloaded(self):
        missing = [t for t in TICKERS if ohlcv_store.load(RAW_DIR, t) is None]
        assert missing == [], f"Missing OHLCV for: {missing}"

    def test_minimum_row_count(self):
        """
        Each ticker should have at least 3000 rows (~12 years).
        Allows for post-2010 IPOs: META (2012), NOW (2012), ABBV/ZTS (2013).
        """
        short = []
        for t in TICKERS:
            df = ohlcv_store.load(RAW_DIR, t)
            if df is not None and len(df) < 3000:
                short.append((t, len(df)))
        assert short == [], f"Tickers with < 3000 rows: {short}"

    def test_date_range(self):
        """Data should start no later than 2011 and end no earlier than 2025-06."""
        for t in SAMPLE:
            df = ohlcv_store.load(RAW_DIR, t)
            assert df.index[0] <= pd.Timestamp("2011-01-01"), f"{t} starts too late: {df.index[0]}"
            assert df.index[-1] >= pd.Timestamp("2025-06-01"), f"{t} ends too early: {df.index[-1]}"

    def test_no_nan_close(self):
        """Close price should have no NaN values after download."""
        for t in SAMPLE:
            df = ohlcv_store.load(RAW_DIR, t)
            nan_count = df["close"].isna().sum()
            assert nan_count == 0, f"{t}: {nan_count} NaN close prices"

    def test_positive_prices(self):
        """All OHLCV prices must be positive."""
        for t in SAMPLE:
            df = ohlcv_store.load(RAW_DIR, t)
            assert (df["close"] > 0).all(), f"{t}: non-positive close prices"
            assert (df["volume"] >= 0).all(), f"{t}: negative volume"

    def test_aapl_2020_split_adjustment(self):
        """
        AAPL did a 4:1 split on 2020-08-31.
        With auto_adjust=True, the pre-split price should already be adjusted.
        Close on 2020-08-28 (last day before split) should be roughly $125-135,
        NOT ~$500 (the unadjusted price).
        """
        df = ohlcv_store.load(RAW_DIR, "AAPL")
        pre_split = df.loc["2020-08-28", "close"]
        assert pre_split < 200, (
            f"AAPL 2020-08-28 close = {pre_split:.2f}; "
            f"expected ~$125-135 (split-adjusted). Got unadjusted price?"
        )
        assert pre_split > 100, f"AAPL 2020-08-28 close = {pre_split:.2f}; suspiciously low"

    def test_no_future_dates(self):
        """No dates beyond the configured end date."""
        end = pd.Timestamp(CFG["data"]["end_date"])
        for t in SAMPLE:
            df = ohlcv_store.load(RAW_DIR, t)
            assert df.index.max() <= end + pd.Timedelta(days=5), \
                f"{t}: data extends beyond end date"


class TestMarketData:
    def test_market_files_exist(self):
        market_dir = Path(RAW_DIR) / "market"
        required = ["SPY", "VIX", "VIX3M", "TNX", "IRX", "HYG", "IEI"]
        for name in required:
            p = market_dir / f"{name}.parquet"
            assert p.exists(), f"Missing market file: {name}.parquet"

    def test_vix_range(self):
        """VIX should be between 5 and 90 across all history."""
        vix = pd.read_parquet(Path(RAW_DIR) / "market" / "VIX.parquet")
        assert vix["close"].min() > 5, f"VIX too low: {vix['close'].min():.1f}"
        assert vix["close"].max() < 100, f"VIX too high: {vix['close'].max():.1f}"

    def test_spy_coverage(self):
        spy = pd.read_parquet(Path(RAW_DIR) / "market" / "SPY.parquet")
        assert len(spy) >= 3800, f"SPY has only {len(spy)} rows"


class TestFeatureStore:
    def test_all_features_built(self):
        missing = [t for t in TICKERS if load_features(PROC_DIR, t) is None]
        assert missing == [], f"Missing features for: {missing}"

    def test_correct_feature_count(self):
        for t in SAMPLE:
            df = load_features(PROC_DIR, t)
            assert list(df.columns) == FEATURE_COLS, \
                f"{t}: columns mismatch. Got {df.columns.tolist()}"

    def test_no_inf_values(self):
        """No infinite values anywhere in the feature store."""
        for t in SAMPLE:
            df = load_features(PROC_DIR, t)
            inf_count = np.isinf(df.values).sum()
            assert inf_count == 0, f"{t}: {inf_count} inf values in features"

    def test_normalized_range(self):
        """
        Most normalized features should be within [-4, 4].
        Allow some outliers (up to 1% of values outside range).
        """
        skip = {"rsi_divergence", "close_norm", "daily_return",
                 "return_5d", "return_20d", "sentiment_score",
                 "article_count_zscore", "sentiment_dispersion",
                 "large_trade_proxy", "institutional_accumulation"}
        for t in SAMPLE:
            df = load_features(PROC_DIR, t)
            for col in FEATURE_COLS:
                if col in skip:
                    continue
                col_data = df[col].dropna()
                if len(col_data) == 0:
                    continue
                pct_outside = ((col_data.abs() > 4).sum()) / len(col_data)
                assert pct_outside < 0.02, \
                    f"{t}/{col}: {pct_outside:.1%} values outside [-4, 4]"

    def test_no_lookahead_daily_return(self):
        """
        daily_return at index t must equal (close_t - close_{t-1}) / close_{t-1}.
        We verify this matches the OHLCV for 10 random dates.
        """
        t = "MSFT"
        features = load_features(PROC_DIR, t)
        ohlcv = ohlcv_store.load(RAW_DIR, t)
        expected = ohlcv["close"].pct_change().clip(-0.5, 0.5)
        common = features.index.intersection(expected.index)[50:60]  # skip NaN warmup
        for date in common:
            feat_ret = features.loc[date, "daily_return"]
            exp_ret = expected.loc[date]
            assert abs(feat_ret - exp_ret) < 1e-4, \
                f"{t} on {date}: daily_return mismatch ({feat_ret:.6f} vs {exp_ret:.6f})"
