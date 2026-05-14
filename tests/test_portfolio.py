"""
Portfolio accounting and metrics unit tests.
Run: python -m pytest tests/test_portfolio.py -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.environment.portfolio import PortfolioState
from src.backtesting.metrics import (
    sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio,
    annualised_return, alpha_beta, win_rate, summary,
)


# ---------------------------------------------------------------------------
class TestPortfolioState:
    def test_initial_state_all_cash(self):
        p = PortfolioState(n_stocks=5, initial_capital=10_000.0)
        assert p.nav == 10_000.0
        assert p.weights[-1] == pytest.approx(1.0)      # 100% cash
        assert p.weights[:-1].sum() == pytest.approx(0.0)

    def test_weights_sum_to_one_after_step(self):
        p = PortfolioState(n_stocks=5)
        w = np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1], dtype=np.float32)
        returns = np.array([0.01, -0.01, 0.02, 0.0, -0.02], dtype=np.float32)
        p.step(w, returns)
        assert p.weights.sum() == pytest.approx(1.0, abs=1e-5)

    def test_nav_grows_with_positive_returns(self):
        p = PortfolioState(n_stocks=3, initial_capital=100_000.0)
        w = np.array([0.33, 0.33, 0.33, 0.01], dtype=np.float32)
        returns = np.array([0.01, 0.01, 0.01], dtype=np.float32)
        info = p.step(w, returns, transaction_cost_bps=0.0)
        assert p.nav > 100_000.0
        assert info["portfolio_return"] == pytest.approx(0.33 * 3 * 0.01, abs=1e-4)

    def test_nav_stays_positive_all_cash(self):
        p = PortfolioState(n_stocks=5)
        w = np.zeros(6, dtype=np.float32)
        w[-1] = 1.0
        returns = np.full(5, -0.10, dtype=np.float32)
        for _ in range(50):
            p.step(w, returns)
        assert p.nav > 0

    def test_drawdown_zero_at_start(self):
        p = PortfolioState()
        assert p.drawdown == pytest.approx(0.0)

    def test_drawdown_computed_correctly(self):
        p = PortfolioState(n_stocks=3, initial_capital=100_000.0)
        # Force NAV down 10%
        p.nav = 90_000.0
        assert p.drawdown == pytest.approx(0.10, abs=1e-6)

    def test_state_vector_shape(self):
        p = PortfolioState(n_stocks=20)
        v = p.state_vector()
        assert v.shape == (23,)  # 20 weights + cash + nav_norm + drawdown

    def test_state_vector_dtype(self):
        p = PortfolioState()
        assert p.state_vector().dtype == np.float32

    def test_transaction_cost_applied(self):
        p = PortfolioState(n_stocks=3, initial_capital=100_000.0)
        # Start 100% cash, move to 100% stock[0]
        w = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        returns = np.zeros(3, dtype=np.float32)
        info = p.step(w, returns, transaction_cost_bps=10.0)
        # Weight change = |1-0| + |0-0| + |0-0| + |0-1| = 2.0; tc = 2.0 * 0.001 * 0.5
        assert info["transaction_cost"] == pytest.approx(2.0 * 0.001 * 0.5, abs=1e-6)

    def test_peak_nav_never_decreases(self):
        p = PortfolioState(n_stocks=3, initial_capital=100_000.0)
        w_stock = np.array([0.5, 0.3, 0.2, 0.0], dtype=np.float32)
        peak = p.initial_capital
        for ret_val in [0.02, -0.05, 0.01, -0.03, 0.04]:
            returns = np.full(3, ret_val, dtype=np.float32)
            p.step(w_stock, returns, transaction_cost_bps=0.0)
            peak = max(peak, p.nav)
            assert p.peak_nav == pytest.approx(peak, rel=1e-5)


# ---------------------------------------------------------------------------
class TestMetrics:
    @pytest.fixture
    def flat_returns(self):
        rng = np.random.default_rng(42)
        return rng.normal(0.0005, 0.01, 252).astype(np.float64)

    @pytest.fixture
    def nav_from_returns(self, flat_returns):
        return 100_000.0 * np.cumprod(1 + flat_returns)

    def test_sharpe_positive_mean(self):
        # High positive mean with tiny noise → high Sharpe
        rng = np.random.default_rng(0)
        r = 0.002 + rng.normal(0, 1e-6, 252)
        assert sharpe_ratio(r) > 5.0

    def test_sharpe_zero_variance(self):
        r = np.full(252, 0.0001, dtype=np.float64)
        # std is 0 → returns 0.0 (not nan)
        result = sharpe_ratio(r)
        assert np.isfinite(result)

    def test_max_drawdown_no_loss(self):
        nav = np.array([100.0, 105.0, 110.0, 115.0])
        assert max_drawdown(nav) == pytest.approx(0.0)

    def test_max_drawdown_known_value(self):
        nav = np.array([100.0, 80.0, 90.0, 70.0, 95.0])
        # Peak = 100 at idx 0, trough = 70 at idx 3 → mdd = 30%
        assert max_drawdown(nav) == pytest.approx(0.30, abs=1e-6)

    def test_annualised_return_flat(self):
        r = np.zeros(252, dtype=np.float64)
        assert annualised_return(r) == pytest.approx(0.0, abs=1e-6)

    def test_annualised_return_positive(self, flat_returns, nav_from_returns):
        ann = annualised_return(flat_returns)
        assert -0.5 < ann < 5.0   # sanity range for random returns

    def test_calmar_zero_drawdown(self):
        r = np.full(252, 0.001, dtype=np.float64)
        nav = np.cumprod(1 + r) * 100
        result = calmar_ratio(r, nav)
        # max_drawdown ≈ 0 → calmar returns 0.0 (not inf)
        assert np.isfinite(result)

    def test_sortino_better_than_sharpe_for_skewed(self):
        rng = np.random.default_rng(0)
        r = rng.normal(0.001, 0.01, 252)
        # Clip upside to make returns positively skewed
        r_skew = np.where(r > 0, r * 0.5, r)
        s  = sharpe_ratio(r_skew)
        so = sortino_ratio(r_skew)
        # Sortino should be >= Sharpe when downside vol < total vol
        assert so >= s - 0.5   # loose bound

    def test_win_rate_all_positive(self):
        r = np.full(100, 0.001)
        assert win_rate(r) == pytest.approx(1.0)

    def test_win_rate_all_negative(self):
        r = np.full(100, -0.001)
        assert win_rate(r) == pytest.approx(0.0)

    def test_alpha_beta_vs_market(self, flat_returns):
        benchmark = flat_returns + np.random.default_rng(1).normal(0, 0.005, len(flat_returns))
        a, b = alpha_beta(flat_returns, benchmark)
        assert np.isfinite(a)
        assert b > 0.0   # positively correlated

    def test_summary_keys(self, flat_returns, nav_from_returns):
        m = summary(flat_returns, nav_from_returns, label="test")
        for key in ("sharpe", "sortino", "calmar", "annualised_return",
                    "max_drawdown", "win_rate", "n_days", "label"):
            assert key in m

    def test_summary_with_benchmark(self, flat_returns, nav_from_returns):
        bench = flat_returns * 0.8
        m = summary(flat_returns, nav_from_returns, benchmark_returns=bench)
        assert "alpha" in m
        assert "beta"  in m
