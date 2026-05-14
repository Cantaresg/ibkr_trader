"""Intraday portfolio accounting with day-level tracking and forced-flat support."""
from __future__ import annotations
import numpy as np

from intraday_trader.constants import N_STOCKS, PORTFOLIO_DIM


class IntradayPortfolioState:
    """
    Tracks intraday portfolio composition and performance.

    weights[0:N_STOCKS] = stock allocations (fraction of NAV)
    weights[N_STOCKS]   = cash fraction
    All weights sum to 1.0.

    Supports day-level reset (day_reset) that records daily PnL while
    keeping NAV continuity across the multi-day episode.
    """

    def __init__(self, n_stocks: int = N_STOCKS, initial_capital: float = 5_000.0):
        self.n_stocks = n_stocks
        self.initial_capital = initial_capital
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Full episode reset — NAV, peak, weights, history."""
        self.nav: float        = self.initial_capital
        self.peak_nav: float   = self.initial_capital
        self.weights           = np.zeros(self.n_stocks + 1, dtype=np.float32)
        self.weights[-1]       = 1.0   # start fully in cash
        self.prev_weights      = self.weights.copy()
        self.step_count: int   = 0
        self._day_start_nav: float = self.initial_capital
        self.daily_pnl_history: list[float] = []

    # ------------------------------------------------------------------
    def day_reset(self) -> None:
        """
        Called at the boundary between trading days.
        Records daily PnL, resets intraday peak, keeps NAV continuity.
        Portfolio should already be flat (cash = 1.0) before this is called.
        """
        daily_pnl = (self.nav - self._day_start_nav) / max(self._day_start_nav, 1e-8)
        self.daily_pnl_history.append(daily_pnl)
        self._day_start_nav = self.nav

    # ------------------------------------------------------------------
    @property
    def drawdown(self) -> float:
        return max(0.0, (self.peak_nav - self.nav) / max(self.peak_nav, 1e-8))

    @property
    def nav_normalized(self) -> float:
        return self.nav / self.initial_capital - 1.0

    # ------------------------------------------------------------------
    def step(
        self,
        new_weights: np.ndarray,
        bar_returns: np.ndarray,   # shape (n_stocks,)
        tc_bps: float = 5.0,
    ) -> dict:
        """
        Apply new_weights, compute bar portfolio return, update state.

        new_weights: shape (n_stocks + 1,), sum to 1, non-negative
        bar_returns: shape (n_stocks,), bar-over-bar returns
        """
        self.prev_weights = self.weights.copy()

        weight_change    = np.abs(new_weights - self.prev_weights)
        tc_rate          = tc_bps / 10_000.0
        transaction_cost = weight_change.sum() * tc_rate * 0.5

        portfolio_return = float(np.dot(new_weights[:self.n_stocks], bar_returns))
        net_return       = portfolio_return - transaction_cost

        self.nav      *= (1.0 + net_return)
        self.peak_nav  = max(self.peak_nav, self.nav)

        updated = new_weights.copy()
        updated[:self.n_stocks] *= (1.0 + bar_returns)
        total = updated.sum()
        if total > 1e-8:
            updated /= total
        self.weights = updated.astype(np.float32)
        self.step_count += 1

        return {
            "portfolio_return": portfolio_return,
            "net_return":       net_return,
            "transaction_cost": transaction_cost,
            "nav":              self.nav,
            "drawdown":         self.drawdown,
        }

    # ------------------------------------------------------------------
    def force_flat(self, tc_bps: float = 5.0) -> float:
        """
        Liquidate all stock positions to cash.
        Returns the one-way transaction cost of doing so.
        """
        stock_weights    = self.weights[:self.n_stocks]
        gross_exposure   = float(stock_weights.sum())
        tc_rate          = tc_bps / 10_000.0
        liquidation_cost = gross_exposure * tc_rate

        self.nav *= (1.0 - liquidation_cost)
        self.peak_nav = max(self.peak_nav, self.nav)

        self.weights = np.zeros(self.n_stocks + 1, dtype=np.float32)
        self.weights[-1] = 1.0
        return liquidation_cost

    # ------------------------------------------------------------------
    def overnight_exposure(self) -> float:
        """Total gross stock weight (used for overnight exposure penalty)."""
        return float(self.weights[:self.n_stocks].sum())

    # ------------------------------------------------------------------
    def state_vector(self) -> np.ndarray:
        """
        Returns a (PORTFOLIO_DIM = n_stocks + 3,) vector:
          [stock_weights..., cash_weight, nav_normalized, drawdown]
        """
        return np.concatenate([
            self.weights,           # (n_stocks + 1,)
            [self.nav_normalized],  # (1,)
            [self.drawdown],        # (1,)
        ]).astype(np.float32)

    # ------------------------------------------------------------------
    def intraday_daily_sharpe(self) -> float:
        """
        Annualized daily Sharpe ratio computed from daily_pnl_history.
        Returns 0.0 if fewer than 2 days recorded.
        """
        if len(self.daily_pnl_history) < 2:
            return 0.0
        pnls = np.array(self.daily_pnl_history, dtype=float)
        mu   = pnls.mean()
        std  = pnls.std(ddof=1)
        if std < 1e-10:
            return 0.0
        return float(mu / std * (252 ** 0.5))
