"""Portfolio accounting: weights, NAV, drawdown tracking."""
import numpy as np


class PortfolioState:
    """
    Tracks portfolio composition and performance during an episode.

    weights[0:n_stocks] = stock allocations (fraction of NAV)
    weights[n_stocks]   = cash fraction
    All weights sum to 1.0.
    """

    def __init__(self, n_stocks: int = 20, initial_capital: float = 100_000.0):
        self.n_stocks = n_stocks
        self.initial_capital = initial_capital
        self.reset()

    def reset(self) -> None:
        self.nav: float = self.initial_capital
        self.peak_nav: float = self.initial_capital
        # Start fully in cash
        self.weights = np.zeros(self.n_stocks + 1, dtype=np.float32)
        self.weights[-1] = 1.0  # cash = 100%
        self.prev_weights = self.weights.copy()
        self.step_count: int = 0

    @property
    def drawdown(self) -> float:
        """Current drawdown from episode peak NAV (0 = no drawdown)."""
        return max(0.0, (self.peak_nav - self.nav) / self.peak_nav)

    @property
    def nav_normalized(self) -> float:
        """NAV relative to initial capital, centred at 0."""
        return self.nav / self.initial_capital - 1.0

    def step(
        self,
        new_weights: np.ndarray,
        stock_returns: np.ndarray,
        transaction_cost_bps: float = 10.0,
    ) -> dict:
        """
        Apply new_weights, compute portfolio return, update state.

        new_weights:       shape (n_stocks + 1,), sum to 1.0, non-negative
        stock_returns:     shape (n_stocks,), daily returns for each stock
        transaction_cost_bps: round-trip cost in basis points

        Returns dict with: portfolio_return, transaction_cost, prev_weights
        """
        self.prev_weights = self.weights.copy()

        # Transaction cost: proportional to total weight change
        weight_change = np.abs(new_weights - self.prev_weights)
        tc_rate = transaction_cost_bps / 10_000.0
        transaction_cost = weight_change.sum() * tc_rate * 0.5  # one-way approximation

        # Portfolio return (stock portion): w_i * r_i, summed
        # Cash earns 0 (simplification; risk-free rate is in the reward, not NAV)
        portfolio_return = float(np.dot(new_weights[:self.n_stocks], stock_returns))

        # Apply transaction cost as a drag on return
        net_return = portfolio_return - transaction_cost

        # Update NAV
        self.nav *= (1.0 + net_return)
        self.peak_nav = max(self.peak_nav, self.nav)

        # Update weights (they drift with returns; re-normalise after step)
        updated = new_weights.copy()
        updated[:self.n_stocks] *= (1.0 + stock_returns)
        total = updated.sum()
        if total > 1e-8:
            updated /= total
        self.weights = updated.astype(np.float32)

        self.step_count += 1

        return {
            "portfolio_return": portfolio_return,
            "net_return": net_return,
            "transaction_cost": transaction_cost,
            "nav": self.nav,
            "drawdown": self.drawdown,
        }

    def state_vector(self) -> np.ndarray:
        """
        Returns a (n_stocks + 3,) vector:
          [stock_weights..., cash_weight, nav_normalized, drawdown]
        """
        return np.concatenate([
            self.weights,                    # (n_stocks + 1,)
            [self.nav_normalized],           # (1,)
            [self.drawdown],                 # (1,)
        ]).astype(np.float32)
