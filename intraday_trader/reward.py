"""
Intraday reward function.

r_t = alpha  * (bar_portfolio_return - risk_free_per_bar)
    - beta   * relu(drawdown - threshold)^2
    - gamma  * transaction_cost
    - delta  * overnight_exposure          [only at end-of-day bar]
    + zeta   * diversification_score       [fraction of available stocks held]

risk_free_per_bar ≈ 0.05 / (252 * 7) ≈ 2.8 bps

diversification_score = n_active_stocks / n_total_stocks ∈ [0, 1]
  Encourages spreading across stocks. Necessary because Gaussian policy +
  softmax creates a perverse dynamic: high ent_coef → high action std →
  more extreme softmax → single-stock bets. A direct diversification bonus
  avoids this by rewarding multi-stock allocation regardless of action distribution.
"""
import numpy as np

RISK_FREE_PER_BAR = 0.05 / (252 * 7)   # ~2.8 basis points per hourly bar


def compute(
    portfolio_return: float,
    transaction_cost: float,
    drawdown: float,
    overnight_exposure: float = 0.0,
    is_eod_bar: bool = False,
    n_active_positions: int = 0,
    n_total_stocks: int = 20,
    alpha: float = 1.0,
    beta: float  = 3.0,
    gamma: float = 0.5,
    delta: float = 0.001,
    zeta:  float = 0.0,
    drawdown_threshold: float = 0.01,
) -> tuple[float, dict]:
    """
    Compute total reward and component breakdown.

    overnight_exposure:  sum of stock weights before forced flat (only at EOD bar).
    n_active_positions:  number of stocks with non-zero weight this bar.
    n_total_stocks:      total available stock slots (for normalising diversity score).
    is_eod_bar:          whether this is the last bar of the trading day.

    Returns (reward, info_dict).
    """
    r_excess = portfolio_return - RISK_FREE_PER_BAR
    r_dd     = max(0.0, drawdown - drawdown_threshold) ** 2
    r_eod    = overnight_exposure if is_eod_bar else 0.0
    r_div    = (n_active_positions / max(n_total_stocks, 1)) if zeta > 0.0 else 0.0

    reward = (
        alpha * r_excess
        - beta  * r_dd
        - gamma * transaction_cost
        - delta * r_eod
        + zeta  * r_div
    )

    info = {
        "r_excess":             r_excess,
        "r_drawdown_penalty":   r_dd,
        "r_transaction_cost":   transaction_cost,
        "r_overnight_exposure": r_eod,
        "r_diversification":    r_div,
        "reward":               float(reward),
    }
    return float(reward), info
