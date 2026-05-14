"""
Reward function (three components, each independently testable).

r_t = alpha * excess_return_t
    - beta  * relu(drawdown_t - threshold)^2
    - gamma * transaction_cost_t
"""
import numpy as np


RISK_FREE_DAILY = 0.05 / 252   # ~5% annual rate, daily equivalent


def excess_return(portfolio_return: float, risk_free: float = RISK_FREE_DAILY) -> float:
    """Daily portfolio return minus risk-free rate."""
    return portfolio_return - risk_free


def drawdown_penalty(drawdown: float, threshold: float = 0.05) -> float:
    """Quadratic penalty for drawdowns exceeding threshold."""
    excess = max(0.0, drawdown - threshold)
    return excess ** 2


def transaction_cost_penalty(tc: float) -> float:
    """Pass-through of the transaction cost computed by PortfolioState."""
    return tc


def compute(
    portfolio_return: float,
    transaction_cost: float,
    drawdown: float,
    alpha: float = 1.0,
    beta: float = 2.0,
    gamma: float = 0.5,
    drawdown_threshold: float = 0.05,
) -> tuple[float, dict]:
    """
    Compute total reward and component breakdown.
    Returns (reward, info_dict).
    """
    r_excess = excess_return(portfolio_return)
    r_dd     = drawdown_penalty(drawdown, drawdown_threshold)
    r_tc     = transaction_cost_penalty(transaction_cost)

    reward = alpha * r_excess - beta * r_dd - gamma * r_tc

    info = {
        "r_excess": r_excess,
        "r_drawdown_penalty": r_dd,
        "r_transaction_cost": r_tc,
        "reward": reward,
    }
    return float(reward), info
