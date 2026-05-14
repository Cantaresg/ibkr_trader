"""
Portfolio performance metrics for backtesting.

All functions accept daily return series (1D numpy arrays or pandas Series).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

TRADING_DAYS = 252
RISK_FREE_ANNUAL = 0.05
RISK_FREE_DAILY  = RISK_FREE_ANNUAL / TRADING_DAYS


# ---------------------------------------------------------------------------
def sharpe_ratio(daily_returns: np.ndarray, risk_free: float = RISK_FREE_DAILY) -> float:
    """Annualised Sharpe ratio."""
    excess = daily_returns - risk_free
    std = excess.std(ddof=1)
    if std < 1e-10:
        return 0.0
    return float(excess.mean() / std * np.sqrt(TRADING_DAYS))


def sortino_ratio(daily_returns: np.ndarray, risk_free: float = RISK_FREE_DAILY) -> float:
    """Annualised Sortino ratio (downside deviation denominator)."""
    excess = daily_returns - risk_free
    downside = excess[excess < 0]
    if len(downside) < 2:
        return 0.0
    downside_std = np.sqrt((downside**2).mean())
    if downside_std < 1e-10:
        return 0.0
    return float(excess.mean() / downside_std * np.sqrt(TRADING_DAYS))


def max_drawdown(nav_series: np.ndarray) -> float:
    """Maximum drawdown from peak to trough (positive value, e.g. 0.15 = 15%)."""
    if len(nav_series) == 0:
        return 0.0
    peak = np.maximum.accumulate(nav_series)
    drawdowns = (peak - nav_series) / np.where(peak > 0, peak, 1.0)
    return float(drawdowns.max())


def calmar_ratio(daily_returns: np.ndarray, nav_series: np.ndarray) -> float:
    """Calmar = annualised return / max drawdown."""
    ann_return = annualised_return(daily_returns)
    mdd = max_drawdown(nav_series)
    if mdd < 1e-10:
        return 0.0
    return float(ann_return / mdd)


def annualised_return(daily_returns: np.ndarray) -> float:
    """Compound annualised growth rate from daily returns."""
    if len(daily_returns) == 0:
        return 0.0
    total = float(np.prod(1.0 + daily_returns))
    n_years = len(daily_returns) / TRADING_DAYS
    if n_years <= 0 or total <= 0:
        return 0.0
    return float(total ** (1.0 / n_years) - 1.0)


def alpha_beta(
    daily_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    risk_free: float = RISK_FREE_DAILY,
) -> tuple[float, float]:
    """
    Jensen's alpha (annualised) and market beta via OLS regression.
    Returns (alpha, beta).
    """
    n = min(len(daily_returns), len(benchmark_returns))
    if n < 5:
        return 0.0, 1.0
    r = daily_returns[:n]
    b = benchmark_returns[:n]
    beta = float(np.cov(r, b)[0, 1] / (np.var(b, ddof=1) + 1e-12))
    alpha_daily = float((r - risk_free).mean() - beta * (b - risk_free).mean())
    alpha_ann   = alpha_daily * TRADING_DAYS
    return alpha_ann, beta


def win_rate(daily_returns: np.ndarray) -> float:
    """Fraction of positive-return days."""
    if len(daily_returns) == 0:
        return 0.0
    return float((daily_returns > 0).mean())


def summary(
    daily_returns: np.ndarray,
    nav_series: np.ndarray,
    benchmark_returns: np.ndarray | None = None,
    label: str = "",
) -> dict:
    """
    Compute full metrics suite and return as a dict.
    Keys: sharpe, sortino, calmar, annualised_return, max_drawdown, win_rate,
          alpha (if benchmark provided), beta (if benchmark provided), label.
    """
    result = {
        "label":            label,
        "sharpe":           sharpe_ratio(daily_returns),
        "sortino":          sortino_ratio(daily_returns),
        "calmar":           calmar_ratio(daily_returns, nav_series),
        "annualised_return": annualised_return(daily_returns),
        "max_drawdown":     max_drawdown(nav_series),
        "win_rate":         win_rate(daily_returns),
        "n_days":           len(daily_returns),
    }
    if benchmark_returns is not None:
        a, b = alpha_beta(daily_returns, benchmark_returns)
        result["alpha"] = a
        result["beta"]  = b
    return result


def print_summary(metrics: dict) -> None:
    """Pretty-print a metrics dict."""
    label = metrics.get("label", "")
    header = f"{'-'*48}\n  {label}\n{'-'*48}" if label else "-" * 48
    print(header)
    print(f"  Sharpe:           {metrics.get('sharpe', 0):.3f}")
    print(f"  Sortino:          {metrics.get('sortino', 0):.3f}")
    print(f"  Calmar:           {metrics.get('calmar', 0):.3f}")
    print(f"  Ann. Return:      {metrics.get('annualised_return', 0):.1%}")
    print(f"  Max Drawdown:     {metrics.get('max_drawdown', 0):.1%}")
    print(f"  Win Rate:         {metrics.get('win_rate', 0):.1%}")
    if "alpha" in metrics:
        print(f"  Alpha (ann):      {metrics.get('alpha', 0):.3f}")
        print(f"  Beta:             {metrics.get('beta', 0):.3f}")
    print(f"  N days:           {metrics.get('n_days', 0)}")
