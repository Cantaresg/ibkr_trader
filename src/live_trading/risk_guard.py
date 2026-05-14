"""
Risk guard: applies pre-trade weight constraints and monitors live risk limits.

All methods are pure (no IB dependency) — they operate on weight vectors and
NAV numbers passed in by the caller.
"""
from __future__ import annotations
import numpy as np
from src.utils.logging_config import get_logger

log = get_logger("live.risk")


class RiskGuard:
    """
    Enforces the risk rules defined in config.yaml ibkr.risk:

      max_position_weight:     0.15   per-stock hard cap
      max_sector_weight:       0.40   sector aggregate cap
      daily_loss_limit:       -0.02   halt day if daily PnL < -2% of SOD NAV
      drawdown_halt_threshold:-0.10   halve target weights if DD > 10%
      drawdown_resume_threshold:-0.05 resume full weights when DD < 5%
    """

    def __init__(self, risk_cfg: dict, ticker_to_sector: dict[str, str]):
        self.max_pos_w   = float(risk_cfg.get("max_position_weight",    0.15))
        self.max_sec_w   = float(risk_cfg.get("max_sector_weight",       0.40))
        self.daily_limit = float(risk_cfg.get("daily_loss_limit",       -0.02))
        self.dd_halt     = float(risk_cfg.get("drawdown_halt_threshold", -0.10))
        self.dd_resume   = float(risk_cfg.get("drawdown_resume_threshold", -0.05))
        self.t2s         = ticker_to_sector   # {ticker: sector_key}
        self._halved     = False              # state: currently in halved-weight regime

    # ------------------------------------------------------------------
    def clip_weights(
        self,
        weights:  np.ndarray,   # shape (n_stocks + 1,): [...stock weights..., cash]
        tickers:  list[str],    # length n_stocks (parallel to weights[:-1])
    ) -> np.ndarray:
        """
        Apply position and sector caps, renormalize so weights sum to 1.
        Cash weight (last element) absorbs the excess.
        """
        w = weights.copy().astype(float)
        n = len(tickers)
        stock_w = w[:n]
        cash_w  = w[n]

        # 1. Per-stock cap
        excess = 0.0
        for i in range(n):
            if stock_w[i] > self.max_pos_w:
                excess       += stock_w[i] - self.max_pos_w
                stock_w[i]    = self.max_pos_w

        # 2. Sector cap (redistribute excess from capped stocks to cash)
        sectors: dict[str, list[int]] = {}
        for i, t in enumerate(tickers):
            s = self.t2s.get(t, "unknown")
            sectors.setdefault(s, []).append(i)

        for sec, idxs in sectors.items():
            sec_total = sum(stock_w[i] for i in idxs)
            if sec_total > self.max_sec_w:
                scale = self.max_sec_w / sec_total
                for i in idxs:
                    excess      += stock_w[i] * (1 - scale)
                    stock_w[i]  *= scale

        # 3. Shift excess into cash, renormalize
        cash_w  = min(1.0, cash_w + excess)
        total   = stock_w.sum() + cash_w
        if total > 0:
            stock_w /= total
            cash_w  /= total

        result = np.concatenate([stock_w, [cash_w]]).astype(np.float32)

        if excess > 0.001:
            log.info("RiskGuard clipped %.1f%% into cash (pos/sector caps)", excess * 100)

        return result

    # ------------------------------------------------------------------
    def apply_drawdown_scale(
        self,
        weights:     np.ndarray,
        current_nav: float,
        peak_nav:    float,
    ) -> np.ndarray:
        """
        If drawdown exceeds halt_threshold: halve all stock weights (shift to cash).
        If drawdown recovers past resume_threshold: restore full weights.
        Returns potentially scaled weight vector.
        """
        if peak_nav <= 0:
            return weights

        dd = (current_nav - peak_nav) / peak_nav   # negative number

        if dd <= self.dd_halt and not self._halved:
            self._halved = True
            log.warning("Drawdown %.1f%% exceeds halt threshold — halving stock weights", dd * 100)

        if dd >= self.dd_resume and self._halved:
            self._halved = False
            log.info("Drawdown recovered to %.1f%% — resuming full weights", dd * 100)

        if self._halved:
            w = weights.copy().astype(float)
            n_stocks = len(w) - 1
            stock_w  = w[:n_stocks] * 0.5
            cash_w   = 1.0 - stock_w.sum()
            return np.concatenate([stock_w, [cash_w]]).astype(np.float32)

        return weights

    # ------------------------------------------------------------------
    def check_daily_loss(self, current_nav: float, sod_nav: float) -> bool:
        """
        Returns True if the daily loss limit has been breached.
        Caller should liquidate all positions to cash if True.
        """
        if sod_nav <= 0:
            return False
        daily_pnl = (current_nav - sod_nav) / sod_nav
        if daily_pnl < self.daily_limit:
            log.warning("Daily loss limit breached: %.2f%% (limit %.2f%%)",
                        daily_pnl * 100, self.daily_limit * 100)
            return True
        return False

    # ------------------------------------------------------------------
    def liquidate_weights(self, n_stocks: int) -> np.ndarray:
        """Return all-cash weight vector (used after daily loss breach)."""
        w = np.zeros(n_stocks + 1, dtype=np.float32)
        w[-1] = 1.0   # 100% cash
        return w
