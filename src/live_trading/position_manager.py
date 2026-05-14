"""
Position manager: translates target portfolio weights into IBKR orders.

Given:
  - target_weights  (N_STOCKS + 1,) from the inference engine (post risk-guard)
  - current_positions from broker.get_positions()
  - current bid/ask prices from broker.get_bid_ask()
  - portfolio NAV from broker.get_nav()

Computes buy/sell orders (whole shares) needed to reach target weights,
respecting a minimum trade size to avoid churning on tiny rebalances.
"""
from __future__ import annotations
import math
from dataclasses import dataclass

import numpy as np

from src.utils.logging_config import get_logger

log = get_logger("live.positions")

MIN_TRADE_DOLLARS = 20.0    # ignore rebalance legs smaller than this
MIN_WEIGHT_DELTA  = 0.005   # ignore weight changes smaller than 0.5%


@dataclass
class Order:
    ticker:   str
    side:     str       # "BUY" or "SELL"
    quantity: int       # whole shares
    limit_px: float     # bid-$0.01 for sells, ask+$0.01 for buys


class PositionManager:
    """
    Computes rebalancing orders.

    Configuration is read from ibkr.execution in config.yaml:
      buy_limit_offset:  0.01  (add to ask)
      sell_limit_offset: 0.01  (subtract from bid)
    """

    def __init__(self, execution_cfg: dict):
        self.buy_offset  = float(execution_cfg.get("buy_limit_offset",  0.01))
        self.sell_offset = float(execution_cfg.get("sell_limit_offset", 0.01))

    # ------------------------------------------------------------------
    def compute_orders(
        self,
        target_weights:      np.ndarray,     # (N_STOCKS + 1,) — last element = cash
        selected_tickers:    list[str],      # N_STOCKS ticker strings (may contain "")
        current_positions:   dict[str, float],  # {ticker: shares_held}
        bid_ask:             dict[str, tuple[float, float]],  # {ticker: (bid, ask)}
        nav:                 float,
    ) -> list[Order]:
        """
        Returns a list of Order objects to rebalance to target_weights.

        Sells are computed before buys so broker ensures buying power exists.
        """
        if nav <= 0:
            log.warning("NAV is zero — skipping order computation")
            return []

        target_values = target_weights[:-1] * nav    # cash is last, skip it
        orders_sell: list[Order] = []
        orders_buy:  list[Order] = []

        for i, ticker in enumerate(selected_tickers):
            if not ticker:
                continue

            bid, ask = bid_ask.get(ticker, (0.0, 0.0))
            mid_px   = (bid + ask) / 2 if bid > 0 and ask > 0 else ask or bid
            if mid_px <= 0:
                log.warning("No valid price for %s — skipping", ticker)
                continue

            target_shares  = int(target_values[i] / mid_px)
            current_shares = int(current_positions.get(ticker, 0))
            delta          = target_shares - current_shares

            # Skip trivial changes
            if abs(delta) == 0:
                continue
            if abs(delta * mid_px) < MIN_TRADE_DOLLARS:
                continue
            weight_delta = abs(target_weights[i] - current_shares * mid_px / nav)
            if weight_delta < MIN_WEIGHT_DELTA:
                continue

            if delta > 0:
                limit_px = round(ask + self.buy_offset, 2)
                orders_buy.append(Order(ticker, "BUY", delta, limit_px))
                log.info("  BUY  %d %s @ limit %.2f (target %.0f%% nav)",
                         delta, ticker, limit_px, target_weights[i] * 100)
            else:
                limit_px = round(bid - self.sell_offset, 2)
                orders_sell.append(Order(ticker, "SELL", abs(delta), limit_px))
                log.info("  SELL %d %s @ limit %.2f (target %.0f%% nav)",
                         abs(delta), ticker, limit_px, target_weights[i] * 100)

        # Sells first (free up cash before buys)
        return orders_sell + orders_buy

    # ------------------------------------------------------------------
    def compute_liquidation_orders(
        self,
        current_positions:  dict[str, float],
        bid_ask:            dict[str, tuple[float, float]],
    ) -> list[Order]:
        """Generate market-at-limit SELL orders to liquidate all positions."""
        orders = []
        for ticker, shares in current_positions.items():
            if shares <= 0:
                continue
            bid, ask = bid_ask.get(ticker, (0.0, 0.0))
            mid      = (bid + ask) / 2 if bid > 0 and ask > 0 else bid or ask or 1.0
            limit_px = round(bid - self.sell_offset, 2) if bid > 0 else round(mid * 0.99, 2)
            orders.append(Order(ticker, "SELL", int(shares), limit_px))
            log.info("  LIQUIDATE %d %s @ %.2f", int(shares), ticker, limit_px)
        return orders

    # ------------------------------------------------------------------
    @staticmethod
    def positions_to_weight_map(
        current_positions: dict[str, float],
        prices:            dict[str, float],
        nav:               float,
        selected_tickers:  list[str],
    ) -> np.ndarray:
        """
        Convert IB positions into a weight vector aligned to selected_tickers.
        Returns (N_STOCKS,) array for use in LiveInferenceEngine.update_portfolio_state().
        """
        weights = np.zeros(len(selected_tickers), dtype=np.float32)
        if nav <= 0:
            return weights
        for i, t in enumerate(selected_tickers):
            if t and t in current_positions and t in prices:
                weights[i] = current_positions[t] * prices[t] / nav
        return weights
