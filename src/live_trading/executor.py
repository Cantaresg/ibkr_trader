"""
Daily execution engine: orchestrates the full rebalance cycle.

Flow each trading day:
  1. Connect to IBKR
  2. Read account NAV, positions, and bid/ask prices
  3. Check daily loss limit (halt → liquidate → disconnect)
  4. Build observation from MarketDataStore → model.predict() → target weights
  5. Apply risk-guard (position cap, sector cap, drawdown scale)
  6. Compute delta orders via PositionManager
  7. Place limit orders, wait for fills, resubmit unfilled as market
  8. Update portfolio state for next prediction
  9. Disconnect
"""
from __future__ import annotations
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

from src.live_trading.broker import IBBroker
from src.live_trading.inference import LiveInferenceEngine
from src.live_trading.position_manager import PositionManager
from src.live_trading.risk_guard import RiskGuard
from src.utils.logging_config import get_logger

log = get_logger("live.executor")

ET = pytz.timezone("America/New_York")


class DailyExecutor:
    """
    Runs one complete rebalance cycle for a single trading day.
    Instantiate once; call run() each day.
    """

    def __init__(
        self,
        broker:           IBBroker,
        inference:        LiveInferenceEngine,
        position_mgr:     PositionManager,
        risk_guard:       RiskGuard,
        initial_capital:  float,
        universe:         list[str] | None = None,
        order_timeout_s:  int = 180,
    ):
        self.broker          = broker
        self.inference       = inference
        self.pos_mgr         = position_mgr
        self.risk_guard      = risk_guard
        self.initial_capital = initial_capital
        self.universe        = universe or []
        self.order_timeout_s = order_timeout_s

        self._sod_nav  = None    # set at start of each day
        self._peak_nav = initial_capital

    # ------------------------------------------------------------------
    def run(self, trade_date: date | None = None) -> dict:
        """
        Execute one rebalance cycle. Returns a summary dict for logging.
        """
        today = trade_date or date.today()
        ts    = pd.Timestamp(today)

        log.info("=== Rebalance cycle: %s ===", today)

        # 1. Connect
        if not self.broker.is_connected():
            try:
                self.broker.connect()
            except Exception as e:
                log.error("ERROR: broker disconnected — connect failed: %s", e)
                raise

        # 2. Account snapshot
        nav       = self.broker.get_nav()
        cash      = self.broker.get_cash()
        positions = self.broker.get_positions()
        pos_map   = {p.ticker: p.shares for p in positions}

        # Cap deployed capital — never use more than initial_capital of broker NAV
        deploy_nav = min(nav, self.initial_capital)

        if self._sod_nav is None:
            self._sod_nav = deploy_nav
        self._peak_nav = max(self._peak_nav, deploy_nav)
        nav_norm       = deploy_nav / self.initial_capital
        drawdown       = max(0.0, (self._peak_nav - deploy_nav) / self._peak_nav)

        log.info("NAV=%.2f  deploy=%.2f  cash=%.2f  positions=%d  DD=%.1f%%",
                 nav, deploy_nav, cash, len(pos_map), drawdown * 100)

        # 3. Daily loss check
        if self.risk_guard.check_daily_loss(nav, self._sod_nav):
            log.warning("ALERT: daily_loss_limit triggered — liquidating all positions")
            return self._liquidate(pos_map, nav)

        # 4. Get prices for all held + candidate tickers
        all_tickers  = list(pos_map.keys())
        weights, selected = self.inference.predict(ts, universe=self.universe)
        candidate_tickers = [t for t in selected if t]
        price_tickers = list(set(all_tickers + candidate_tickers))
        bid_ask_raw   = self.broker.get_bid_ask(price_tickers)
        bid_ask       = {t: (ba.bid, ba.ask) for t, ba in bid_ask_raw.items()}
        mid_prices    = {
            t: (ba.bid + ba.ask) / 2 if ba.bid > 0 and ba.ask > 0 else ba.ask or ba.bid
            for t, ba in bid_ask_raw.items()
        }

        # 5. Update portfolio state for next prediction
        stock_weights = PositionManager.positions_to_weight_map(
            pos_map, mid_prices, nav, selected
        )
        self.inference.update_portfolio_state(
            stock_weights, cash / nav, nav_norm, drawdown
        )

        # 6. Risk-guard: clip weights, apply drawdown scale
        weights = self.risk_guard.clip_weights(weights, selected)
        weights = self.risk_guard.apply_drawdown_scale(weights, nav, self._peak_nav)
        if self.risk_guard._halved:
            log.warning("ALERT: drawdown_halt at %.2f%% — stock weights halved", drawdown * 100)

        # 7. Compute orders (against deploy_nav, not full broker NAV)
        orders = self.pos_mgr.compute_orders(
            target_weights=weights,
            selected_tickers=selected,
            current_positions=pos_map,
            bid_ask=bid_ask,
            nav=deploy_nav,
        )

        if not orders:
            log.info("No rebalance orders needed.")
            return {"date": str(today), "nav": nav, "orders": 0, "status": "no_rebalance"}

        # 8. Place limit orders
        log.info("Placing %d limit orders...", len(orders))
        results = []
        for o in orders:
            try:
                r = self.broker.place_limit_order(o.ticker, o.side, o.quantity, o.limit_px)
                results.append(r)
            except Exception as e:
                log.error("Failed to place order %s %d %s: %s", o.side, o.quantity, o.ticker, e)

        # 9. Wait for fills
        results = self.broker.wait_for_fills(results, timeout_seconds=self.order_timeout_s)

        # 10. Resubmit unfilled as market
        unfilled = [r for r in results if not r.filled]
        if unfilled:
            log.warning("WARN: %d orders unfilled after timeout — resubmitting as market", len(unfilled))
            results = self.broker.cancel_unfilled_and_resubmit_market(results)
            results = self.broker.wait_for_fills(results, timeout_seconds=60)

        filled_count = sum(1 for r in results if r.filled)
        log.info("Rebalance complete: %d/%d orders filled.", filled_count, len(results))

        self._sod_nav = None   # reset for next day

        return {
            "date":         str(today),
            "nav":          nav,
            "orders_total": len(results),
            "orders_filled": filled_count,
            "status":       "ok",
        }

    # ------------------------------------------------------------------
    def _liquidate(self, pos_map: dict, nav: float) -> dict:
        all_tickers = list(pos_map.keys())
        if not all_tickers:
            return {"status": "liquidate_no_positions"}
        bid_ask_raw = self.broker.get_bid_ask(all_tickers)
        bid_ask     = {t: (ba.bid, ba.ask) for t, ba in bid_ask_raw.items()}
        orders      = self.pos_mgr.compute_liquidation_orders(pos_map, bid_ask)
        results     = []
        for o in orders:
            try:
                r = self.broker.place_limit_order(o.ticker, o.side, o.quantity, o.limit_px)
                results.append(r)
            except Exception as e:
                log.error("Liquidation order failed %s: %s", o.ticker, e)
        results = self.broker.wait_for_fills(results, timeout_seconds=self.order_timeout_s)
        results = self.broker.cancel_unfilled_and_resubmit_market(results)
        self.broker.wait_for_fills(results, timeout_seconds=60)
        return {"status": "liquidated", "nav": nav}
