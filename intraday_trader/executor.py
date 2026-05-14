"""
IntradayExecutor: hourly decision loop for intraday DRL trading.

Flow each trading session:
  1. Connect to IBKR (stays connected all session)
  2. Verify all positions are flat from prior day (liquidate any overnight leftovers)
  3. reset_for_new_day() on inference engine
  4. For each hourly bar [9:35, 10:35, 11:35, 12:35, 13:35, 14:35] ET:
     a. Wait until bar time
     b. Snapshot NAV + positions
     c. Check intraday loss limit (halt → liquidate → end session)
     d. model.predict() → target weights
     e. Risk-guard clip + drawdown scale
     f. Compute orders via PositionManager
     g. Place + fill (45s timeout, then market fallback)
     h. Update inference engine portfolio state
  5. Force liquidation at 15:45 ET
  6. Disconnect
"""
from __future__ import annotations
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

from src.live_trading.broker import IBBroker
from src.live_trading.position_manager import PositionManager
from src.live_trading.risk_guard import RiskGuard
from intraday_trader.inference import IntradayInferenceEngine
from src.utils.logging_config import get_logger

log = get_logger("intraday.executor")

ET = pytz.timezone("America/New_York")

_DEFAULT_BAR_TIMES_ET  = [(9, 35), (10, 35), (11, 35), (12, 35), (13, 35), (14, 35)]
_DEFAULT_EOD_LIQUIDATE = (15, 45)


class IntradayRiskGuard(RiskGuard):
    """
    Tighter risk limits for intraday trading.
    Inherits all methods from RiskGuard; only config values differ.
    """


class IntradayExecutor:
    """
    Runs one complete intraday session (SOD to EOD).
    Instantiate once; call run_session() each trading day.
    """

    def __init__(
        self,
        broker:              IBBroker,
        inference:           IntradayInferenceEngine,
        position_mgr:        PositionManager,
        risk_guard:          RiskGuard,
        initial_capital:     float,
        universe:            list[str],
        order_timeout_s:     int = 45,
        bar_times_et:        list[tuple[int, int]] | None = None,
        eod_liquidate_et:    tuple[int, int] = _DEFAULT_EOD_LIQUIDATE,
    ):
        self.broker          = broker
        self.inference       = inference
        self.pos_mgr         = position_mgr
        self.risk_guard      = risk_guard
        self.initial_capital = initial_capital
        self.universe        = universe
        self.order_timeout_s = order_timeout_s
        self.bar_times_et    = bar_times_et or _DEFAULT_BAR_TIMES_ET
        self.eod_et          = eod_liquidate_et
        self._peak_nav       = initial_capital

    # ------------------------------------------------------------------
    def run_session(self, trade_date: date | None = None) -> dict:
        """Execute one complete intraday session. Returns a summary dict."""
        today = trade_date or date.today()
        log.info("=== Intraday session: %s ===", today)

        if not self.broker.is_connected():
            self.broker.connect()

        self._verify_flat_at_sod()
        sod_nav   = self.broker.get_nav()
        self._peak_nav = max(self._peak_nav, sod_nav)
        log.info("SOD NAV=%.2f  peak=%.2f", sod_nav, self._peak_nav)

        self.inference.reset_for_new_day()
        session_results = []

        for bar_hour, bar_min in self.bar_times_et:
            bar_et = _make_et_datetime(today, bar_hour, bar_min)
            if not self._wait_until_et(bar_et):
                log.info("Session clock past target bar time — skipping remaining bars")
                break

            log.info("--- Bar %02d:%02d ET ---", bar_hour, bar_min)
            result = self.run_bar(pd.Timestamp(bar_et))
            session_results.append(result)

            if result.get("session_halted"):
                log.warning("Session halted after bar %02d:%02d", bar_hour, bar_min)
                break

        eod_et = _make_et_datetime(today, self.eod_et[0], self.eod_et[1])
        self._wait_until_et(eod_et)
        log.info("--- EOD Liquidation at %02d:%02d ET ---", self.eod_et[0], self.eod_et[1])
        eod_result = self.run_eod_liquidation()

        self.broker.disconnect()

        nav_final = eod_result.get("nav_after", sod_nav)
        day_pnl   = (nav_final - sod_nav) / max(sod_nav, 1) * 100
        log.info("Session complete: SOD=%.2f  EOD=%.2f  PnL=%.2f%%",
                 sod_nav, nav_final, day_pnl)

        return {
            "date":        str(today),
            "sod_nav":     sod_nav,
            "eod_nav":     nav_final,
            "day_pnl_pct": day_pnl,
            "bar_results": session_results,
            "eod_result":  eod_result,
        }

    # ------------------------------------------------------------------
    def run_bar(self, bar_ts: pd.Timestamp) -> dict:
        """Execute one hourly bar decision cycle."""
        nav       = self.broker.get_nav()
        positions = self.broker.get_positions()
        pos_map   = {p.ticker: p.shares for p in positions}
        deploy_nav = min(nav, self.initial_capital)
        self._peak_nav = max(self._peak_nav, deploy_nav)

        sod_nav = self.initial_capital
        if self.risk_guard.check_daily_loss(deploy_nav, sod_nav):
            log.warning("ALERT: intraday loss limit — liquidating")
            self._liquidate(pos_map)
            return {"bar_ts": str(bar_ts), "session_halted": True, "reason": "daily_loss"}

        weights, tickers = self.inference.predict(bar_ts)

        all_tickers = list(set(list(pos_map.keys()) + [t for t in tickers if t]))
        bid_ask_raw = self.broker.get_bid_ask(all_tickers)
        bid_ask     = {t: (ba.bid, ba.ask) for t, ba in bid_ask_raw.items()}
        mid_prices  = {
            t: (ba.bid + ba.ask) / 2 if ba.bid > 0 and ba.ask > 0 else ba.ask or ba.bid
            for t, ba in bid_ask_raw.items()
        }

        stock_weights = PositionManager.positions_to_weight_map(
            pos_map, mid_prices, deploy_nav, tickers
        )
        nav_norm = deploy_nav / self.initial_capital - 1.0
        drawdown = max(0.0, (self._peak_nav - deploy_nav) / max(self._peak_nav, 1))
        self.inference.update_portfolio_state(stock_weights, 1.0 - stock_weights.sum(), nav_norm, drawdown)

        weights = self.risk_guard.clip_weights(weights, tickers)
        weights = self.risk_guard.apply_drawdown_scale(weights, deploy_nav, self._peak_nav)

        orders = self.pos_mgr.compute_orders(
            target_weights    = weights,
            selected_tickers  = tickers,
            current_positions = pos_map,
            bid_ask           = bid_ask,
            nav               = deploy_nav,
        )

        if not orders:
            log.info("  No orders at this bar.")
            return {"bar_ts": str(bar_ts), "orders": 0, "nav": nav}

        log.info("  Placing %d limit orders...", len(orders))
        results = []
        for o in orders:
            try:
                r = self.broker.place_limit_order(o.ticker, o.side, o.quantity, o.limit_px)
                results.append(r)
            except Exception as e:
                log.error("  Order failed %s %s: %s", o.side, o.ticker, e)

        results = self.broker.wait_for_fills(results, timeout_seconds=self.order_timeout_s)
        unfilled = [r for r in results if not r.filled]
        if unfilled:
            log.warning("  %d unfilled — resubmitting as market", len(unfilled))
            results = self.broker.cancel_unfilled_and_resubmit_market(results)
            results = self.broker.wait_for_fills(results, timeout_seconds=30)

        filled = sum(1 for r in results if r.filled)
        log.info("  Bar %s: %d/%d orders filled", bar_ts, filled, len(results))

        return {
            "bar_ts":        str(bar_ts),
            "orders":        len(results),
            "orders_filled": filled,
            "nav":           nav,
        }

    # ------------------------------------------------------------------
    def run_eod_liquidation(self) -> dict:
        """Force flat all positions at end of session."""
        positions = self.broker.get_positions()
        pos_map   = {p.ticker: p.shares for p in positions}

        if not pos_map:
            log.info("EOD: already flat — no liquidation needed")
            return {"nav_after": self.broker.get_nav(), "orders": 0}

        all_tickers = list(pos_map.keys())
        bid_ask_raw = self.broker.get_bid_ask(all_tickers)
        bid_ask     = {t: (ba.bid, ba.ask) for t, ba in bid_ask_raw.items()}
        orders      = self.pos_mgr.compute_liquidation_orders(pos_map, bid_ask)

        results = []
        for o in orders:
            try:
                r = self.broker.place_limit_order(o.ticker, o.side, o.quantity, o.limit_px)
                results.append(r)
            except Exception as e:
                log.error("EOD liquidation order failed %s: %s", o.ticker, e)

        results = self.broker.wait_for_fills(results, timeout_seconds=45)
        results = self.broker.cancel_unfilled_and_resubmit_market(results)
        results = self.broker.wait_for_fills(results, timeout_seconds=30)

        nav_after = self.broker.get_nav()
        filled    = sum(1 for r in results if r.filled)
        log.info("EOD liquidation: %d/%d filled  NAV=%.2f", filled, len(results), nav_after)
        return {"nav_after": nav_after, "orders": len(results), "orders_filled": filled}

    # ------------------------------------------------------------------
    def _verify_flat_at_sod(self) -> None:
        """Ensure no leftover overnight positions exist; liquidate if found."""
        positions = self.broker.get_positions()
        pos_map   = {p.ticker: p.shares for p in positions if p.shares > 0}
        if pos_map:
            log.warning("SOD: found %d non-zero positions — liquidating before session start", len(pos_map))
            self._liquidate(pos_map)

    # ------------------------------------------------------------------
    def _liquidate(self, pos_map: dict) -> None:
        if not pos_map:
            return
        tickers     = list(pos_map.keys())
        bid_ask_raw = self.broker.get_bid_ask(tickers)
        bid_ask     = {t: (ba.bid, ba.ask) for t, ba in bid_ask_raw.items()}
        orders      = self.pos_mgr.compute_liquidation_orders(pos_map, bid_ask)
        results     = []
        for o in orders:
            try:
                r = self.broker.place_limit_order(o.ticker, o.side, o.quantity, o.limit_px)
                results.append(r)
            except Exception as e:
                log.error("Liquidation failed %s: %s", o.ticker, e)
        results = self.broker.wait_for_fills(results, timeout_seconds=self.order_timeout_s)
        self.broker.cancel_unfilled_and_resubmit_market(results)
        self.broker.wait_for_fills(results, timeout_seconds=30)

    # ------------------------------------------------------------------
    def _wait_until_et(self, target: datetime) -> bool:
        return _wait_until_et(target)


# ------------------------------------------------------------------
def _make_et_datetime(today: date, hour: int, minute: int) -> datetime:
    dt = datetime(today.year, today.month, today.day, hour, minute, tzinfo=ET)
    return dt


def _wait_until_et(target: datetime) -> bool:
    """Block until target ET datetime. Returns True if target is in the future."""
    now  = datetime.now(ET)
    diff = (target - now).total_seconds()
    if diff <= 0:
        return False
    log.info("Waiting %.0fs until %s ET", diff, target.strftime("%H:%M"))
    time.sleep(diff)
    return True
