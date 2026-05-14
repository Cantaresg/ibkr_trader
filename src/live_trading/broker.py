"""
IBKR broker connection and order management via ib_async.

Wraps ib_async in a synchronous interface. All methods block until the
operation completes or times out.
"""
from __future__ import annotations
import time
import asyncio
from dataclasses import dataclass, field
from typing import Optional

from ib_async import IB, Stock, LimitOrder, MarketOrder, Trade
from ib_async import util as ib_util

from src.utils.logging_config import get_logger

log = get_logger("live.broker")


@dataclass
class Position:
    ticker:  str
    shares:  float
    avg_cost: float


@dataclass
class BidAsk:
    ticker: str
    bid:    float
    ask:    float
    last:   float


@dataclass
class OrderResult:
    ticker:   str
    side:     str           # "BUY" or "SELL"
    quantity: int
    limit_px: Optional[float]
    trade:    Trade
    filled:   bool = False
    fill_px:  float = 0.0


class IBBroker:
    """
    Synchronous wrapper around ib_async for daily portfolio rebalancing.

    Connects once at startup and reuses the connection across the trading day.
    All price/order operations are synchronous (blocking).
    """

    def __init__(
        self,
        host:       str   = "127.0.0.1",
        port:       int   = 7497,
        client_id:  int   = 1,
        timeout:    int   = 30,
    ):
        self.host      = host
        self.port      = port
        self.client_id = client_id
        self.timeout   = timeout
        self._ib       = IB()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    def connect(self) -> None:
        log.info("Connecting to IBKR %s:%d (clientId=%d)...", self.host, self.port, self.client_id)
        self._ib.connect(self.host, self.port, clientId=self.client_id, timeout=self.timeout)
        log.info("Connected. Account: %s", self._ib.managedAccounts())

    def disconnect(self) -> None:
        self._ib.disconnect()
        log.info("Disconnected from IBKR.")

    def is_connected(self) -> bool:
        return self._ib.isConnected()

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------
    def get_nav(self) -> float:
        """Net liquidation value of the account in USD."""
        for av in self._ib.accountValues():
            if av.tag == "NetLiquidation" and av.currency == "USD":
                return float(av.value)
        raise RuntimeError("NetLiquidation not found in account values")

    def get_cash(self) -> float:
        """Available cash (TotalCashValue) in USD."""
        for av in self._ib.accountValues():
            if av.tag == "TotalCashValue" and av.currency == "USD":
                return float(av.value)
        return 0.0

    def get_positions(self) -> list[Position]:
        """Current equity positions (excludes cash)."""
        result = []
        for pos in self._ib.positions():
            if pos.contract.secType == "STK" and pos.position != 0:
                result.append(Position(
                    ticker   = pos.contract.symbol,
                    shares   = pos.position,
                    avg_cost = pos.avgCost,
                ))
        return result

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------
    def get_bid_ask(self, tickers: list[str], timeout: float = 5.0) -> dict[str, BidAsk]:
        """
        Request live bid/ask for a list of tickers.
        Returns dict keyed by ticker. Uses last-trade price as fallback.
        """
        contracts = [Stock(t, "SMART", "USD") for t in tickers]
        self._ib.qualifyContracts(*contracts)

        mkt_data = {c.symbol: self._ib.reqMktData(c, "", False, False) for c in contracts}
        self._ib.sleep(timeout)   # let quotes arrive

        result = {}
        for ticker, td in mkt_data.items():
            bid  = td.bid  if td.bid  and td.bid  > 0 else td.last
            ask  = td.ask  if td.ask  and td.ask  > 0 else td.last
            last = td.last if td.last and td.last > 0 else 0.0
            result[ticker] = BidAsk(ticker=ticker, bid=bid or 0.0, ask=ask or 0.0, last=last)
            self._ib.cancelMktData(mkt_data[ticker].contract if hasattr(mkt_data[ticker], 'contract') else
                                   [c for c in contracts if c.symbol == ticker][0])
        return result

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------
    def _qualify_contract(self, ticker: str) -> Stock:
        contract = Stock(ticker, "SMART", "USD")
        self._ib.qualifyContracts(contract)
        return contract

    def place_limit_order(
        self,
        ticker:    str,
        side:      str,         # "BUY" or "SELL"
        quantity:  int,
        limit_px:  float,
    ) -> OrderResult:
        if quantity <= 0:
            raise ValueError(f"quantity must be > 0, got {quantity}")
        contract = self._qualify_contract(ticker)
        order    = LimitOrder(side, quantity, round(limit_px, 2))
        trade    = self._ib.placeOrder(contract, order)
        log.info("Placed LIMIT %s %d %s @ %.2f (orderId=%s)",
                 side, quantity, ticker, limit_px, trade.order.orderId)
        return OrderResult(ticker=ticker, side=side, quantity=quantity,
                           limit_px=limit_px, trade=trade)

    def place_market_order(
        self,
        ticker:   str,
        side:     str,
        quantity: int,
    ) -> OrderResult:
        if quantity <= 0:
            raise ValueError(f"quantity must be > 0, got {quantity}")
        contract = self._qualify_contract(ticker)
        order    = MarketOrder(side, quantity)
        trade    = self._ib.placeOrder(contract, order)
        log.info("Placed MARKET %s %d %s (orderId=%s)",
                 side, quantity, ticker, trade.order.orderId)
        return OrderResult(ticker=ticker, side=side, quantity=quantity,
                           limit_px=None, trade=trade)

    def cancel_order(self, order_result: OrderResult) -> None:
        self._ib.cancelOrder(order_result.trade.order)
        log.info("Cancelled order %s %s (orderId=%s)",
                 order_result.side, order_result.ticker,
                 order_result.trade.order.orderId)

    def wait_for_fills(
        self,
        order_results: list[OrderResult],
        timeout_seconds: int = 180,
    ) -> list[OrderResult]:
        """
        Wait up to timeout_seconds for all orders to fill.
        Returns updated OrderResult list with fill status.
        """
        deadline = time.monotonic() + timeout_seconds
        pending  = list(order_results)

        while pending and time.monotonic() < deadline:
            self._ib.sleep(2)
            still_pending = []
            for r in pending:
                status = r.trade.orderStatus.status
                if status in ("Filled", "ApiCancelled", "Cancelled", "Inactive"):
                    r.filled  = (status == "Filled")
                    r.fill_px = r.trade.orderStatus.avgFillPrice or 0.0
                    log.info("Order %s %s %s: %s @ %.2f",
                             r.side, r.quantity, r.ticker, status, r.fill_px)
                else:
                    still_pending.append(r)
            pending = still_pending

        return order_results

    def cancel_unfilled_and_resubmit_market(
        self,
        order_results: list[OrderResult],
    ) -> list[OrderResult]:
        """Cancel any limit orders that did not fill and resubmit as market."""
        new_results = []
        for r in order_results:
            status = r.trade.orderStatus.status
            if not r.filled and status not in ("Filled",):
                self.cancel_order(r)
                self._ib.sleep(0.5)
                mkt = self.place_market_order(r.ticker, r.side, r.quantity)
                new_results.append(mkt)
            else:
                new_results.append(r)
        return new_results
