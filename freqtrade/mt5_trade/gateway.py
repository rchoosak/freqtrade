from __future__ import annotations

import logging
from dataclasses import replace
from types import ModuleType
from typing import Any

from freqtrade.exceptions import OperationalException
from freqtrade.mt5_trade.models import (
    BrokerOrder,
    BrokerPosition,
    MT5BridgeConfig,
    MT5OrderRequest,
    MT5OrderResult,
    normalize_lot_size,
)


logger = logging.getLogger(__name__)


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


class LazyMT5Gateway:
    def __init__(self, config: MT5BridgeConfig, mt5_module: ModuleType | Any | None = None) -> None:
        self._config = config
        self._mt5 = mt5_module
        self._connected = False

    @property
    def mt5(self) -> Any:
        if self._mt5 is None:
            try:
                import MetaTrader5 as mt5
            except ImportError as exc:
                raise OperationalException(
                    "MetaTrader5 is not installed. Install it in the runtime environment that "
                    "hosts the MT5 terminal before using live MT5 execution."
                ) from exc
            self._mt5 = mt5
        return self._mt5

    def connect(self) -> None:
        if self._connected:
            return

        kwargs: dict[str, Any] = {}
        if self._config.terminal_path:
            kwargs["path"] = self._config.terminal_path

        initialized = False
        try:
            if not self.mt5.initialize(**kwargs):
                raise OperationalException(f"MT5 initialize failed: {self.mt5.last_error()}")
            initialized = True

            if self._config.login is not None:
                logged_in = self.mt5.login(
                    self._config.login,
                    password=self._config.password,
                    server=self._config.server,
                )
                if not logged_in:
                    raise OperationalException(f"MT5 login failed: {self.mt5.last_error()}")

            for mapping in self._config.symbols:
                if not self.mt5.symbol_select(mapping.mt5_symbol, True):
                    raise OperationalException(f"MT5 symbol is not available: {mapping.mt5_symbol}")

            self._connected = True
        except Exception:
            if initialized:
                self.mt5.shutdown()
            self._connected = False
            raise

    def shutdown(self) -> None:
        if self._mt5 is not None and self._connected:
            self._mt5.shutdown()
        self._connected = False

    def is_healthy(self) -> bool:
        """Cheap liveness probe: a connected terminal returns terminal info."""
        if not self._connected:
            return False
        try:
            return self.mt5.terminal_info() is not None
        except Exception:
            return False

    def ensure_connected(self) -> None:
        """
        Connect if needed, and transparently reconnect if the terminal dropped.

        Live terminals can restart or lose their link mid-session; this re-establishes the
        connection so the next call works instead of failing on a stale handle.
        """
        if not self._connected:
            self.connect()
            return
        if not self.is_healthy():
            logger.warning("MT5 terminal connection lost; reconnecting.")
            try:
                self.shutdown()
            except Exception:
                self._connected = False
            self.connect()

    def order_send(self, order: MT5OrderRequest) -> MT5OrderResult:
        try:
            request = self.build_order_send_request(order)
        except (ValueError, OperationalException, KeyError) as exc:
            # Building the request normalizes against the broker's symbol_info, which can be
            # stricter than the config (below min lot, missing symbol/tick, bad lot step). Return
            # a rejected result so the bot follows its normal reject path instead of the loop's
            # generic exception handler.
            return MT5OrderResult(
                accepted=False, order_id=None, message=f"order build failed: {exc}"
            )
        result = self._dispatch(request, description=f"order_send {order.symbol} {order.side}")
        # request["volume"] is normalized against the broker's symbol_info, so report it back —
        # it can differ from the bot's config-normalized request when broker rules differ.
        return replace(result, requested_volume=float(request["volume"]))

    def _dispatch(self, request: dict[str, Any], *, description: str) -> MT5OrderResult:
        """Send a trade request with one transparent reconnect+retry on a dropped link."""
        for attempt in (1, 2):
            self.ensure_connected()
            response = self.mt5.order_send(request)
            if response is not None:
                return self._parse_result(response)
            # None usually means the request never reached a live terminal.
            if attempt == 1 and not self.is_healthy():
                logger.warning("MT5 %s returned None; retrying after reconnect.", description)
                continue
            return MT5OrderResult(
                accepted=False,
                order_id=None,
                message=f"MT5 {description} returned None: {self.mt5.last_error()}",
            )
        raise AssertionError("unreachable")  # pragma: no cover

    def _parse_result(self, response: Any) -> MT5OrderResult:
        retcode = getattr(response, "retcode", None)
        done_code = getattr(self.mt5, "TRADE_RETCODE_DONE", None)
        placed_code = getattr(self.mt5, "TRADE_RETCODE_PLACED", None)
        success_codes = {code for code in (done_code, placed_code) if code is not None}
        accepted = retcode is not None and retcode in success_codes
        is_pending = accepted and placed_code is not None and retcode == placed_code
        order_attr = getattr(response, "order", None)
        deal_attr = getattr(response, "deal", None)
        order_id = order_attr if order_attr is not None else deal_attr
        comment = getattr(response, "comment", None)
        return MT5OrderResult(
            accepted=accepted,
            order_id=str(order_id) if order_id is not None else None,
            retcode=retcode,
            message=comment,
            filled_volume=_optional_float(getattr(response, "volume", None)),
            fill_price=_optional_float(getattr(response, "price", None)),
            is_pending=is_pending,
            raw=response,
        )

    def account_balance(self) -> float | None:
        """Current account balance from the terminal (for risk-percent sizing that compounds)."""
        self.ensure_connected()
        info = self.mt5.account_info()
        return _optional_float(getattr(info, "balance", None)) if info is not None else None

    def account_equity(self) -> float | None:
        """Current account equity, including floating PnL, for conservative live risk sizing."""
        self.ensure_connected()
        info = self.mt5.account_info()
        return _optional_float(getattr(info, "equity", None)) if info is not None else None

    def symbol_spread_points(self, symbol: str) -> int | None:
        """Current spread (in points) for ``symbol`` from the terminal, or None if unavailable."""
        self.ensure_connected()
        info = self.mt5.symbol_info(symbol)
        if info is None:
            return None
        spread = getattr(info, "spread", None)
        return int(spread) if spread is not None else None

    def open_positions(self) -> list[BrokerPosition]:
        """Return the broker's currently open positions (used for reconciliation)."""
        self.ensure_connected()
        raw = self.mt5.positions_get()
        if not raw:
            return []
        buy_type = getattr(self.mt5, "POSITION_TYPE_BUY", 0)
        positions: list[BrokerPosition] = []
        for pos in raw:
            side: Any = "buy" if getattr(pos, "type", buy_type) == buy_type else "sell"
            positions.append(
                BrokerPosition(
                    symbol=str(getattr(pos, "symbol", "")),
                    side=side,
                    volume=float(getattr(pos, "volume", 0.0)),
                    price=_optional_float(getattr(pos, "price_open", None)),
                    ticket=_optional_int(getattr(pos, "ticket", None)),
                )
            )
        return positions

    def open_orders(self) -> list[BrokerOrder]:
        """Return the broker's currently resting (pending) orders, for reconciliation."""
        self.ensure_connected()
        raw = self.mt5.orders_get()
        if not raw:
            return []
        buy_types = {
            getattr(self.mt5, name)
            for name in ("ORDER_TYPE_BUY", "ORDER_TYPE_BUY_LIMIT", "ORDER_TYPE_BUY_STOP")
            if hasattr(self.mt5, name)
        }
        orders: list[BrokerOrder] = []
        for order in raw:
            order_type = getattr(order, "type", None)
            # Fall back to MT5's even=buy / odd=sell convention if constants are unavailable.
            is_buy = order_type in buy_types if buy_types else (int(order_type or 0) % 2 == 0)
            side: Any = "buy" if is_buy else "sell"
            orders.append(
                BrokerOrder(
                    symbol=str(getattr(order, "symbol", "")),
                    side=side,
                    volume=float(getattr(order, "volume_current", getattr(order, "volume", 0.0))),
                    price=_optional_float(getattr(order, "price_open", None)),
                    ticket=_optional_int(getattr(order, "ticket", None)),
                )
            )
        return orders

    def modify_position_sltp(
        self,
        symbol: str,
        stop_loss: float | None,
        take_profit: float | None,
        *,
        position_ticket: int | None = None,
    ) -> MT5OrderResult:
        """Set/clear the broker-side stop-loss and take-profit on an open position."""
        mapping = self._config.mapping_for(symbol)
        self.ensure_connected()
        if position_ticket is None:
            positions = self._matching_positions(mapping.mt5_symbol)
            if not positions:
                return MT5OrderResult(
                    accepted=False,
                    order_id=None,
                    message=f"No open MT5 position for {mapping.mt5_symbol} to modify.",
                )
            if len(positions) > 1:
                return MT5OrderResult(
                    accepted=False,
                    order_id=None,
                    message=(
                        f"Multiple open MT5 positions for {mapping.mt5_symbol}; "
                        "position_ticket is required for SL/TP."
                    ),
                )
            position_ticket = positions[0].ticket
        if position_ticket is None:
            return MT5OrderResult(
                accepted=False,
                order_id=None,
                message=f"Open MT5 position for {mapping.mt5_symbol} has no ticket.",
            )
        request: dict[str, Any] = {
            "action": getattr(self.mt5, "TRADE_ACTION_SLTP", 6),
            "symbol": mapping.mt5_symbol,
            "position": position_ticket,
            "sl": round(stop_loss, mapping.price_precision) if stop_loss is not None else 0.0,
            "tp": round(take_profit, mapping.price_precision) if take_profit is not None else 0.0,
        }
        return self._dispatch(request, description=f"modify_sltp {mapping.mt5_symbol}")

    def cancel_order(self, ticket: int) -> MT5OrderResult:
        """Cancel a pending order by its ticket id."""
        request: dict[str, Any] = {
            "action": getattr(self.mt5, "TRADE_ACTION_REMOVE", 8),
            "order": ticket,
        }
        return self._dispatch(request, description=f"cancel_order {ticket}")

    def _matching_positions(self, mt5_symbol: str) -> list[BrokerPosition]:
        return [position for position in self.open_positions() if position.symbol == mt5_symbol]

    def build_order_send_request(self, order: MT5OrderRequest) -> dict[str, Any]:
        mapping = self._config.mapping_for(order.symbol)
        symbol_info = self._symbol_info(mapping.mt5_symbol)
        order_type = self._order_type(order)
        request: dict[str, Any] = {
            "action": self._action(order),
            "symbol": mapping.mt5_symbol,
            "volume": self._normalize_volume(mapping, order.volume, symbol_info),
            "type": order_type,
            "deviation": order.deviation if order.deviation is not None else self._config.deviation,
            "magic": order.magic if order.magic is not None else self._config.magic,
            "comment": order.comment or order.client_order_id or self._config.comment,
            "type_time": self._type_time(order),
            "type_filling": self._type_filling(order, symbol_info),
        }

        if order.order_kind == "market":
            tick = self.mt5.symbol_info_tick(mapping.mt5_symbol)
            if tick is None:
                raise OperationalException(f"MT5 tick is not available for {mapping.mt5_symbol}.")
            request["price"] = getattr(tick, "ask" if order.side == "buy" else "bid")
        elif order.price is None:
            raise ValueError(f"{order.order_kind} orders require an explicit price.")
        else:
            request["price"] = round(order.price, mapping.price_precision)

        if order.stop_loss is not None:
            request["sl"] = round(order.stop_loss, mapping.price_precision)
        if order.take_profit is not None:
            request["tp"] = round(order.take_profit, mapping.price_precision)
        if order.position_ticket is not None:
            request["position"] = order.position_ticket
        # A broker-side expiry only applies to resting (pending) orders.
        if order.expiration is not None and order.order_kind != "market":
            request["expiration"] = order.expiration

        return request

    def _action(self, order: MT5OrderRequest) -> int:
        if order.order_kind == "market":
            return getattr(self.mt5, "TRADE_ACTION_DEAL", 1)
        return getattr(self.mt5, "TRADE_ACTION_PENDING", 5)

    def _order_type(self, order: MT5OrderRequest) -> int:
        if order.order_kind == "market":
            return getattr(self.mt5, "ORDER_TYPE_BUY" if order.side == "buy" else "ORDER_TYPE_SELL")
        if order.order_kind == "limit":
            return getattr(
                self.mt5,
                "ORDER_TYPE_BUY_LIMIT" if order.side == "buy" else "ORDER_TYPE_SELL_LIMIT",
            )
        return getattr(
            self.mt5,
            "ORDER_TYPE_BUY_STOP" if order.side == "buy" else "ORDER_TYPE_SELL_STOP",
        )

    def _type_time(self, order: MT5OrderRequest) -> int:
        # An expiry timestamp switches the order to "specified time"; otherwise good-till-cancel.
        if order.expiration is not None and order.order_kind != "market":
            return getattr(self.mt5, "ORDER_TIME_SPECIFIED", 2)
        return getattr(self.mt5, "ORDER_TIME_GTC", 0)

    def _type_filling(self, order: MT5OrderRequest, symbol_info: Any) -> int:
        """
        Pick the MT5 order filling policy.

        FOK and IOC differ in partial-fill semantics, so an explicit ``time_in_force`` is
        honored exactly — never silently substituted. If the broker does not support it the
        order is rejected with a clear retcode, which is safer than changing the trader's
        intent. Only the implicit default (GTC) is auto-selected: market deals must fill
        immediately (``ORDER_FILLING_RETURN`` is rejected by most brokers, retcode 10030), so
        an immediate-or-cancel policy the broker advertises is chosen instead.
        """
        fok = getattr(self.mt5, "ORDER_FILLING_FOK", 0)
        ioc = getattr(self.mt5, "ORDER_FILLING_IOC", 1)
        ret = getattr(self.mt5, "ORDER_FILLING_RETURN", 2)

        if order.time_in_force == "FOK":
            return fok
        if order.time_in_force == "IOC":
            return ioc

        # Resting pending order: RETURN is the standard policy.
        if order.order_kind != "market":
            return ret

        # Default (GTC) market order: prefer IOC, then FOK, from what the symbol advertises.
        # ``symbol_info.filling_mode`` is a bitmask; an unknown (0) mask means the broker did
        # not report capabilities, so default to IOC and let the broker have the final say.
        mask = getattr(symbol_info, "filling_mode", 0) or 0
        symbol_fok = getattr(self.mt5, "SYMBOL_FILLING_FOK", 1)
        symbol_ioc = getattr(self.mt5, "SYMBOL_FILLING_IOC", 2)
        if not mask or mask & symbol_ioc:
            return ioc
        if mask & symbol_fok:
            return fok
        return ret

    def _symbol_info(self, mt5_symbol: str) -> Any:
        symbol_info = self.mt5.symbol_info(mt5_symbol)
        if symbol_info is None:
            raise OperationalException(
                f"MT5 symbol info is not available for {mt5_symbol}: {self.mt5.last_error()}"
            )
        return symbol_info

    def _normalize_volume(self, mapping, volume: float, symbol_info: Any) -> float:
        return normalize_lot_size(
            volume,
            min_lot=getattr(symbol_info, "volume_min", mapping.min_lot),
            lot_step=getattr(symbol_info, "volume_step", mapping.lot_step),
            max_lot=getattr(symbol_info, "volume_max", None),
            symbol=mapping.mt5_symbol,
        )
