from __future__ import annotations

from types import ModuleType
from typing import Any

from freqtrade.exceptions import OperationalException
from freqtrade.mt5_trade.models import (
    MT5BridgeConfig,
    MT5OrderRequest,
    MT5OrderResult,
    normalize_lot_size,
)


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

    def order_send(self, order: MT5OrderRequest) -> MT5OrderResult:
        self.connect()
        request = self.build_order_send_request(order)
        response = self.mt5.order_send(request)
        if response is None:
            return MT5OrderResult(
                accepted=False,
                order_id=None,
                message=f"MT5 order_send returned None: {self.mt5.last_error()}",
            )

        retcode = getattr(response, "retcode", None)
        done_code = getattr(self.mt5, "TRADE_RETCODE_DONE", None)
        placed_code = getattr(self.mt5, "TRADE_RETCODE_PLACED", None)
        success_codes = {code for code in (done_code, placed_code) if code is not None}
        accepted = retcode is not None and retcode in success_codes
        order_attr = getattr(response, "order", None)
        deal_attr = getattr(response, "deal", None)
        order_id = order_attr if order_attr is not None else deal_attr
        comment = getattr(response, "comment", None)
        return MT5OrderResult(
            accepted=accepted,
            order_id=str(order_id) if order_id is not None else None,
            retcode=retcode,
            message=comment,
            raw=response,
        )

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
            "type_time": self._type_time(),
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

    def _type_time(self) -> int:
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
