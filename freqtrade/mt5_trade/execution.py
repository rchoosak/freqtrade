from __future__ import annotations

from freqtrade.mt5_trade.gateway import LazyMT5Gateway
from freqtrade.mt5_trade.models import (
    BrokerOrder,
    BrokerPosition,
    MT5BridgeConfig,
    MT5OrderRequest,
    MT5OrderResult,
    normalize_lot_size,
)


class MT5ExecutionBridge:
    def __init__(self, config: MT5BridgeConfig, gateway: LazyMT5Gateway | None = None) -> None:
        self._config = config
        self._gateway = gateway or LazyMT5Gateway(config)

    def submit_order(self, order: MT5OrderRequest) -> MT5OrderResult:
        if self._config.dry_run:
            mapping = self._config.mapping_for(order.symbol)
            # Dry-run cannot simulate a resting order waiting to be touched; treating a
            # limit/stop entry as an instant fill would diverge from both live (which rests the
            # order) and backtest-mt5 (which fills it when price reaches the level). Reject it.
            if order.order_kind != "market":
                return MT5OrderResult(
                    accepted=False,
                    order_id=None,
                    message=(
                        f"Dry-run does not simulate {order.order_kind} (pending) entries; "
                        "use backtest-mt5 for limit/stop fills."
                    ),
                )
            # Apply the same lot normalization/validation as live so a volume that the
            # broker would reject does not silently "pass" in dry-run.
            volume = normalize_lot_size(
                order.volume,
                min_lot=mapping.min_lot,
                lot_step=mapping.lot_step,
                symbol=mapping.mt5_symbol,
            )
            return MT5OrderResult(
                accepted=True,
                order_id=f"dry-run:{mapping.mt5_symbol}:{order.side}:{volume}",
                message="Dry-run order accepted locally; no MT5 order was sent.",
            )
        return self._gateway.order_send(order)

    def modify_sltp(
        self,
        symbol: str,
        stop_loss: float | None,
        take_profit: float | None,
        *,
        position_ticket: int | None = None,
    ) -> MT5OrderResult:
        if self._config.dry_run:
            return MT5OrderResult(
                accepted=True,
                order_id=None,
                message="Dry-run SL/TP modify accepted locally; no MT5 request was sent.",
            )
        return self._gateway.modify_position_sltp(
            symbol, stop_loss, take_profit, position_ticket=position_ticket
        )

    def cancel_order(self, ticket: int) -> MT5OrderResult:
        if self._config.dry_run:
            return MT5OrderResult(
                accepted=True,
                order_id=str(ticket),
                message="Dry-run cancel accepted locally; no MT5 request was sent.",
            )
        return self._gateway.cancel_order(ticket)

    def broker_positions(self) -> list[BrokerPosition] | None:
        """
        Open positions from the broker, or None in dry-run (no broker to reconcile against).

        None is meaningfully different from an empty list: empty means the broker is flat,
        None means reconciliation does not apply.
        """
        if self._config.dry_run:
            return None
        return self._gateway.open_positions()

    def account_balance(self) -> float | None:
        """Live account balance, or None in dry-run (no broker)."""
        if self._config.dry_run:
            return None
        return self._gateway.account_balance()

    def account_equity(self) -> float | None:
        """Live account equity, or None in dry-run (no broker)."""
        if self._config.dry_run:
            return None
        return self._gateway.account_equity()

    def current_spread_points(self, symbol: str) -> int | None:
        """Live spread in points for ``symbol``, or None in dry-run / when unavailable."""
        if self._config.dry_run:
            return None
        return self._gateway.symbol_spread_points(symbol)

    def broker_orders(self) -> list[BrokerOrder] | None:
        """Resting pending orders from the broker, or None in dry-run (nothing to reconcile)."""
        if self._config.dry_run:
            return None
        return self._gateway.open_orders()

    def close(self) -> None:
        self._gateway.shutdown()
