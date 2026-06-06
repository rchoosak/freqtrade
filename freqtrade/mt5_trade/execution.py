from __future__ import annotations

from freqtrade.mt5_trade.gateway import LazyMT5Gateway
from freqtrade.mt5_trade.models import (
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

    def close(self) -> None:
        self._gateway.shutdown()
