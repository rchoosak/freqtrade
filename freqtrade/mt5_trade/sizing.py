from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from freqtrade.mt5_trade.models import MT5SymbolMapping, OrderSide, normalize_lot_size


PositionSizingMode = Literal["fixed", "risk_percent"]

_ENTRY_SIDE: dict[str, OrderSide] = {"enter_long": "buy", "enter_short": "sell"}


@dataclass(frozen=True)
class SizingDecision:
    volume: float | None
    reason: str | None = None

    @property
    def skipped(self) -> bool:
        return self.volume is None


@dataclass(frozen=True)
class PositionSizer:
    mode: PositionSizingMode = "fixed"
    fixed_lot_size: float = 0.01
    risk_per_trade: float = 1.0
    contract_size: float = 1.0
    min_lot: float | None = None
    lot_step: float | None = None
    max_lot: float | None = None
    skip_if_min_lot_exceeds_risk: bool = True

    @classmethod
    def from_config(
        cls,
        extra: dict[str, Any],
        *,
        default_lot_size: float,
        contract_size: float = 1.0,
    ) -> PositionSizer:
        raw = extra.get("position_sizing", {})
        if not isinstance(raw, dict):
            raise ValueError("position_sizing must be an object.")

        mode = str(raw.get("mode", "fixed")).lower()
        if mode not in {"fixed", "risk_percent"}:
            raise ValueError("position_sizing.mode must be 'fixed' or 'risk_percent'.")

        fixed_lot_size = _as_float(
            raw.get("lot_size", raw.get("fixed_lot_size", default_lot_size)), default_lot_size
        )
        sizing_contract_size = _as_float(raw.get("contract_size", contract_size), contract_size)
        risk_per_trade = _as_float(raw.get("risk_per_trade", raw.get("risk_percent", 1.0)), 1.0)

        sizer = cls(
            mode=mode,  # type: ignore[arg-type]
            fixed_lot_size=fixed_lot_size,
            risk_per_trade=risk_per_trade,
            contract_size=sizing_contract_size,
            min_lot=_optional_float(raw.get("min_lot")),
            lot_step=_optional_float(raw.get("lot_step")),
            max_lot=_optional_float(raw.get("max_lot")),
            skip_if_min_lot_exceeds_risk=bool(
                raw.get("skip_if_min_lot_exceeds_risk", True)
            ),
        )
        sizer.validate()
        return sizer

    def validate(self) -> None:
        if self.fixed_lot_size <= 0:
            raise ValueError("position_sizing lot size must be positive.")
        if self.risk_per_trade <= 0:
            raise ValueError("position_sizing.risk_per_trade must be positive.")
        if self.contract_size <= 0:
            raise ValueError("position_sizing.contract_size must be positive.")
        if self.min_lot is not None and self.min_lot <= 0:
            raise ValueError("position_sizing.min_lot must be positive.")
        if self.lot_step is not None and self.lot_step <= 0:
            raise ValueError("position_sizing.lot_step must be positive.")
        if self.max_lot is not None and self.max_lot <= 0:
            raise ValueError("position_sizing.max_lot must be positive.")

    @property
    def requires_balance(self) -> bool:
        return self.mode == "risk_percent"

    def size_entry(
        self,
        *,
        symbol: str,
        side: OrderSide,
        entry_price: float,
        stop_loss: float | None,
        balance: float | None,
        mapping: MT5SymbolMapping | None = None,
    ) -> SizingDecision:
        min_lot = self.min_lot if self.min_lot is not None else _mapping_min_lot(mapping)
        lot_step = self.lot_step if self.lot_step is not None else _mapping_lot_step(mapping)
        max_lot = self.max_lot

        if self.mode == "fixed":
            return self._normalize(self.fixed_lot_size, min_lot, lot_step, max_lot, symbol)

        if balance is None:
            raise ValueError("risk_percent position sizing requires starting_balance.")
        if stop_loss is None:
            return SizingDecision(None, f"{symbol}: risk_percent sizing requires stop_loss.")

        stop_distance = _risk_stop_distance(side, entry_price, stop_loss)
        if stop_distance <= 0:
            return SizingDecision(
                None,
                f"{symbol}: stop_loss must be beyond entry price for {side} risk sizing.",
            )

        risk_amount = balance * self.risk_per_trade / 100
        raw_volume = risk_amount / (stop_distance * self.contract_size)
        if raw_volume < min_lot and self.skip_if_min_lot_exceeds_risk:
            actual_risk = min_lot * stop_distance * self.contract_size
            return SizingDecision(
                None,
                (
                    f"{symbol}: min_lot {min_lot} risks {actual_risk:.2f}, above target "
                    f"{risk_amount:.2f}."
                ),
            )

        requested = max(raw_volume, min_lot)
        if max_lot is not None:
            requested = min(requested, max_lot)
        return self._normalize(requested, min_lot, lot_step, max_lot, symbol)

    def _normalize(
        self,
        volume: float,
        min_lot: float,
        lot_step: float,
        max_lot: float | None,
        symbol: str,
    ) -> SizingDecision:
        try:
            return SizingDecision(
                normalize_lot_size(
                    volume,
                    min_lot=min_lot,
                    lot_step=lot_step,
                    max_lot=max_lot,
                    symbol=symbol,
                )
            )
        except ValueError as exc:
            return SizingDecision(None, str(exc))


def entry_side_for_action(action: str) -> OrderSide | None:
    return _ENTRY_SIDE.get(action)


def _risk_stop_distance(side: OrderSide, entry_price: float, stop_loss: float) -> float:
    if side == "buy":
        return entry_price - stop_loss
    return stop_loss - entry_price


def _mapping_min_lot(mapping: MT5SymbolMapping | None) -> float:
    return mapping.min_lot if mapping is not None else 0.01


def _mapping_lot_step(mapping: MT5SymbolMapping | None) -> float:
    return mapping.lot_step if mapping is not None else 0.01


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def _as_float(value: Any, default: float) -> float:
    return float(value) if value is not None else default
