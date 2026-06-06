from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_FLOOR, Decimal
from typing import Any, Literal, get_args

from freqtrade.exceptions import OperationalException


OrderSide = Literal["buy", "sell"]
OrderKind = Literal["market", "limit", "stop"]
TimeInForce = Literal["GTC", "IOC", "FOK"]


def _validate_literal(name: str, value: str, allowed_values: tuple[str, ...]) -> None:
    if value not in allowed_values:
        allowed = ", ".join(allowed_values)
        raise ValueError(f"Invalid {name} {value!r}. Expected one of: {allowed}.")


@dataclass(frozen=True)
class MT5SymbolMapping:
    base: str
    quote: str
    mt5_symbol: str
    venue: str = "MT5"
    price_precision: int = 5
    lot_precision: int = 2
    min_lot: float = 0.01
    lot_step: float = 0.01

    @property
    def pair(self) -> str:
        return f"{self.base.upper()}/{self.quote.upper()}"

    @property
    def instrument_id(self) -> str:
        return f"{self.mt5_symbol}.{self.venue.upper()}"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MT5SymbolMapping:
        try:
            base = str(data["base"])
            quote = str(data["quote"])
            mt5_symbol = str(data["mt5_symbol"])
        except KeyError as exc:
            raise OperationalException(
                f"MT5 symbol mapping is missing required field {exc.args[0]!r}."
            ) from exc
        return cls(
            base=base,
            quote=quote,
            mt5_symbol=mt5_symbol,
            venue=str(data.get("venue", "MT5")),
            price_precision=int(data.get("price_precision", 5)),
            lot_precision=int(data.get("lot_precision", 2)),
            min_lot=float(data.get("min_lot", 0.01)),
            lot_step=float(data.get("lot_step", 0.01)),
        )


@dataclass(frozen=True)
class MT5BridgeConfig:
    symbols: tuple[MT5SymbolMapping, ...]
    terminal_path: str | None = None
    login: int | None = field(default=None, repr=False)
    password: str | None = field(default=None, repr=False)
    server: str | None = None
    dry_run: bool = True
    default_lot_size: float = 0.01
    deviation: int = 20
    magic: int = 20260606
    comment: str = "mt5-trade"
    extra: dict[str, Any] = field(default_factory=dict)

    def mapping_for(self, symbol_or_instrument_id: str) -> MT5SymbolMapping:
        raw_symbol = symbol_or_instrument_id.strip()
        for mapping in self.symbols:
            normalized = _normalize_config_symbol(raw_symbol, mapping.venue)
            pair = f"{mapping.base}{mapping.quote}".upper()
            if normalized in {mapping.mt5_symbol.upper(), pair}:
                return mapping
        raise KeyError(f"No MT5 symbol mapping configured for {symbol_or_instrument_id!r}.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MT5BridgeConfig:
        raw_symbols = data.get("symbols")
        if not raw_symbols:
            raise OperationalException("MT5 config requires a non-empty 'symbols' list.")
        symbols = tuple(MT5SymbolMapping.from_dict(item) for item in raw_symbols)
        known = {
            "symbols",
            "terminal_path",
            "login",
            "password",
            "server",
            "dry_run",
            "default_lot_size",
            "deviation",
            "magic",
            "comment",
        }
        login = data.get("login")
        return cls(
            symbols=symbols,
            terminal_path=data.get("terminal_path"),
            login=int(login) if login is not None else None,
            password=data.get("password"),
            server=data.get("server"),
            dry_run=bool(data.get("dry_run", True)),
            default_lot_size=float(data.get("default_lot_size", 0.01)),
            deviation=int(data.get("deviation", 20)),
            magic=int(data.get("magic", 20260606)),
            comment=str(data.get("comment", "mt5-trade")),
            extra={k: v for k, v in data.items() if k not in known},
        )


@dataclass(frozen=True)
class MT5BotConfig:
    """Runtime settings for the MT5 forex bot loop (separate from connection settings)."""

    timeframe: str = "M5"
    poll_interval: float = 5.0
    warmup_bars: int = 200
    db_path: str = "mt5_trade.sqlite"
    reconcile_interval: int = 0
    symbols: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.poll_interval <= 0:
            raise ValueError(f"Invalid poll_interval {self.poll_interval!r}. Expected > 0.")
        if self.warmup_bars <= 0:
            raise ValueError(f"Invalid warmup_bars {self.warmup_bars!r}. Expected > 0.")
        if self.reconcile_interval < 0:
            raise ValueError(
                f"Invalid reconcile_interval {self.reconcile_interval!r}. Expected >= 0."
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MT5BotConfig:
        return cls(
            timeframe=str(data.get("timeframe", "M5")),
            poll_interval=float(data.get("poll_interval", 5.0)),
            warmup_bars=int(data.get("warmup_bars", 200)),
            db_path=str(data.get("db_path", "mt5_trade.sqlite")),
            reconcile_interval=int(data.get("reconcile_interval", 0)),
            symbols=tuple(str(s) for s in data.get("symbols", ())),
        )


@dataclass(frozen=True)
class MT5OrderRequest:
    symbol: str
    side: OrderSide
    volume: float
    order_kind: OrderKind = "market"
    price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    time_in_force: TimeInForce = "GTC"
    client_order_id: str | None = None
    deviation: int | None = None
    magic: int | None = None
    comment: str | None = None

    def __post_init__(self) -> None:
        _validate_literal("side", self.side, get_args(OrderSide))
        _validate_literal("order_kind", self.order_kind, get_args(OrderKind))
        _validate_literal("time_in_force", self.time_in_force, get_args(TimeInForce))
        if self.volume <= 0:
            raise ValueError(f"Invalid volume {self.volume!r}. Expected a positive lot size.")


@dataclass(frozen=True)
class BrokerPosition:
    """An open position as reported by the broker terminal."""

    symbol: str
    side: OrderSide
    volume: float
    price: float | None = None
    ticket: int | None = None


@dataclass(frozen=True)
class MT5OrderResult:
    accepted: bool
    order_id: str | None
    retcode: int | None = None
    message: str | None = None
    raw: Any = None


def _normalize_config_symbol(symbol: str, venue: str) -> str:
    cleaned = symbol.upper()
    venue_suffix = f".{venue.upper()}"
    if cleaned.endswith(venue_suffix):
        cleaned = cleaned[: -len(venue_suffix)]
    return cleaned.replace("/", "").replace("_", "").replace("-", "")


def normalize_lot_size(
    volume: float,
    *,
    min_lot: float,
    lot_step: float,
    max_lot: float | None = None,
    symbol: str = "",
) -> float:
    """
    Snap a requested volume down to a valid broker lot size and validate it against the
    min/max bounds. Shared by the live gateway (using broker ``symbol_info``) and the
    dry-run path (using the configured mapping defaults) so both reject the same volumes.
    """
    step = Decimal(str(lot_step))
    minimum = Decimal(str(min_lot))
    # A non-positive max means the broker does not cap volume (some report 0 = unlimited).
    maximum = Decimal(str(max_lot)) if max_lot is not None and max_lot > 0 else None
    suffix = f" for {symbol}" if symbol else ""

    if step <= 0:
        raise OperationalException(f"Invalid MT5 lot step{suffix}: {step}.")

    requested = Decimal(str(volume))
    normalized = (requested / step).to_integral_value(rounding=ROUND_FLOOR) * step
    normalized = normalized.quantize(step)

    if normalized < minimum:
        raise ValueError(
            f"Requested lot size {volume} normalizes to {normalized}, below minimum "
            f"{minimum}{suffix}."
        )
    if maximum is not None and normalized > maximum:
        raise ValueError(
            f"Requested lot size {volume} normalizes to {normalized}, above maximum "
            f"{maximum}{suffix}."
        )

    return float(normalized)
