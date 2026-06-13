from __future__ import annotations

from types import SimpleNamespace

import pytest

from freqtrade.exceptions import OperationalException
from freqtrade.mt5_trade.execution import MT5ExecutionBridge
from freqtrade.mt5_trade.gateway import LazyMT5Gateway
from freqtrade.mt5_trade.models import MT5BridgeConfig, MT5OrderRequest, MT5SymbolMapping
from freqtrade.mt5_trade.symbols import (
    instrument_id_to_mt5_symbol,
    normalize_forex_symbol,
    to_instrument_id,
)


class FakeMT5:
    TRADE_ACTION_DEAL = 1
    TRADE_ACTION_PENDING = 5
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TYPE_BUY_LIMIT = 2
    ORDER_TYPE_SELL_LIMIT = 3
    ORDER_TYPE_BUY_STOP = 4
    ORDER_TYPE_SELL_STOP = 5
    ORDER_TIME_GTC = 0
    ORDER_TIME_DAY = 1
    ORDER_TIME_SPECIFIED = 2
    ORDER_FILLING_FOK = 0
    ORDER_FILLING_IOC = 1
    ORDER_FILLING_RETURN = 2
    SYMBOL_FILLING_FOK = 1
    SYMBOL_FILLING_IOC = 2
    TRADE_RETCODE_DONE = 10009
    TRADE_RETCODE_PLACED = 10008
    TRADE_ACTION_SLTP = 6
    TRADE_ACTION_REMOVE = 8
    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1

    def __init__(self) -> None:
        self.request = None
        self.shutdown_called = False
        self.positions: list = []
        self.pending_orders: list = []

    def initialize(self, **kwargs):
        return True

    def login(self, *args, **kwargs):
        return True

    def terminal_info(self):
        return SimpleNamespace(connected=True)

    def positions_get(self, *args, **kwargs):
        return list(self.positions)

    def orders_get(self, *args, **kwargs):
        return list(self.pending_orders)

    def last_error(self):
        return (0, "ok")

    def symbol_select(self, symbol, enabled):
        return symbol == "EURUSD" and enabled

    def symbol_info_tick(self, symbol):
        assert symbol == "EURUSD"
        return SimpleNamespace(ask=1.08501, bid=1.08499)

    def symbol_info(self, symbol):
        assert symbol == "EURUSD"
        return SimpleNamespace(volume_min=0.01, volume_max=100.0, volume_step=0.01)

    def order_send(self, request):
        self.request = request
        return SimpleNamespace(retcode=self.TRADE_RETCODE_DONE, order=12345, comment="filled")

    def shutdown(self):
        self.shutdown_called = True
        return None


class LoginFailMT5(FakeMT5):
    def login(self, *args, **kwargs):
        return False


class SymbolSelectFailMT5(FakeMT5):
    def symbol_select(self, symbol, enabled):
        return False


class FokOnlyMT5(FakeMT5):
    """Broker that only advertises FOK filling for the symbol."""

    def symbol_info(self, symbol):
        assert symbol == "EURUSD"
        return SimpleNamespace(
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
            filling_mode=FakeMT5.SYMBOL_FILLING_FOK,
        )


class IocOnlyMT5(FakeMT5):
    """Broker that only advertises IOC filling for the symbol."""

    def symbol_info(self, symbol):
        assert symbol == "EURUSD"
        return SimpleNamespace(
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
            filling_mode=FakeMT5.SYMBOL_FILLING_IOC,
        )


class UnlimitedVolumeMT5(FakeMT5):
    def symbol_info(self, symbol):
        assert symbol == "EURUSD"
        return SimpleNamespace(volume_min=0.01, volume_max=0.0, volume_step=0.01)


class SymbolInfoMissingMT5(FakeMT5):
    def symbol_info(self, symbol):
        return None


class NoRetcodeMT5(FakeMT5):
    def order_send(self, request):
        self.request = request
        return SimpleNamespace(order=12345, comment="filled")


@pytest.fixture
def bridge_config() -> MT5BridgeConfig:
    return MT5BridgeConfig(
        symbols=(MT5SymbolMapping(base="EUR", quote="USD", mt5_symbol="EURUSD"),),
        dry_run=True,
    )


def test_symbol_mapping_helpers() -> None:
    assert normalize_forex_symbol("eur/usd") == "EURUSD"
    assert normalize_forex_symbol("EURUSD.MT5") == "EURUSD"
    assert normalize_forex_symbol("EURUSD.a.MT5") == "EURUSD.a"
    assert to_instrument_id("eur_usd") == "EURUSD.MT5"
    assert to_instrument_id("EURUSD.a") == "EURUSD.a.MT5"
    assert instrument_id_to_mt5_symbol("EURUSD.MT5") == "EURUSD"
    assert instrument_id_to_mt5_symbol("EURUSD.a.MT5") == "EURUSD.a"


def test_bridge_config_finds_mapping(bridge_config: MT5BridgeConfig) -> None:
    mapping = bridge_config.mapping_for("EUR/USD")
    assert mapping.mt5_symbol == "EURUSD"
    assert mapping.instrument_id == "EURUSD.MT5"


def test_bridge_config_preserves_broker_symbol_suffix() -> None:
    config = MT5BridgeConfig(
        symbols=(MT5SymbolMapping(base="EUR", quote="USD", mt5_symbol="EURUSD.a"),),
    )

    mapping = config.mapping_for("EURUSD.a.MT5")
    assert mapping.mt5_symbol == "EURUSD.a"
    assert mapping.instrument_id == "EURUSD.a.MT5"


def test_dry_run_order_does_not_call_gateway(bridge_config: MT5BridgeConfig) -> None:
    bridge = MT5ExecutionBridge(bridge_config)
    result = bridge.submit_order(MT5OrderRequest(symbol="EURUSD.MT5", side="buy", volume=0.01))

    assert result.accepted is True
    assert result.order_id == "dry-run:EURUSD:buy:0.01"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"side": "BUY"}, "Invalid side"),
        ({"order_kind": "instant"}, "Invalid order_kind"),
        ({"time_in_force": "DAY"}, "Invalid time_in_force"),
        ({"volume": 0}, "Invalid volume"),
        ({"position_ticket": 0}, "Invalid position_ticket"),
        ({"order_kind": "limit", "price": 1.0, "position_ticket": 1}, "position_ticket"),
    ],
)
def test_order_request_validates_runtime_values(kwargs, message) -> None:
    request_kwargs = {"symbol": "EURUSD", "side": "buy", "volume": 0.01}
    request_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        MT5OrderRequest(**request_kwargs)


def test_gateway_builds_market_order_request(bridge_config: MT5BridgeConfig) -> None:
    gateway = LazyMT5Gateway(bridge_config, mt5_module=FakeMT5())
    request = gateway.build_order_send_request(
        MT5OrderRequest(
            symbol="EUR/USD",
            side="buy",
            volume=0.017,
            stop_loss=1.08,
            take_profit=1.09,
            client_order_id="order-1",
        )
    )

    assert request["action"] == FakeMT5.TRADE_ACTION_DEAL
    assert request["symbol"] == "EURUSD"
    assert request["volume"] == 0.01
    assert request["type"] == FakeMT5.ORDER_TYPE_BUY
    assert request["price"] == 1.08501
    assert request["sl"] == 1.08
    assert request["tp"] == 1.09
    assert request["comment"] == "order-1"
    assert request["type_time"] == FakeMT5.ORDER_TIME_GTC
    assert "position" not in request
    # Default (GTC) market deals must fill immediately; RETURN would be rejected by most
    # brokers, so the gateway picks an immediate-or-cancel policy instead.
    assert request["type_filling"] == FakeMT5.ORDER_FILLING_IOC


def test_gateway_sets_position_ticket_on_market_close(
    bridge_config: MT5BridgeConfig,
) -> None:
    gateway = LazyMT5Gateway(bridge_config, mt5_module=FakeMT5())

    request = gateway.build_order_send_request(
        MT5OrderRequest(symbol="EUR/USD", side="sell", volume=0.01, position_ticket=42)
    )

    assert request["action"] == FakeMT5.TRADE_ACTION_DEAL
    assert request["position"] == 42


@pytest.mark.parametrize(
    ("time_in_force", "expected_filling"),
    [
        ("IOC", FakeMT5.ORDER_FILLING_IOC),
        ("FOK", FakeMT5.ORDER_FILLING_FOK),
    ],
)
def test_gateway_maps_fill_policy_separately_from_order_time(
    bridge_config: MT5BridgeConfig,
    time_in_force: str,
    expected_filling: int,
) -> None:
    gateway = LazyMT5Gateway(bridge_config, mt5_module=FakeMT5())

    request = gateway.build_order_send_request(
        MT5OrderRequest(
            symbol="EUR/USD",
            side="buy",
            volume=0.01,
            time_in_force=time_in_force,
        )
    )

    assert request["type_time"] == FakeMT5.ORDER_TIME_GTC
    assert request["type_filling"] == expected_filling


def test_gateway_rejects_volume_below_broker_minimum(bridge_config: MT5BridgeConfig) -> None:
    gateway = LazyMT5Gateway(bridge_config, mt5_module=FakeMT5())

    with pytest.raises(ValueError, match="below minimum"):
        gateway.build_order_send_request(
            MT5OrderRequest(symbol="EUR/USD", side="buy", volume=0.009)
        )


def test_gateway_sends_order(bridge_config: MT5BridgeConfig) -> None:
    fake_mt5 = FakeMT5()
    live_config = MT5BridgeConfig(symbols=bridge_config.symbols, dry_run=False)
    gateway = LazyMT5Gateway(live_config, mt5_module=fake_mt5)

    result = gateway.order_send(MT5OrderRequest(symbol="EURUSD", side="sell", volume=0.01))

    assert result.accepted is True
    assert result.order_id == "12345"
    assert fake_mt5.request["type"] == FakeMT5.ORDER_TYPE_SELL


def test_gateway_shuts_down_after_login_failure(bridge_config: MT5BridgeConfig) -> None:
    fake_mt5 = LoginFailMT5()
    config = MT5BridgeConfig(symbols=bridge_config.symbols, login=123456, dry_run=False)
    gateway = LazyMT5Gateway(config, mt5_module=fake_mt5)

    with pytest.raises(OperationalException, match="MT5 login failed"):
        gateway.connect()

    assert fake_mt5.shutdown_called is True


def test_gateway_shuts_down_after_symbol_select_failure(bridge_config: MT5BridgeConfig) -> None:
    fake_mt5 = SymbolSelectFailMT5()
    gateway = LazyMT5Gateway(bridge_config, mt5_module=fake_mt5)

    with pytest.raises(OperationalException, match="MT5 symbol is not available"):
        gateway.connect()

    assert fake_mt5.shutdown_called is True


def test_gateway_honors_broker_filling_mode_bitmask(bridge_config: MT5BridgeConfig) -> None:
    # Broker advertises FOK only, so even a default GTC market order must use FOK.
    gateway = LazyMT5Gateway(bridge_config, mt5_module=FokOnlyMT5())

    request = gateway.build_order_send_request(
        MT5OrderRequest(symbol="EUR/USD", side="buy", volume=0.01)
    )

    assert request["type_filling"] == FakeMT5.ORDER_FILLING_FOK


@pytest.mark.parametrize(
    ("mt5_module", "requested_tif", "expected_filling"),
    [
        # Explicit FOK is honored even when the broker only advertises IOC, and vice versa,
        # so partial-fill semantics are never silently swapped.
        (IocOnlyMT5(), "FOK", FakeMT5.ORDER_FILLING_FOK),
        (FokOnlyMT5(), "IOC", FakeMT5.ORDER_FILLING_IOC),
    ],
)
def test_gateway_honors_explicit_fill_policy_even_if_unsupported(
    bridge_config: MT5BridgeConfig,
    mt5_module,
    requested_tif: str,
    expected_filling: int,
) -> None:
    gateway = LazyMT5Gateway(bridge_config, mt5_module=mt5_module)

    request = gateway.build_order_send_request(
        MT5OrderRequest(symbol="EUR/USD", side="buy", volume=0.01, time_in_force=requested_tif)
    )

    assert request["type_filling"] == expected_filling


def test_gateway_treats_zero_volume_max_as_unlimited(bridge_config: MT5BridgeConfig) -> None:
    gateway = LazyMT5Gateway(bridge_config, mt5_module=UnlimitedVolumeMT5())

    request = gateway.build_order_send_request(
        MT5OrderRequest(symbol="EUR/USD", side="buy", volume=50.0)
    )

    assert request["volume"] == 50.0


def test_gateway_raises_when_symbol_info_unavailable(bridge_config: MT5BridgeConfig) -> None:
    gateway = LazyMT5Gateway(bridge_config, mt5_module=SymbolInfoMissingMT5())

    with pytest.raises(OperationalException, match="symbol info is not available"):
        gateway.build_order_send_request(
            MT5OrderRequest(symbol="EUR/USD", side="buy", volume=0.01)
        )


def test_gateway_rejects_response_without_retcode(bridge_config: MT5BridgeConfig) -> None:
    live_config = MT5BridgeConfig(symbols=bridge_config.symbols, dry_run=False)
    gateway = LazyMT5Gateway(live_config, mt5_module=NoRetcodeMT5())

    result = gateway.order_send(MT5OrderRequest(symbol="EURUSD", side="buy", volume=0.01))

    assert result.accepted is False


def test_dry_run_rejects_volume_below_minimum(bridge_config: MT5BridgeConfig) -> None:
    bridge = MT5ExecutionBridge(bridge_config)

    with pytest.raises(ValueError, match="below minimum"):
        bridge.submit_order(MT5OrderRequest(symbol="EURUSD.MT5", side="buy", volume=0.009))


def test_dry_run_normalizes_volume_to_lot_step(bridge_config: MT5BridgeConfig) -> None:
    bridge = MT5ExecutionBridge(bridge_config)

    result = bridge.submit_order(MT5OrderRequest(symbol="EURUSD.MT5", side="buy", volume=0.017))

    assert result.order_id == "dry-run:EURUSD:buy:0.01"


def test_bridge_config_repr_hides_credentials() -> None:
    config = MT5BridgeConfig(
        symbols=(MT5SymbolMapping(base="EUR", quote="USD", mt5_symbol="EURUSD"),),
        login=123456,
        password="super-secret",
    )

    rendered = repr(config)
    assert "super-secret" not in rendered
    assert "123456" not in rendered


# --- Phase 3: reconciliation, SL/TP, cancel, reconnect ---


def _live_gateway(fake: FakeMT5) -> LazyMT5Gateway:
    config = MT5BridgeConfig(
        symbols=(MT5SymbolMapping(base="EUR", quote="USD", mt5_symbol="EURUSD"),),
        dry_run=False,
    )
    return LazyMT5Gateway(config, mt5_module=fake)


def test_gateway_open_positions_maps_side_and_fields() -> None:
    fake = FakeMT5()
    fake.positions = [
        SimpleNamespace(symbol="EURUSD", type=FakeMT5.POSITION_TYPE_SELL, volume=0.20,
                        price_open=1.085, ticket=555),
    ]
    gateway = _live_gateway(fake)

    positions = gateway.open_positions()

    assert len(positions) == 1
    assert positions[0].side == "sell"
    assert positions[0].volume == 0.20
    assert positions[0].ticket == 555


def test_gateway_modify_position_sltp_sends_sltp_action() -> None:
    fake = FakeMT5()
    fake.positions = [
        SimpleNamespace(symbol="EURUSD", type=FakeMT5.POSITION_TYPE_BUY, volume=0.10,
                        price_open=1.08, ticket=42),
    ]
    gateway = _live_gateway(fake)

    result = gateway.modify_position_sltp("EURUSD", stop_loss=1.07, take_profit=1.10)

    assert result.accepted is True
    assert fake.request["action"] == FakeMT5.TRADE_ACTION_SLTP
    assert fake.request["position"] == 42
    assert fake.request["sl"] == 1.07
    assert fake.request["tp"] == 1.10


def test_gateway_modify_position_sltp_uses_explicit_position_ticket() -> None:
    fake = FakeMT5()
    fake.positions = [
        SimpleNamespace(symbol="EURUSD", type=FakeMT5.POSITION_TYPE_BUY, volume=0.10,
                        price_open=1.08, ticket=42),
        SimpleNamespace(symbol="EURUSD", type=FakeMT5.POSITION_TYPE_BUY, volume=0.20,
                        price_open=1.09, ticket=43),
    ]
    gateway = _live_gateway(fake)

    result = gateway.modify_position_sltp(
        "EURUSD", stop_loss=1.07, take_profit=None, position_ticket=43
    )

    assert result.accepted is True
    assert fake.request["position"] == 43


def test_gateway_modify_sltp_rejects_ambiguous_symbol() -> None:
    fake = FakeMT5()
    fake.positions = [
        SimpleNamespace(symbol="EURUSD", type=FakeMT5.POSITION_TYPE_BUY, volume=0.10,
                        price_open=1.08, ticket=42),
        SimpleNamespace(symbol="EURUSD", type=FakeMT5.POSITION_TYPE_SELL, volume=0.20,
                        price_open=1.09, ticket=43),
    ]
    gateway = _live_gateway(fake)

    result = gateway.modify_position_sltp("EURUSD", stop_loss=1.07, take_profit=None)

    assert result.accepted is False
    assert "Multiple open MT5 positions" in result.message


def test_gateway_modify_sltp_without_position_is_rejected() -> None:
    gateway = _live_gateway(FakeMT5())

    result = gateway.modify_position_sltp("EURUSD", stop_loss=1.07, take_profit=None)

    assert result.accepted is False
    assert "No open MT5 position" in result.message


def test_gateway_cancel_order_sends_remove_action() -> None:
    fake = FakeMT5()
    gateway = _live_gateway(fake)

    result = gateway.cancel_order(777)

    assert result.accepted is True
    assert fake.request["action"] == FakeMT5.TRADE_ACTION_REMOVE
    assert fake.request["order"] == 777


class DropThenRecoverMT5(FakeMT5):
    """terminal_info reports a dropped link until reconnected; order_send returns None once."""

    def __init__(self) -> None:
        super().__init__()
        self._healthy = True
        self._order_calls = 0
        self.reconnects = 0

    def terminal_info(self):
        return SimpleNamespace(connected=True) if self._healthy else None

    def initialize(self, **kwargs):
        # A reconnect makes the terminal healthy again.
        self._healthy = True
        self.reconnects += 1
        return True

    def order_send(self, request):
        self._order_calls += 1
        if self._order_calls == 1:
            # Simulate the link dropping on the first attempt.
            self._healthy = False
            return None
        self.request = request
        return SimpleNamespace(retcode=self.TRADE_RETCODE_DONE, order=999, comment="ok")


def test_gateway_reconnects_and_retries_on_dropped_link() -> None:
    fake = DropThenRecoverMT5()
    gateway = _live_gateway(fake)

    result = gateway.order_send(MT5OrderRequest(symbol="EURUSD", side="buy", volume=0.01))

    assert result.accepted is True
    assert result.order_id == "999"
    # initialize() runs once on first connect and again on the reconnect.
    assert fake.reconnects >= 2


def test_gateway_open_orders_maps_pending_orders() -> None:
    fake = FakeMT5()
    fake.pending_orders = [
        SimpleNamespace(symbol="EURUSD", type=FakeMT5.ORDER_TYPE_BUY_LIMIT, volume_current=0.10,
                        price_open=1.07, ticket=900),
        SimpleNamespace(symbol="EURUSD", type=FakeMT5.ORDER_TYPE_SELL_STOP, volume_current=0.20,
                        price_open=1.09, ticket=901),
    ]
    gateway = _live_gateway(fake)

    orders = gateway.open_orders()

    assert orders[0].side == "buy"
    assert orders[0].volume == 0.10
    assert orders[0].ticket == 900
    assert orders[1].side == "sell"
    assert orders[1].ticket == 901


def test_gateway_sets_expiration_on_pending_order(bridge_config: MT5BridgeConfig) -> None:
    gateway = LazyMT5Gateway(bridge_config, mt5_module=FakeMT5())

    request = gateway.build_order_send_request(
        MT5OrderRequest(
            symbol="EUR/USD", side="buy", volume=0.01,
            order_kind="limit", price=1.07, expiration=1700000000,
        )
    )

    assert request["type_time"] == FakeMT5.ORDER_TIME_SPECIFIED
    assert request["expiration"] == 1700000000


def test_gateway_ignores_expiration_on_market_order(bridge_config: MT5BridgeConfig) -> None:
    gateway = LazyMT5Gateway(bridge_config, mt5_module=FakeMT5())

    request = gateway.build_order_send_request(
        MT5OrderRequest(symbol="EUR/USD", side="buy", volume=0.01, expiration=1700000000)
    )

    assert request["type_time"] == FakeMT5.ORDER_TIME_GTC
    assert "expiration" not in request


def test_split_lot_snaps_off_grid_legs() -> None:
    from freqtrade.mt5_trade.models import split_lot

    # 0.03 * 0.5 = 0.015 is off the 0.01 grid -> close floored to 0.01, remainder 0.02.
    assert split_lot(0.03, 0.5, min_lot=0.01, lot_step=0.01) == (0.01, 0.02)


def test_split_lot_returns_none_when_close_below_min() -> None:
    from freqtrade.mt5_trade.models import split_lot

    assert split_lot(0.01, 0.5, min_lot=0.01, lot_step=0.01) is None


def test_split_lot_returns_none_when_remainder_below_min() -> None:
    from freqtrade.mt5_trade.models import split_lot

    # 0.03 * 0.95 -> close 0.02 (>= min) but remainder 0.01 < min 0.02 -> no valid split.
    assert split_lot(0.03, 0.95, min_lot=0.02, lot_step=0.01) is None


def test_split_lot_without_grid_uses_raw_fraction() -> None:
    from freqtrade.mt5_trade.models import split_lot

    assert split_lot(1.0, 0.5, min_lot=0.0, lot_step=0.0) == (0.5, 0.5)


def test_dry_run_rejects_pending_entry(bridge_config: MT5BridgeConfig) -> None:
    bridge = MT5ExecutionBridge(bridge_config)

    result = bridge.submit_order(
        MT5OrderRequest(symbol="EURUSD", side="buy", volume=0.01, order_kind="limit", price=1.05)
    )

    assert result.accepted is False
    assert result.is_pending is False
    assert "limit" in result.message


def test_gateway_reports_normalized_request_volume(bridge_config: MT5BridgeConfig) -> None:
    fake = FakeMT5()
    live_config = MT5BridgeConfig(symbols=bridge_config.symbols, dry_run=False)
    gateway = LazyMT5Gateway(live_config, mt5_module=fake)

    result = gateway.order_send(MT5OrderRequest(symbol="EURUSD", side="buy", volume=0.017))

    # 0.017 is normalized to the broker's 0.01 lot step and reported back.
    assert result.requested_volume == 0.01


class FillPriceMT5(FakeMT5):
    def order_send(self, request):
        self.request = request
        return SimpleNamespace(
            retcode=self.TRADE_RETCODE_DONE, order=12345, comment="filled",
            price=1.23456, volume=0.01,
        )


def test_gateway_reports_fill_price(bridge_config: MT5BridgeConfig) -> None:
    live_config = MT5BridgeConfig(symbols=bridge_config.symbols, dry_run=False)
    gateway = LazyMT5Gateway(live_config, mt5_module=FillPriceMT5())

    result = gateway.order_send(MT5OrderRequest(symbol="EURUSD", side="buy", volume=0.01))

    assert result.fill_price == 1.23456


class StrictBrokerMT5(FakeMT5):
    """Broker whose minimum lot (0.1) is stricter than the config default (0.01)."""

    def symbol_info(self, symbol):
        return SimpleNamespace(volume_min=0.1, volume_max=100.0, volume_step=0.01)


def test_gateway_order_send_rejects_on_broker_normalization_error(
    bridge_config: MT5BridgeConfig,
) -> None:
    live_config = MT5BridgeConfig(symbols=bridge_config.symbols, dry_run=False)
    gateway = LazyMT5Gateway(live_config, mt5_module=StrictBrokerMT5())

    # 0.05 is below the broker's 0.1 minimum -> build raises, but order_send must convert that
    # into a rejected result rather than letting the exception escape to the bot loop.
    result = gateway.order_send(MT5OrderRequest(symbol="EURUSD", side="buy", volume=0.05))

    assert result.accepted is False
    assert "below minimum" in result.message


class AccountMT5(FakeMT5):
    def account_info(self):
        return SimpleNamespace(balance=1234.5)


def test_gateway_account_balance(bridge_config: MT5BridgeConfig) -> None:
    live_config = MT5BridgeConfig(symbols=bridge_config.symbols, dry_run=False)
    gateway = LazyMT5Gateway(live_config, mt5_module=AccountMT5())

    assert gateway.account_balance() == 1234.5


def test_bridge_account_balance_is_none_in_dry_run(bridge_config: MT5BridgeConfig) -> None:
    bridge = MT5ExecutionBridge(bridge_config)  # dry_run=True

    assert bridge.account_balance() is None
