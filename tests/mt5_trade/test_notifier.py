from __future__ import annotations

import logging

from freqtrade.mt5_trade.notifier import (
    LoggingNotifier,
    NullNotifier,
    RPCNotifier,
)


def test_null_notifier_is_silent() -> None:
    # Should not raise; returns None.
    assert NullNotifier().send("anything") is None


def test_logging_notifier_emits(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="freqtrade.mt5_trade.notifier"):
        LoggingNotifier().send("hello")
    assert "hello" in caplog.text


class FakeRPCManager:
    def __init__(self) -> None:
        self.messages: list = []

    def send_msg(self, msg) -> None:
        self.messages.append(msg)


def test_rpc_notifier_forwards_status_message() -> None:
    from freqtrade.enums import RPCMessageType

    rpc = FakeRPCManager()
    RPCNotifier(rpc).send("position opened")

    assert len(rpc.messages) == 1
    assert rpc.messages[0]["type"] == RPCMessageType.STATUS
    assert rpc.messages[0]["status"] == "position opened"
