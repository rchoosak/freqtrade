from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any


logger = logging.getLogger(__name__)


class Notifier(ABC):
    """Sink for human-facing bot events (fills, errors, reconciliation)."""

    @abstractmethod
    def send(self, message: str) -> None: ...


class NullNotifier(Notifier):
    """Drops every message. Default when notifications are disabled."""

    def send(self, message: str) -> None:
        return None


class LoggingNotifier(Notifier):
    """Emits messages to the standard logger."""

    def send(self, message: str) -> None:
        logger.info("[notify] %s", message)


class RPCNotifier(Notifier):
    """
    Adapter that forwards messages through freqtrade's RPCManager (Telegram/Discord/webhook).

    Kept thin and optional so the bot does not hard-depend on the RPC stack; the manager is
    injected (and easily faked in tests).
    """

    def __init__(self, rpc_manager: Any) -> None:
        self._rpc = rpc_manager

    def send(self, message: str) -> None:
        from freqtrade.enums import RPCMessageType

        self._rpc.send_msg({"type": RPCMessageType.STATUS, "status": message})
