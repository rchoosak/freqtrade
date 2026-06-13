from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass

from freqtrade.mt5_trade.models import MT5OrderRequest, MT5OrderResult


@dataclass(frozen=True)
class OpenPosition:
    symbol: str
    side: str
    volume: float
    entry_price: float | None
    ticket: int | None = None


@dataclass(frozen=True)
class ManagedPosition:
    symbol: str
    side: str
    entry_price: float
    tp1: float
    close_fraction: float
    move_be: bool
    scaled: bool


class MT5TradeStore:
    """
    Lightweight sqlite store for MT5 orders and open positions.

    Deliberately standalone (stdlib ``sqlite3``) rather than reusing freqtrade's SQLAlchemy
    Trade/Order models, which mirror CCXT exchange semantics. Records are for tracking and
    auditing the forex bot; reconciliation against the broker is a later phase.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        # check_same_thread=False keeps the single-threaded bot loop simple if reused elsewhere.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS mt5_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                client_order_id TEXT,
                order_id TEXT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                volume REAL NOT NULL,
                order_kind TEXT,
                price REAL,
                expiration INTEGER,
                accepted INTEGER NOT NULL,
                retcode INTEGER,
                message TEXT,
                ts REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS mt5_positions (
                symbol TEXT PRIMARY KEY,
                side TEXT NOT NULL,
                volume REAL NOT NULL,
                entry_price REAL,
                ticket INTEGER,
                opened_ts REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS mt5_managed_positions (
                symbol TEXT PRIMARY KEY,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                tp1 REAL NOT NULL,
                close_fraction REAL NOT NULL,
                move_be INTEGER NOT NULL,
                scaled INTEGER NOT NULL,
                updated_ts REAL NOT NULL
            );
            """
        )
        self._ensure_column("mt5_positions", "ticket", "ticket INTEGER")
        self._conn.commit()

    def _ensure_column(self, table: str, column: str, ddl: str) -> None:
        columns = {
            row["name"] for row in self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")

    def record_order(self, order: MT5OrderRequest, result: MT5OrderResult) -> None:
        self._conn.execute(
            """
            INSERT INTO mt5_orders
                (client_order_id, order_id, symbol, side, volume, order_kind, price,
                 expiration, accepted, retcode, message, ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                order.client_order_id,
                result.order_id,
                order.symbol,
                order.side,
                order.volume,
                order.order_kind,
                order.price,
                order.expiration,
                1 if result.accepted else 0,
                result.retcode,
                result.message,
                time.time(),
            ),
        )
        self._conn.commit()

    def open_position(
        self,
        symbol: str,
        side: str,
        volume: float,
        entry_price: float | None,
        ticket: int | None = None,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO mt5_positions (symbol, side, volume, entry_price, ticket, opened_ts)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                side=excluded.side,
                volume=excluded.volume,
                entry_price=excluded.entry_price,
                ticket=excluded.ticket,
                opened_ts=excluded.opened_ts
            """,
            (symbol, side, volume, entry_price, ticket, time.time()),
        )
        self._conn.commit()

    def close_position(self, symbol: str) -> None:
        self._conn.execute("DELETE FROM mt5_positions WHERE symbol = ?", (symbol,))
        self.clear_managed_position(symbol)
        self._conn.commit()

    def open_positions(self) -> dict[str, OpenPosition]:
        rows = self._conn.execute(
            "SELECT symbol, side, volume, entry_price, ticket FROM mt5_positions"
        ).fetchall()
        return {
            row["symbol"]: OpenPosition(
                symbol=row["symbol"],
                side=row["side"],
                volume=row["volume"],
                entry_price=row["entry_price"],
                ticket=row["ticket"],
            )
            for row in rows
        }

    def order_count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM mt5_orders").fetchone()[0])

    def set_managed_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        tp1: float,
        close_fraction: float,
        move_be: bool,
        scaled: bool,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO mt5_managed_positions
                (symbol, side, entry_price, tp1, close_fraction, move_be, scaled, updated_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                side=excluded.side,
                entry_price=excluded.entry_price,
                tp1=excluded.tp1,
                close_fraction=excluded.close_fraction,
                move_be=excluded.move_be,
                scaled=excluded.scaled,
                updated_ts=excluded.updated_ts
            """,
            (
                symbol,
                side,
                entry_price,
                tp1,
                close_fraction,
                1 if move_be else 0,
                1 if scaled else 0,
                time.time(),
            ),
        )
        self._conn.commit()

    def clear_managed_position(self, symbol: str) -> None:
        self._conn.execute("DELETE FROM mt5_managed_positions WHERE symbol = ?", (symbol,))
        self._conn.commit()

    def managed_positions(self) -> dict[str, ManagedPosition]:
        rows = self._conn.execute(
            """
            SELECT symbol, side, entry_price, tp1, close_fraction, move_be, scaled
            FROM mt5_managed_positions
            """
        ).fetchall()
        return {
            row["symbol"]: ManagedPosition(
                symbol=row["symbol"],
                side=row["side"],
                entry_price=row["entry_price"],
                tp1=row["tp1"],
                close_fraction=row["close_fraction"],
                move_be=bool(row["move_be"]),
                scaled=bool(row["scaled"]),
            )
            for row in rows
        }

    def close(self) -> None:
        self._conn.close()
