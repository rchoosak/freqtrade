# mt5-trade Forex Integration

This document describes the architecture and implementation plan for trading forex instruments
through a MetaTrader 5 (MT5) terminal, using a dedicated bot that connects to MT5 **directly**
via the official `MetaTrader5` Python package.

## Context from the existing system

Freqtrade is built around a candle-driven bot loop:

- Configuration selects strategy, exchange, stake, pricing, risk, and runtime mode.
- Strategies calculate indicators and entry/exit signals from completed OHLCV candles.
- The live bot loop refreshes market data, updates open orders/trades, evaluates exits,
  adjusts positions, and finally evaluates new entries.
- Exchange access is abstracted behind exchange classes, which currently assume CCXT-like
  crypto exchange semantics.
- Backtesting, hyperopt, persistence, RPC, and UI are separate from the exchange connector.

MT5 forex has a different boundary: it is a broker terminal with lot-based sizing, not a CCXT
crypto exchange. The safest design is therefore **not** to force MT5 into Freqtrade's CCXT
exchange contract. Instead, `freqtrade.mt5_trade` is a self-contained bridge + bot package that
talks to the MT5 terminal directly and runs its own trading loop.

## Design

```mermaid
flowchart LR
    Strategy["MT5Strategy (on_bar -> Signal)"] --> Bot["MT5ForexBot loop"]
    Feed["MT5DataFeed"] --> Bot
    Bot --> Bridge["MT5ExecutionBridge"]
    Bridge --> Gateway["LazyMT5Gateway"]
    Gateway --> Terminal["MT5 terminal / broker"]
    Feed -.live.-> Gateway
    Bot --> Store["MT5TradeStore (sqlite)"]
```

### Responsibilities

- `models.py` — typed config (`MT5BridgeConfig`, `MT5BotConfig`), order/result models, and
  shared lot-size normalization. `from_dict` builders parse the config section.
- `config.py` — `load_mt5_config()` loads + validates the `mt5_trade` config section (local JSON
  schema, decoupled from the crypto-bot `CONF_SCHEMA`).
- `symbols.py` — maps `EUR/USD`, `EURUSD`, and `EURUSD.MT5` to consistent MT5 symbols.
- `gateway.py` — `LazyMT5Gateway` lazy-loads `MetaTrader5`, opens the terminal connection,
  selects symbols, normalizes lots, and sends MT5 order requests.
- `execution.py` — `MT5ExecutionBridge` submits orders (dry-run locally, or via the gateway).
- `data.py` — `MT5DataFeed`: `LiveMT5DataFeed` (live bars from `copy_rates_from_pos`) and
  `ReplayDataFeed` (deterministic offline bars for dry-run and tests).
- `strategy.py` — `MT5Strategy` interface (`on_bar -> Signal`) plus a `SmaCrossStrategy` demo.
- `bot.py` — `MT5ForexBot`: the loop that pulls bars, asks the strategy, manages one position
  per symbol, submits orders, and records them.
- `persistence.py` — `MT5TradeStore`, a lightweight stdlib-`sqlite3` store for orders/positions
  (independent of the CCXT-coupled SQLAlchemy `Trade`/`Order` models).
- `runner.py` — `MT5TradeRuntime` assembles the bot from config and runs it.

### Running

```bash
freqtrade trade-mt5 --config mt5-config.json
```

The `trade-mt5` command loads the standalone `mt5_trade` config directly (not through the
crypto-bot config validation) and starts the bot. In dry-run with a `replay_data` file it runs
fully offline; for live trading it connects to a running MT5 terminal.

## Configuration Shape

```json
{
  "mt5_trade": {
    "terminal_path": "C:/Program Files/MetaTrader 5/terminal64.exe",
    "login": 123456,
    "password": "secret",
    "server": "Broker-Demo",
    "dry_run": true,
    "default_lot_size": 0.01,
    "deviation": 20,
    "magic": 20260606,
    "timeframe": "M5",
    "poll_interval": 5.0,
    "warmup_bars": 200,
    "db_path": "mt5_trade.sqlite",
    "strategy": {"fast": 10, "slow": 30},
    "replay_data": "user_data/mt5_bars.json",
    "trade_symbols": ["EURUSD"],
    "symbols": [
      {"venue": "MT5", "base": "EUR", "quote": "USD", "mt5_symbol": "EURUSD"},
      {"venue": "MT5", "base": "GBP", "quote": "USD", "mt5_symbol": "GBPUSD"}
    ]
  }
}
```

- `trade_symbols` selects which mapped symbols the bot trades (defaults to all mapped symbols).
- `replay_data` (optional) points to a JSON file mapping `symbol -> [[time,o,h,l,c,v], ...]`;
  when present the bot replays it offline instead of using the live MT5 feed.
- `strategy` parameters are passed to the default `SmaCrossStrategy`.

## Execution Plan

**Phase 1 — Bridge primitives (done)**
1. Typed models, forex symbol mapping, lazy `MetaTrader5` gateway, execution bridge.
2. Tests for symbol mapping, config defaults, lot normalization, and MT5 order construction.

**Phase 2 — Direct-MT5 forex bot (this work)**
1. Config loader + JSON-schema validation for the `mt5_trade` section.
2. Market data feed abstraction: live (`copy_rates_from_pos`) + offline replay.
3. `MT5Strategy` interface + `SmaCrossStrategy` demo.
4. `MT5ForexBot` loop with one-position-per-symbol management (open / reverse / exit).
5. Lightweight sqlite persistence for orders and open positions.
6. `trade-mt5` CLI command.
7. End-to-end dry-run tests (offline replay + dry-run bridge).

**Phase 3 — Live hardening (future; requires Windows + MT5 terminal)**
1. Reconciliation: align cached positions/orders with broker reality on startup and on the fly.
2. Order modify/cancel and broker-side stop-loss / take-profit management.
3. Reconnect and error recovery (terminal restarts, dropped connections, retry policy).
4. RPC / Telegram notifications (reuse `RPCManager` via an adapter shim).
5. Backtesting / replay against downloaded historical MT5 data.

## Operational Constraints

- Start in `dry_run=true`; switch to live only after demo validation.
- Dry-run has no broker connection, so it validates lot sizes against the configured
  `min_lot`/`lot_step` in each symbol mapping — not the broker's live `symbol_info`. Keep those
  mapping values aligned with the broker, otherwise a volume that passes in dry-run can still be
  rejected live (and vice versa).
- The `MetaTrader5` package is Windows-only and requires a running/login-capable MT5 terminal,
  so the live path can only be exercised on Windows. Imports are kept lazy so the rest of
  Freqtrade stays importable elsewhere; the dry-run/replay path runs on any platform.
- Forex sizing is in lots, not crypto base units. Risk controls must explicitly convert strategy
  risk to broker lot sizes.
- Reconciliation (Phase 3) is mandatory before relying on live trading, because broker orders
  and positions may exist outside the bot process.
