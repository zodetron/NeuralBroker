"""
engine/feed.py — Historical fill + live WebSocket feed for BTC/USDT 1M candles.

Boot sequence:
  1. Create candles_1m table if not present.
  2. Fetch all missing 1M history from Binance via ccxt (2019-01-01 → now).
  3. Open WebSocket stream; on each closed bar insert candle and fire callbacks.

Callbacks fire on every closed 1M bar:
  on_bar(ts: pd.Timestamp, bar: dict)
  ts  — UTC close timestamp of the just-closed bar
  bar — {"open", "high", "low", "close", "volume"}

Usage:
  from engine.feed import Feed
  feed = Feed(on_bar=my_callback)
  asyncio.run(feed.run())
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Callable, Optional

import duckdb
import pandas as pd
import websockets
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from config import BINANCE_WS_URL, DB_PATH, HIST_START, SYMBOL

log = logging.getLogger(__name__)

# ── DuckDB helpers ─────────────────────────────────────────────────────────────

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS candles_1m (
    timestamp TIMESTAMPTZ PRIMARY KEY,
    open      DOUBLE NOT NULL,
    high      DOUBLE NOT NULL,
    low       DOUBLE NOT NULL,
    close     DOUBLE NOT NULL,
    volume    DOUBLE NOT NULL
);
"""

_INSERT_SQL = """
INSERT OR IGNORE INTO candles_1m (timestamp, open, high, low, close, volume)
VALUES (?, ?, ?, ?, ?, ?)
"""


def _init_db() -> None:
    con = duckdb.connect(DB_PATH)
    try:
        con.execute(_CREATE_TABLE)
    finally:
        con.close()


def _get_latest_ts() -> Optional[datetime]:
    """Return max timestamp in candles_1m, or None if table is empty."""
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        row = con.execute("SELECT max(timestamp) FROM candles_1m").fetchone()
        val = row[0] if row else None
        if val is None:
            return None
        if isinstance(val, datetime):
            return val.replace(tzinfo=timezone.utc) if val.tzinfo is None else val
        return pd.Timestamp(val).to_pydatetime().replace(tzinfo=timezone.utc)
    finally:
        con.close()


def _bulk_insert(rows: list[tuple]) -> None:
    """Insert a batch of (timestamp, o, h, l, c, v) tuples."""
    if not rows:
        return
    con = duckdb.connect(DB_PATH)
    try:
        con.executemany(_INSERT_SQL, rows)
    finally:
        con.close()


def _insert_one(ts: datetime, o: float, h: float, l: float, c: float, v: float) -> None:
    con = duckdb.connect(DB_PATH)
    try:
        con.execute(_INSERT_SQL, [ts, o, h, l, c, v])
    finally:
        con.close()


# ── Historical fill ────────────────────────────────────────────────────────────

def fill_history(progress_every: int = 50_000) -> int:
    """
    Fetch all 1M candles from HIST_START to now that are not already stored.
    Returns the number of new rows inserted.
    """
    import ccxt  # local import — not needed unless filling

    exchange = ccxt.binance({"enableRateLimit": True})
    symbol   = SYMBOL.replace("USDT", "/USDT")   # BTCUSDT → BTC/USDT

    latest = _get_latest_ts()
    if latest is None:
        since_ms = int(pd.Timestamp(HIST_START, tz="UTC").timestamp() * 1000)
        log.info("Empty DB — fetching from %s", HIST_START)
    else:
        # Start 1 minute after the last stored bar to avoid duplicate
        since_ms = int((latest.timestamp() + 60) * 1000)
        log.info("Resuming fill from %s", latest.isoformat())

    now_ms    = int(time.time() * 1000)
    inserted  = 0
    batch_buf: list[tuple] = []

    while since_ms < now_ms:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, "1m", since=since_ms, limit=1000)
        except Exception as exc:
            log.warning("ccxt fetch error: %s — retrying in 5s", exc)
            time.sleep(5)
            continue

        if not ohlcv:
            break

        for bar in ohlcv:
            ts_ms, o, h, l, c, v = bar
            if ts_ms >= now_ms:
                # Don't store candles that are still forming
                break
            ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            batch_buf.append((ts, float(o), float(h), float(l), float(c), float(v)))

        if batch_buf:
            _bulk_insert(batch_buf)
            inserted += len(batch_buf)
            since_ms  = int(batch_buf[-1][0].timestamp() * 1000) + 60_000
            batch_buf.clear()

            if inserted % progress_every < 1000:
                log.info("  … %d rows inserted", inserted)
        else:
            break

        # Respect Binance rate limit (1200 req/min → 50ms floor)
        time.sleep(0.05)

    log.info("History fill complete — %d new rows inserted", inserted)
    return inserted


# ── WebSocket live feed ────────────────────────────────────────────────────────

class Feed:
    """
    Async WebSocket feed.  Call `asyncio.run(feed.run())` to start.
    `on_bar` fires on every closed 1M candle with the bar dict.
    """

    def __init__(
        self,
        on_bar: Optional[Callable] = None,
        fill_on_start: bool = True,
        reconnect_delay: float = 3.0,
    ):
        self.on_bar         = on_bar
        self.fill_on_start  = fill_on_start
        self.reconnect_delay = reconnect_delay
        self._running       = False

    async def run(self) -> None:
        _init_db()

        if self.fill_on_start:
            log.info("Starting historical fill…")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, fill_history)
            log.info("Historical fill done — starting live stream")

        self._running = True
        while self._running:
            try:
                await self._stream_loop()
            except (ConnectionClosedError, ConnectionClosedOK) as exc:
                log.warning("WebSocket closed (%s) — reconnecting in %.1fs", exc, self.reconnect_delay)
            except Exception as exc:
                log.error("Feed error: %s — reconnecting in %.1fs", exc, self.reconnect_delay)
            if self._running:
                await asyncio.sleep(self.reconnect_delay)

    async def _stream_loop(self) -> None:
        log.info("Connecting to %s", BINANCE_WS_URL)
        async with websockets.connect(
            BINANCE_WS_URL,
            ping_interval=20,
            ping_timeout=10,
        ) as ws:
            log.info("WebSocket connected")
            async for raw in ws:
                msg = json.loads(raw)
                k   = msg.get("k", {})
                if not k.get("x", False):
                    # Candle still forming — ignore
                    continue
                await self._handle_closed_bar(k)

    async def _handle_closed_bar(self, k: dict) -> None:
        """Process a confirmed-closed Binance kline payload."""
        ts = datetime.fromtimestamp(k["t"] / 1000, tz=timezone.utc)
        o, h, l, c, v = float(k["o"]), float(k["h"]), float(k["l"]), float(k["c"]), float(k["v"])

        _insert_one(ts, o, h, l, c, v)

        bar = {"open": o, "high": h, "low": l, "close": c, "volume": v}
        log.debug("Closed bar %s  C=%.2f  V=%.2f", ts.isoformat(), c, v)

        if self.on_bar is not None:
            try:
                result = self.on_bar(pd.Timestamp(ts), bar)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                log.error("on_bar callback raised: %s", exc)

    def stop(self) -> None:
        self._running = False


# ── CLI self-test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
    )

    def _on_bar(ts: pd.Timestamp, bar: dict) -> None:
        print(f"  BAR {ts}  C={bar['close']:.2f}  V={bar['volume']:.2f}")

    feed = Feed(on_bar=_on_bar, fill_on_start=True)
    print("Running feed (Ctrl-C to stop)…")
    try:
        asyncio.run(feed.run())
    except KeyboardInterrupt:
        print("Stopped.")
