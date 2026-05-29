"""
api/server.py — FastAPI REST + WebSocket server.

REST endpoints:
  GET  /status          — system health + account summary
  GET  /trades          — recent trade log
  GET  /candles/{tf}    — OHLCV bars for charting (tf = 1, 5, 15, 60, 240, 1440)
  GET  /stack           — latest TF analysis snapshot
  GET  /position        — open position (or null)

WebSocket:
  WS   /ws/feed         — real-time push on each closed 5M bar
      pushes JSON: { type: "bar"|"signal"|"trade", payload: ... }

All responses are JSON.  CORS is open to dashboard dev origin (5173).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import json
import logging
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from config import API_PORT, CORS_ORIGINS

log = logging.getLogger(__name__)

app = FastAPI(title="MTF Live", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins     = CORS_ORIGINS,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Global shared state (injected by main.py) ──────────────────────────────────
# These are populated at startup by main.py before uvicorn starts.
_state: dict[str, Any] = {
    "tf_stack":       {},          # latest TF snapshots
    "last_signal":    None,        # most recent signal dict
    "trader":         None,        # PaperTrader instance
    "feed_running":   False,
    "candle_count":   0,
}

_ws_clients: list[WebSocket] = []


def set_state(key: str, value: Any) -> None:
    """Called by main.py to inject live objects."""
    _state[key] = value


# ── REST endpoints ─────────────────────────────────────────────────────────────

@app.get("/status")
async def status():
    trader = _state.get("trader")
    account = {}
    if trader is not None:
        account = trader.stats()

    return {
        "feed_running":  _state["feed_running"],
        "candle_count":  _state["candle_count"],
        "account":       account,
        "ws_clients":    len(_ws_clients),
    }


@app.get("/trades")
async def trades(n: int = 50):
    trader = _state.get("trader")
    if trader is None:
        return []
    return trader.recent_trades(n)


@app.get("/candles/{tf}")
async def candles(tf: int, n: int = 200):
    """
    tf : timeframe in minutes (1, 5, 15, 60, 240, 1440)
    n  : number of bars to return
    """
    valid_tfs = {1, 5, 15, 60, 240, 1440}
    if tf not in valid_tfs:
        return {"error": f"tf must be one of {valid_tfs}"}

    from engine.builder import build_candles
    df = build_candles(tf, n_bars=n)
    if df.empty:
        return []

    return df.assign(
        timestamp=df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    ).to_dict(orient="records")


@app.get("/stack")
async def stack():
    return _state.get("tf_stack", {})


@app.get("/position")
async def position():
    trader = _state.get("trader")
    if trader is None:
        return None
    return trader.open_position


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws/feed")
async def ws_feed(websocket: WebSocket):
    await websocket.accept()
    _ws_clients.append(websocket)
    log.info("WS client connected (total %d)", len(_ws_clients))
    try:
        while True:
            # Keep connection alive; messages are pushed by broadcast()
            await asyncio.sleep(30)
            await websocket.send_text('{"type":"ping"}')
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.remove(websocket)
        log.info("WS client disconnected (total %d)", len(_ws_clients))


async def broadcast(msg_type: str, payload: Any) -> None:
    """Push a message to all connected WS clients."""
    if not _ws_clients:
        return
    data = json.dumps({"type": msg_type, "payload": payload}, default=str)
    dead: list[WebSocket] = []
    for ws in _ws_clients:
        try:
            await ws.send_text(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        try:
            _ws_clients.remove(ws)
        except ValueError:
            pass


# ── Server entry (used only when run directly, not via main.py) ───────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=API_PORT, log_level="info")
