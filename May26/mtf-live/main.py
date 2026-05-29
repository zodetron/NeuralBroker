"""
main.py — MTF Live Trading System entry point.

Startup sequence:
  1. Initialise DuckDB and run historical fill (Binance 2019-01-01 → now).
  2. Start FastAPI server in a background thread.
  3. Open live WebSocket feed; on each closed 1M bar:
       a. Run all TF analysis modules.
       b. On 5M bar boundary: run signal generator + ML filter.
       c. Update paper trader.
       d. Push state to WS clients.

Usage:
  python main.py                  — normal start (fills history, then goes live)
  python main.py --no-fill        — skip historical fill (DB already populated)
  python main.py --dry-run        — analysis only, no paper trades
"""

import argparse
import asyncio
import logging
import threading
from datetime import datetime, timezone

import pandas as pd
import uvicorn

from config import API_PORT
from engine.builder import build_candles
from engine.feed import Feed
from engine.tf_1d    import analyse_1d
from engine.tf_4h    import analyse_4h
from engine.tf_1h    import analyse_1h
from engine.tf_15m   import analyse_15m
from engine.tf_5m    import analyse_5m
from engine.tf_swappy import update_swappy

from signals.generator   import SignalGenerator
from signals.ml_filter   import MLFilter
from signals.paper_trader import PaperTrader

from api.server import app as fastapi_app, set_state, broadcast

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
)
log = logging.getLogger("main")


# ── Analysis + signal pipeline ─────────────────────────────────────────────────

class Pipeline:
    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run   = dry_run
        self.generator = SignalGenerator()
        self.ml_filter = MLFilter()
        self.trader    = PaperTrader() if not dry_run else None
        self._bar_count = 0

        if self.trader is not None:
            set_state("trader", self.trader)

    def on_bar(self, ts: pd.Timestamp, bar: dict) -> None:
        """Called on every closed 1M bar by Feed."""
        self._bar_count += 1
        set_state("candle_count", self._bar_count)

        # Only run full analysis every 5M bar
        is_5m_bar = (ts.minute % 5 == 4)   # closed bar whose minute ends at X4
        if not is_5m_bar:
            return

        # ── Build TF snapshots ─────────────────────────────────────────────────
        snap_1d  = analyse_1d()
        snap_4h  = analyse_4h()
        snap_1h  = analyse_1h()
        snap_15m = analyse_15m()
        snap_5m  = analyse_5m()

        tf_stack = {
            "1d":  snap_1d,
            "4h":  snap_4h,
            "1h":  snap_1h,
            "15m": snap_15m,
            "5m":  snap_5m,
        }
        set_state("tf_stack", tf_stack)

        # ── Swappy ICT ─────────────────────────────────────────────────────────
        df15         = build_candles(15, n_bars=60)
        nearest_4h   = snap_4h.get("nearest_support", float("nan"))
        swappy_sig   = update_swappy(df15, nearest_4h)

        # ── Signal generator ───────────────────────────────────────────────────
        signal = self.generator.evaluate(
            snap_1d, snap_4h, snap_1h, snap_15m, snap_5m, swappy_sig
        )

        if signal is not None:
            signal = self.ml_filter.score(signal)
            set_state("last_signal", signal)
            if signal["approved"]:
                log.info("SIGNAL  %s  dir=%+d  entry=%.2f  rr=%.2f  score=%.3f",
                         signal["type"], signal["direction"],
                         signal["entry"], signal["rr"], signal["raw_score"])
            asyncio.get_event_loop().call_soon_threadsafe(
                lambda: asyncio.ensure_future(broadcast("signal", signal))
            )

        # ── Paper trader ───────────────────────────────────────────────────────
        current_price = float(bar["close"])
        if self.trader is not None:
            approved_sig = signal if (signal and signal.get("approved")) else None
            result = self.trader.update(current_price, ts, signal=approved_sig)
            if result is not None:
                asyncio.get_event_loop().call_soon_threadsafe(
                    lambda r=result: asyncio.ensure_future(broadcast("trade", r))
                )

        # ── Push bar to WS ─────────────────────────────────────────────────────
        asyncio.get_event_loop().call_soon_threadsafe(
            lambda b=bar, t=ts: asyncio.ensure_future(
                broadcast("bar", {"timestamp": t.isoformat(), **b})
            )
        )


# ── FastAPI thread ─────────────────────────────────────────────────────────────

def _start_api() -> None:
    uvicorn.run(fastapi_app, host="0.0.0.0", port=API_PORT, log_level="warning")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MTF Live Trading System")
    parser.add_argument("--no-fill",  action="store_true", help="Skip historical fill")
    parser.add_argument("--dry-run",  action="store_true", help="Analysis only — no paper trades")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("MTF Live  |  BTC/USDT  |  5M signal TF")
    log.info("=" * 60)

    pipeline = Pipeline(dry_run=args.dry_run)
    set_state("feed_running", True)

    # Start API in a daemon thread so it runs alongside the async feed
    api_thread = threading.Thread(target=_start_api, daemon=True)
    api_thread.start()
    log.info("API server started on port %d", API_PORT)

    feed = Feed(
        on_bar        = pipeline.on_bar,
        fill_on_start = not args.no_fill,
    )

    try:
        asyncio.run(feed.run())
    except KeyboardInterrupt:
        log.info("Shutting down…")
    finally:
        set_state("feed_running", False)


if __name__ == "__main__":
    main()
