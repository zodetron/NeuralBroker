"""
signals/paper_trader.py — Paper trading engine with DuckDB persistence.

Tables (in paper_trades.duckdb):
  account  — single-row equity ledger
  trades   — full trade log with entry/exit/result

Position management:
  - Max 1 open trade at a time.
  - Exits: TP hit | SL hit | TIME_EXIT_BARS × 5M bars elapsed.
  - Commission + slippage applied at entry and exit.
  - Daily loss limit halts new entries for the rest of the UTC day.

All prices come from the 5M candle stream (build_candles(5, n_bars=2)).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import duckdb
import pandas as pd

from config import (
    STARTING_BALANCE,
    RISK_PCT_STANDARD, RISK_PCT_CONFLUENCE,
    MAX_TRADES_PER_DAY, DAILY_LOSS_LIMIT_PCT,
    COMMISSION_RT, TIME_EXIT_BARS,
    TRADES_DB_PATH,
)

log = logging.getLogger(__name__)

_INIT_SQL = """
CREATE TABLE IF NOT EXISTS account (
    id              INTEGER PRIMARY KEY,
    balance         DOUBLE NOT NULL,
    peak_balance    DOUBLE NOT NULL,
    trades_today    INTEGER NOT NULL DEFAULT 0,
    day_open_bal    DOUBLE NOT NULL,
    last_reset_day  DATE
);

CREATE TABLE IF NOT EXISTS trades (
    id            INTEGER PRIMARY KEY,
    signal_type   VARCHAR,
    direction     INTEGER,
    entry_price   DOUBLE,
    stop_price    DOUBLE,
    tp_price      DOUBLE,
    entry_ts      TIMESTAMPTZ,
    exit_ts       TIMESTAMPTZ,
    exit_price    DOUBLE,
    units         DOUBLE,
    gross_pnl     DOUBLE,
    costs         DOUBLE,
    net_pnl       DOUBLE,
    exit_reason   VARCHAR,      -- TP | SL | TIME
    balance_after DOUBLE,
    rr            DOUBLE,
    confluence    BOOLEAN
);
"""

_INSERT_TRADE = """
INSERT INTO trades (
    signal_type, direction, entry_price, stop_price, tp_price,
    entry_ts, exit_ts, exit_price, units, gross_pnl, costs, net_pnl,
    exit_reason, balance_after, rr, confluence
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_COMMISSION = COMMISSION_RT / 2  # half round-trip at entry, half at exit
_SLIPPAGE   = 0.0002             # 0.02% adverse fill


@dataclass
class _Position:
    signal_type: str
    direction:   int
    entry_price: float
    stop_price:  float
    tp_price:    float
    entry_ts:    pd.Timestamp
    units:       float
    bars_held:   int
    rr:          float
    confluence:  bool


class PaperTrader:
    """
    Holds state in memory and flushes to DuckDB on every update.
    Call update(signal, current_price, current_ts) on each 5M bar.
    """

    def __init__(self) -> None:
        os.makedirs(os.path.dirname(TRADES_DB_PATH), exist_ok=True)
        self._init_db()
        self._balance, self._peak, self._trades_today, self._day_open_bal, self._last_reset = (
            self._load_account()
        )
        self._position: Optional[_Position] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(
        self,
        current_price: float,
        current_ts: pd.Timestamp,
        signal: Optional[dict] = None,
    ) -> Optional[dict]:
        """
        Must be called on every closed 5M bar.
        Returns a trade result dict if a position was just closed, else None.
        """
        self._maybe_reset_day(current_ts)

        # ── Manage open position ───────────────────────────────────────────────
        if self._position is not None:
            result = self._check_exit(current_price, current_ts)
            if result is not None:
                return result

        # ── Maybe open new position ────────────────────────────────────────────
        if signal is not None and signal.get("approved") and self._position is None:
            self._try_enter(signal, current_price, current_ts)

        return None

    @property
    def balance(self) -> float:
        return self._balance

    @property
    def open_position(self) -> Optional[dict]:
        if self._position is None:
            return None
        p = self._position
        return {
            "signal_type": p.signal_type,
            "direction":   p.direction,
            "entry_price": p.entry_price,
            "stop_price":  p.stop_price,
            "tp_price":    p.tp_price,
            "entry_ts":    p.entry_ts.isoformat(),
            "bars_held":   p.bars_held,
            "units":       p.units,
        }

    def stats(self) -> dict:
        """Summary statistics from the full trade log."""
        con = duckdb.connect(TRADES_DB_PATH, read_only=True)
        try:
            row = con.execute("""
                SELECT
                    count(*)                            AS total_trades,
                    sum(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) AS wins,
                    sum(net_pnl)                        AS total_net_pnl,
                    min(balance_after)                  AS min_balance,
                    max(balance_after)                  AS max_balance
                FROM trades
            """).fetchone()
        finally:
            con.close()

        if row is None or row[0] == 0:
            return {"total_trades": 0, "win_rate": 0.0, "total_net_pnl": 0.0}

        total, wins, net_pnl, min_bal, max_bal = row
        return {
            "total_trades":  total,
            "wins":          wins,
            "losses":        total - wins,
            "win_rate":      round(wins / total, 4) if total else 0.0,
            "total_net_pnl": round(net_pnl, 2),
            "min_balance":   round(min_bal, 2),
            "max_balance":   round(max_bal, 2),
            "current_balance": round(self._balance, 2),
        }

    def recent_trades(self, n: int = 20) -> list[dict]:
        con = duckdb.connect(TRADES_DB_PATH, read_only=True)
        try:
            rows = con.execute(f"""
                SELECT signal_type, direction, entry_price, exit_price,
                       net_pnl, exit_reason, entry_ts, exit_ts, rr, confluence
                FROM trades
                ORDER BY id DESC
                LIMIT {n}
            """).fetchall()
        finally:
            con.close()

        keys = ["signal_type", "direction", "entry_price", "exit_price",
                "net_pnl", "exit_reason", "entry_ts", "exit_ts", "rr", "confluence"]
        return [dict(zip(keys, r)) for r in rows]

    # ── Entry logic ────────────────────────────────────────────────────────────

    def _try_enter(self, signal: dict, price: float, ts: pd.Timestamp) -> None:
        if self._trades_today >= MAX_TRADES_PER_DAY:
            log.debug("Max daily trades reached")
            return

        # Daily loss limit
        loss_pct = (self._balance - self._day_open_bal) / self._day_open_bal
        if loss_pct <= -(DAILY_LOSS_LIMIT_PCT / 100):
            log.info("Daily loss limit hit — no new entries today")
            return

        direction = signal["direction"]
        entry     = signal["entry"]
        stop      = signal["stop"]
        tp        = signal["tp"]
        conf      = signal.get("confluence", False)

        risk_pct = RISK_PCT_CONFLUENCE if conf else RISK_PCT_STANDARD
        risk_amt = self._balance * risk_pct / 100

        sl_dist = abs(entry - stop)
        if sl_dist < 1e-8:
            return

        units = risk_amt / sl_dist

        # Apply entry cost
        entry_cost = entry * units * (_COMMISSION + _SLIPPAGE)
        self._balance -= entry_cost

        self._position = _Position(
            signal_type = signal["type"],
            direction   = direction,
            entry_price = entry,
            stop_price  = stop,
            tp_price    = tp,
            entry_ts    = ts,
            units       = units,
            bars_held   = 0,
            rr          = signal.get("rr", 0.0),
            confluence  = conf,
        )
        self._trades_today += 1
        self._flush_account()
        log.info(
            "ENTER %s  %s  entry=%.2f  sl=%.2f  tp=%.2f  units=%.6f",
            signal["type"], "LONG" if direction == 1 else "SHORT",
            entry, stop, tp, units,
        )

    # ── Exit logic ─────────────────────────────────────────────────────────────

    def _check_exit(self, price: float, ts: pd.Timestamp) -> Optional[dict]:
        p = self._position
        assert p is not None

        p.bars_held += 1
        exit_reason: Optional[str] = None

        if p.direction == 1:
            if price >= p.tp_price:
                exit_reason = "TP"
                exit_price  = p.tp_price
            elif price <= p.stop_price:
                exit_reason = "SL"
                exit_price  = p.stop_price
        else:
            if price <= p.tp_price:
                exit_reason = "TP"
                exit_price  = p.tp_price
            elif price >= p.stop_price:
                exit_reason = "SL"
                exit_price  = p.stop_price
            else:
                exit_price = price

        if exit_reason is None:
            if p.bars_held >= TIME_EXIT_BARS:
                exit_reason = "TIME"
                exit_price  = price
        else:
            pass  # exit_price already set above

        if exit_reason is None:
            if p.direction == 1:
                exit_price = price
            return None

        return self._close(p, exit_price, exit_reason, ts)

    def _close(
        self,
        p: _Position,
        exit_price: float,
        reason: str,
        ts: pd.Timestamp,
    ) -> dict:
        pnl_per_unit = (exit_price - p.entry_price) * p.direction
        gross_pnl    = pnl_per_unit * p.units
        exit_cost    = exit_price * p.units * (_COMMISSION + _SLIPPAGE)
        costs        = exit_cost  # entry cost already deducted at open
        net_pnl      = gross_pnl - costs
        self._balance += gross_pnl - costs
        if self._balance > self._peak:
            self._peak = self._balance

        self._position = None
        self._flush_account()
        self._write_trade(p, exit_price, ts, gross_pnl, costs, net_pnl, reason)

        log.info(
            "EXIT  %s  %s  @%.2f  reason=%s  net_pnl=%+.2f  bal=%.2f",
            p.signal_type, "LONG" if p.direction == 1 else "SHORT",
            exit_price, reason, net_pnl, self._balance,
        )

        return {
            "signal_type": p.signal_type,
            "direction":   p.direction,
            "entry_price": p.entry_price,
            "exit_price":  exit_price,
            "net_pnl":     round(net_pnl, 2),
            "exit_reason": reason,
            "bars_held":   p.bars_held,
            "balance":     round(self._balance, 2),
        }

    # ── DB helpers ─────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        con = duckdb.connect(TRADES_DB_PATH)
        try:
            con.execute(_INIT_SQL)
        finally:
            con.close()

    def _load_account(self) -> tuple:
        con = duckdb.connect(TRADES_DB_PATH, read_only=True)
        try:
            row = con.execute("SELECT balance, peak_balance, trades_today, day_open_bal, last_reset_day FROM account WHERE id=1").fetchone()
        finally:
            con.close()
        if row:
            bal, peak, today, day_open, last_day = row
            return bal, peak, today, day_open, last_day
        # First run — seed account
        con = duckdb.connect(TRADES_DB_PATH)
        try:
            con.execute(
                "INSERT INTO account VALUES (1, ?, ?, 0, ?, ?)",
                [STARTING_BALANCE, STARTING_BALANCE, STARTING_BALANCE, None],
            )
        finally:
            con.close()
        return STARTING_BALANCE, STARTING_BALANCE, 0, STARTING_BALANCE, None

    def _flush_account(self) -> None:
        con = duckdb.connect(TRADES_DB_PATH)
        try:
            con.execute(
                "UPDATE account SET balance=?, peak_balance=?, trades_today=?, day_open_bal=? WHERE id=1",
                [self._balance, self._peak, self._trades_today, self._day_open_bal],
            )
        finally:
            con.close()

    def _write_trade(
        self, p: _Position, exit_price: float, exit_ts: pd.Timestamp,
        gross_pnl: float, costs: float, net_pnl: float, reason: str,
    ) -> None:
        con = duckdb.connect(TRADES_DB_PATH)
        try:
            con.execute(_INSERT_TRADE, [
                p.signal_type, p.direction, p.entry_price, p.stop_price, p.tp_price,
                p.entry_ts.to_pydatetime(), exit_ts.to_pydatetime(), exit_price,
                p.units, gross_pnl, costs, net_pnl,
                reason, self._balance, p.rr, p.confluence,
            ])
        finally:
            con.close()

    def _maybe_reset_day(self, ts: pd.Timestamp) -> None:
        today = ts.date()
        if self._last_reset != today:
            self._trades_today = 0
            self._day_open_bal = self._balance
            self._last_reset   = today
            con = duckdb.connect(TRADES_DB_PATH)
            try:
                con.execute(
                    "UPDATE account SET trades_today=0, day_open_bal=?, last_reset_day=? WHERE id=1",
                    [self._balance, today],
                )
            finally:
                con.close()
