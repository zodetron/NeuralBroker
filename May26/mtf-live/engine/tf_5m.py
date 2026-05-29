"""
engine/tf_5m.py — 5M trigger detection (entry confirmation signals).

Triggers:
  ENGULFING      — current bar body engulfs prior bar body, aligned with 15M momentum
  BREAKOUT       — close breaks above/below 10-bar high/low with vol > 1.3× avg
  RSI_CROSS      — RSI crosses 50 from the pullback side (30→50 bull / 70→50 bear)
  NONE           — no trigger this bar

Outputs:
  trigger, direction (+1/-1/0), ema20, rsi, atr, avg_vol,
  last_close, timestamp
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import ta
from engine.builder import build_candles
from config import BREAKOUT_VOL_5M, ENGULF_BODY_MULT


def analyse_5m(n_bars: int = 60) -> dict:
    df = build_candles(5, n_bars=n_bars)
    if df.empty or len(df) < 20:
        return _empty()

    close  = df["close"]
    open_  = df["open"]
    high   = df["high"]
    low    = df["low"]
    vol    = df["volume"]

    ema20   = float(ta.trend.ema_indicator(close, window=20).iloc[-1])
    rsi_s   = ta.momentum.RSIIndicator(close, window=14).rsi()
    atr_val = float(ta.volatility.average_true_range(high, low, close, window=14).iloc[-1])
    avg_vol = float(vol.rolling(20).mean().iloc[-1])

    rsi_val  = float(rsi_s.iloc[-1])
    rsi_prev = float(rsi_s.iloc[-2])

    last_c = float(close.iloc[-1])
    last_o = float(open_.iloc[-1])
    prev_c = float(close.iloc[-2])
    prev_o = float(open_.iloc[-2])

    trigger, direction = _detect(
        last_c, last_o, prev_c, prev_o,
        high, low, close, vol, avg_vol,
        rsi_val, rsi_prev,
    )

    return {
        "timestamp":  df["timestamp"].iloc[-1],
        "trigger":    trigger,
        "direction":  direction,
        "ema20":      ema20,
        "rsi":        rsi_val,
        "atr":        atr_val,
        "avg_vol":    avg_vol,
        "last_close": last_c,
    }


def _detect(
    last_c, last_o, prev_c, prev_o,
    high, low, close, vol, avg_vol,
    rsi, rsi_prev,
):
    # ── Engulfing ──────────────────────────────────────────────────────────────
    last_body = abs(last_c - last_o)
    prev_body = abs(prev_c - prev_o)
    avg_body  = float(abs(close - close.shift(1)).rolling(10).mean().iloc[-1]) or 1.0

    bull_engulf = (
        last_c > last_o                      # current bullish
        and prev_c < prev_o                  # prior bearish
        and last_c >= prev_o                 # engulfs
        and last_o <= prev_c
        and last_body >= prev_body * ENGULF_BODY_MULT
    )
    bear_engulf = (
        last_c < last_o
        and prev_c > prev_o
        and last_c <= prev_o
        and last_o >= prev_c
        and last_body >= prev_body * ENGULF_BODY_MULT
    )

    if bull_engulf:
        return "ENGULFING", 1
    if bear_engulf:
        return "ENGULFING", -1

    # ── Breakout ───────────────────────────────────────────────────────────────
    roll_high = float(high.iloc[-11:-1].max())
    roll_low  = float(low.iloc[-11:-1].min())
    last_vol  = float(vol.iloc[-1])
    vol_surge = last_vol > avg_vol * BREAKOUT_VOL_5M

    if last_c > roll_high and vol_surge:
        return "BREAKOUT", 1
    if last_c < roll_low and vol_surge:
        return "BREAKOUT", -1

    # ── RSI Cross 50 ──────────────────────────────────────────────────────────
    if rsi_prev < 50 <= rsi:
        return "RSI_CROSS", 1
    if rsi_prev > 50 >= rsi:
        return "RSI_CROSS", -1

    return "NONE", 0


def _empty() -> dict:
    return {
        "timestamp": None,
        "trigger": "NONE", "direction": 0,
        "ema20": float("nan"),
        "rsi": 50.0, "atr": 0.0, "avg_vol": 0.0, "last_close": 0.0,
    }


if __name__ == "__main__":
    result = analyse_5m()
    for k, v in result.items():
        print(f"  {k:<20} {v}")
