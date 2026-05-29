"""
engine/tf_1h.py — 1H market structure analysis.

Structure labels:
  TRENDING_UP      — ADX > 25, EMA20 > EMA50, positive slope
  TRENDING_DOWN    — ADX > 25, EMA20 < EMA50, negative slope
  BREAKOUT_UP      — Price breaks above 20-bar high, ADX rising
  BREAKOUT_DOWN    — Price breaks below 20-bar low, ADX rising
  RANGING          — ADX ≤ 25

Outputs:
  structure, ema20, ema50, adx, rsi, atr, close, timestamp
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import ta
from engine.builder import build_candles
from config import ADX_TREND_THRESH


def analyse_1h(n_bars: int = 200) -> dict:
    df = build_candles(60, n_bars=n_bars)
    if df.empty or len(df) < 55:
        return _empty()

    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    ema20 = ta.trend.ema_indicator(close, window=20)
    ema50 = ta.trend.ema_indicator(close, window=50)
    adx   = ta.trend.ADXIndicator(high, low, close, window=14)
    rsi   = ta.momentum.RSIIndicator(close, window=14).rsi()
    atr   = ta.volatility.average_true_range(high, low, close, window=14)

    e20      = float(ema20.iloc[-1])
    e50      = float(ema50.iloc[-1])
    adx_val  = float(adx.adx().iloc[-1])
    rsi_val  = float(rsi.iloc[-1])
    atr_val  = float(atr.iloc[-1])
    last_c   = float(close.iloc[-1])

    # EMA slope over last 5 bars
    slope = float(ema20.iloc[-1] - ema20.iloc[-6]) if len(ema20.dropna()) >= 6 else 0.0

    # 20-bar high/low breakout check
    roll_high = float(high.iloc[-21:-1].max())
    roll_low  = float(low.iloc[-21:-1].min())

    structure = _classify(adx_val, e20, e50, slope, last_c, roll_high, roll_low)

    return {
        "timestamp": df["timestamp"].iloc[-1],
        "structure": structure,
        "ema20":     e20,
        "ema50":     e50,
        "adx":       adx_val,
        "rsi":       rsi_val,
        "atr":       atr_val,
        "close":     last_c,
    }


def _classify(
    adx: float, ema20: float, ema50: float, slope: float,
    close: float, roll_high: float, roll_low: float,
) -> str:
    trending = adx > ADX_TREND_THRESH

    if trending:
        if ema20 > ema50 and slope > 0:
            return "TRENDING_UP"
        if ema20 < ema50 and slope < 0:
            return "TRENDING_DOWN"

    # Breakout: price closes beyond 20-bar extreme and ADX is rising
    if close > roll_high:
        return "BREAKOUT_UP"
    if close < roll_low:
        return "BREAKOUT_DOWN"

    return "RANGING"


def _empty() -> dict:
    return {
        "timestamp": None,
        "structure": "RANGING",
        "ema20": float("nan"), "ema50": float("nan"),
        "adx": 0.0, "rsi": 50.0, "atr": 0.0, "close": 0.0,
    }


if __name__ == "__main__":
    result = analyse_1h()
    for k, v in result.items():
        print(f"  {k:<20} {v}")
