"""
engine/tf_15m.py — 15M momentum state.

States:
  STRONG_BULL    — ADX > 25, EMA20 rising, RSI > 55
  PULLBACK_BULL  — EMA20 rising, RSI 35–55 (pullback into trend)
  STRONG_BEAR    — ADX > 25, EMA20 falling, RSI < 45
  PULLBACK_BEAR  — EMA20 falling, RSI 45–65 (pullback into downtrend)
  EXHAUSTED      — RSI divergence or extreme reading (> 75 or < 25)
  NEUTRAL        — none of the above

Outputs:
  momentum, ema20, adx, rsi, atr, avg_atr, close, timestamp
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import ta
from engine.builder import build_candles
from config import ADX_TREND_THRESH


def analyse_15m(n_bars: int = 100) -> dict:
    df = build_candles(15, n_bars=n_bars)
    if df.empty or len(df) < 30:
        return _empty()

    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    ema20     = ta.trend.ema_indicator(close, window=20)
    adx_obj   = ta.trend.ADXIndicator(high, low, close, window=14)
    rsi       = ta.momentum.RSIIndicator(close, window=14).rsi()
    atr_s     = ta.volatility.average_true_range(high, low, close, window=14)

    e20     = float(ema20.iloc[-1])
    adx_val = float(adx_obj.adx().iloc[-1])
    rsi_val = float(rsi.iloc[-1])
    atr_val = float(atr_s.iloc[-1])
    avg_atr = float(atr_s.rolling(20).mean().iloc[-1])
    last_c  = float(close.iloc[-1])

    # Slope: 5-bar EMA direction
    slope = float(ema20.iloc[-1] - ema20.iloc[-6]) if len(ema20.dropna()) >= 6 else 0.0

    momentum = _classify(adx_val, slope, rsi_val)

    return {
        "timestamp": df["timestamp"].iloc[-1],
        "momentum":  momentum,
        "ema20":     e20,
        "adx":       adx_val,
        "rsi":       rsi_val,
        "atr":       atr_val,
        "avg_atr":   avg_atr,
        "close":     last_c,
    }


def _classify(adx: float, slope: float, rsi: float) -> str:
    if rsi > 75 or rsi < 25:
        return "EXHAUSTED"

    trending = adx > ADX_TREND_THRESH

    if slope > 0:
        if trending and rsi > 55:
            return "STRONG_BULL"
        if 35 <= rsi <= 55:
            return "PULLBACK_BULL"
    else:
        if trending and rsi < 45:
            return "STRONG_BEAR"
        if 45 <= rsi <= 65:
            return "PULLBACK_BEAR"

    return "NEUTRAL"


def _empty() -> dict:
    return {
        "timestamp": None,
        "momentum": "NEUTRAL",
        "ema20": float("nan"),
        "adx": 0.0, "rsi": 50.0, "atr": 0.0, "avg_atr": 0.0, "close": 0.0,
    }


if __name__ == "__main__":
    result = analyse_15m()
    for k, v in result.items():
        print(f"  {k:<20} {v}")
