"""
engine/tf_1d.py — Daily-timeframe analysis.

Outputs:
  bias     : +1 (bullish) | -1 (bearish) | 0 (neutral)
  atr      : 14-bar ATR value of current daily candle
  sma20/50/200 : daily SMAs
  daily_range_pct : (high-low)/close of most recent bar (used by signal gate)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import ta
from engine.builder import build_candles


def analyse_1d(n_bars: int = 250) -> dict:
    """
    Build n_bars of daily candles and return a snapshot dict.

    Returns
    -------
    dict with keys:
      bias, atr, sma20, sma50, sma200,
      daily_range_pct, close, timestamp
    """
    df = build_candles(1440, n_bars=n_bars)
    if df.empty or len(df) < 20:
        return _empty()

    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    # ATR (14)
    atr_series = ta.volatility.average_true_range(high, low, close, window=14)
    atr = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0

    # SMAs
    sma20  = float(close.rolling(20).mean().iloc[-1])
    sma50  = float(close.rolling(50).mean().iloc[-1]) if len(df) >= 50  else float("nan")
    sma200 = float(close.rolling(200).mean().iloc[-1]) if len(df) >= 200 else float("nan")

    last_close = float(close.iloc[-1])

    # Bias: price above/below sma50 or sma200
    if not (pd.isna(sma50) or pd.isna(sma200)):
        if last_close > sma50 and last_close > sma200:
            bias = 1
        elif last_close < sma50 and last_close < sma200:
            bias = -1
        else:
            bias = 0
    elif not pd.isna(sma50):
        bias = 1 if last_close > sma50 else -1
    else:
        bias = 1 if last_close > sma20 else -1

    last_bar = df.iloc[-1]
    daily_range_pct = float((last_bar["high"] - last_bar["low"]) / last_close)

    return {
        "timestamp":        df["timestamp"].iloc[-1],
        "bias":             bias,
        "atr":              atr,
        "sma20":            sma20,
        "sma50":            sma50,
        "sma200":           sma200,
        "close":            last_close,
        "daily_range_pct":  daily_range_pct,
    }


def _empty() -> dict:
    return {
        "timestamp": None,
        "bias": 0, "atr": 0.0,
        "sma20": float("nan"), "sma50": float("nan"), "sma200": float("nan"),
        "close": 0.0, "daily_range_pct": 0.0,
    }


if __name__ == "__main__":
    result = analyse_1d()
    for k, v in result.items():
        print(f"  {k:<20} {v}")
