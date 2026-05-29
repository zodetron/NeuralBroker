"""
engine/tf_4h.py — 4H zone detection (demand / supply / order-block / midzone).

Zone types:
  DEMAND   — prior swing low area (potential buy zone)
  SUPPLY   — prior swing high area (potential sell zone)
  OB_BULL  — bullish order block (down bar before strong move up)
  OB_BEAR  — bearish order block (up bar before strong move down)
  MIDZONE  — between two identified zones (no strong bias)

Outputs:
  zone        : current zone label
  nearest_support  : price of nearest demand/OB_BULL level below current price
  nearest_resist   : price of nearest supply/OB_BEAR level above current price
  ema20 / ema50    : 4H EMAs
  close / timestamp
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import ta
from engine.builder import build_candles
from config import OB_RALLY_PCT, ZONE_RANGE_PCT


def analyse_4h(n_bars: int = 300) -> dict:
    df = build_candles(240, n_bars=n_bars)
    if df.empty or len(df) < 50:
        return _empty()

    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    ema20 = float(ta.trend.ema_indicator(close, window=20).iloc[-1])
    ema50 = float(ta.trend.ema_indicator(close, window=50).iloc[-1]) if len(df) >= 50 else float("nan")

    last_close = float(close.iloc[-1])

    supports, resistances = _find_key_levels(df)

    nearest_support = _nearest_below(supports, last_close)
    nearest_resist  = _nearest_above(resistances, last_close)

    zone = _classify_zone(last_close, nearest_support, nearest_resist)

    return {
        "timestamp":        df["timestamp"].iloc[-1],
        "zone":             zone,
        "nearest_support":  nearest_support,
        "nearest_resist":   nearest_resist,
        "ema20":            ema20,
        "ema50":            ema50,
        "close":            last_close,
        "supports":         supports,
        "resistances":      resistances,
    }


# ── Level detection ────────────────────────────────────────────────────────────

def _find_key_levels(df: pd.DataFrame) -> tuple[list[float], list[float]]:
    """
    Identify demand (support) and supply (resistance) levels from:
      - Swing highs/lows (3-bar pivot)
      - Order blocks (last down/up bar before a strong directional move)
    """
    highs  = df["high"].values
    lows   = df["low"].values
    opens  = df["open"].values
    closes = df["close"].values
    n      = len(df)

    supports:    list[float] = []
    resistances: list[float] = []

    # Swing pivots (use last 100 bars to avoid stale levels)
    lookback = min(100, n - 2)
    for i in range(1, lookback):
        # Swing low → support
        if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
            supports.append(float(lows[i]))
        # Swing high → resistance
        if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            resistances.append(float(highs[i]))

    # Order blocks (last 60 bars)
    rally_threshold = OB_RALLY_PCT   # 1% move in next 3 bars qualifies
    for i in range(1, min(60, n - 3)):
        body_i   = abs(closes[i] - opens[i])
        range_3  = max(highs[i+1:i+4]) - min(lows[i+1:i+4]) if n > i + 3 else 0.0

        # Bullish OB: down candle just before a strong up move
        if closes[i] < opens[i]:
            fwd_move = (max(highs[i+1:i+4]) - closes[i]) / closes[i] if n > i + 3 else 0.0
            if fwd_move >= rally_threshold:
                supports.append(float(lows[i]))   # bottom of OB as support

        # Bearish OB: up candle just before a strong down move
        if closes[i] > opens[i]:
            fwd_move = (closes[i] - min(lows[i+1:i+4])) / closes[i] if n > i + 3 else 0.0
            if fwd_move >= rally_threshold:
                resistances.append(float(highs[i]))  # top of OB as resistance

    return _dedupe(supports), _dedupe(resistances)


def _dedupe(levels: list[float], pct: float = 0.005) -> list[float]:
    """Merge levels within pct% of each other into a single average."""
    if not levels:
        return []
    levels = sorted(levels)
    merged = [levels[0]]
    for lvl in levels[1:]:
        if (lvl - merged[-1]) / merged[-1] < pct:
            merged[-1] = (merged[-1] + lvl) / 2
        else:
            merged.append(lvl)
    return merged


def _nearest_below(levels: list[float], price: float) -> float:
    below = [l for l in levels if l < price]
    return max(below) if below else float("nan")


def _nearest_above(levels: list[float], price: float) -> float:
    above = [l for l in levels if l > price]
    return min(above) if above else float("nan")


def _classify_zone(
    price: float,
    support: float,
    resist: float,
    zone_pct: float = ZONE_RANGE_PCT,
) -> str:
    in_support = not pd.isna(support) and abs(price - support) / price <= zone_pct
    in_resist  = not pd.isna(resist)  and abs(price - resist)  / price <= zone_pct

    if in_support and in_resist:
        return "MIDZONE"
    if in_support:
        return "DEMAND"
    if in_resist:
        return "SUPPLY"
    return "MIDZONE"


def _empty() -> dict:
    return {
        "timestamp": None,
        "zone": "MIDZONE",
        "nearest_support": float("nan"),
        "nearest_resist":  float("nan"),
        "ema20": float("nan"),
        "ema50": float("nan"),
        "close": 0.0,
        "supports": [],
        "resistances": [],
    }


if __name__ == "__main__":
    result = analyse_4h()
    for k, v in result.items():
        if not isinstance(v, list):
            print(f"  {k:<22} {v}")
    print(f"  {'supports':<22} {result['supports'][-3:]} (last 3)")
    print(f"  {'resistances':<22} {result['resistances'][:3]} (first 3)")
