"""
strategy/sizing.py — Volatility-adjusted position sizing and trade levels.

Sizing formula (all params from config.STRAT):
  risk_usd  = capital × risk_per_trade          e.g. 0.02 × $10,000 = $200
  stop_dist = atr_stop_mult × ATR_14            e.g. 1.5 × ATR  (in price $)
  units     = risk_usd / stop_dist              units of the asset to buy/sell
  value     = units × price                     $ market value of position
  value     = min(value, capital × max_pos_pct) hard cap at 25% of capital

This ensures every trade risks exactly risk_per_trade% of current capital,
regardless of how volatile the market is at entry time.

Stop / Take-profit levels (in price terms):
  Long  entry at E:
    SL = E − atr_stop_mult × ATR
    TP = E + atr_tp_mult   × ATR

  Short entry at E:
    SL = E + atr_stop_mult × ATR
    TP = E − atr_tp_mult   × ATR
"""

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import STRAT, BT


# ── SINGLE-TRADE SIZING ───────────────────────────────────────────────────────

def size_trade(
    capital: float,
    entry_price: float,
    atr: float,
    direction: int,          # +1 long, -1 short
    asset_key: str = "BTC",  # used to look up per-asset stop/TP overrides
) -> Dict:
    """
    Compute position size, dollar value, stop-loss, and take-profit for one trade.

    Parameters
    ----------
    capital     : current account equity
    entry_price : execution price (after slippage, applied by backtest engine)
    atr         : ATR_14 at signal bar (used for stop and TP distance)
    direction   : +1 for long, -1 for short

    Returns
    -------
    dict with keys:
      units      – number of asset units
      value      – $ value of position
      stop_price – stop-loss price
      tp_price   – take-profit price
      stop_dist  – $ distance to stop from entry
      tp_dist    – $ distance to TP from entry
      risk_usd   – dollars at risk (≈ risk_per_trade × capital)
    """
    key_lo    = asset_key.lower()
    stop_mult = STRAT.get(f"atr_stop_mult_{key_lo}", STRAT["atr_stop_mult"])
    tp_mult   = STRAT.get(f"atr_tp_mult_{key_lo}",   STRAT["atr_tp_mult"])

    risk_usd  = capital * STRAT["risk_per_trade"]
    stop_dist = stop_mult * atr
    tp_dist   = tp_mult   * atr

    # Avoid division by zero or degenerate ATR
    if stop_dist < 1e-9 or atr < 1e-9:
        return _zero_trade()

    units = risk_usd / stop_dist

    # Cap position value at max_position_pct of capital
    max_value = capital * STRAT["max_position_pct"]
    value     = units * entry_price
    if value > max_value:
        value = max_value
        units = value / entry_price

    # Require at least a minimum viable position
    if units <= 0 or value <= 0:
        return _zero_trade()

    # Stop and TP in absolute price terms
    if direction == 1:          # long
        stop_price = entry_price - stop_dist
        tp_price   = entry_price + tp_dist
    else:                       # short
        stop_price = entry_price + stop_dist
        tp_price   = entry_price - tp_dist

    return {
        "units":      units,
        "value":      value,
        "stop_price": stop_price,
        "tp_price":   tp_price,
        "stop_dist":  stop_dist,
        "tp_dist":    tp_dist,
        "risk_usd":   units * stop_dist,   # actual $ at risk (may differ from target if capped)
    }


def _zero_trade() -> Dict:
    """Return a null trade dict when sizing is not possible."""
    return {
        "units": 0.0, "value": 0.0,
        "stop_price": np.nan, "tp_price": np.nan,
        "stop_dist": np.nan,  "tp_dist": np.nan,
        "risk_usd": 0.0,
    }


# ── EXECUTION PRICE (slippage) ────────────────────────────────────────────────

def apply_slippage(price: float, direction: int, slippage: float = BT["slippage"]) -> float:
    """
    Simulate adverse slippage on fill.
    Long  entries: price * (1 + slippage)  — buy slightly higher
    Short entries: price * (1 - slippage)  — sell slightly lower
    """
    return price * (1 + direction * slippage)


# ── COMMISSION ────────────────────────────────────────────────────────────────

def commission_cost(value: float, rate: float = BT["commission"]) -> float:
    """Dollar commission for a one-sided fill at given position value."""
    return value * rate


# ── VECTORISED SIZING (for diagnostics / phase 5 output) ─────────────────────

def compute_sizing_series(df: pd.DataFrame, capital: float) -> pd.DataFrame:
    """
    Compute position sizes, SL and TP levels for every signal bar.
    Used for validation in Phase 5 — the backtest engine calls size_trade()
    bar-by-bar with the live equity curve instead.

    Returns df with added columns:
      pos_units, pos_value, sl_price, tp_price, rr_ratio
    """
    atr_col = f"atr_{STRAT['atr_window']}"
    out     = df.copy()

    has_signal = df["signal"] != 0

    # Vectorised approximation (assumes fixed capital for diagnostics)
    risk_usd  = capital * STRAT["risk_per_trade"]
    stop_dist = STRAT["atr_stop_mult"] * df[atr_col]
    tp_dist   = STRAT["atr_tp_mult"]   * df[atr_col]

    units = (risk_usd / stop_dist).clip(upper=capital * STRAT["max_position_pct"] / df["close"])
    value = (units * df["close"]).clip(upper=capital * STRAT["max_position_pct"])

    d = df["signal"]   # +1 or -1

    out["pos_units"]  = np.where(has_signal, units,  np.nan)
    out["pos_value"]  = np.where(has_signal, value,  np.nan)
    out["sl_price"]   = np.where(has_signal, df["close"] - d * stop_dist, np.nan)
    out["tp_price"]   = np.where(has_signal, df["close"] + d * tp_dist,   np.nan)
    out["rr_ratio"]   = (tp_dist / stop_dist).where(has_signal)

    return out


# ── SIZING SUMMARY ────────────────────────────────────────────────────────────

def print_sizing_summary(df: pd.DataFrame, asset_key: str, capital: float) -> None:
    """Print a summary of position sizing parameters for signal bars."""
    atr_col  = f"atr_{STRAT['atr_window']}"
    sig_bars = df[df["signal"] != 0].copy()

    if sig_bars.empty:
        print(f"  {asset_key}: no signal bars to size.")
        return

    stop_d = STRAT["atr_stop_mult"] * sig_bars[atr_col]
    tp_d   = STRAT["atr_tp_mult"]   * sig_bars[atr_col]
    units  = (capital * STRAT["risk_per_trade"] / stop_d).clip(
                upper=capital * STRAT["max_position_pct"] / sig_bars["close"]
             )
    value  = (units * sig_bars["close"]).clip(
                upper=capital * STRAT["max_position_pct"]
             )

    print(f"\n  {'─'*58}")
    print(f"  {asset_key} Position Sizing  (capital = ${capital:,.0f})")
    print(f"  {'─'*58}")
    print(f"  Risk per trade    : {STRAT['risk_per_trade']*100:.1f}% "
          f"= ${capital * STRAT['risk_per_trade']:,.0f}")
    print(f"  SL multiplier     : {STRAT['atr_stop_mult']}× ATR{STRAT['atr_window']}")
    print(f"  TP multiplier     : {STRAT['atr_tp_mult']}× ATR{STRAT['atr_window']}")
    print(f"  R:R ratio         : {STRAT['atr_tp_mult']/STRAT['atr_stop_mult']:.1f}:1")
    print(f"  Max pos size      : {STRAT['max_position_pct']*100:.0f}% of capital "
          f"= ${capital * STRAT['max_position_pct']:,.0f}")
    print(f"\n  Signal-bar stats  (N = {len(sig_bars):,} bars):")
    print(f"  {'Metric':<22} {'Min':>10} {'Mean':>10} {'Max':>10}")
    print(f"  {'─'*54}")
    rows = [
        ("ATR14 ($)",      sig_bars[atr_col]),
        ("Stop dist ($)",  stop_d),
        ("TP dist ($)",    tp_d),
        ("Units",          units),
        ("Position ($)",   value),
        ("% of capital",   value / capital * 100),
    ]
    for name, s in rows:
        print(f"  {name:<22} {s.min():>10.3f} {s.mean():>10.3f} {s.max():>10.3f}")
    print(f"  {'─'*54}")
