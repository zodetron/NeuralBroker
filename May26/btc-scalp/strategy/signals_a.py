"""
strategy/signals_a.py — Strategy A: Momentum Burst

Active in: HIGH_VOL_CHOP state only.

Long entry  : close breaks above 10-bar highest high, volume > 2× avg
Short entry : close breaks below 10-bar lowest low,  volume > 2× avg
1H alignment: if 1H trend OPPOSES direction → skip entirely
              if 1H trend ALIGNS            → +0.05 confidence bonus
              if 1H trend NEUTRAL (0)       → no bonus, still trade

All conditions computed on bar t (closed); entry on bar t+1 open.
No lookahead: high/low lookback uses shift(1).rolling(10) so bar t is excluded.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import STRAT_A

# State constants
_HIGH_VOL_CHOP = "HIGH_VOL_CHOP"


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate raw Momentum Burst signals on the classified 15M DataFrame.

    Parameters
    ----------
    df : output of classify_market_state — must contain market_state,
         ema20, atr_14, trend_1h, align_bonus columns.

    Returns
    -------
    pd.DataFrame indexed by bar timestamp (signal fires at t, enters t+1 open).
    Columns: signal_id, strategy, direction, market_state, close, atr_14,
             roll_level, vol_ratio, hour, trend_1h, align_1h, conf_bonus.
    """
    d = df.copy()

    lb   = STRAT_A["lookback_bars"]     # 10
    vmul = STRAT_A["vol_mult"]          # 2.0
    bonus= STRAT_A["trend_align_bonus"] # 0.05

    # ── Lookback levels (no lookahead: shift(1) excludes current bar) ────────
    roll_high = d["high"].shift(1).rolling(lb).max()
    roll_low  = d["low"].shift(1).rolling(lb).min()
    vol_ma20  = d["volume"].rolling(20).mean()

    # ── Core conditions ───────────────────────────────────────────────────────
    state_ok    = d["market_state"] == _HIGH_VOL_CHOP
    vol_ok      = d["volume"] > vmul * vol_ma20
    long_break  = d["close"] > roll_high
    short_break = d["close"] < roll_low

    # 1H alignment gate: skip if 1H trend opposes intended direction
    # Long  requires trend_1h != -1 (not bearish 1H)
    # Short requires trend_1h !=  1 (not bullish 1H)
    long_mask  = state_ok & vol_ok & long_break  & (d["trend_1h"] != -1)
    short_mask = state_ok & vol_ok & short_break & (d["trend_1h"] !=  1)

    # Handle the (extremely rare) simultaneous long+short: keep 1H-aligned side
    both = long_mask & short_mask
    long_mask  = long_mask  & ~(both & (d["trend_1h"] == -1))
    short_mask = short_mask & ~(both & (d["trend_1h"] ==  1))
    # If both still fire with neutral 1H, prefer long
    short_mask = short_mask & ~(long_mask & short_mask)

    any_signal = long_mask | short_mask
    if not any_signal.any():
        return pd.DataFrame()

    # ── Build signal rows (fully vectorized) ─────────────────────────────────
    sig_df = pd.DataFrame(index=d.index[any_signal])
    sig_df["direction"]    = np.where(long_mask[any_signal], 1, -1)
    sig_df["market_state"] = d.loc[any_signal, "market_state"].values
    sig_df["close"]        = d.loc[any_signal, "close"].values
    sig_df["atr_14"]       = d.loc[any_signal, "atr_14"].values
    sig_df["hour"]         = d.index[any_signal].hour
    sig_df["trend_1h"]     = d.loc[any_signal, "trend_1h"].values.astype(int)

    # roll_level = the breakout level that was breached
    sig_df["roll_level"] = np.where(
        sig_df["direction"] == 1,
        roll_high[any_signal].values,
        roll_low[any_signal].values,
    )
    sig_df["vol_ratio"] = (d.loc[any_signal, "volume"] / (vol_ma20[any_signal] + 1e-9)).values

    # align_1h: 1 if 1H trend matches direction
    sig_df["align_1h"] = (
        ((sig_df["direction"] ==  1) & (sig_df["trend_1h"] ==  1)) |
        ((sig_df["direction"] == -1) & (sig_df["trend_1h"] == -1))
    ).astype(np.int8)

    sig_df["conf_bonus"] = sig_df["align_1h"] * bonus

    # Assign sequential IDs
    sig_df.insert(0, "strategy", "A")
    sig_df.insert(0, "signal_id", [f"A{n:04d}" for n in range(1, len(sig_df) + 1)])
    sig_df.index.name = "ts"

    return sig_df


def signal_stats(sig: pd.DataFrame, df: pd.DataFrame) -> None:
    """Print statistics for Strategy A signals."""
    if sig.empty:
        print("  Strategy A: no signals generated.")
        return

    n      = len(sig)
    n_long = (sig["direction"] == 1).sum()
    n_days = (df.index[-1] - df.index[0]).days
    avg_pd = n / n_days

    print(f"\n  {'─'*56}")
    print(f"  Strategy A — Momentum Burst")
    print(f"  {'─'*56}")
    print(f"  Total signals   : {n:>7,}")
    print(f"  Long / Short    : {n_long:>7,} / {n - n_long:,}")
    print(f"  Avg per day     : {avg_pd:>7.2f}")
    print(f"  With 1H bonus   : {sig['align_1h'].sum():>7,}  "
          f"({sig['align_1h'].mean()*100:.1f}%)")
    print(f"  Avg vol ratio   : {sig['vol_ratio'].mean():>7.2f}×")

    # Sanity: all should be HIGH_VOL_CHOP
    states = sig["market_state"].value_counts()
    print(f"\n  Market state breakdown:")
    for s, cnt in states.items():
        print(f"    {s:<22} {cnt:>6,}  ({cnt/n*100:.1f}%)")

    # Hour distribution — top 5
    hour_counts = sig["hour"].value_counts().sort_values(ascending=False)
    print(f"\n  Top 5 hours (UTC):")
    for hr, cnt in hour_counts.head(5).items():
        bar = "█" * int(cnt / hour_counts.iloc[0] * 20)
        print(f"    {hr:02d}:00  {cnt:>5,}  {bar}")
