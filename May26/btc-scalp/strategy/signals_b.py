"""
strategy/signals_b.py — Strategy B: VWAP Reversion

Active in: LOW_VOL_RANGE state only.

VWAP resets at 00:00 UTC every day (BTC 24/7 — no session anchor).
Typical price = (H + L + C) / 3 for VWAP weighting.

Long  : price < VWAP × (1 - 0.35%) AND RSI14 < 42 AND volume < 1.5× avg
Short : price > VWAP × (1 + 0.35%) AND RSI14 > 58 AND volume < 1.5× avg

Safety filter: skip if any of the last 2 bars was HIGH_VOL_CHOP
(prevents catching a falling knife after a vol spike settles back to range).

All conditions computed on bar t; entry on bar t+1 open.
No lookahead: VWAP is strictly cumulative within the current day.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import STRAT_B

_LOW_VOL_RANGE = "LOW_VOL_RANGE"
_HIGH_VOL_CHOP = "HIGH_VOL_CHOP"


# ── RSI (Wilder EMA) ──────────────────────────────────────────────────────────

def _rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=window - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=window - 1, adjust=False).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - 100 / (1 + rs)


# ── VWAP (daily reset at 00:00 UTC) ──────────────────────────────────────────

def _compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Cumulative VWAP resetting at UTC midnight each day.
    Uses typical price = (H + L + C) / 3.
    Strictly causal: VWAP[t] uses only bars from 00:00 UTC up to and including t.
    """
    d = df.copy()
    d["_tp"]    = (d["high"] + d["low"] + d["close"]) / 3
    d["_tpvol"] = d["_tp"] * d["volume"]

    # Group by UTC date (normalize() floors to midnight, keeps timezone)
    date_key = d.index.normalize()
    d["_cum_tpvol"] = d.groupby(date_key)["_tpvol"].cumsum()
    d["_cum_vol"]   = d.groupby(date_key)["volume"].cumsum()
    vwap = d["_cum_tpvol"] / (d["_cum_vol"] + 1e-9)
    return vwap


# ── MAIN SIGNAL GENERATOR ─────────────────────────────────────────────────────

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate raw VWAP Reversion signals on the classified 15M DataFrame.

    Parameters
    ----------
    df : output of classify_market_state.

    Returns
    -------
    pd.DataFrame indexed by bar timestamp.
    Columns: signal_id, strategy, direction, market_state, close, atr_14,
             vwap, vwap_dev_pct, rsi14, vol_ratio, hour, trend_1h, align_1h,
             conf_bonus.
    """
    d = df.copy()

    band_pct  = STRAT_B["vwap_band_pct"]      # 0.0035
    rsi_lo    = STRAT_B["rsi_long_max"]        # 42
    rsi_hi    = STRAT_B["rsi_short_min"]       # 58
    vol_spike = STRAT_B["vol_spike_mult"]      # 1.5
    rsi_w     = STRAT_B["rsi_period"]          # 14

    # ── Indicators ────────────────────────────────────────────────────────────
    d["vwap"]      = _compute_vwap(d)
    d["rsi14"]     = _rsi(d["close"], rsi_w)
    vol_ma20       = d["volume"].rolling(20).mean()

    # ── Conditions ────────────────────────────────────────────────────────────
    state_ok   = d["market_state"] == _LOW_VOL_RANGE
    vol_quiet  = d["volume"] < vol_spike * vol_ma20    # no vol spikes

    vwap_upper = d["vwap"] * (1 + band_pct)
    vwap_lower = d["vwap"] * (1 - band_pct)

    long_dev   = d["close"] < vwap_lower
    short_dev  = d["close"] > vwap_upper

    # Safety: skip if last 2 bars were HIGH_VOL_CHOP (post-spike knife)
    prev_chop = (
        (d["market_state"].shift(1) == _HIGH_VOL_CHOP) |
        (d["market_state"].shift(2) == _HIGH_VOL_CHOP)
    )

    long_mask  = state_ok & vol_quiet & long_dev  & (d["rsi14"] < rsi_lo) & ~prev_chop
    short_mask = state_ok & vol_quiet & short_dev & (d["rsi14"] > rsi_hi) & ~prev_chop

    any_signal = long_mask | short_mask
    if not any_signal.any():
        return pd.DataFrame()

    # ── Build signal rows ─────────────────────────────────────────────────────
    idx = d.index[any_signal]
    sig_df = pd.DataFrame(index=idx)
    sig_df["direction"]    = np.where(long_mask[any_signal], 1, -1)
    sig_df["market_state"] = d.loc[any_signal, "market_state"].values
    sig_df["close"]        = d.loc[any_signal, "close"].values
    sig_df["atr_14"]       = d.loc[any_signal, "atr_14"].values
    sig_df["hour"]         = idx.hour
    sig_df["trend_1h"]     = d.loc[any_signal, "trend_1h"].values.astype(int)
    sig_df["vwap"]         = d.loc[any_signal, "vwap"].values

    # VWAP deviation % at signal bar
    sig_df["vwap_dev_pct"] = (
        (d.loc[any_signal, "close"] - d.loc[any_signal, "vwap"])
        / (d.loc[any_signal, "vwap"] + 1e-9) * 100
    ).values
    sig_df["rsi14"]    = d.loc[any_signal, "rsi14"].values
    sig_df["vol_ratio"]= (d.loc[any_signal, "volume"] / (vol_ma20[any_signal] + 1e-9)).values

    # Strategy B has no directional 1H alignment requirement
    # (mean-reversion is regime-driven, not trend-driven)
    sig_df["align_1h"]  = d.loc[any_signal, "align_bonus"].values.astype(np.int8)
    sig_df["conf_bonus"] = 0.0

    sig_df.insert(0, "strategy",  "B")
    sig_df.insert(0, "signal_id", [f"B{n:04d}" for n in range(1, len(sig_df) + 1)])
    sig_df.index.name = "ts"

    return sig_df


def signal_stats(sig: pd.DataFrame, df: pd.DataFrame) -> None:
    """Print statistics for Strategy B signals."""
    if sig.empty:
        print("  Strategy B: no signals generated.")
        return

    n      = len(sig)
    n_long = (sig["direction"] == 1).sum()
    n_days = (df.index[-1] - df.index[0]).days
    avg_pd = n / n_days

    print(f"\n  {'─'*56}")
    print(f"  Strategy B — VWAP Reversion")
    print(f"  {'─'*56}")
    print(f"  Total signals   : {n:>7,}")
    print(f"  Long / Short    : {n_long:>7,} / {n - n_long:,}")
    print(f"  Avg per day     : {avg_pd:>7.2f}")
    print(f"  Avg VWAP dev%   : {sig['vwap_dev_pct'].abs().mean():>7.3f}%")
    print(f"  Avg RSI14       : {sig['rsi14'].mean():>7.2f}")
    print(f"  Avg vol ratio   : {sig['vol_ratio'].mean():>7.2f}×")

    states = sig["market_state"].value_counts()
    print(f"\n  Market state breakdown (should be all LOW_VOL_RANGE):")
    for s, cnt in states.items():
        print(f"    {s:<22} {cnt:>6,}  ({cnt/n*100:.1f}%)")

    hour_counts = sig["hour"].value_counts().sort_values(ascending=False)
    print(f"\n  Top 5 hours (UTC):")
    for hr, cnt in hour_counts.head(5).items():
        bar = "█" * int(cnt / hour_counts.iloc[0] * 20)
        print(f"    {hr:02d}:00  {cnt:>5,}  {bar}")
