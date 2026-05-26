"""
strategy/signals.py — Asset-specific, regime-aware signal generation.

Signal encoding:
   1  = go long
  -1  = go short
   0  = flat / no position

BTC (Improvement 1 + 3):
  Bull regime (P(Bull) > bull_prob_threshold):
    → LONG when:
      1. close > 20-day prior high  (breakout confirmation)
      2. volume > vol_mult × 20d avg volume  (volume confirmation)
      3. ADX_14 > adx_bull_min  (trending market, not noise)
  Bear regime:
    → SHORT if allow_shorting AND close < 20d prior low AND same vol/ADX

XAU (Improvement 2 + 3):
  Sideways OR Bull regime:
    → LONG when close <= bb_lower AND ADX_14 < adx_bear_max
  Sideways regime only:
    → SHORT when close >= bb_upper AND ADX_14 < adx_bear_max (if allow_shorting)
  Bear regime:
    → FLAT (gold crashes are violent — sit out)

Lookahead contract:
  All features at row t are computed from data ≤ bar t.
  The backtest engine shifts signals by +1 bar so entries fill at t+1.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import STRAT


# ── REGIME SELECTOR ───────────────────────────────────────────────────────────

def _regime_masks(df: pd.DataFrame):
    """
    Return (bull, side, bear) boolean Series.
    Uses soft HMM probability columns when available (v2+);
    falls back to hard regime labels.
    """
    thresh = STRAT.get("bull_prob_threshold", 0.60)

    if "bull_prob" in df.columns and df["bull_prob"].notna().any():
        bull = df["bull_prob"] > thresh
        bear = df["bear_prob"] > thresh if "bear_prob" in df.columns else (df["regime"] == "Bear")
        side = ~bull & ~bear
    else:
        bull = df["regime"] == "Bull"
        side = df["regime"] == "Sideways"
        bear = df["regime"] == "Bear"

    return bull, side, bear


# ── BTC SIGNAL (volume-confirmed breakout) ────────────────────────────────────

def _btc_signals(df: pd.DataFrame) -> pd.Series:
    """
    Improvement 1: volume-confirmed N-day breakout.
    Improvement 3: ADX trend-strength gate.
    """
    allow    = STRAT["allow_shorting"]
    vol_mult = STRAT.get("vol_mult", 1.5)
    adx_min  = STRAT.get("adx_bull_min", 25)

    bull, side, bear = _regime_masks(df)

    sig = pd.Series(0, index=df.index, dtype=int)

    # Volume confirmation mask (applies to both long and short)
    vol_ok = df["vol_ratio_20"] > vol_mult

    # ADX trend confirmation (strong trend required for breakout)
    adx_ok = df["adx_14"] > adx_min

    # Long: breakout above 20d high in Bull regime
    long_cond = (
        bull
        & (df["close"] > df["roll_high_20"])   # closes above prior 20d high
        & vol_ok
        & adx_ok
    )
    sig[long_cond] = 1

    # Short: break below 20d low in Bear regime
    if allow:
        short_cond = (
            bear
            & (df["close"] < df["roll_low_20"])
            & vol_ok
            & adx_ok
        )
        sig[short_cond] = -1

    return sig


# ── XAU SIGNAL (Bollinger Band mean-reversion) ────────────────────────────────

def _xau_signals(df: pd.DataFrame) -> pd.Series:
    """
    Improvement 2: BB mean-reversion on daily bars with sticky hold.
    Improvement 3: ADX ranging gate (ADX < adx_bear_max) on ENTRY only.

    Sticky logic: once price touches lower BB (entry trigger), signal stays
    LONG until price reaches the upper BB or regime turns Bear.  This lets
    trades run to TP/SL instead of exiting the moment price leaves the lower
    band (which averaged 2.7 days — not enough for 3×ATR target).
    """
    allow   = STRAT["allow_shorting"]
    adx_max = STRAT.get("adx_bear_max", 25)

    bull, side, bear = _regime_masks(df)

    # Numpy arrays for the inner loop (faster than .iloc)
    close     = df["close"].values
    bb_lower  = df["bb_lower"].values
    bb_upper  = df["bb_upper"].values
    adx       = df["adx_14"].values
    is_bear   = bear.values
    is_long_r = (side | bull).values   # Bull or Sideways = allowed regime

    # SMA200 trend filter: block entries in persistent downtrends.
    # Gold 2013-2015 crash (−40%) and 2022 decline generated false bb_lower
    # touches that continued falling. Filtering when >8% below 200d SMA avoids
    # the worst "falling knife" entries without sacrificing normal corrections.
    if "sma200_dist" in df.columns:
        sma200_ok = df["sma200_dist"].values > -8.0
    else:
        sma200_ok = np.ones(len(df), dtype=bool)

    # "First touch" — price must cross FROM above TO at/below lower band.
    # This prevents re-entry while already below the band (e.g., after a SL hit
    # drove price further below bb_lower, which was the re-entry bug causing 143 trades).
    prev_above_lower = np.concatenate([[True], close[:-1] > bb_lower[:-1]])

    # Entry: first crossing below lower BB, ranging market, non-Bear, not deep downtrend
    entry = (
        is_long_r
        & (close <= bb_lower)
        & prev_above_lower          # must have been above on previous bar
        & (adx < adx_max)
        & sma200_ok                 # not in persistent downtrend (>8% below SMA200)
    )
    # Exit: ONLY when price reaches upper BB (natural mean-reversion target).
    # Bear regime does NOT force exit — HMM is too noisy (3.4-day avg duration,
    # 1,738 transitions), so Bear labels pop up mid-session and cause premature
    # exits well before price reaches bb_upper.
    # The engine's SL (3×ATR) handles genuine bear-market risk.
    # New entries are still blocked during Bear (entry requires is_long_r).
    exit_ = close >= bb_upper

    sig_arr = np.zeros(len(df), dtype=int)
    active_long = False
    for i in range(len(df)):
        if exit_[i]:
            active_long = False
        if entry[i]:
            active_long = True
        if active_long:
            sig_arr[i] = 1

    return pd.Series(sig_arr, index=df.index, dtype=int)


# ── UNIFIED ENTRY POINT ───────────────────────────────────────────────────────

def compute_signals(feat_df: pd.DataFrame, asset_key: str = "BTC") -> pd.DataFrame:
    """
    Compute regime-aware trading signals for every bar.

    Parameters
    ----------
    feat_df   : DataFrame with all feature columns, regime, and HMM probabilities.
    asset_key : "BTC" → breakout strategy; "XAU" → BB mean-reversion strategy.

    Returns
    -------
    DataFrame with added columns:
      signal    : final entry signal (0 / 1 / -1)
      raw_signal: same, before NaN masking (diagnostic)
    """
    df = feat_df.copy()

    if asset_key == "BTC":
        sig = _btc_signals(df)
    else:
        sig = _xau_signals(df)

    # Mask rows where key features are NaN
    bad_mask = (
        df["regime"].isna()
        | df["adx_14"].isna()
        | df["bb_lower"].isna()
        | (df["roll_high_20"].isna() if "roll_high_20" in df.columns else False)
    )
    sig[bad_mask] = 0

    df["raw_signal"] = sig
    df["signal"]     = sig

    return df


# ── SIGNAL SUMMARY ────────────────────────────────────────────────────────────

def print_signal_summary(df: pd.DataFrame, asset_key: str) -> None:
    """Print signal distribution by type and by regime."""
    sig  = df["signal"]
    reg  = df["regime"]
    strategy = "Breakout" if asset_key == "BTC" else "BB Mean-Rev"

    total   = len(df)
    n_long  = (sig ==  1).sum()
    n_short = (sig == -1).sum()
    n_flat  = (sig ==  0).sum()

    changes       = sig.diff().fillna(0)
    entries_long  = ((changes > 0) & (sig == 1)).sum()
    entries_short = ((changes < 0) & (sig == -1)).sum()
    exits         = ((sig == 0) & (changes != 0)).sum()

    print(f"\n  {'─'*60}")
    print(f"  {asset_key} Signal Summary  [{strategy}]")
    print(f"  {'─'*60}")
    print(f"  Total bars    : {total:,}")
    print(f"  Long bars     : {n_long:,}  ({n_long/total*100:.1f}%)")
    print(f"  Short bars    : {n_short:,}  ({n_short/total*100:.1f}%)")
    print(f"  Flat bars     : {n_flat:,}  ({n_flat/total*100:.1f}%)")
    print(f"\n  Signal transitions (trade entries):")
    print(f"    Long  entries : {entries_long:,}")
    print(f"    Short entries : {entries_short:,}")
    print(f"    Exits to flat : {exits:,}")

    print(f"\n  Signal by regime:")
    print(f"  {'Regime':<12} {'Long':>7} {'Short':>7} {'Flat':>7} {'Total':>7}")
    print(f"  {'─'*40}")
    for regime in ["Bull", "Sideways", "Bear"]:
        mask = reg == regime
        n    = mask.sum()
        if n == 0:
            continue
        nl = (sig[mask] ==  1).sum()
        ns = (sig[mask] == -1).sum()
        nf = (sig[mask] ==  0).sum()
        print(f"  {regime:<12} {nl:>7,} {ns:>7,} {nf:>7,} {n:>7,}")
    print(f"  {'─'*40}")

    recent = df[sig != 0].tail(5)
    if len(recent) == 0:
        print(f"\n  (no active signals in history)")
        return

    cols = ["close", "regime", "adx_14", "signal"]
    if asset_key == "BTC":
        cols += ["roll_high_dist", "vol_ratio_20"]
    else:
        cols += ["bb_pct", "bb_lower", "bb_upper"]
    cols = [c for c in cols if c in recent.columns]

    recent = recent[cols].copy()
    recent["signal_str"] = recent["signal"].map({1: "LONG", -1: "SHORT"})
    print(f"\n  Last 5 active signals:")
    for ts, row in recent.iterrows():
        print(f"  {str(ts.date()):<14} close={row['close']:>10,.2f}  "
              f"regime={row['regime']:<10}  adx={row['adx_14']:>5.1f}  "
              f"{row['signal_str']}")
