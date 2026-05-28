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


# ── BTC SIGNAL (breakout + pullback) ─────────────────────────────────────────

def _btc_signals(df: pd.DataFrame):
    """
    Mode 1 — volume-confirmed N-day breakout (Bull) + breakdown (Bear).
    Mode 2 — pullback entry in Bull regime:
        price between SMA20 and SMA50, RSI 42-58, ADX > 22,
        minimum 3 bars since last entry of any type.

    Returns (sig, mode) — both pd.Series aligned to df.index.
    mode: 0=flat, 1=breakout/breakdown, 2=pullback long.
    """
    allow    = STRAT["allow_shorting"]
    vol_mult = STRAT.get("vol_mult", 1.5)
    adx_min  = STRAT.get("adx_bull_min", 25)

    pb_rsi_lo = STRAT.get("pullback_rsi_min", 42)
    pb_rsi_hi = STRAT.get("pullback_rsi_max", 58)
    pb_adx    = STRAT.get("pullback_adx_min", 22)
    pb_gap    = STRAT.get("pullback_min_days", 3)

    bull, side, bear = _regime_masks(df)

    vol_ok = df["vol_ratio_20"] > vol_mult
    adx_ok = df["adx_14"] > adx_min

    # ── Mode 1 long: breakout above 20d high in Bull regime ──────────────────
    breakout_long = (
        bull
        & (df["close"] > df["roll_high_20"])
        & vol_ok
        & adx_ok
    ).values

    # ── Mode 2: pullback entry in Bull regime ─────────────────────────────────
    # price pulled back below SMA20 but still above SMA50 → in the "pullback zone"
    # RSI cooling (42-58), ADX still showing trend (>22)
    pullback_cond = (
        bull
        & (df["close"] < df["bb_mid"])        # below SMA20: pulled back
        & (df["sma50_dist"] > 0)              # still above SMA50: trend intact
        & (df["rsi_14"] >= pb_rsi_lo)
        & (df["rsi_14"] <= pb_rsi_hi)
        & (df["adx_14"] > pb_adx)
    ).values

    # ── Mode 1 short: breakdown below 20d low in Bear regime ──────────────────
    if allow:
        breakdown_short = (
            bear
            & (df["close"] < df["roll_low_20"])
            & vol_ok
            & adx_ok
        ).values
    else:
        breakdown_short = np.zeros(len(df), dtype=bool)

    # ── Combine with min_days_since_entry gap for Mode 2 ──────────────────────
    sig_arr  = np.zeros(len(df), dtype=int)
    mode_arr = np.zeros(len(df), dtype=int)
    last_entry = -999   # bar index of last entry (any type)

    for i in range(len(df)):
        if breakout_long[i]:
            sig_arr[i]  = 1
            mode_arr[i] = 1
            last_entry  = i
        elif pullback_cond[i] and (i - last_entry) >= pb_gap:
            sig_arr[i]  = 1
            mode_arr[i] = 2
            last_entry  = i

        if breakdown_short[i]:
            sig_arr[i]  = -1
            mode_arr[i] = 1   # shorts are always Mode 1

    sig  = pd.Series(sig_arr,  index=df.index, dtype=int)
    mode = pd.Series(mode_arr, index=df.index, dtype=int)
    return sig, mode


# ── XAU SIGNAL (v3b hybrid: Bull=momentum, Sideways=BB mean-reversion) ────────

def _xau_signals(df: pd.DataFrame) -> pd.Series:
    """
    v3b hybrid regime strategy for XAU:

    BULL regime   → 20d-high momentum breakout + ADX > adx_bull_min_xau.
                    Bar-by-bar signal: stays 1 while close > 20d high AND Bull.
                    Gold in Bull regimes trends cleanly — ride it, don't fade it.

    SIDEWAYS regime → BB lower-band mean-reversion, sticky hold to bb_upper.
                    Entry: first touch below bb_lower (ADX < adx_bear_max).
                    Exit: close reaches bb_upper (natural reversion target).

    BEAR regime   → flat (unchanged: violent gold crashes, stay out).

    Interaction: if a Sideways sticky position is open when regime turns Bull
    and a breakout fires, the position continues held (signal stays 1 from both
    conditions). The engine doesn't add to an open position (no pyramiding).
    """
    adx_max_mr = STRAT.get("adx_bear_max", 25)       # mean-rev entry: ADX < 25
    adx_min_mo = STRAT.get("adx_bull_min_xau", 20)   # Bull momentum: ADX > 20

    bull, side, bear = _regime_masks(df)

    close    = df["close"].values
    bb_lower = df["bb_lower"].values
    bb_upper = df["bb_upper"].values
    adx      = df["adx_14"].values
    is_bull  = bull.values
    is_side  = side.values

    # === BULL REGIME: bar-by-bar 20d-high momentum breakout ==================
    roll_high = (df["roll_high_20"].values if "roll_high_20" in df.columns
                 else np.full(len(df), np.nan))
    bull_momentum = (
        is_bull
        & (close > roll_high)
        & (adx > adx_min_mo)
    )

    # === SIDEWAYS REGIME: sticky BB mean-reversion ===========================
    # "First touch" prevents re-entry while already below the band.
    prev_above_lower = np.concatenate([[True], close[:-1] > bb_lower[:-1]])

    entry_mr = (
        is_side             # ONLY in Sideways; Bull regime gets momentum signal
        & (close <= bb_lower)
        & prev_above_lower
        & (adx < adx_max_mr)
    )
    exit_mr = close >= bb_upper    # hold until price reaches upper band

    # === COMBINE ============================================================
    sig_arr   = np.zeros(len(df), dtype=int)
    active_mr = False              # is a Sideways sticky mean-rev trade open?

    for i in range(len(df)):
        # Mean-rev exit (sticky)
        if exit_mr[i]:
            active_mr = False
        # Mean-rev entry (new first-touch in Sideways)
        if entry_mr[i]:
            active_mr = True
        # Signal = 1 if either sticky mean-rev OR Bull momentum is active
        if active_mr or bull_momentum[i]:
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
        sig, mode = _btc_signals(df)
    else:
        sig  = _xau_signals(df)
        mode = pd.Series(0, index=df.index, dtype=int)

    # Mask rows where key features are NaN
    bad_mask = (
        df["regime"].isna()
        | df["adx_14"].isna()
        | df["bb_lower"].isna()
        | (df["roll_high_20"].isna() if "roll_high_20" in df.columns else False)
    )
    sig[bad_mask]  = 0
    mode[bad_mask] = 0

    df["raw_signal"]   = sig
    df["signal"]       = sig
    df["signal_mode"]  = mode

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
