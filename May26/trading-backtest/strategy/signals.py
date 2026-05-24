"""
strategy/signals.py — Regime-aware signal generation.

Signal encoding:
   1  = go long
  -1  = go short
   0  = flat / no position

Rules (all thresholds live in config.STRAT — nothing hardcoded here):

  BULL regime:
    → LONG when composite momentum score > 0
      score = average of (normalised ROC, normalised RSI)
      Extra guard: RSI must exceed momentum_rsi_min (avoids chasing
      already-overbought entries)

  SIDEWAYS regime:
    → LONG  when RSI_14 < rsi_oversold  (mean-reversion buy)
    → SHORT when RSI_14 > rsi_overbought (mean-reversion sell)
      [SHORT arm only active when STRAT["allow_shorting"] is True]

  BEAR regime:
    → SHORT if allow_shorting else FLAT

Lookahead contract:
  Features at row t are computed from data available at close of bar t.
  Signals computed here use features[t] → legitimate (no future data).
  The backtest engine MUST shift signals by 1 bar before execution so
  trades are entered at the open of bar t+1.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import STRAT


# ── MOMENTUM SCORE ────────────────────────────────────────────────────────────

def _momentum_score(df: pd.DataFrame) -> pd.Series:
    """
    Composite momentum score ∈ (-∞, +∞), positive = bullish.

    Components:
      roc_norm  : ROC normalised by its rolling 252-day std
                  → scale-independent across assets and time
      rsi_norm  : (RSI_14 − 50) / 50  → maps [0, 100] to [−1, +1]
    """
    roc_col  = f"roc_{STRAT['momentum_roc_period']}"

    # Normalise ROC: rolling std with 60-bar min-period so signal fires early
    roc_std  = df[roc_col].rolling(252, min_periods=60).std().replace(0, np.nan)
    roc_norm = df[roc_col] / roc_std

    rsi_norm = (df["rsi_14"] - 50.0) / 50.0

    return (roc_norm + rsi_norm) / 2.0


# ── SIGNAL GENERATOR ─────────────────────────────────────────────────────────

def compute_signals(feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute regime-aware trading signals for every bar in feat_df.

    Parameters
    ----------
    feat_df : pd.DataFrame
        Must contain columns: regime, rsi_14, roc_{period}, atr_{window}.
        Rows with NaN regime are assigned signal = 0.

    Returns
    -------
    pd.DataFrame with added columns:
      momentum_score : continuous composite score (diagnostic)
      raw_signal     : signal before any NaN-masking
      signal         : final signal (0 where regime/features are NaN)
    """
    df    = feat_df.copy()
    allow = STRAT["allow_shorting"]

    mom   = _momentum_score(df)
    df["momentum_score"] = mom

    sig   = pd.Series(0, index=df.index, dtype=int)

    # ── Bull: momentum long ───────────────────────────────────────────────────
    bull = df["regime"] == "Bull"
    bull_long = (
        bull
        & (mom > 0)
        & (df["rsi_14"] > STRAT["momentum_rsi_min"])   # not already overbought
    )
    sig[bull_long] = 1

    # ── Sideways: mean reversion ──────────────────────────────────────────────
    side = df["regime"] == "Sideways"
    sig[side & (df["rsi_14"] < STRAT["rsi_oversold"])]  = 1    # buy dip
    if allow:
        sig[side & (df["rsi_14"] > STRAT["rsi_overbought"])] = -1  # sell spike

    # ── Bear: short or flat ───────────────────────────────────────────────────
    bear = df["regime"] == "Bear"
    if allow:
        sig[bear] = -1
    # else: remain 0 (flat) in Bear — default

    # ── Mask rows where regime or key features are NaN ───────────────────────
    bad_mask = (
        df["regime"].isna()
        | df["rsi_14"].isna()
        | mom.isna()
    )
    sig[bad_mask] = 0

    df["raw_signal"] = sig
    df["signal"]     = sig

    return df


# ── SIGNAL SUMMARY ────────────────────────────────────────────────────────────

def print_signal_summary(df: pd.DataFrame, asset_key: str) -> None:
    """Print signal distribution by type and by regime."""
    sig = df["signal"]
    reg = df["regime"]

    total    = len(df)
    n_long   = (sig ==  1).sum()
    n_short  = (sig == -1).sum()
    n_flat   = (sig ==  0).sum()

    # Count signal *changes* (actual trade triggers)
    changes  = sig.diff().fillna(0)
    entries_long  = ((changes > 0) & (sig == 1)).sum()
    entries_short = ((changes < 0) & (sig == -1)).sum()
    exits         = ((sig == 0) & (changes != 0)).sum()

    print(f"\n  {'─'*58}")
    print(f"  {asset_key} Signal Summary")
    print(f"  {'─'*58}")
    print(f"  Total bars    : {total:,}")
    print(f"  Long bars     : {n_long:,}  ({n_long/total*100:.1f}%)")
    print(f"  Short bars    : {n_short:,}  ({n_short/total*100:.1f}%)")
    print(f"  Flat bars     : {n_flat:,}  ({n_flat/total*100:.1f}%)")
    print(f"\n  Signal transitions (trade entries):")
    print(f"    Long  entries : {entries_long:,}")
    print(f"    Short entries : {entries_short:,}")
    print(f"    Exits to flat : {exits:,}")

    # Breakdown by regime
    print(f"\n  Signal by regime:")
    print(f"  {'Regime':<12} {'Long':>7} {'Short':>7} {'Flat':>7} {'Total':>7}")
    print(f"  {'─'*40}")
    for regime in ["Bull", "Sideways", "Bear"]:
        mask  = reg == regime
        n     = mask.sum()
        if n == 0:
            continue
        nl = (sig[mask] ==  1).sum()
        ns = (sig[mask] == -1).sum()
        nf = (sig[mask] ==  0).sum()
        print(f"  {regime:<12} {nl:>7,} {ns:>7,} {nf:>7,} {n:>7,}")
    print(f"  {'─'*40}")

    # Recent 5 signals with context
    recent = df[sig != 0].tail(5)[
        ["close", "regime", "rsi_14", "momentum_score", "signal"]
    ].copy()
    recent["signal_str"] = recent["signal"].map({1: "LONG", -1: "SHORT"})
    print(f"\n  Last 5 active signals:")
    print(f"  {'Date':<14} {'Close':>10} {'Regime':<10} {'RSI':>6} "
          f"{'MomScore':>10} {'Signal':>7}")
    print(f"  {'─'*58}")
    for ts, row in recent.iterrows():
        print(f"  {str(ts.date()):<14} {row['close']:>10,.2f} "
              f"{row['regime']:<10} {row['rsi_14']:>6.1f} "
              f"{row['momentum_score']:>10.3f} {row['signal_str']:>7}")
