"""
strategy/market_state.py — Fast vectorized market state classifier.

Labels every 15M bar as one of four states:
  TRENDING_UP   : price > EMA20, EMA20 slope > 0, ADX > 25
  TRENDING_DOWN : price < EMA20, EMA20 slope < 0, ADX > 25
  HIGH_VOL_CHOP : ADX < 25, ATR > 1.5× 20-bar avg ATR
  LOW_VOL_RANGE : ADX < 25, ATR < 1.0× 20-bar avg ATR
  (default)     : HIGH_VOL_CHOP for any unclassified bars

Also adds:
  trend_1h      : direction of 1H EMA20 (+1 up, -1 down, 0 flat)
  align_1h      : 1 if 15M signal direction matches 1H trend

No lookahead: EMA and rolling features use only past data.
ADX uses Wilder smoothing (same as the standard definition).
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MS

# State labels
TRENDING_UP   = "TRENDING_UP"
TRENDING_DOWN = "TRENDING_DOWN"
HIGH_VOL_CHOP = "HIGH_VOL_CHOP"
LOW_VOL_RANGE = "LOW_VOL_RANGE"

ALL_STATES = [TRENDING_UP, TRENDING_DOWN, HIGH_VOL_CHOP, LOW_VOL_RANGE]


# ── INDICATOR HELPERS ─────────────────────────────────────────────────────────

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Wilder ATR (EWM with span=period)."""
    h, l, cp = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([h - l, (h - cp).abs(), (l - cp).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _adx(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Average Directional Index using Wilder EMA smoothing.
    Returns ADX values (0-100 scale).
    """
    h  = df["high"]
    l  = df["low"]
    tr = pd.concat([
        h - l,
        (h - df["close"].shift(1)).abs(),
        (l - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)

    up   = (h - h.shift(1)).clip(lower=0)
    down = (l.shift(1) - l).clip(lower=0)

    plus_dm  = np.where(up > down, up,   0.0)
    minus_dm = np.where(down > up, down, 0.0)

    plus_dm_s  = pd.Series(plus_dm,  index=df.index)
    minus_dm_s = pd.Series(minus_dm, index=df.index)

    com = period - 1
    tr14      = tr.ewm(com=com, adjust=False).mean()
    plus_di   = 100 * plus_dm_s.ewm(com=com, adjust=False).mean() / (tr14 + 1e-9)
    minus_di  = 100 * minus_dm_s.ewm(com=com, adjust=False).mean() / (tr14 + 1e-9)

    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    return dx.ewm(com=com, adjust=False).mean()


# ── MAIN CLASSIFIER ───────────────────────────────────────────────────────────

def classify_market_state(
    df_15m: pd.DataFrame,
    df_1h:  pd.DataFrame,
) -> pd.DataFrame:
    """
    Classify every 15M bar and attach 1H trend context.

    Parameters
    ----------
    df_15m : 15M OHLCV DataFrame with UTC DatetimeIndex
    df_1h  : 1H  OHLCV DataFrame with UTC DatetimeIndex

    Returns
    -------
    df_15m copy with additional columns:
      ema20, ema20_slope, atr_14, atr_avg, adx_14,
      market_state, trend_1h, align_bonus
    """
    df = df_15m.copy()

    period_ema  = MS["ema_period"]        # 20
    period_adx  = MS["adx_period"]        # 14
    period_atr  = MS["atr_period"]        # 20
    adx_thresh  = MS["adx_trend_thresh"]  # 25
    slope_bars  = MS["slope_bars"]        # 5
    atr_hi      = MS["atr_high_mult"]     # 1.5
    atr_lo      = MS["atr_low_mult"]      # 1.0

    # ── Core indicators on 15M ───────────────────────────────────────────────
    df["ema20"]       = _ema(df["close"], period_ema)
    # Slope: % change of EMA over slope_bars bars — positive = upward tilt
    df["ema20_slope"] = df["ema20"].diff(slope_bars) / (df["ema20"].shift(slope_bars) + 1e-9)
    df["atr_14"]      = _atr(df, period_atr)
    # Rolling mean of ATR — used as the "normal" baseline for vol regime
    df["atr_avg"]     = df["atr_14"].rolling(period_atr * 2).mean()
    df["adx_14"]      = _adx(df, period_adx)

    # ── Classification (vectorized, no lookahead) ────────────────────────────
    close     = df["close"].values
    ema20     = df["ema20"].values
    slope     = df["ema20_slope"].values
    adx       = df["adx_14"].values
    atr_ratio = (df["atr_14"] / (df["atr_avg"] + 1e-9)).values

    trending_up   = (adx > adx_thresh) & (close > ema20) & (slope > 0)
    trending_down = (adx > adx_thresh) & (close < ema20) & (slope < 0)
    high_chop     = (adx <= adx_thresh) & (atr_ratio > atr_hi)
    low_range     = (adx <= adx_thresh) & (atr_ratio < atr_lo)

    state = np.select(
        [trending_up, trending_down, high_chop, low_range],
        [TRENDING_UP, TRENDING_DOWN, HIGH_VOL_CHOP, LOW_VOL_RANGE],
        default=HIGH_VOL_CHOP,   # mid-range ATR, non-trending → treat as choppy
    )
    df["market_state"] = state

    # ── 1H trend context ─────────────────────────────────────────────────────
    h1 = df_1h.copy()
    h1["ema20_1h"]   = _ema(h1["close"], period_ema)
    h1["slope_1h"]   = h1["ema20_1h"].diff(slope_bars) / (h1["ema20_1h"].shift(slope_bars) + 1e-9)
    # +1 = up trend, -1 = down trend, 0 = flat/ambiguous
    h1["trend_1h"]   = np.where(h1["slope_1h"] > 0, 1, np.where(h1["slope_1h"] < 0, -1, 0))

    # Merge into 15M — forward-fill so each 15M bar inherits the most recent
    # completed 1H bar's trend. Merge on index using reindex + ffill.
    trend_series = h1["trend_1h"].reindex(df.index, method="ffill")
    df["trend_1h"] = trend_series.values

    # align_bonus: 1H trend matches the 15M market state direction
    df["align_bonus"] = np.where(
        ((df["market_state"] == TRENDING_UP)   & (df["trend_1h"] ==  1)) |
        ((df["market_state"] == TRENDING_DOWN) & (df["trend_1h"] == -1)),
        1, 0
    ).astype(np.int8)

    return df


# ── SUMMARY & CHART ───────────────────────────────────────────────────────────

def print_state_summary(df: pd.DataFrame) -> None:
    """Print state distribution and alignment stats."""
    n = len(df)

    print(f"\n  {'─'*60}")
    print(f"  Market State Distribution  ({n:,} bars)")
    print(f"  {'─'*60}")
    print(f"  {'State':<20} {'Bars':>8} {'Pct':>7}  {'Align 1H':>9}")
    print(f"  {'─'*60}")

    for state in ALL_STATES:
        mask  = df["market_state"] == state
        cnt   = mask.sum()
        pct   = cnt / n * 100
        # alignment only meaningful for trending states
        if state in (TRENDING_UP, TRENDING_DOWN):
            alg = df.loc[mask, "align_bonus"].mean() * 100
            alg_str = f"{alg:>8.1f}%"
        else:
            alg_str = f"{'—':>9}"
        print(f"  {state:<20} {cnt:>8,} {pct:>6.1f}%  {alg_str}")

    print(f"  {'─'*60}")

    uncl = n - df["market_state"].isin(ALL_STATES).sum()
    if uncl:
        print(f"  Unclassified: {uncl:,}  (NaN warmup rows)")

    trend_up   = (df["trend_1h"] ==  1).sum()
    trend_down = (df["trend_1h"] == -1).sum()
    trend_flat = (df["trend_1h"] ==  0).sum()
    print(f"\n  1H trend context:")
    print(f"    Up   : {trend_up:>8,}  ({trend_up/n*100:.1f}%)")
    print(f"    Down : {trend_down:>8,}  ({trend_down/n*100:.1f}%)")
    print(f"    Flat : {trend_flat:>8,}  ({trend_flat/n*100:.1f}%)")

    overall_align = df["align_bonus"].mean() * 100
    print(f"    1H–15M trend alignment (trending bars): "
          f"{overall_align:.1f}% of all bars have matching 1H trend")

    print(f"\n  Indicator ranges (15M):")
    for col, label in [("adx_14", "ADX"), ("atr_14", "ATR"), ("ema20_slope", "EMA slope")]:
        if col in df.columns:
            s = df[col].dropna()
            print(f"    {label:<12}  min={s.min():>8.3f}  "
                  f"mean={s.mean():>8.3f}  max={s.max():>8.3f}")


def plot_state_distribution(df: pd.DataFrame, out_path) -> None:
    """
    Two charts:
      1. Stacked area of market state over time (sampled monthly).
      2. Pie chart of overall state distribution.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    STATE_COLORS = {
        TRENDING_UP:   "#4CAF50",   # green
        TRENDING_DOWN: "#F44336",   # red
        HIGH_VOL_CHOP: "#FF9800",   # orange
        LOW_VOL_RANGE: "#2196F3",   # blue
    }

    # Monthly resampled state fractions
    dummies  = pd.get_dummies(df["market_state"])
    for s in ALL_STATES:
        if s not in dummies.columns:
            dummies[s] = 0
    monthly  = dummies[ALL_STATES].resample("ME").mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("BTC/USDT 15M — Market State Classifier", fontsize=14, fontweight="bold")

    # Stacked area
    bottom = None
    for state in ALL_STATES:
        vals = monthly[state].values
        ax1.fill_between(
            monthly.index, vals if bottom is None else bottom + vals,
            bottom if bottom is not None else np.zeros(len(vals)),
            color=STATE_COLORS[state], alpha=0.7, label=state,
        )
        bottom = vals if bottom is None else bottom + vals

    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Fraction of bars", fontsize=10)
    ax1.set_title("State composition over time (monthly)", fontsize=11)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.2)

    # Pie chart
    counts = [int((df["market_state"] == s).sum()) for s in ALL_STATES]
    colors = [STATE_COLORS[s] for s in ALL_STATES]
    wedges, texts, autotexts = ax2.pie(
        counts, labels=ALL_STATES, colors=colors,
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax2.set_title("Overall distribution", fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  State distribution chart → {Path(out_path).name}")
