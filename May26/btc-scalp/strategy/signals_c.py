"""
strategy/signals_c.py — Strategy C: EMA Pullback

Active in: TRENDING_UP (longs) or TRENDING_DOWN (shorts) only.

Long (TRENDING_UP only):
  - Price within 0.1% of EMA20 (pullback touch zone)
  - RSI14 between 38–52  (cooling, not crashing)
  - At least 2 of the last 3 bars were bearish (close < open)
  - 1H trend MUST be UP (hard requirement — no trade if 1H is down/flat)
  - 1H ADX > 20 (higher-timeframe trend has enough strength)

Short (TRENDING_DOWN only): mirror logic
  - Price within 0.1% of EMA20
  - RSI14 between 48–62
  - At least 2 of the last 3 bars were bullish (close > open)
  - 1H trend MUST be DOWN (hard requirement)
  - 1H ADX > 20

All conditions on bar t; entry on bar t+1 open.
No lookahead: EMA20 already uses only past data; 1H context is forward-filled
from the last completed 1H bar.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import STRAT_C

_TRENDING_UP   = "TRENDING_UP"
_TRENDING_DOWN = "TRENDING_DOWN"


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=window - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=window - 1, adjust=False).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - 100 / (1 + rs)


def _adx(df: pd.DataFrame, period: int) -> pd.Series:
    """Wilder ADX — same implementation as market_state.py."""
    h, l = df["high"], df["low"]
    tr = pd.concat([
        h - l,
        (h - df["close"].shift(1)).abs(),
        (l - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)

    up   = (h - h.shift(1)).clip(lower=0)
    down = (l.shift(1) - l).clip(lower=0)
    pdm  = pd.Series(np.where(up > down, up, 0.0), index=df.index)
    ndm  = pd.Series(np.where(down > up, down, 0.0), index=df.index)

    com = period - 1
    tr14   = tr.ewm(com=com, adjust=False).mean()
    pdi    = 100 * pdm.ewm(com=com, adjust=False).mean() / (tr14 + 1e-9)
    ndi    = 100 * ndm.ewm(com=com, adjust=False).mean() / (tr14 + 1e-9)
    dx     = 100 * (pdi - ndi).abs() / (pdi + ndi + 1e-9)
    return dx.ewm(com=com, adjust=False).mean()


# ── MAIN SIGNAL GENERATOR ─────────────────────────────────────────────────────

def generate_signals(df: pd.DataFrame, df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Generate raw EMA Pullback signals on the classified 15M DataFrame.

    Parameters
    ----------
    df     : output of classify_market_state (15M bars).
    df_1h  : raw 1H OHLCV DataFrame (used to compute 1H ADX context).

    Returns
    -------
    pd.DataFrame indexed by bar timestamp.
    Columns: signal_id, strategy, direction, market_state, close, atr_14,
             ema20, ema_dist_pct, rsi14, adx_1h, hour, trend_1h, align_1h,
             conf_bonus.
    """
    d = df.copy()

    ema_window  = STRAT_C["ema_period"]   # 20
    rsi_window  = STRAT_C["rsi_period"]    # 14
    touch_pct   = STRAT_C["ema_touch_pct"] # ±0.3% of EMA20 (widened)
    rsi_lo_L    = STRAT_C["rsi_pb_lo"]    # 38
    rsi_hi_L    = STRAT_C["rsi_pb_hi"]    # 52
    rsi_lo_S    = 48                      # short RSI range (48–62 per spec)
    rsi_hi_S    = 62
    red_req     = STRAT_C["red_bars_required"]   # 2

    # ── 1H ADX (merged from 1H bars) ─────────────────────────────────────────
    h1 = df_1h.copy()
    h1["adx_1h"] = _adx(h1, 14)
    adx_1h = h1["adx_1h"].reindex(d.index, method="ffill")
    d["adx_1h"] = adx_1h.values

    # ── RSI14 on 15M ─────────────────────────────────────────────────────────
    d["rsi14"] = _rsi(d["close"], rsi_window)

    # ── EMA distance (use ema20 computed in classify_market_state) ────────────
    # ema20 is already in df from Phase 3 classification
    d["ema_dist_pct"] = (d["close"] - d["ema20"]) / (d["ema20"] + 1e-9) * 100

    # EMA touch zone: |dist| ≤ 0.1%
    touches_ema = d["ema_dist_pct"].abs() <= touch_pct

    # ── Candle color for the bar-pattern filter ────────────────────────────────
    is_red   = (d["close"] < d["open"]).astype(int)
    is_green = (d["close"] > d["open"]).astype(int)

    # "2 of last 3 bars" = bars [t-1, t-2, t-3]
    prev_red_sum   = is_red.shift(1)   + is_red.shift(2)   + is_red.shift(3)
    prev_green_sum = is_green.shift(1) + is_green.shift(2) + is_green.shift(3)

    # ── Long conditions (TRENDING_UP) ─────────────────────────────────────────
    long_mask = (
        (d["market_state"] == _TRENDING_UP)
        & touches_ema
        & (d["ema_dist_pct"] >= -touch_pct)    # touching from above (≥ −0.1%)
        & (d["rsi14"] >= rsi_lo_L)
        & (d["rsi14"] <= rsi_hi_L)
        & (prev_red_sum >= red_req)
        & (d["trend_1h"] == 1)                  # hard 1H up requirement
        & (d["adx_1h"] > 20)
    )

    # ── Short conditions (TRENDING_DOWN) ──────────────────────────────────────
    short_mask = (
        (d["market_state"] == _TRENDING_DOWN)
        & touches_ema
        & (d["ema_dist_pct"] <= touch_pct)     # touching from below (≤ +0.1%)
        & (d["rsi14"] >= rsi_lo_S)
        & (d["rsi14"] <= rsi_hi_S)
        & (prev_green_sum >= red_req)
        & (d["trend_1h"] == -1)                 # hard 1H down requirement
        & (d["adx_1h"] > 20)
    )

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
    sig_df["ema20"]        = d.loc[any_signal, "ema20"].values
    sig_df["ema_dist_pct"] = d.loc[any_signal, "ema_dist_pct"].values
    sig_df["rsi14"]        = d.loc[any_signal, "rsi14"].values
    sig_df["adx_1h"]       = d.loc[any_signal, "adx_1h"].values

    # Strategy C always aligns with 1H trend (it's a hard requirement)
    sig_df["align_1h"]   = np.int8(1)
    sig_df["conf_bonus"] = 0.0

    sig_df.insert(0, "strategy",  "C")
    sig_df.insert(0, "signal_id", [f"C{n:04d}" for n in range(1, len(sig_df) + 1)])
    sig_df.index.name = "ts"

    return sig_df


def signal_stats(sig: pd.DataFrame, df: pd.DataFrame) -> None:
    """Print statistics for Strategy C signals."""
    if sig.empty:
        print("  Strategy C: no signals generated.")
        return

    n      = len(sig)
    n_long = (sig["direction"] == 1).sum()
    n_days = (df.index[-1] - df.index[0]).days
    avg_pd = n / n_days

    print(f"\n  {'─'*56}")
    print(f"  Strategy C — EMA Pullback")
    print(f"  {'─'*56}")
    print(f"  Total signals    : {n:>7,}")
    print(f"  Long / Short     : {n_long:>7,} / {n - n_long:,}")
    print(f"  Avg per day      : {avg_pd:>7.2f}")
    print(f"  Avg EMA dist%    : {sig['ema_dist_pct'].abs().mean():>7.4f}%")
    print(f"  Avg RSI14        : {sig['rsi14'].mean():>7.2f}")
    print(f"  Avg 1H ADX       : {sig['adx_1h'].mean():>7.2f}")
    print(f"  1H aligned       : 100% (hard requirement)")

    states = sig["market_state"].value_counts()
    print(f"\n  Market state breakdown (should be only TRENDING_UP/DOWN):")
    for s, cnt in states.items():
        print(f"    {s:<22} {cnt:>6,}  ({cnt/n*100:.1f}%)")

    hour_counts = sig["hour"].value_counts().sort_values(ascending=False)
    print(f"\n  Top 5 hours (UTC):")
    for hr, cnt in hour_counts.head(5).items():
        bar_chart = "█" * int(cnt / hour_counts.iloc[0] * 20)
        print(f"    {hr:02d}:00  {cnt:>5,}  {bar_chart}")
