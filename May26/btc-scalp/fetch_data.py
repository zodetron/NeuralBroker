"""
fetch_data.py — Phase 2: fetch and cache BTC/USDT 15M + 1H data from Binance.
Run once; subsequent runs load from parquet cache.

Usage:
    venv/bin/python3 fetch_data.py
    venv/bin/python3 fetch_data.py --refresh   # force re-fetch
"""

import sys
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import argparse
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from data.pipeline import load_15m, load_1h
from config import DATA_CFG, BT

def estimate_signal_frequency(df_15m: pd.DataFrame) -> None:
    """
    Quick estimate of raw signal frequency for each strategy before filters.
    Uses approximate conditions — exact computation is in Phase 4.
    """
    print("\n" + "═"*62)
    print("  SIGNAL FREQUENCY ESTIMATE  (pre-filter, approximate)")
    print("═"*62)

    c    = df_15m["close"]
    vol  = df_15m["volume"]
    n    = len(df_15m)
    bars_per_day = 96   # 24h × 4 bars/hr

    # ── Simple EMA + ADX approximation ───────────────────────────────────────
    ema20 = c.ewm(span=20, adjust=False).mean()
    above_ema = (c > ema20)

    # ATR-based ADX proxy: rolling std of % returns as a rough trending proxy
    log_ret  = np.log(c / c.shift(1))
    atr_raw  = df_15m["high"] - df_15m["low"]
    atr20    = atr_raw.rolling(20).mean()
    atr_avg  = atr20.rolling(20).mean()

    # Market state proxies
    ema_slope = ema20.diff(5)
    trending_up   = above_ema & (ema_slope > 0)
    trending_down = (~above_ema) & (ema_slope < 0)
    atr_ratio     = atr20 / (atr_avg + 1e-9)
    # Use rolling std of price changes as ADX proxy
    vol20     = log_ret.rolling(14).std() * np.sqrt(14)
    high_vol  = atr_ratio > 1.5
    low_vol   = atr_ratio < 1.0

    chop      = (~trending_up) & (~trending_down)
    high_chop = chop & high_vol
    low_range = chop & low_vol

    state_counts = {
        "TRENDING_UP":     int(trending_up.sum()),
        "TRENDING_DOWN":   int(trending_down.sum()),
        "HIGH_VOL_CHOP":   int(high_chop.sum()),
        "LOW_VOL_RANGE":   int(low_range.sum()),
    }

    total_labeled = sum(state_counts.values())
    print(f"\n  Market state distribution (~{n:,} bars total):")
    for state, cnt in state_counts.items():
        pct = cnt / n * 100
        print(f"    {state:<20} {cnt:>8,}  ({pct:>5.1f}%)")

    # ── Strategy A: breakout signals ─────────────────────────────────────────
    roll_high10 = c.shift(1).rolling(10).max()
    roll_low10  = c.shift(1).rolling(10).min()
    vol_ma20    = vol.rolling(20).mean()
    breakout_L  = (c > roll_high10) & (vol > 2.0 * vol_ma20)
    breakout_S  = (c < roll_low10)  & (vol > 2.0 * vol_ma20)
    strat_a_raw = breakout_L | breakout_S
    strat_a_n   = int(strat_a_raw.sum())

    # ── Strategy B: VWAP reversion proxy (using EMA as VWAP proxy) ───────────
    # Rough proxy: price deviation from EMA20 > 0.35%
    dev_pct = (c - ema20) / (ema20 + 1e-9) * 100
    from ta.momentum import RSIIndicator
    rsi14 = RSIIndicator(c, window=14).rsi()
    vwap_long  = (dev_pct < -0.35) & (rsi14 < 42) & low_range
    vwap_short = (dev_pct >  0.35) & (rsi14 > 58) & low_range
    strat_b_raw = (vwap_long | vwap_short).sum()

    # ── Strategy C: EMA pullback ──────────────────────────────────────────────
    # Touch EMA20 in trending state + RSI 38-52
    touch_ema = (c - ema20).abs() / (ema20 + 1e-9) * 100 < 0.15
    rsi_pb    = (rsi14 >= 38) & (rsi14 <= 52)
    strat_c_raw = int((touch_ema & rsi_pb & (trending_up | trending_down)).sum())

    print(f"\n  Raw signal estimates (pre ML-filter, ~0.54 threshold ≈ keeps 60%):")
    print(f"  {'Strategy':<28} {'Raw signals':>12} {'Per day':>9} {'Est. filtered':>14}")
    print(f"  {'─'*64}")

    n_days    = (df_15m.index[-1] - df_15m.index[0]).days
    for name, raw in [
        ("A — Momentum Burst", strat_a_n),
        ("B — VWAP Reversion", int(strat_b_raw)),
        ("C — EMA Pullback",   strat_c_raw),
    ]:
        per_day  = raw / n_days
        filtered = raw * 0.60
        print(f"  {name:<28} {raw:>12,} {per_day:>8.1f}/d {filtered:>13,.0f}")

    total_raw = strat_a_n + int(strat_b_raw) + strat_c_raw
    print(f"  {'─'*64}")
    print(f"  {'TOTAL (all strategies)':<28} {total_raw:>12,} "
          f"{total_raw/n_days:>8.1f}/d {total_raw*0.60:>13,.0f}")

    print(f"\n  Note: estimates use simplified conditions. Phase 4 adds full "
          f"ADX,\n  exact VWAP, consecutive-bar checks and 1H alignment filters.")
    print(f"  Expected final signals after all filters: "
          f"~{total_raw*0.60*0.5/n_days:.1f} – {total_raw*0.60/n_days:.1f} per day")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-fetch from Binance (ignores cache)")
    args = parser.parse_args()

    print("\n" + "═"*62)
    print("  BTC SCALP — PHASE 2: DATA")
    print("═"*62)
    print(f"  Fetching: {DATA_CFG['symbol_15m']} "
          f"{DATA_CFG['tf_15m']} + {DATA_CFG['tf_1h']}  |  "
          f"from {DATA_CFG['start']}  |  exchange: Binance")
    print(f"  Cache dir: {DATA_CFG['parquet_15m'].parent}\n")

    # ── Fetch / load ──────────────────────────────────────────────────────────
    df_15m = load_15m(force_refresh=args.refresh)
    df_1h  = load_1h(force_refresh=args.refresh)

    # ── Pre-flight check ──────────────────────────────────────────────────────
    overlap_start = max(df_15m.index[0], df_1h.index[0])
    overlap_end   = min(df_15m.index[-1], df_1h.index[-1])
    print(f"\n  Overlapping range for both timeframes:")
    print(f"    {overlap_start.date()}  →  {overlap_end.date()}")

    # ── Signal frequency estimate ─────────────────────────────────────────────
    estimate_signal_frequency(df_15m)

    print("\n" + "═"*62)
    print("  PHASE 2 COMPLETE")
    print("═"*62)
    print(f"  15M parquet : {DATA_CFG['parquet_15m'].stat().st_size / 1e6:.1f} MB")
    print(f"  1H  parquet : {DATA_CFG['parquet_1h'].stat().st_size / 1e6:.1f} MB")
    print(f"\n  Ready for Phase 3 — Market State Detector.")


if __name__ == "__main__":
    main()
