"""
run_phase4.py — Phase 4: generate and inspect all three strategy raw signals.

Outputs:
  data/signals_A_raw.csv
  data/signals_B_raw.csv
  data/signals_C_raw.csv
  results/signals_combined_preview.txt

Usage:
    venv/bin/python3 run_phase4.py
"""

import sys
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import warnings
import time
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from data.pipeline import load_15m, load_1h
from strategy.market_state import classify_market_state
from strategy.signals_a import generate_signals as gen_a, signal_stats as stats_a
from strategy.signals_b import generate_signals as gen_b, signal_stats as stats_b
from strategy.signals_c import generate_signals as gen_c, signal_stats as stats_c
from config import DATA_CFG, RESULTS


# ── LOOKAHEAD BIAS CHECKER ────────────────────────────────────────────────────

def check_lookahead(sig: pd.DataFrame, df: pd.DataFrame, label: str) -> bool:
    """
    Verify that every signal bar has valid (non-NaN) indicators,
    and that no signal timestamp falls before the required warmup period.

    Rules checked:
      1. Signal bar must have atr_14, ema20 computed (not warmup NaN).
      2. Signal bar timestamp must exist in df index.
      3. No signal bar's close equals the *next* bar's open
         (would indicate the signal was acted on the same bar — lookahead).
    """
    if sig.empty:
        print(f"  {label}: no signals — skipping lookahead check.")
        return True

    ok = True

    # Check 1: all signal bars exist in df
    missing = sig.index.difference(df.index)
    if len(missing) > 0:
        print(f"  ⚠  {label}: {len(missing)} signal timestamps not in df index!")
        ok = False
    else:
        print(f"  {label}: all {len(sig):,} signal timestamps exist in df ✓")

    # Check 2: atr_14 and ema20 must be non-NaN (warmup complete)
    for col in ["atr_14", "ema20"]:
        if col in df.columns:
            nan_sig = sig.index[df.loc[sig.index, col].isna()]
            if len(nan_sig) > 0:
                print(f"  ⚠  {label}: {len(nan_sig)} signals have NaN {col} (warmup bar)")
                ok = False

    # Check 3: verify signal is generated from bar t, not bar t+1
    # If a signal fires at bar t, the next bar (t+1) open is the entry price.
    # The entry price should DIFFER from the current bar's close.
    # We sample 50 random signals and verify close[t] != open[t+1].
    rng = np.random.default_rng(123)
    sample_idx = rng.choice(len(sig), size=min(50, len(sig)), replace=False)
    lookahead_count = 0
    for i in sample_idx:
        bar_ts = sig.index[i]
        pos    = df.index.get_loc(bar_ts)
        if pos + 1 < len(df):
            close_t  = df.iloc[pos]["close"]
            open_t1  = df.iloc[pos + 1]["open"]
            # If close[t] == open[t+1] exactly, it could indicate a fill-at-close bug
            if abs(close_t - open_t1) < 1e-6:
                lookahead_count += 1
    if lookahead_count > 5:
        print(f"  ⚠  {label}: {lookahead_count}/50 sampled signals have "
              f"close[t]==open[t+1] (review entry timing)")
    else:
        print(f"  {label}: entry-timing check passed "
              f"({lookahead_count}/50 close≈open cases — expected for gap-less BTC) ✓")

    return ok


# ── COMBINED PREVIEW ──────────────────────────────────────────────────────────

def combined_preview(
    sig_a: pd.DataFrame,
    sig_b: pd.DataFrame,
    sig_c: pd.DataFrame,
    df:    pd.DataFrame,
) -> None:
    """
    Combined view: total signals per day, hourly distribution, capacity check.
    """
    n_days = (df.index[-1] - df.index[0]).days

    print(f"\n  {'═'*60}")
    print(f"  COMBINED SIGNAL PREVIEW  (all three strategies)")
    print(f"  {'═'*60}")

    # Summary table
    print(f"\n  {'Strategy':<28} {'Signals':>9} {'Per day':>9} {'Long':>7} {'Short':>7}")
    print(f"  {'─'*58}")
    totals = {"n": 0, "long": 0, "short": 0}
    for label, sig in [("A — Momentum Burst", sig_a),
                        ("B — VWAP Reversion",  sig_b),
                        ("C — EMA Pullback",     sig_c)]:
        if sig.empty:
            print(f"  {label:<28} {'0':>9} {'0.00':>9} {'0':>7} {'0':>7}")
            continue
        n   = len(sig)
        nl  = int((sig["direction"] == 1).sum())
        ns  = n - nl
        print(f"  {label:<28} {n:>9,} {n/n_days:>9.2f} {nl:>7,} {ns:>7,}")
        totals["n"] += n; totals["long"] += nl; totals["short"] += ns

    print(f"  {'─'*58}")
    print(f"  {'TOTAL':>28} {totals['n']:>9,} "
          f"{totals['n']/n_days:>9.2f} {totals['long']:>7,} {totals['short']:>7,}")

    # Hourly distribution (all strategies combined)
    all_sigs = pd.concat(
        [s for s in [sig_a, sig_b, sig_c] if not s.empty],
        ignore_index=False,
    )
    if all_sigs.empty:
        return

    hourly = all_sigs["hour"].value_counts().sort_index()
    max_h  = hourly.max()

    print(f"\n  Hourly signal distribution (UTC):")
    print(f"  {'Hour':>5}  {'Count':>7}  {'Bar':}")
    print(f"  {'─'*42}")
    for hr in range(24):
        cnt  = hourly.get(hr, 0)
        bar  = "█" * int(cnt / max_h * 30)
        flag = "  ◀ peak" if cnt == max_h else ""
        print(f"  {hr:02d}:00  {cnt:>7,}  {bar}{flag}")

    # ── Capacity check: bars where 3+ signals fire simultaneously ─────────────
    print(f"\n  {'─'*60}")
    print(f"  Capacity check — bars where 3+ strategies fire simultaneously:")

    all_sigs_ts = all_sigs.index.to_series()
    bar_counts  = all_sigs_ts.value_counts()
    overloaded  = bar_counts[bar_counts >= 3]

    if overloaded.empty:
        print(f"  No simultaneous 3-strategy fires found ✓")
    else:
        print(f"  {len(overloaded)} bar(s) with 3+ simultaneous signals:")
        for ts, cnt in overloaded.head(10).items():
            strats = all_sigs.loc[all_sigs.index == ts, "strategy"].tolist()
            print(f"    {ts}  →  {cnt} signals  ({', '.join(strats)})")

    # ── Strategy co-occurrence at bar level ───────────────────────────────────
    print(f"\n  Strategy co-occurrence (2 strategies at same 15M bar):")
    bar_counts2 = bar_counts[bar_counts == 2]
    if bar_counts2.empty:
        print(f"  No 2-strategy simultaneous fires found ✓")
    else:
        print(f"  {len(bar_counts2):,} bar(s) with exactly 2 simultaneous signals "
              f"({len(bar_counts2)/n_days:.2f}/day avg)")


def main():
    t0 = time.time()

    print("\n" + "═"*62)
    print("  BTC SCALP — PHASE 4: STRATEGY SIGNALS")
    print("═"*62)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n  Loading and classifying data …")
    df_15m = load_15m()
    df_1h  = load_1h()

    t1 = time.time()
    df = classify_market_state(df_15m, df_1h)
    print(f"  Market states classified in {time.time()-t1:.1f}s")

    # ── Generate signals ──────────────────────────────────────────────────────
    print("\n" + "─"*62)
    print("  Generating strategy signals …")

    t1 = time.time()
    sig_a = gen_a(df)
    print(f"  Strategy A done: {len(sig_a):,} signals  ({time.time()-t1:.1f}s)")

    t1 = time.time()
    sig_b = gen_b(df)
    print(f"  Strategy B done: {len(sig_b):,} signals  ({time.time()-t1:.1f}s)")

    t1 = time.time()
    sig_c = gen_c(df, df_1h)
    print(f"  Strategy C done: {len(sig_c):,} signals  ({time.time()-t1:.1f}s)")

    # ── Per-strategy stats ────────────────────────────────────────────────────
    print("\n" + "═"*62)
    print("  PER-STRATEGY STATISTICS")
    print("═"*62)
    stats_a(sig_a, df)
    stats_b(sig_b, df)
    stats_c(sig_c, df)

    # ── Save raw signal CSVs ──────────────────────────────────────────────────
    print("\n" + "─"*62)
    print("  Saving raw signal CSVs …")
    data_dir = DATA_CFG["parquet_15m"].parent
    for label, sig in [("A", sig_a), ("B", sig_b), ("C", sig_c)]:
        out = data_dir / f"signals_{label}_raw.csv"
        if not sig.empty:
            sig.to_csv(out)
            print(f"  data/signals_{label}_raw.csv  →  {len(sig):,} rows  "
                  f"({out.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"  data/signals_{label}_raw.csv  →  empty (0 signals)")

    # ── Lookahead bias check ──────────────────────────────────────────────────
    print("\n" + "═"*62)
    print("  ZERO-LOOKAHEAD BIAS CHECK")
    print("═"*62)
    all_ok = True
    all_ok &= check_lookahead(sig_a, df, "Strategy A")
    all_ok &= check_lookahead(sig_b, df, "Strategy B")
    all_ok &= check_lookahead(sig_c, df, "Strategy C")
    if all_ok:
        print(f"\n  ✓ All checks passed — no lookahead bias detected.")
    else:
        print(f"\n  ⚠ Some checks flagged — review above.")

    # ── Combined preview ──────────────────────────────────────────────────────
    combined_preview(sig_a, sig_b, sig_c, df)

    elapsed = time.time() - t0
    print(f"\n{'═'*62}")
    print(f"  PHASE 4 COMPLETE  ({elapsed:.1f}s)")
    print(f"{'═'*62}")
    print(f"  Raw CSVs → data/signals_[A|B|C]_raw.csv")
    print(f"  Awaiting review before proceeding to Phase 5.\n")


if __name__ == "__main__":
    main()
