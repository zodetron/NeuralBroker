"""
run_phase5.py — Phase 5: ML filter (separate XGBoost per strategy).

Changes vs previous run:
  Strategy A: threshold 0.56 → 0.54
  Strategy B: min_samples 120 → 60, SL 0.5×ATR → 1.0×ATR, TP = 50% to VWAP
  Strategy C: ML skipped entirely (all signals pass)
  Strategy D: NEW — Manipulation Fade (ICT Fib 2.5/4.0), XGBoost WFO

Usage:
    venv/bin/python3 run_phase5.py
"""

import sys
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import time
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from data.pipeline import load_15m, load_1h
from strategy.market_state import classify_market_state
from strategy.signals_a import generate_signals as gen_a, signal_stats as stats_a
from strategy.signals_b import generate_signals as gen_b, signal_stats as stats_b
from strategy.signals_c import generate_signals as gen_c, signal_stats as stats_c
from strategy.signals_d import generate_signals as gen_d, signal_stats as stats_d
from strategy.signals_b import _compute_vwap
from models.ml_filter import run_walk_forward
from config import DATA_CFG, RESULTS, ML_PER_STRAT


GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def _print_phase5_summary(
    sig_a: pd.DataFrame,
    sig_b: pd.DataFrame,
    sig_c: pd.DataFrame,
    sig_d: pd.DataFrame,
    df:    pd.DataFrame,
) -> None:
    n_days = (df.index[-1] - df.index[0]).days

    targets = {"A": (0.4, 0.6), "B": (1.5, 2.5), "C": (0.3, 0.5), "D": (0.3, 0.6)}

    print(f"\n  {'═'*76}")
    print(f"  PHASE 5 SUMMARY — ML FILTER RESULTS")
    print(f"  {'═'*76}")
    print(f"\n  {'Strategy':<30} {'Raw':>7} {'OOS':>7} {'Passed':>8} "
          f"{'Raw/d':>7} {'Post/d':>8} {'Target':>14} {'OK?':>5}")
    print(f"  {'─'*76}")

    combined_post = 0
    for label, sig, key in [
        ("A — Momentum Burst",        sig_a, "A"),
        ("B — VWAP Reversion",        sig_b, "B"),
        ("C — EMA Pullback (no ML)",  sig_c, "C"),
        ("D — Manipulation Fade",     sig_d, "D"),
    ]:
        n_raw   = len(sig)
        n_oos   = int(sig["ml_confidence"].notna().sum()) if not sig.empty else 0
        n_pass  = int(sig["ml_signal"].sum()) if not sig.empty else 0
        raw_pd  = n_raw  / n_days
        post_pd = n_pass / n_days
        lo, hi  = targets[key]
        ok      = lo <= post_pd <= hi
        mark    = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
        tgt_str = f"{lo:.1f}–{hi:.1f}/d"
        combined_post += n_pass
        print(f"  {label:<30} {n_raw:>7,} {n_oos:>7,} {n_pass:>8,} "
              f"{raw_pd:>7.2f} {post_pd:>8.2f} {tgt_str:>14} {mark}")

    total_post_pd = combined_post / n_days
    print(f"  {'─'*76}")
    print(f"  {'COMBINED':<30} {'':>7} {'':>7} {combined_post:>8,} "
          f"{'':>7} {total_post_pd:>8.2f} {'2.5–4.0/d':>14}")

    lo_tot, hi_tot = 2.5, 4.0
    overall_ok = lo_tot <= total_post_pd <= hi_tot
    status = f"{GREEN}✓ In target range{RESET}" if overall_ok else (
             f"{YELLOW}⚠ Outside target{RESET}")
    print(f"\n  Combined: {total_post_pd:.2f} signals/day  →  {status}")
    interval = 24 / total_post_pd if total_post_pd > 0 else float("inf")
    print(f"  Average interval: 1 signal every {interval:.1f} hours")

    if not overall_ok:
        # Find and print the single biggest bottleneck
        rates = {
            "A": (len(sig_a), int(sig_a["ml_signal"].sum()) if not sig_a.empty else 0, targets["A"]),
            "B": (len(sig_b), int(sig_b["ml_signal"].sum()) if not sig_b.empty else 0, targets["B"]),
            "C": (len(sig_c), int(sig_c["ml_signal"].sum()) if not sig_c.empty else 0, targets["C"]),
            "D": (len(sig_d), int(sig_d["ml_signal"].sum()) if not sig_d.empty else 0, targets["D"]),
        }
        bottleneck = None
        max_gap    = 0.0
        for k, (raw, passed, (lo, hi)) in rates.items():
            target_mid = (lo + hi) / 2
            actual     = passed / n_days
            gap        = target_mid - actual
            if gap > max_gap:
                max_gap    = gap
                bottleneck = k
        if bottleneck:
            k = bottleneck
            raw, passed, (lo, hi) = rates[k]
            print(f"\n  Biggest bottleneck: Strategy {k}  "
                  f"({passed/n_days:.2f}/d actual vs {lo:.1f}–{hi:.1f}/d target)")


def _save_filtered_signals(
    sig_a: pd.DataFrame,
    sig_b: pd.DataFrame,
    sig_c: pd.DataFrame,
    sig_d: pd.DataFrame,
) -> None:
    data_dir = DATA_CFG["parquet_15m"].parent
    for label, sig in [("A", sig_a), ("B", sig_b), ("C", sig_c), ("D", sig_d)]:
        out_raw = data_dir / f"signals_{label}_raw.csv"
        out_fil = data_dir / f"signals_{label}_filtered.csv"
        if not sig.empty:
            sig.to_csv(out_raw)
            filtered = sig[sig["ml_signal"] == 1]
            filtered.to_csv(out_fil)
            print(f"  signals_{label}_raw.csv      → {len(sig):,} rows")
            print(f"  signals_{label}_filtered.csv → {len(filtered):,} rows")
        else:
            print(f"  signals_{label}: empty — no files written")


def main():
    t_start = time.time()

    print("\n" + "═"*70)
    print("  BTC SCALP — PHASE 5: ML FILTER (PER-STRATEGY XGBoost)")
    print("═"*70)
    print(f"  Strategy A threshold : {ML_PER_STRAT['A']['threshold']}  "
          f"(train {ML_PER_STRAT['A']['train_weeks']}w)")
    print(f"  Strategy B threshold : {ML_PER_STRAT['B']['threshold']}  "
          f"(train {ML_PER_STRAT['B']['train_weeks']}w, min_samples={ML_PER_STRAT['B']['min_samples']})")
    print(f"  Strategy C           : ML SKIPPED (all signals pass)")
    print(f"  Strategy D threshold : {ML_PER_STRAT['D']['threshold']}  "
          f"(train {ML_PER_STRAT['D']['train_weeks']}w, HC threshold=0.50)")

    # ── Load & classify ───────────────────────────────────────────────────────
    print("\n  Loading data and classifying market states …")
    df_15m = load_15m()
    df_1h  = load_1h()
    df     = classify_market_state(df_15m, df_1h)

    df["vwap"] = _compute_vwap(df)
    print(f"  Classified + VWAP computed  ({len(df):,} bars)")

    # ── Generate signals ──────────────────────────────────────────────────────
    print("\n" + "─"*70)
    print("  Generating signals …")

    t1    = time.time()
    sig_a = gen_a(df)
    sig_b = gen_b(df)
    sig_c = gen_c(df, df_1h)
    sig_d = gen_d(df)
    print(f"  A: {len(sig_a):,}  |  B: {len(sig_b):,}  |  "
          f"C: {len(sig_c):,}  |  D: {len(sig_d):,}   ({time.time()-t1:.1f}s)")

    # ── Per-strategy stats ────────────────────────────────────────────────────
    print("\n" + "─"*70)
    print("  Raw signal statistics:")
    stats_a(sig_a, df)
    stats_b(sig_b, df)
    stats_c(sig_c, df)
    stats_d(sig_d, df)

    # ── ML Walk-Forward ───────────────────────────────────────────────────────
    print("\n" + "═"*70)
    print("  ML WALK-FORWARD  (separate XGBoost per strategy)")
    print("═"*70)

    t1    = time.time()
    sig_a = run_walk_forward(sig_a, df, "A")
    print(f"  Strategy A WFO done  ({time.time()-t1:.1f}s)")

    t1    = time.time()
    sig_b = run_walk_forward(sig_b, df, "B")
    print(f"  Strategy B WFO done  ({time.time()-t1:.1f}s)")

    t1    = time.time()
    sig_c = run_walk_forward(sig_c, df, "C")
    print(f"  Strategy C (skip ML) done  ({time.time()-t1:.1f}s)")

    t1    = time.time()
    sig_d = run_walk_forward(sig_d, df, "D")
    print(f"  Strategy D WFO done  ({time.time()-t1:.1f}s)")

    # ── Save files ────────────────────────────────────────────────────────────
    print("\n" + "─"*70)
    print("  Saving signal files …")
    _save_filtered_signals(sig_a, sig_b, sig_c, sig_d)

    # ── Final summary ─────────────────────────────────────────────────────────
    _print_phase5_summary(sig_a, sig_b, sig_c, sig_d, df)

    elapsed = time.time() - t_start
    print(f"\n{'═'*70}")
    print(f"  PHASE 5 COMPLETE  ({elapsed:.0f}s)")
    print(f"{'═'*70}")
    print(f"  Filtered CSVs → data/signals_[A|B|C|D]_filtered.csv\n")


if __name__ == "__main__":
    main()
