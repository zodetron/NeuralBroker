"""
run_phase5.py — Phase 5: ML filter (separate XGBoost per strategy).

Updated parameters vs Phase 4:
  Strategy B: vwap_band_pct 0.35% → 0.50%
  Strategy C: ema_touch_pct 0.10% → 0.30%
  ML: tiered thresholds A=0.56 / B=0.51 / C=0.52 (separate models)

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
    df:    pd.DataFrame,
) -> None:
    """Final summary: before/after filter counts and daily rates."""
    n_days = (df.index[-1] - df.index[0]).days

    targets = {"A": (0.4, 0.6), "B": (1.5, 2.5), "C": (0.3, 0.5)}

    print(f"\n  {'═'*70}")
    print(f"  PHASE 5 SUMMARY — ML FILTER RESULTS")
    print(f"  {'═'*70}")
    print(f"\n  {'Strategy':<28} {'Raw':>7} {'OOS':>7} {'Passed':>8} "
          f"{'Raw/d':>7} {'Post/d':>8} {'Target':>14} {'OK?':>5}")
    print(f"  {'─'*70}")

    combined_post = 0
    for label, sig, key in [
        ("A — Momentum Burst",  sig_a, "A"),
        ("B — VWAP Reversion",  sig_b, "B"),
        ("C — EMA Pullback",    sig_c, "C"),
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
        print(f"  {label:<28} {n_raw:>7,} {n_oos:>7,} {n_pass:>8,} "
              f"{raw_pd:>7.2f} {post_pd:>8.2f} {tgt_str:>14} {mark}")

    total_post_pd = combined_post / n_days
    print(f"  {'─'*70}")
    print(f"  {'COMBINED':<28} {'':>7} {'':>7} {combined_post:>8,} "
          f"{'':>7} {total_post_pd:>8.2f} {'2.5–3.5/d':>14}")

    lo_tot, hi_tot = 2.5, 3.5
    overall_ok = lo_tot <= total_post_pd <= hi_tot
    status = f"{GREEN}✓ In target range{RESET}" if overall_ok else (
             f"{YELLOW}⚠ Outside target — see note{RESET}")
    print(f"\n  Combined: {total_post_pd:.2f} signals/day  →  {status}")
    interval = 24 / total_post_pd if total_post_pd > 0 else float("inf")
    print(f"  Average interval: 1 signal every {interval:.1f} hours")


def _save_filtered_signals(
    sig_a: pd.DataFrame,
    sig_b: pd.DataFrame,
    sig_c: pd.DataFrame,
) -> None:
    """Save ML-filtered signal files."""
    data_dir = DATA_CFG["parquet_15m"].parent
    for label, sig in [("A", sig_a), ("B", sig_b), ("C", sig_c)]:
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
          f"(train {ML_PER_STRAT['B']['train_weeks']}w)  [band 0.50%]")
    print(f"  Strategy C threshold : {ML_PER_STRAT['C']['threshold']}  "
          f"(train {ML_PER_STRAT['C']['train_weeks']}w)  [touch ±0.30%]")

    # ── Load & classify ───────────────────────────────────────────────────────
    print("\n  Loading data and classifying market states …")
    df_15m = load_15m()
    df_1h  = load_1h()
    df     = classify_market_state(df_15m, df_1h)

    # Add VWAP to classified df (needed for Strategy B label construction)
    df["vwap"] = _compute_vwap(df)
    print(f"  Classified + VWAP computed  ({len(df):,} bars)")

    # ── Generate signals (updated params) ────────────────────────────────────
    print("\n" + "─"*70)
    print("  Generating signals with updated parameters …")

    t1    = time.time()
    sig_a = gen_a(df)
    sig_b = gen_b(df)
    sig_c = gen_c(df, df_1h)
    print(f"  A: {len(sig_a):,}  |  B: {len(sig_b):,}  |  C: {len(sig_c):,}  "
          f"  ({time.time()-t1:.1f}s)")

    print("\n  Updated signal counts vs Phase 4:")
    print(f"    B: {len(sig_b):,} (was 9,683 at 0.35% band — now 0.50%)")
    print(f"    C: {len(sig_c):,} (was 40 at 0.10% touch — now 0.30%)")

    # ── Per-strategy stats ────────────────────────────────────────────────────
    print("\n" + "─"*70)
    print("  Raw signal statistics:")
    stats_a(sig_a, df)
    stats_b(sig_b, df)
    stats_c(sig_c, df)

    # ── ML Walk-Forward — one model per strategy ──────────────────────────────
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
    print(f"  Strategy C WFO done  ({time.time()-t1:.1f}s)")

    # ── Save files ────────────────────────────────────────────────────────────
    print("\n" + "─"*70)
    print("  Saving signal files …")
    _save_filtered_signals(sig_a, sig_b, sig_c)

    # ── Final summary ─────────────────────────────────────────────────────────
    _print_phase5_summary(sig_a, sig_b, sig_c, df)

    elapsed = time.time() - t_start
    print(f"\n{'═'*70}")
    print(f"  PHASE 5 COMPLETE  ({elapsed:.0f}s)")
    print(f"{'═'*70}")
    print(f"  Filtered CSVs → data/signals_[A|B|C]_filtered.csv")
    print(f"  Feature charts → results/strategy_[a|b|c]_feature_importance.png\n")


if __name__ == "__main__":
    main()
