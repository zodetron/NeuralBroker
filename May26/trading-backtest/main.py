"""
main.py — Entry point. Run phases sequentially or individually.

Usage:
    venv/bin/python3 main.py            # run all phases
    venv/bin/python3 main.py --phase 2  # run a specific phase
"""

import sys
import argparse
from pathlib import Path

# Ensure project root is on sys.path regardless of working directory
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


# ── PHASE 1 — Project Structure ───────────────────────────────────────────────
def phase1():
    print("\n" + "═"*60)
    print("  PHASE 1 — Project Setup")
    print("═"*60)

    folders = ["data", "features", "models", "strategy", "backtest", "results"]
    for folder in folders:
        path = ROOT / folder
        path.mkdir(exist_ok=True)
        print(f"  ✓  {folder}/")

    print(f"\n  venv  : {ROOT / 'venv'}")

    # Verify all key packages are importable
    packages = [
        ("pandas",       "pd"),
        ("numpy",        "np"),
        ("scipy",        None),
        ("yfinance",     "yf"),
        ("ccxt",         None),
        ("hmmlearn",     None),
        ("xgboost",      None),
        ("sklearn",      None),
        ("matplotlib",   None),
        ("seaborn",      None),
        ("plotly",       None),
        ("ta",           None),
        ("vectorbt",     None),
    ]

    print("\n  Package verification:")
    all_ok = True
    for pkg, alias in packages:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            print(f"    ✓  {pkg:<14} {ver}")
        except ImportError as e:
            print(f"    ✗  {pkg:<14} NOT FOUND — {e}")
            all_ok = False

    status = "all packages verified" if all_ok else "some packages missing — check output above"
    print(f"\n✅ Phase 1 complete — project structure created, venv ready ({status})")


# ── PHASE 2 — Data Pipeline ───────────────────────────────────────────────────
def phase2():
    print("\n" + "═"*60)
    print("  PHASE 2 — Data Pipeline")
    print("═"*60)

    from data.pipeline import load_btc, load_xau

    print("\n[2.1] Loading BTC/USD …")
    btc = load_btc()

    print("\n[2.2] Loading XAU/USD …")
    xau = load_xau()

    # Cross-asset date overlap info (useful for Phase 5+)
    overlap_start = max(btc.index[0], xau.index[0])
    overlap_end   = min(btc.index[-1], xau.index[-1])
    overlap_days  = (overlap_end - overlap_start).days
    print(f"\n  Shared date range: {overlap_start.date()} → {overlap_end.date()}")
    print(f"  Overlapping days : {overlap_days:,}")

    print(f"\n✅ Phase 2 complete — BTC ({len(btc):,} rows) and "
          f"XAU ({len(xau):,} rows) loaded and cached to /data")

    return btc, xau


# ── PHASE 3 — Feature Engineering ────────────────────────────────────────────
def phase3(btc=None, xau=None):
    print("\n" + "═"*60)
    print("  PHASE 3 — Feature Engineering")
    print("═"*60)

    if btc is None or xau is None:
        btc, xau = phase2()

    from features.engineer import build_features, FEATURE_COLS

    print(f"\n[3.1] BTC features …")
    btc_feat = build_features(btc, "BTC")

    print(f"\n[3.2] XAU features …")
    xau_feat = build_features(xau, "XAU")

    print(f"\n  Feature columns ({len(FEATURE_COLS)}): {', '.join(FEATURE_COLS)}")
    print(f"\n✅ Phase 3 complete — {len(FEATURE_COLS)} features computed for "
          f"BTC and XAU, saved to /data")

    return btc_feat, xau_feat


# ── PHASE 4 — Regime Detection ────────────────────────────────────────────────
def phase4(btc_feat=None, xau_feat=None):
    print("\n" + "═"*60)
    print("  PHASE 4 — Regime Detection (HMM)")
    print("═"*60)

    if btc_feat is None or xau_feat is None:
        btc_feat, xau_feat = phase3()

    from models.regime import detect_regimes

    print("\n[4.1] BTC regime detection …")
    btc_reg = detect_regimes(btc_feat, "BTC")

    print("\n[4.2] XAU regime detection …")
    xau_reg = detect_regimes(xau_feat, "XAU")

    print(f"\n✅ Phase 4 complete — regimes labelled for BTC and XAU, "
          f"charts saved to /results")

    return btc_reg, xau_reg


# ── PHASE 5 — Strategy Logic ──────────────────────────────────────────────────
def phase5(btc_reg=None, xau_reg=None):
    print("\n" + "═"*60)
    print("  PHASE 5 — Strategy Logic")
    print("═"*60)

    if btc_reg is None or xau_reg is None:
        btc_reg, xau_reg = phase4()

    from strategy.signals import compute_signals, print_signal_summary
    from strategy.sizing  import compute_sizing_series, print_sizing_summary
    from config import BT, STRAT

    capital = BT["initial_capital"]

    print(f"\n  Config snapshot:")
    print(f"    allow_shorting    : {STRAT['allow_shorting']}")
    print(f"    risk_per_trade    : {STRAT['risk_per_trade']*100:.0f}%")
    print(f"    atr_stop_mult     : {STRAT['atr_stop_mult']}×")
    print(f"    atr_tp_mult       : {STRAT['atr_tp_mult']}×  →  R:R = "
          f"{STRAT['atr_tp_mult']/STRAT['atr_stop_mult']:.1f}:1")
    print(f"    rsi_oversold      : {STRAT['rsi_oversold']}")
    print(f"    rsi_overbought    : {STRAT['rsi_overbought']}")
    print(f"    momentum_roc_period: {STRAT['momentum_roc_period']}d")

    results = {}
    for key, df_reg in [("BTC", btc_reg), ("XAU", xau_reg)]:
        print(f"\n[5.{'1' if key=='BTC' else '2'}] {key} signals …")
        df_sig  = compute_signals(df_reg)
        df_full = compute_sizing_series(df_sig, capital)
        print_signal_summary(df_full, key)
        print_sizing_summary(df_full, key, capital)
        results[key] = df_full

    print(f"\n✅ Phase 5 complete — signals and position sizes computed for BTC and XAU")
    return results


# ── PHASE 6 — ML Filter ───────────────────────────────────────────────────────
def phase6(strat_results=None):
    print("\n" + "═"*60)
    print("  PHASE 6 — ML Filter (XGBoost Walk-Forward)")
    print("═"*60)

    if strat_results is None:
        btc_reg, xau_reg = phase4()
        strat_results    = phase5(btc_reg, xau_reg)

    from models.ml_filter import run_walk_forward
    from config import ML

    print(f"\n  Config: forward_days={ML['forward_days']}  "
          f"min_confidence={ML['min_confidence']}  "
          f"train={ML['wf_train_months']}mo / test={ML['wf_test_months']}mo")

    ml_results = {}
    for key in ["BTC", "XAU"]:
        print(f"\n[6.{'1' if key=='BTC' else '2'}] {key} walk-forward ML …")
        ml_results[key] = run_walk_forward(strat_results[key], key)

    print(f"\n✅ Phase 6 complete — ML filter applied, "
          f"feature importance charts saved to /results")
    return ml_results


# ── PHASE 7 — Backtest Engine ─────────────────────────────────────────────────
def phase7(ml_results=None):
    print("\n" + "═"*60)
    print("  PHASE 7 — Backtest Engine")
    print("═"*60)

    if ml_results is None:
        ml_results = phase6()

    from backtest.engine import run_backtest
    from config import BT

    print(f"\n  Config snapshot:")
    print(f"    initial_capital : ${BT['initial_capital']:,.0f}")
    print(f"    commission      : {BT['commission']*100:.2f}% per side")
    print(f"    slippage        : {BT['slippage']*100:.3f}% adverse")
    print(f"    ML gate         : ml_signal == 1 (confidence > {__import__('config').ML['min_confidence']})")
    print(f"    signal shift    : +1 bar (strict no-lookahead)")

    bt_results = run_backtest(ml_results)

    print(f"\n✅ Phase 7 complete — backtest finished, equity charts and "
          f"trade logs saved to /results")
    return bt_results


# ── DISPATCHER ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Regime-Aware Backtest Framework")
    parser.add_argument("--phase", type=int, default=0,
                        help="Run a specific phase (1-9). Default: run 1+2.")
    args = parser.parse_args()

    if args.phase == 0 or args.phase == 1:
        phase1()
    if args.phase == 0 or args.phase == 2:
        btc, xau = phase2()
    if args.phase == 0 or args.phase == 3:
        if args.phase == 3:
            btc, xau = phase2()
        btc_feat, xau_feat = phase3(btc, xau)
    if args.phase == 0 or args.phase == 4:
        if args.phase == 4:
            btc_feat, xau_feat = phase3()
        btc_reg, xau_reg = phase4(btc_feat, xau_feat)
    if args.phase == 0 or args.phase == 5:
        if args.phase == 5:
            btc_reg, xau_reg = phase4()
        strat_results = phase5(btc_reg, xau_reg)
    if args.phase == 0 or args.phase == 6:
        if args.phase == 6:
            strat_results = phase5()
        ml_results = phase6(strat_results)
    if args.phase == 0 or args.phase == 7:
        if args.phase == 7:
            ml_results = phase6()
        phase7(ml_results)


if __name__ == "__main__":
    main()
