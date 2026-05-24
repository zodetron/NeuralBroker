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

    from data.pipeline import load_btc, load_xau, resample_to_weekly
    from config import XAU_WEEKLY

    print("\n[2.1] Loading BTC/USD …")
    btc = load_btc()

    print("\n[2.2] Loading XAU/USD …")
    xau = load_xau()

    if XAU_WEEKLY:
        print("\n[2.3] Fix 3: Resampling XAU from daily → weekly bars …")
        xau = resample_to_weekly(xau)
        print(f"  XAU weekly: {len(xau):,} bars  "
              f"({xau.index[0].date()} → {xau.index[-1].date()})")
        print(f"  (Reduced from daily to weekly — ~5× fewer trades, "
              f"commission drag cut proportionally)")

    # Cross-asset date overlap info (useful for Phase 5+)
    overlap_start = max(btc.index[0], xau.index[0])
    overlap_end   = min(btc.index[-1], xau.index[-1])
    overlap_days  = (overlap_end - overlap_start).days
    print(f"\n  Shared date range: {overlap_start.date()} → {overlap_end.date()}")
    print(f"  BTC bars : {len(btc):,} (daily)")
    print(f"  XAU bars : {len(xau):,} ({'weekly' if XAU_WEEKLY else 'daily'})")
    print(f"  Overlap  : {overlap_days:,} calendar days")

    print(f"\n✅ Phase 2 complete — BTC ({len(btc):,} rows) and "
          f"XAU ({len(xau):,} rows) loaded")

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
        df_sig  = compute_signals(df_reg, key)          # asset_key routes to correct strategy
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

    # ── BEFORE vs AFTER comparison table ────────────────────────────────────
    _print_before_after(bt_results)

    print(f"\n✅ Phase 7 complete — backtest finished, equity charts and "
          f"trade logs saved to /results/v3")
    return bt_results


# v2 baseline metrics (daily BTC / weekly XAU, min_confidence=0.62, min_hold=5)
_V2_METRICS = {
    "BTC": {"sharpe_ratio": 0.082, "win_rate_pct": 50.0,  "total_trades": 62,  "profit_factor": 1.070,
            "annual_return_pct":  0.36, "max_drawdown_pct": -11.95, "avg_trade_duration_days": 5.4},
    "XAU": {"sharpe_ratio": 0.028, "win_rate_pct": 45.5,  "total_trades": 11,  "profit_factor": 1.028,
            "annual_return_pct":  0.01, "max_drawdown_pct":  -2.56, "avg_trade_duration_days": 29.9},
}

# Live-trading readiness targets
_V3_TARGETS = {
    "BTC": {"sharpe_ratio": 0.5,  "win_rate_pct": 52.0, "profit_factor": 1.3, "total_trades": 40},
    "XAU": {"sharpe_ratio": 0.4,  "win_rate_pct": 50.0, "profit_factor": 1.2, "total_trades": 30},
}


def _print_before_after(bt_results: dict) -> None:
    """Print BEFORE (v2) vs AFTER (v3) comparison with live-trading target check."""
    metrics = [
        ("Sharpe Ratio",           "sharpe_ratio",           "{:+.3f}",  True),
        ("Win Rate %",             "win_rate_pct",           "{:.1f}%",  True),
        ("Profit Factor",          "profit_factor",          "{:.3f}",   True),
        ("Total Trades",           "total_trades",           "{:d}",     None),
        ("Annual Return %",        "annual_return_pct",      "{:+.2f}%", True),
        ("Max Drawdown %",         "max_drawdown_pct",       "{:.2f}%",  False),
        ("Avg Trade Duration (d)", "avg_trade_duration_days","{:.1f}",   None),
    ]

    BOLD  = "\033[1m"
    GREEN = "\033[92m"
    RED   = "\033[91m"
    YELLOW= "\033[93m"
    RESET = "\033[0m"

    def _col(val_str, improved):
        if improved is None:
            return val_str
        return f"{GREEN}{val_str}{RESET}" if improved else f"{RED}{val_str}{RESET}"

    print(f"\n{BOLD}{'─'*76}{RESET}")
    print(f"{BOLD}  BEFORE (v2) vs AFTER (v3) — 4-Improvement Summary{RESET}")
    print(f"{BOLD}{'─'*76}{RESET}")

    for asset_key in ["BTC", "XAU"]:
        if asset_key not in bt_results:
            continue
        prev  = _V2_METRICS[asset_key]
        curr  = bt_results[asset_key]["metrics"]
        targs = _V3_TARGETS[asset_key]

        strat_v2 = "weekly" if asset_key == "XAU" else "daily"
        print(f"\n  {BOLD}{asset_key}{RESET}  (v2={strat_v2}/0.62  →  v3=daily/0.62/breakout+BB+ADX+trail)")
        print(f"  {'Metric':<28} {'v2 (Before)':>14} {'v3 (After)':>14} {'Change':>12} {'Target':>10}")
        print(f"  {'─'*74}")

        for label, key, fmt, higher_is_better in metrics:
            pv  = prev.get(key, float("nan"))
            cv  = curr.get(key, float("nan"))
            tgt = targs.get(key)

            def _f(v):
                if isinstance(v, float) and (v != v):
                    return "—"
                try:
                    return fmt.format(int(v) if "d}" in fmt else v)
                except Exception:
                    return str(v)

            pvs = _f(pv)
            cvs = _f(cv)
            tgt_s = _f(tgt) if tgt is not None else ""

            try:
                delta     = cv - pv
                delta_str = f"{delta:+.3f}"
                improved  = (delta > 0) if higher_is_better is True else \
                            (delta < 0) if higher_is_better is False else None
            except Exception:
                delta_str = "—"
                improved  = None

            # Highlight if target met
            if tgt is not None:
                target_met = cv >= tgt if key != "max_drawdown_pct" else cv >= tgt
                tgt_marker = f"{GREEN}✓{RESET}" if target_met else f"{RED}✗{RESET}"
                tgt_s = f"{tgt_s} {tgt_marker}"

            cv_colored = _col(cvs, improved)
            print(f"  {label:<28} {pvs:>14} {cv_colored:>24} {delta_str:>12} {tgt_s:>12}")

        print(f"  {'─'*74}")

        # Target evaluation
        target_keys = list(targs.keys())
        hits = []
        misses = []
        for tk in target_keys:
            tv  = targs[tk]
            cv2 = curr.get(tk, 0)
            if cv2 >= tv:
                hits.append(tk)
            else:
                misses.append((tk, tv, cv2))

        all_hit = len(hits) == len(target_keys)
        if all_hit:
            print(f"\n  {GREEN}{BOLD}★ {asset_key} READY FOR PAPER TRADING ★{RESET}")
            print(f"  {GREEN}  All 4 targets met: Sharpe, Win Rate, Profit Factor, Trades{RESET}")
        else:
            print(f"\n  {RED}  {asset_key}: {len(hits)}/{len(target_keys)} targets met{RESET}")
            # Biggest remaining weakness = the one furthest from target in % terms
            worst_key, worst_tgt, worst_val = max(
                misses, key=lambda x: abs(x[1] - x[2]) / (abs(x[1]) + 1e-9)
            )
            gap = worst_val - worst_tgt
            print(f"  {YELLOW}  Biggest weakness: {worst_key}  "
                  f"(need {worst_tgt}, got {worst_val:.3f}, gap {gap:+.3f}){RESET}")

    print(f"\n{BOLD}{'─'*76}{RESET}")


# ── PHASE 8 — Walk-Forward Optimization ──────────────────────────────────────
def phase8(ml_results=None):
    print("\n" + "═"*60)
    print("  PHASE 8 — Walk-Forward Optimization")
    print("═"*60)

    if ml_results is None:
        ml_results = phase6()

    from backtest.wfo import run_wfo
    from config import WFO

    print(f"\n  Config snapshot:")
    print(f"    train_months      : {WFO['train_months']}mo")
    print(f"    test_months       : {WFO['test_months']}mo")
    print(f"    atr_stop_mult grid: {WFO['param_grid']['atr_stop_mult']}")
    print(f"    momentum_roc grid : {WFO['param_grid']['momentum_roc_period']}")

    wfo_results = run_wfo(ml_results)

    print(f"\n  Recommended params (mode across OOS windows):")
    for key, res in wfo_results.items():
        if res:
            bp = res["best_params"]
            print(f"    {key}: atr_stop_mult={bp['atr_stop_mult']}  "
                  f"momentum_roc_period={bp['momentum_roc_period']}")

    print(f"\n✅ Phase 8 complete — WFO finished, OOS equity curves and "
          f"per-window tables saved to /results")
    return wfo_results


# ── PHASE 9 — Results Dashboard ──────────────────────────────────────────────
def phase9(bt_results=None, wfo_results=None):
    print("\n" + "═"*60)
    print("  PHASE 9 — Results Dashboard")
    print("═"*60)

    if bt_results is None:
        ml_results = phase6()
        bt_results = phase7(ml_results)
    if wfo_results is None:
        ml_results_for_wfo = phase6()
        wfo_results = phase8(ml_results_for_wfo)

    from results.dashboard import build_dashboard, print_summary_report

    print("\n[9.1] Rendering dashboard chart …")
    build_dashboard(bt_results, wfo_results)

    print("\n[9.2] Writing monthly returns CSVs …")
    # Monthly CSVs are written inside build_dashboard; this is a no-op confirmation.

    print("\n[9.3] Final performance summary:")
    print_summary_report(bt_results, wfo_results)

    print(f"\n✅ Phase 9 complete — dashboard and full report generated in /results")


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
        bt_results = phase7(ml_results)
    if args.phase == 0 or args.phase == 8:
        if args.phase == 8:
            ml_results = phase6()
        wfo_results = phase8(ml_results)
    if args.phase == 0 or args.phase == 9:
        if args.phase == 9:
            ml_results = phase6()
            bt_results  = phase7(ml_results)
            wfo_results = phase8(ml_results)
        phase9(bt_results, wfo_results)


if __name__ == "__main__":
    main()
