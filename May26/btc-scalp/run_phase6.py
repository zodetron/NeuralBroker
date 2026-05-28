"""
run_phase6.py — Phase 6: Full bar-by-bar backtest.

Loads Phase 5 filtered signals (A/B/C) plus all D signals (skip ML for D —
insufficient OOS coverage at 4.3%), runs the event-loop engine, and prints
a full metrics report plus saves charts and trade log CSV.

Usage:
    venv/bin/python3 run_phase6.py
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
from strategy.signals_a import generate_signals as gen_a
from strategy.signals_b import generate_signals as gen_b, _compute_vwap
from strategy.signals_c import generate_signals as gen_c
from strategy.signals_d import generate_signals as gen_d
from backtest.engine import run_backtest, compute_metrics, plot_results
from config import DATA_CFG, RESULTS, BT

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def _load_filtered_signals(data_dir: Path) -> dict:
    """Load Phase 5 filtered CSVs, falling back to raw if missing."""
    sigs = {}
    for key in ["A", "B", "C"]:
        fpath = data_dir / f"signals_{key}_filtered.csv"
        if fpath.exists():
            df = pd.read_csv(fpath, index_col="ts", parse_dates=True)
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            sigs[key] = df
            print(f"  signals_{key}_filtered.csv  →  {len(df):,} rows")
        else:
            print(f"  ⚠ signals_{key}_filtered.csv not found — generating raw signals")
            sigs[key] = pd.DataFrame()
    return sigs


def print_metrics(metrics: dict) -> None:
    """Print the full metrics report."""
    print(f"\n  {'═'*62}")
    print(f"  PHASE 6 BACKTEST RESULTS")
    print(f"  {'═'*62}")

    # Overall
    print(f"\n  {'─'*62}")
    print(f"  OVERALL PERFORMANCE")
    print(f"  {'─'*62}")
    ret   = metrics["total_return_pct"]
    cagr  = metrics["cagr_pct"]
    sharpe= metrics["sharpe"]
    sortino = metrics["sortino"]
    dd    = metrics["max_dd_pct"]
    n     = metrics["n_trades"]
    wr    = metrics["win_rate"]
    tp_r  = metrics["tp_rate"]
    pf    = metrics["profit_factor"]
    aw    = metrics["avg_win"]
    al    = metrics["avg_loss"]
    halts = metrics["n_halt_days"]

    def c(val, good_if): return GREEN if good_if(val) else RED

    print(f"  {'Total Return':<26} {c(ret, lambda v: v>0)}{ret:>+8.1f}%{RESET}")
    print(f"  {'CAGR':<26} {c(cagr, lambda v: v>0)}{cagr:>+8.1f}%{RESET}")
    print(f"  {'Sharpe Ratio':<26} {c(sharpe, lambda v: v>0.5)}{sharpe:>8.2f}{RESET}")
    print(f"  {'Sortino Ratio':<26} {c(sortino, lambda v: v>0.8)}{sortino:>8.2f}{RESET}")
    print(f"  {'Max Drawdown':<26} {c(dd, lambda v: v>-20)}{dd:>+8.1f}%{RESET}")
    print(f"  {'Total Trades':<26} {n:>8,}")
    print(f"  {'Win Rate':<26} {c(wr, lambda v: v>50)}{wr:>8.1f}%{RESET}")
    print(f"  {'TP Rate':<26} {tp_r:>8.1f}%")
    print(f"  {'Profit Factor':<26} {c(pf, lambda v: v>1.1)}{pf:>8.2f}{RESET}")
    print(f"  {'Avg Win':<26} ${aw:>8,.2f}")
    print(f"  {'Avg Loss':<26} ${al:>8,.2f}")
    print(f"  {'Avg Win / Avg Loss':<26} {abs(aw/al):>8.2f}" if al < 0 else "")
    print(f"  {'Daily Halt Days':<26} {halts:>8,}")

    # Per-strategy
    print(f"\n  {'─'*62}")
    print(f"  PER-STRATEGY ATTRIBUTION")
    print(f"  {'─'*62}")
    print(f"  {'Strategy':<22} {'Trades':>7} {'Net PnL':>10} {'Win%':>7} {'TP%':>7} {'Avg PnL':>9}")
    print(f"  {'─'*62}")
    for key, label in [("A", "A — Momentum Burst"),
                        ("B", "B — VWAP Reversion"),
                        ("C", "C — EMA Pullback"),
                        ("D", "D — Manip. Fade")]:
        s = metrics["per_strategy"].get(key, {})
        if not s or s["n"] == 0:
            print(f"  {label:<22} {'0':>7} {'$0':>10}")
            continue
        pnl_c = GREEN if s["net_pnl"] >= 0 else RED
        print(f"  {label:<22} {s['n']:>7,} "
              f"{pnl_c}${s['net_pnl']:>9,.0f}{RESET} "
              f"{s['win_rate']*100:>7.1f}% "
              f"{s.get('tp_rate', 0)*100:>7.1f}% "
              f"${s.get('avg_pnl', 0):>8,.2f}")

    # Session breakdown
    print(f"\n  {'─'*62}")
    print(f"  SESSION PERFORMANCE")
    print(f"  {'─'*62}")
    print(f"  {'Session':<14} {'Trades':>7} {'Net PnL':>10} {'Win%':>7}")
    print(f"  {'─'*62}")
    for sess, data in metrics.get("sessions", {}).items():
        if data["n"] == 0:
            continue
        pnl_c = GREEN if data["net_pnl"] >= 0 else RED
        print(f"  {sess:<14} {data['n']:>7,} "
              f"{pnl_c}${data['net_pnl']:>9,.0f}{RESET} "
              f"{data['win_rate']*100:>7.1f}%")


def main():
    t_start = time.time()

    print("\n" + "═"*66)
    print("  BTC SCALP — PHASE 6: BACKTEST ENGINE")
    print("═"*66)
    print(f"  Initial capital : ${BT['initial_capital']:,}")
    print(f"  Commission      : {BT['commission']*100:.3f}% / side")
    print(f"  Slippage        : {BT['slippage']*100:.3f}% / side")
    print(f"  Max concurrent  : {BT['max_concurrent']} (1 per strategy)")
    print(f"  Daily loss halt : {BT['daily_loss_limit_pct']:.1f}%")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n  Loading and classifying data …")
    df_15m = load_15m()
    df_1h  = load_1h()
    df     = classify_market_state(df_15m, df_1h)
    df["vwap"] = _compute_vwap(df)
    print(f"  Classified  ({len(df):,} bars)")

    # ── Load filtered signals (Phase 5 output) ────────────────────────────────
    print("\n  Loading Phase 5 filtered signals …")
    data_dir = DATA_CFG["parquet_15m"].parent
    filtered = _load_filtered_signals(data_dir)

    signals_a = filtered.get("A", pd.DataFrame())
    signals_b = filtered.get("B", pd.DataFrame())
    signals_c = filtered.get("C", pd.DataFrame())

    # Strategy D: regenerate raw (skip ML — insufficient OOS coverage)
    print("  Strategy D: regenerating raw signals (skipping ML) …")
    t1 = time.time()
    signals_d = gen_d(df)
    print(f"  D: {len(signals_d):,} signals  ({time.time()-t1:.1f}s)")

    print(f"\n  Signal counts entering backtest:")
    print(f"    A (ML-filtered)  : {len(signals_a):,}")
    print(f"    B (ML-filtered)  : {len(signals_b):,}")
    print(f"    C (all, no ML)   : {len(signals_c):,}")
    print(f"    D (all, no ML)   : {len(signals_d):,}")

    total_sigs = len(signals_a) + len(signals_b) + len(signals_c) + len(signals_d)
    n_days     = (df.index[-1] - df.index[0]).days
    print(f"    TOTAL            : {total_sigs:,}  ({total_sigs/n_days:.2f}/day)")

    # ── Run backtest ──────────────────────────────────────────────────────────
    print("\n" + "─"*66)
    print("  Running bar-by-bar backtest …")
    t1 = time.time()
    results = run_backtest(df, signals_a, signals_b, signals_c, signals_d)
    bt_time = time.time() - t1
    print(f"  Backtest complete  ({bt_time:.1f}s)")

    # ── Compute metrics ───────────────────────────────────────────────────────
    metrics = compute_metrics(results)

    # ── Print report ──────────────────────────────────────────────────────────
    print_metrics(metrics)

    # ── Save outputs ──────────────────────────────────────────────────────────
    print(f"\n  {'─'*66}")
    print(f"  Saving outputs …")

    # Trade log CSV
    if not results["trades"].empty:
        trade_log_path = RESULTS / "phase6_trade_log.csv"
        results["trades"].to_csv(trade_log_path, index=False)
        print(f"  Trade log → results/phase6_trade_log.csv  "
              f"({len(results['trades']):,} rows)")

    # Charts
    plot_results(results, metrics)

    # Equity curve CSV
    eq_path = RESULTS / "phase6_equity_curve.csv"
    results["equity"].to_csv(eq_path)
    print(f"  Equity curve → results/phase6_equity_curve.csv")

    elapsed = time.time() - t_start
    print(f"\n{'═'*66}")
    print(f"  PHASE 6 COMPLETE  ({elapsed:.0f}s)")
    print(f"{'═'*66}")
    print(f"  Final equity: ${results['final_equity']:,.2f}  "
          f"(from ${results['initial']:,})")
    print(f"  Results → results/phase6_*\n")


if __name__ == "__main__":
    main()
