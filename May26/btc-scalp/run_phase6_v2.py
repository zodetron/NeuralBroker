"""
run_phase6_v2.py — Phase 6 v2: 4 structural fixes applied.

Fixes vs v1:
  A  — Variable risk: 1% in HIGH_VOL_CHOP, 2% in all other states
  B  — SL reverted to 0.5×ATR; TP = VWAP midpoint; time_exit 6→12 bars
  C  — Limit entry at EMA20; maker costs (comm 0.02%, slip 0.01%, spread 0)
  D  — TP/SL anchored to fib_2_5 reference price, not bar open; skip if
       bar_open > fib_2_5 × 1.003 (price bounced past entry zone)

Signals reused from Phase 5 output (no re-run needed).
Results saved to results/phase6_v2/

Usage:
    venv/bin/python3 run_phase6_v2.py
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
from strategy.signals_b import _compute_vwap
from strategy.signals_d import generate_signals as gen_d
from backtest.engine import run_backtest, compute_metrics, plot_results
from config import DATA_CFG, RESULTS, BT

OUT_DIR = RESULTS / "phase6_v2"
OUT_DIR.mkdir(exist_ok=True)

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

_TARGETS = {"A": 1.3, "B": 1.0, "C": 1.0, "D": 1.0, "combined": 1.1}


def _load_filtered_signals(data_dir: Path) -> dict:
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
            print(f"  WARNING: signals_{key}_filtered.csv not found")
            sigs[key] = pd.DataFrame()
    return sigs


def _gross_pf(trades: pd.DataFrame) -> float:
    wins   = trades[trades["gross_pnl"] > 0]["gross_pnl"].sum()
    losses = abs(trades[trades["gross_pnl"] < 0]["gross_pnl"].sum())
    return wins / max(losses, 1e-9)


def _net_pf(trades: pd.DataFrame) -> float:
    wins   = trades[trades["net_pnl"] > 0]["net_pnl"].sum()
    losses = abs(trades[trades["net_pnl"] <= 0]["net_pnl"].sum())
    return wins / max(losses, 1e-9)


def print_v2_report(results: dict, metrics: dict) -> None:
    trades = results["trades"]

    print(f"\n{'═'*76}")
    print(f"  PHASE 6 v2 — STRATEGY ATTRIBUTION TABLE")
    print(f"{'═'*76}")

    # ── Main attribution table ─────────────────────────────────────────────────
    hdr = f"  {'Strategy':<22} {'Gross PF':>8} {'Gross P&L':>11} {'Costs':>10} {'Net P&L':>10} {'Net PF':>7} {'Verdict':>10}"
    print(hdr)
    print(f"  {'─'*72}")

    strat_labels = {
        "A": "A — Momentum Burst",
        "B": "B — VWAP Reversion",
        "C": "C — EMA Pullback",
        "D": "D — Manip. Fade",
    }

    for key in ["A", "B", "C", "D"]:
        t = trades[trades["strategy"] == key] if not trades.empty else pd.DataFrame()
        if t.empty:
            print(f"  {strat_labels[key]:<22} {'—':>8} {'$0':>11} {'$0':>10} {'$0':>10} {'—':>7} {'NO DATA':>10}")
            continue

        gpf   = _gross_pf(t)
        gross = t["gross_pnl"].sum()
        costs = t["cost"].sum()
        net   = t["net_pnl"].sum()
        npf   = _net_pf(t)
        target = _TARGETS[key]
        met    = npf >= target
        verdict = f"{GREEN}PASS{RESET}" if met else f"{RED}FAIL{RESET}"

        gpf_c  = GREEN if gpf  >= 1.0 else RED
        npf_c  = GREEN if npf  >= target else RED
        net_c  = GREEN if net  >= 0 else RED
        gross_c= GREEN if gross >= 0 else RED

        print(
            f"  {strat_labels[key]:<22} "
            f"{gpf_c}{gpf:>8.3f}{RESET} "
            f"{gross_c}${gross:>10,.0f}{RESET} "
            f"${costs:>9,.0f} "
            f"{net_c}${net:>9,.0f}{RESET} "
            f"{npf_c}{npf:>7.3f}{RESET} "
            f"{verdict:>10}"
        )

    # Combined
    print(f"  {'─'*72}")
    if not trades.empty:
        gpf_all = _gross_pf(trades)
        gross_all = trades["gross_pnl"].sum()
        costs_all = trades["cost"].sum()
        net_all   = trades["net_pnl"].sum()
        npf_all   = _net_pf(trades)
        target_c  = _TARGETS["combined"]
        met_c     = npf_all >= target_c
        verdict_c = f"{GREEN}PASS{RESET}" if met_c else f"{RED}FAIL{RESET}"

        gpf_c  = GREEN if gpf_all  >= 1.0 else RED
        npf_c  = GREEN if npf_all  >= target_c else RED
        net_c  = GREEN if net_all  >= 0 else RED
        gross_c= GREEN if gross_all >= 0 else RED

        print(
            f"  {'COMBINED':<22} "
            f"{gpf_c}{gpf_all:>8.3f}{RESET} "
            f"{gross_c}${gross_all:>10,.0f}{RESET} "
            f"${costs_all:>9,.0f} "
            f"{net_c}${net_all:>9,.0f}{RESET} "
            f"{npf_c}{npf_all:>7.3f}{RESET} "
            f"{verdict_c:>10}"
        )

    # ── Per-strategy detail rows ───────────────────────────────────────────────
    print(f"\n{'═'*76}")
    print(f"  PER-STRATEGY DETAIL")
    print(f"{'═'*76}")

    for key in ["A", "B", "C", "D"]:
        t = trades[trades["strategy"] == key] if not trades.empty else pd.DataFrame()
        if t.empty:
            continue

        n           = len(t)
        avg_cost    = t["cost"].mean()
        avg_gross   = t["gross_pnl"].mean()
        cost_pct    = (t["cost"].sum() / abs(t["gross_pnl"].sum()) * 100) if t["gross_pnl"].sum() != 0 else float("nan")
        wr          = (t["net_pnl"] > 0).mean() * 100
        avg_hold_h  = t["bars_held"].mean() * 15 / 60  # 15M bars → hours
        tp_rate     = (t["exit_reason"] == "tp").sum() / n * 100
        sl_rate     = (t["exit_reason"] == "sl").sum() / n * 100
        time_rate   = (t["exit_reason"] == "time").sum() / n * 100

        print(f"\n  {'─'*56}")
        print(f"  {strat_labels[key]}  ({n:,} trades)")
        print(f"  {'─'*56}")
        print(f"  {'Avg cost / trade':<32} ${avg_cost:>8,.2f}")
        print(f"  {'Avg gross P&L / trade':<32} ${avg_gross:>8,.2f}")
        cost_pct_str = f"{cost_pct:.1f}%" if not np.isnan(cost_pct) else "—"
        print(f"  {'Cost as % of |gross P&L|':<32} {cost_pct_str:>9}")
        print(f"  {'Win rate (net)':<32} {wr:>8.1f}%")
        print(f"  {'TP rate':<32} {tp_rate:>8.1f}%")
        print(f"  {'SL rate':<32} {sl_rate:>8.1f}%")
        print(f"  {'Time-exit rate':<32} {time_rate:>8.1f}%")
        print(f"  {'Avg hold time':<32} {avg_hold_h:>8.2f}h")

    # ── Overall summary ────────────────────────────────────────────────────────
    print(f"\n{'═'*76}")
    print(f"  OVERALL PERFORMANCE")
    print(f"{'═'*76}")

    def c(val, ok): return GREEN if ok(val) else RED

    ret    = metrics["total_return_pct"]
    cagr   = metrics["cagr_pct"]
    sharpe = metrics["sharpe"]
    dd     = metrics["max_dd_pct"]
    n_t    = metrics["n_trades"]
    wr_all = metrics["win_rate"]
    pf_all = metrics["profit_factor"]
    halts  = metrics["n_halt_days"]

    print(f"  {'Total Return':<26} {c(ret,  lambda v: v>0)}{ret:>+8.1f}%{RESET}")
    print(f"  {'CAGR':<26} {c(cagr, lambda v: v>0)}{cagr:>+8.1f}%{RESET}")
    print(f"  {'Sharpe Ratio':<26} {c(sharpe,lambda v: v>0.5)}{sharpe:>8.2f}{RESET}")
    print(f"  {'Max Drawdown':<26} {c(dd,   lambda v: v>-25)}{dd:>+8.1f}%{RESET}")
    print(f"  {'Total Trades':<26} {n_t:>8,}")
    print(f"  {'Win Rate':<26} {c(wr_all, lambda v: v>50)}{wr_all:>8.1f}%{RESET}")
    print(f"  {'Profit Factor (net)':<26} {c(pf_all,lambda v: v>1.1)}{pf_all:>8.2f}{RESET}")
    print(f"  {'Daily Halt Days':<26} {halts:>8,}")


def main():
    t_start = time.time()

    print("\n" + "═"*70)
    print("  BTC SCALP — PHASE 6 v2: BACKTEST WITH STRUCTURAL FIXES")
    print("═"*70)
    print(f"  Fixes applied:")
    print(f"    A  — 1% risk in HIGH_VOL_CHOP, 2% elsewhere")
    print(f"    B  — SL 0.5×ATR, TP = VWAP midpoint, time_exit 12 bars")
    print(f"    C  — Limit entry at EMA20, maker costs (comm 0.02%, slip 0.01%)")
    print(f"    D  — TP/SL distances from fib_2_5 ref; skip if bar_open >0.3% past fib_2_5")
    print(f"  Output: results/phase6_v2/")

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

    # Strategy D: regenerate raw (ML had only 4.3% OOS coverage)
    print("  Strategy D: regenerating raw signals (skipping ML) …")
    t1 = time.time()
    signals_d = gen_d(df)
    print(f"  D: {len(signals_d):,} signals  ({time.time()-t1:.1f}s)")

    print(f"\n  Signal counts entering backtest:")
    print(f"    A (ML-filtered)  : {len(signals_a):,}")
    print(f"    B (ML-filtered)  : {len(signals_b):,}")
    print(f"    C (all, no ML)   : {len(signals_c):,}")
    print(f"    D (all, no ML)   : {len(signals_d):,}")
    n_days = (df.index[-1] - df.index[0]).days
    total  = len(signals_a) + len(signals_b) + len(signals_c) + len(signals_d)
    print(f"    TOTAL            : {total:,}  ({total/n_days:.2f}/day)")

    # ── Run backtest ──────────────────────────────────────────────────────────
    print("\n" + "─"*70)
    print("  Running bar-by-bar backtest (v2 fixes active) …")
    t1 = time.time()
    results = run_backtest(df, signals_a, signals_b, signals_c, signals_d)
    print(f"  Backtest complete  ({time.time()-t1:.1f}s)")

    # ── Compute metrics ───────────────────────────────────────────────────────
    metrics = compute_metrics(results)

    # ── Print v2 report ───────────────────────────────────────────────────────
    print_v2_report(results, metrics)

    # ── Save outputs ──────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  Saving outputs to results/phase6_v2/ …")

    if not results["trades"].empty:
        trade_path = OUT_DIR / "phase6_v2_trade_log.csv"
        results["trades"].to_csv(trade_path, index=False)
        print(f"  Trade log  → results/phase6_v2/phase6_v2_trade_log.csv  "
              f"({len(results['trades']):,} rows)")

    plot_results(results, metrics, out_dir=OUT_DIR)

    eq_path = OUT_DIR / "phase6_v2_equity_curve.csv"
    results["equity"].to_csv(eq_path)
    print(f"  Equity curve → results/phase6_v2/phase6_v2_equity_curve.csv")

    elapsed = time.time() - t_start
    print(f"\n{'═'*70}")
    print(f"  PHASE 6 v2 COMPLETE  ({elapsed:.0f}s)")
    print(f"{'═'*70}")
    print(f"  Final equity : ${results['final_equity']:>12,.2f}  "
          f"(started ${results['initial']:,})")
    print(f"  Return       : {metrics['total_return_pct']:>+.1f}%\n")


if __name__ == "__main__":
    main()
