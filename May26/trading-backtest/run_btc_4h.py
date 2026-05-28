"""
run_btc_4h.py — BTC 4-hour bar strategy backtest.

Mode 1 only: HMM regime detection + volume-confirmed 120-bar high breakout
(≈ 20 calendar days × 6 bars/day) + ADX > 25 + ML confidence > 0.58,
on 4H Binance bars from 2018.

Sharpe is annualized using sqrt(365×6 = 2190) for 24/7 4H crypto,
not the stock-market sqrt(252) used by the daily engine.

Usage:
    venv/bin/python3 run_btc_4h.py
"""

# ── Config override — MUST happen before any project module is imported ───────
import sys
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config as _cfg
_cfg.RESULTS = ROOT / "results" / "btc_4h"
_cfg.RESULTS.mkdir(parents=True, exist_ok=True)
_cfg.FEAT["breakout_window"] = 120  # 20 calendar days × 6 bars/day
_cfg.STRAT["min_hold_bars"] = 18   # 3 calendar days × 6 bars/day = 72H min hold
_cfg.ML["min_confidence"]   = 0.58

# ── Standard imports (after config override) ──────────────────────────────────
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from data.pipeline import load_btc_4h
from features.engineer import build_features, FEATURE_COLS
from models.regime import detect_regimes
from strategy.signals import compute_signals
from strategy.sizing import compute_sizing_series
from models.ml_filter import run_walk_forward
from backtest.engine import run_backtest, _run_asset, _compute_metrics, _buy_hold_metrics
from backtest.wfo import run_wfo
from config import BT, STRAT, ML, RESULTS

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────
BARS_PER_YEAR = 365 * 6   # 4H crypto 24/7: 2190 bars/year
ANN_FACTOR    = np.sqrt(BARS_PER_YEAR)

V4_TARGETS = {
    "sharpe_ratio":  0.5,
    "win_rate_pct":  55.0,
    "profit_factor": 1.5,
    "total_trades":  40,
}

GREEN = "\033[92m"
RED   = "\033[91m"
YELLOW= "\033[93m"
BOLD  = "\033[1m"
RESET = "\033[0m"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sharpe_4h(equity: pd.Series) -> float:
    """Annualized Sharpe using 4H crypto convention (sqrt(2190))."""
    r = equity.pct_change().dropna()
    if r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * ANN_FACTOR)


def _recompute_sharpe(bt_results: dict) -> None:
    """Patch the stored metrics dict with correct 4H-annualized Sharpe."""
    for key, res in bt_results.items():
        equity = res["equity"]
        res["metrics"]["sharpe_ratio"] = _sharpe_4h(equity)
        # Also fix buy & hold Sharpe
        bh_eq = res["df"]["close"] / res["df"]["close"].iloc[0] * BT["initial_capital"]
        r_bh  = bh_eq.pct_change().dropna()
        if r_bh.std() > 0:
            res["bh_metrics"]["sharpe_ratio"] = float(
                r_bh.mean() / r_bh.std() * ANN_FACTOR
            )


def _print_trade_year_breakdown(trades: list) -> None:
    """Print trade count, WR, and P&L broken down by entry year."""
    if not trades:
        print("  No trades to summarize.")
        return

    by_year: dict = {}
    for t in trades:
        yr = t.entry_date.year
        if yr not in by_year:
            by_year[yr] = {"n": 0, "wins": 0, "pnl": 0.0}
        by_year[yr]["n"]    += 1
        by_year[yr]["pnl"]  += t.pnl
        if t.pnl > 0:
            by_year[yr]["wins"] += 1

    print(f"\n  {'─'*52}")
    print(f"  {'Year':<6} {'Trades':>7} {'Wins':>6} {'Losses':>7} "
          f"{'WR%':>7} {'P&L':>10}")
    print(f"  {'─'*52}")
    total_n = total_w = 0
    total_pnl = 0.0
    for yr in sorted(by_year):
        y = by_year[yr]
        wr = y["wins"] / y["n"] * 100
        total_n  += y["n"]
        total_w  += y["wins"]
        total_pnl+= y["pnl"]
        print(f"  {yr:<6} {y['n']:>7} {y['wins']:>6} {y['n']-y['wins']:>7} "
              f"{wr:>6.0f}% {y['pnl']:>+10.2f}")
    print(f"  {'─'*52}")
    total_wr = total_w / total_n * 100 if total_n else 0
    print(f"  {'Total':<6} {total_n:>7} {total_w:>6} {total_n-total_w:>7} "
          f"{total_wr:>6.0f}% {total_pnl:>+10.2f}")
    print(f"  {'─'*52}")


def _plot_equity_4h(equity: pd.Series, df: pd.DataFrame, metrics: dict) -> None:
    """Equity curve vs buy-and-hold for BTC 4H."""
    bh_eq = df["close"] / df["close"].iloc[0] * BT["initial_capital"]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(18, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    fig.suptitle(
        f"BTC/USD 4H — Strategy vs Buy & Hold\n"
        f"Sharpe {metrics['sharpe_ratio']:.3f}  |  "
        f"Ann.Return {metrics['annual_return_pct']:.1f}%  |  "
        f"MaxDD {metrics['max_drawdown_pct']:.1f}%  |  "
        f"Trades {metrics['total_trades']}  |  "
        f"WR {metrics['win_rate_pct']:.1f}%  |  "
        f"PF {metrics['profit_factor']:.3f}",
        fontsize=12, fontweight="bold",
    )

    ax1.plot(equity.index, equity.values, color="#F7931A",
             linewidth=1.2, label="Strategy (Mode 1 Breakout)")
    ax1.plot(bh_eq.index, bh_eq.values, color="#888888",
             linewidth=0.8, linestyle="--", alpha=0.7, label="Buy & Hold")
    ax1.axhline(BT["initial_capital"], color="#ccc", linewidth=0.5, linestyle=":")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.set_ylabel("Portfolio Value (USD)", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.25)

    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max * 100.0
    ax2.fill_between(dd.index, dd.values, 0, color="#E53935", alpha=0.5)
    ax2.set_ylabel("Drawdown %", fontsize=9)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    out = RESULTS / "btc_4h_equity.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Equity chart → results/btc_4h/{out.name}")


def _print_results_vs_targets(metrics: dict, bh_metrics: dict, trades: list) -> None:
    """Print final metrics table, per-year breakdown, and READY check."""
    print(f"\n  {'═'*64}")
    print(f"  {'BTC 4H — Final Results':^64}")
    print(f"  {'═'*64}")
    print(f"\n  {'Metric':<28} {'Strategy':>14} {'Buy&Hold':>14}")
    print(f"  {'─'*60}")

    rows = [
        ("Sharpe Ratio",       "sharpe_ratio",       "{:.3f}"),
        ("Annual Return %",    "annual_return_pct",  "{:.2f}%"),
        ("Max Drawdown %",     "max_drawdown_pct",   "{:.2f}%"),
        ("Win Rate %",         "win_rate_pct",        "{:.1f}%"),
        ("Profit Factor",      "profit_factor",       "{:.3f}"),
        ("Total Trades",       "total_trades",        "{:d}"),
        ("Avg Duration (bars)","avg_trade_duration_days", "{:.1f}"),
    ]
    for label, key, fmt in rows:
        sv = metrics.get(key, float("nan"))
        bv = bh_metrics.get(key, float("nan"))
        def _f(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "—"
            return (fmt.format(int(v)) if "{:d}" in fmt else fmt.format(v))
        print(f"  {label:<28} {_f(sv):>14} {_f(bv):>14}")

    print(f"  {'─'*60}")
    print(f"  (Sharpe annualized with sqrt({BARS_PER_YEAR}) for 4H 24/7 crypto)")

    # Trade attribution
    if trades:
        m1 = [t for t in trades if t.signal_mode == 1]
        m2 = [t for t in trades if t.signal_mode == 2]
        print(f"\n  Trade attribution:")
        if m1:
            print(f"    Mode 1 — Breakout : {len(m1):>3} trades  "
                  f"WR {len([t for t in m1 if t.pnl>0])/len(m1)*100:.0f}%  "
                  f"avg P&L ${np.mean([t.pnl for t in m1]):.2f}")
        if m2:
            print(f"    Mode 2 — Pullback : {len(m2):>3} trades  "
                  f"WR {len([t for t in m2 if t.pnl>0])/len(m2)*100:.0f}%  "
                  f"avg P&L ${np.mean([t.pnl for t in m2]):.2f}")

        wins   = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl < 0]
        avg_w  = np.mean([t.pnl for t in wins])   if wins   else 0.0
        avg_l  = np.mean([t.pnl for t in losses]) if losses else 0.0
        ratio  = avg_w / abs(avg_l) if avg_l != 0 else float("inf")
        print(f"\n  Avg win  : ${avg_w:>8.2f}  ({len(wins):>3} trades)")
        print(f"  Avg loss : ${avg_l:>8.2f}  ({len(losses):>3} trades)")
        print(f"  W/L ratio: {ratio:.3f}")

        reasons = pd.Series([t.exit_reason for t in trades]).value_counts()
        print(f"\n  Exit reasons:")
        for reason, cnt in reasons.items():
            print(f"    {reason:<12} {cnt:>5}  ({cnt/len(trades)*100:.1f}%)")

    # Per-year breakdown
    print(f"\n  Trade count by year:")
    _print_trade_year_breakdown(trades)

    # Target check
    print(f"\n  {'─'*60}")
    print(f"  {'Metric':<28} {'Value':>10} {'Target':>10} {'Status':>6}")
    print(f"  {'─'*60}")
    hits = 0
    for key, tgt in V4_TARGETS.items():
        val = metrics.get(key, 0)
        met = bool(val >= tgt)
        hits += int(met)
        mark = f"{GREEN}✓{RESET}" if met else f"{RED}✗{RESET}"
        fmt_v = f"{val:.3f}" if isinstance(val, float) else str(int(val))
        fmt_t = f"{tgt:.1f}"
        print(f"  {key:<28} {fmt_v:>10} {fmt_t:>10} {mark}")
    print(f"  {'─'*60}")

    if hits == 4:
        print(f"\n  {BOLD}{GREEN}★  READY FOR PAPER TRADING  ★{RESET}")
        print(f"  {GREEN}  All 4 targets met: Sharpe, WR, PF, Trades{RESET}")
    else:
        print(f"\n  {RED}  {hits}/4 targets met{RESET}")
        gaps = [(abs(tgt - metrics.get(k, 0)) / (abs(tgt) + 1e-9), k, tgt, metrics.get(k, 0))
                for k, tgt in V4_TARGETS.items()
                if metrics.get(k, 0) < tgt]
        if gaps:
            _, wk, wt, wv = max(gaps)
            fmt_wt = f"{wt:.1f}"
            fmt_wv = f"{wv:.3f}" if isinstance(wv, float) else str(int(wv))
            print(f"  {YELLOW}  Biggest weakness: {wk}  (need {fmt_wt}, got {fmt_wv}){RESET}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═"*60)
    print("  BTC 4H STRATEGY BACKTEST")
    print("═"*60)
    print(f"  Timeframe : 4-hour bars  (Binance BTC/USDT from 2018)")
    print(f"  Strategy  : Mode 1 — volume-confirmed 120-bar breakout (≈ 20 calendar days)")
    print(f"  Regime    : HMM 3-state (Bull/Sideways/Bear)")
    print(f"  ML gate   : XGBoost walk-forward, confidence > {ML['min_confidence']}")
    print(f"  Min hold  : {STRAT['min_hold_bars']} bars × 4H = "
          f"{STRAT['min_hold_bars'] * 4}H minimum hold")
    print(f"  SL / TP   : {STRAT['atr_stop_mult']}× / {STRAT['atr_tp_mult']}× ATR")
    print(f"  Results   → results/btc_4h/")

    # ── Phase 2: Data ─────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  PHASE 2 — Data (4H)")
    print("═"*60)
    btc_4h = load_btc_4h()

    # ── Phase 3: Features ─────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  PHASE 3 — Feature Engineering")
    print("═"*60)
    print("\n  Building features on 4H bars …")
    btc_feat = build_features(btc_4h, "BTC")
    print(f"  Feature columns ({len(FEATURE_COLS)}): {', '.join(FEATURE_COLS[:8])} …")

    # ── Phase 4: Regimes ──────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  PHASE 4 — Regime Detection (HMM)")
    print("═"*60)
    print("\n  Fitting HMM on 4H BTC log-return + vol features …")
    btc_reg = detect_regimes(btc_feat, "BTC")

    # ── Phase 5: Signals (Mode 1 only) ────────────────────────────────────────
    print("\n" + "═"*60)
    print("  PHASE 5 — Strategy Signals")
    print("═"*60)
    btc_sig = compute_signals(btc_reg, "BTC")

    # Strip out any Mode 2 pullback signals — 4H run is Mode 1 breakout only
    mode2_bars = (btc_sig["signal_mode"] == 2).sum()
    if mode2_bars > 0:
        btc_sig.loc[btc_sig["signal_mode"] == 2, "signal"]      = 0
        btc_sig.loc[btc_sig["signal_mode"] == 2, "signal_mode"] = 0
        print(f"  Filtered out {mode2_bars:,} Mode 2 pullback bars (4H = Mode 1 only)")

    sig     = btc_sig["signal"]
    n_long  = (sig == 1).sum()
    n_short = (sig == -1).sum()
    n_flat  = (sig == 0).sum()
    print(f"\n  BTC 4H Signal Summary:")
    print(f"    Total bars : {len(sig):,}")
    print(f"    Long       : {n_long:,}  ({n_long/len(sig)*100:.1f}%)")
    print(f"    Short      : {n_short:,}  ({n_short/len(sig)*100:.1f}%)")
    print(f"    Flat       : {n_flat:,}  ({n_flat/len(sig)*100:.1f}%)")

    changes       = sig.diff().fillna(0)
    entries_long  = ((changes > 0)  & (sig ==  1)).sum()
    entries_short = ((changes < 0)  & (sig == -1)).sum()
    print(f"    Long  entries (transitions to +1) : {entries_long:,}")
    print(f"    Short entries (transitions to -1) : {entries_short:,}")

    btc_sized = compute_sizing_series(btc_sig, BT["initial_capital"])

    # ── Phase 6: ML Filter ────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  PHASE 6 — ML Filter (XGBoost Walk-Forward)")
    print("═"*60)
    print(f"\n  Config: forward_days={ML['forward_days']}  "
          f"min_confidence={ML['min_confidence']}  "
          f"train={ML['wf_train_months']}mo / test={ML['wf_test_months']}mo")
    print(f"  Note: on 4H bars, {ML['wf_train_months']}mo train ≈ "
          f"{ML['wf_train_months']*30*6:,} bars  |  "
          f"forward_{ML['forward_days']} bars = {ML['forward_days']*4}H look-ahead")
    btc_ml = run_walk_forward(btc_sized, "BTC")

    # ── Phase 7: Backtest ─────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  PHASE 7 — Backtest Engine")
    print("═"*60)
    print(f"\n  initial_capital : ${BT['initial_capital']:,.0f}")
    print(f"  commission      : {BT['commission']*100:.2f}% per side")
    print(f"  slippage        : {BT['slippage']*100:.3f}% adverse")
    print(f"  ML gate         : ml_signal == 1 (confidence > {ML['min_confidence']})")
    print(f"  signal shift    : +1 bar (strict no-lookahead)")

    bt_results = run_backtest({"BTC": btc_ml})

    # Recompute Sharpe with correct 4H annualization
    _recompute_sharpe(bt_results)

    metrics    = bt_results["BTC"]["metrics"]
    bh_metrics = bt_results["BTC"]["bh_metrics"]
    trades     = bt_results["BTC"]["trades"]
    equity     = bt_results["BTC"]["equity"]

    # Custom equity chart with 4H label
    _plot_equity_4h(equity, btc_ml, metrics)

    # ── Phase 8: WFO ─────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  PHASE 8 — Walk-Forward Optimization")
    print("═"*60)
    wfo_results = run_wfo({"BTC": btc_ml})

    # ── Phase 9: Final Report ─────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  PHASE 9 — Results vs Targets")
    print("═"*60)
    _print_results_vs_targets(metrics, bh_metrics, trades)

    print(f"\n✅ BTC 4H backtest complete — all results in results/btc_4h/\n")


if __name__ == "__main__":
    main()
