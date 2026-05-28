"""
run_phase6_v3.py — Phase 6 v3: Definitive backtest (A + C only).

Changes vs v2:
  B removed  — no gross edge at scale; costs mathematically unrecoverable
  D removed  — 17.1% TP rate; origin unreachable on 15M (revisit on 1H/4H)
  A          — limit entry at signal bar's high (bull) / low (bear)
               maker costs: comm 0.02%, spread 0.005%, slip 0.005% (~0.06% RT)
               3-bar cancellation window
  C          — atr_tp_mult 1.5 → 1.8; ema_touch_pct 0.4% → 0.6%
               signals regenerated fresh from config; maker costs retained

Targets (READY FOR PAPER TRADING):
  Combined Net PF  > 1.2
  Combined Sharpe  > 0.5
  Max Drawdown     < 25%
  Signals per day  > 1.5
  A fill rate      > 60%

Usage:
    venv/bin/python3 run_phase6_v3.py
"""

import sys
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

from data.pipeline import load_15m, load_1h
from strategy.market_state import classify_market_state
from strategy.signals_b import _compute_vwap
from strategy.signals_c import generate_signals as gen_c
from backtest.engine import run_backtest, compute_metrics, plot_results
from config import DATA_CFG, RESULTS

OUT_DIR = RESULTS / "phase6_v3"
OUT_DIR.mkdir(exist_ok=True)

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# Paper-trading readiness targets
TARGETS = {
    "net_pf":      1.2,
    "sharpe":      0.5,
    "max_dd":     -25.0,    # drawdown is negative
    "signals_day": 1.5,
    "fill_rate":   0.60,
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _gross_pf(t: pd.DataFrame) -> float:
    w = t[t["gross_pnl"] > 0]["gross_pnl"].sum()
    l = abs(t[t["gross_pnl"] < 0]["gross_pnl"].sum())
    return w / max(l, 1e-9)

def _net_pf(t: pd.DataFrame) -> float:
    w = t[t["net_pnl"] > 0]["net_pnl"].sum()
    l = abs(t[t["net_pnl"] <= 0]["net_pnl"].sum())
    return w / max(l, 1e-9)

def _load_a_signals(data_dir: Path) -> pd.DataFrame:
    fpath = data_dir / "signals_A_filtered.csv"
    if not fpath.exists():
        raise FileNotFoundError(f"signals_A_filtered.csv not found in {data_dir}")
    df = pd.read_csv(fpath, index_col="ts", parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_main_table(results_ac: dict, sig_a: pd.DataFrame, sig_c: pd.DataFrame,
                     n_days: int) -> None:
    trades = results_ac["trades"]

    print(f"\n{'═'*78}")
    print(f"  PHASE 6 v3 — MAIN ATTRIBUTION TABLE")
    print(f"{'═'*78}")
    print(f"  {'Strategy':<14} {'Trades':>7} {'Fill%':>7} {'Gross PF':>9} "
          f"{'Costs':>10} {'Net PF':>8} {'Net P&L':>10}")
    print(f"  {'─'*74}")

    rows = {}
    for key, n_sigs in [("A", len(sig_a)), ("C", len(sig_c))]:
        t = trades[trades["strategy"] == key] if not trades.empty else pd.DataFrame()
        if t.empty:
            rows[key] = {"n_sigs": n_sigs, "n_trades": 0}
            continue
        n_t     = len(t)
        fill    = n_t / n_sigs * 100 if n_sigs > 0 else 0.0
        gpf     = _gross_pf(t)
        costs   = t["cost"].sum()
        npf     = _net_pf(t)
        net     = t["net_pnl"].sum()

        gpf_c = GREEN if gpf >= 1.0 else RED
        npf_c = GREEN if npf >= 1.0 else RED
        net_c = GREEN if net >= 0   else RED
        fill_c = GREEN if fill >= 60 else YELLOW

        print(f"  {'A — Momentum Burst' if key=='A' else 'C — EMA Pullback':<14} "
              f"{n_t:>7,} "
              f"{fill_c}{fill:>6.1f}%{RESET} "
              f"{gpf_c}{gpf:>9.3f}{RESET} "
              f"${costs:>9,.0f} "
              f"{npf_c}{npf:>8.3f}{RESET} "
              f"{net_c}${net:>9,.0f}{RESET}")
        rows[key] = {"n_sigs": n_sigs, "n_trades": n_t, "fill": fill,
                     "gpf": gpf, "costs": costs, "npf": npf, "net": net}

    # Combined
    print(f"  {'─'*74}")
    if not trades.empty:
        tot_sigs = len(sig_a) + len(sig_c)
        n_tot    = len(trades)
        fill_tot = n_tot / tot_sigs * 100 if tot_sigs > 0 else 0.0
        gpf_tot  = _gross_pf(trades)
        costs_tot= trades["cost"].sum()
        npf_tot  = _net_pf(trades)
        net_tot  = trades["net_pnl"].sum()

        gpf_c = GREEN if gpf_tot >= 1.0 else RED
        npf_c = GREEN if npf_tot >= TARGETS["net_pf"] else RED
        net_c = GREEN if net_tot >= 0 else RED

        print(f"  {'TOTAL':<14} {n_tot:>7,} {fill_tot:>6.1f}%  "
              f"{gpf_c}{gpf_tot:>9.3f}{RESET} "
              f"${costs_tot:>9,.0f} "
              f"{npf_c}{npf_tot:>8.3f}{RESET} "
              f"{net_c}${net_tot:>9,.0f}{RESET}")
    return rows


def print_detail(results_ac: dict, sig_a: pd.DataFrame, sig_c: pd.DataFrame,
                 metrics: dict, n_days: int) -> None:
    trades = results_ac["trades"]

    strat_labels = {"A": "A — Momentum Burst (limit entry)", "C": "C — EMA Pullback (limit entry)"}
    n_sigs = {"A": len(sig_a), "C": len(sig_c)}

    for key in ["A", "C"]:
        t = trades[trades["strategy"] == key] if not trades.empty else pd.DataFrame()
        if t.empty:
            continue

        n_t      = len(t)
        fill     = n_t / n_sigs[key] * 100
        avg_cost = t["cost"].mean()
        avg_gross= t["gross_pnl"].mean()
        cost_pct = abs(t["cost"].sum() / t["gross_pnl"].sum() * 100) if t["gross_pnl"].sum() != 0 else float("nan")
        wr       = (t["net_pnl"] > 0).mean() * 100
        avg_hold = t["bars_held"].mean() * 15 / 60
        tp_r     = (t["exit_reason"] == "tp").mean() * 100
        sl_r     = (t["exit_reason"] == "sl").mean() * 100
        time_r   = (t["exit_reason"] == "time").mean() * 100

        print(f"\n  {'─'*60}")
        print(f"  {strat_labels[key]}  ({n_t:,} trades / {n_sigs[key]:,} signals)")
        print(f"  {'─'*60}")
        fill_c = GREEN if fill >= 60 else YELLOW
        print(f"  {'Fill rate':<34} {fill_c}{fill:>6.1f}%{RESET}  ({n_t:,} filled / {n_sigs[key]:,} queued)")
        print(f"  {'Avg cost / trade':<34} ${avg_cost:>8,.2f}")
        print(f"  {'Avg gross P&L / trade':<34} ${avg_gross:>8,.2f}")
        cp_str = f"{cost_pct:.1f}%" if not np.isnan(cost_pct) else "—"
        print(f"  {'Cost as % of |gross P&L|':<34} {cp_str:>9}")
        print(f"  {'Win rate (net)':<34} {wr:>8.1f}%")
        print(f"  {'TP / SL / Time exit':<34} {tp_r:.1f}% / {sl_r:.1f}% / {time_r:.1f}%")
        print(f"  {'Avg hold time':<34} {avg_hold:>8.2f}h")
        print(f"  {'Signals / day (queued)':<34} {n_sigs[key]/n_days:>8.2f}")

    # Cost breakdown
    if not trades.empty:
        comm_rt = 0.0002 * 2  # both A and C: 0.04% RT
        slip_a  = 0.00005 * 2
        slip_c  = 0.0001  * 2
        spr_a   = 0.00005 * 2
        spr_c   = 0.0     * 2

        ta = trades[trades["strategy"] == "A"]
        tc = trades[trades["strategy"] == "C"]
        tv_a = (ta["entry_price"] * ta["units"]).sum() if not ta.empty else 0
        tv_c = (tc["entry_price"] * tc["units"]).sum() if not tc.empty else 0

        total_comm = comm_rt * (tv_a + tv_c)
        total_slip = slip_a * tv_a + slip_c * tv_c
        total_spr  = spr_a  * tv_a + spr_c  * tv_c

        print(f"\n{'═'*78}")
        print(f"  COST BREAKDOWN (estimated from trade values)")
        print(f"{'═'*78}")
        print(f"  {'Component':<30} {'Amount':>12}")
        print(f"  {'─'*44}")
        print(f"  {'Commission (0.02%/side × 2)':<30} ${total_comm:>10,.0f}")
        print(f"  {'Slippage (A:0.005%, C:0.01% × 2)':<30} ${total_slip:>10,.0f}")
        print(f"  {'Spread  (A:0.005%, C:0% × 2)':<30} ${total_spr:>10,.0f}")
        print(f"  {'─'*44}")
        print(f"  {'TOTAL (actual from ledger)':<30} ${trades['cost'].sum():>10,.0f}")


def print_monthly_table(results_ac: dict) -> None:
    trades = results_ac["trades"]
    if trades.empty:
        return

    trades_c = trades.copy()
    trades_c["year"]  = pd.to_datetime(trades_c["exit_ts"]).dt.year
    trades_c["month"] = pd.to_datetime(trades_c["exit_ts"]).dt.month
    monthly = trades_c.groupby(["year", "month"])["net_pnl"].sum().unstack(fill_value=0)

    months = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]

    print(f"\n{'═'*78}")
    print(f"  MONTHLY NET P&L ($)")
    print(f"{'═'*78}")

    header = f"  {'Year':<6} " + "".join(f"{m:>7}" for m in months) + f"{'Total':>8}"
    print(header)
    print(f"  {'─'*76}")

    for yr in sorted(monthly.index):
        row_parts = []
        yr_total = 0.0
        for m in range(1, 13):
            val = monthly.loc[yr, m] if m in monthly.columns else 0.0
            yr_total += val
            if val > 0:
                row_parts.append(f"{GREEN}{val:>6.0f}{RESET} ")
            elif val < 0:
                row_parts.append(f"{RED}{val:>6.0f}{RESET} ")
            else:
                row_parts.append(f"{'—':>7}")
        yr_c = GREEN if yr_total > 0 else RED
        print(f"  {yr:<6} " + "".join(row_parts) + f" {yr_c}{yr_total:>7.0f}{RESET}")


def print_readiness(metrics: dict, results_ac: dict, sig_a: pd.DataFrame,
                    sig_c: pd.DataFrame, n_days: int) -> bool:
    trades  = results_ac["trades"]
    net_pf  = metrics["profit_factor"]
    sharpe  = metrics["sharpe"]
    max_dd  = metrics["max_dd_pct"]
    n_sigs  = (len(sig_a) + len(sig_c)) / n_days
    n_trades_a = len(trades[trades["strategy"] == "A"]) if not trades.empty else 0
    fill_rate  = n_trades_a / len(sig_a) if len(sig_a) > 0 else 0.0

    checks = {
        "Combined Net PF > 1.2":   (net_pf,    TARGETS["net_pf"],    net_pf    >= TARGETS["net_pf"]),
        "Combined Sharpe > 0.5":   (sharpe,    TARGETS["sharpe"],    sharpe    >= TARGETS["sharpe"]),
        "Max Drawdown < -25%":     (max_dd,    TARGETS["max_dd"],    max_dd    >= TARGETS["max_dd"]),
        "Signals / day > 1.5":     (n_sigs,    TARGETS["signals_day"],n_sigs   >= TARGETS["signals_day"]),
        "A fill rate > 60%":       (fill_rate, TARGETS["fill_rate"], fill_rate >= TARGETS["fill_rate"]),
    }

    print(f"\n{'═'*78}")
    print(f"  PAPER-TRADING READINESS CHECK")
    print(f"{'═'*78}")
    print(f"  {'Target':<36} {'Actual':>10} {'Required':>10} {'Status':>8}")
    print(f"  {'─'*66}")

    all_pass = True
    for label, (actual, required, passed) in checks.items():
        status_c = GREEN if passed else RED
        status   = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        if "fill" in label.lower():
            print(f"  {label:<36} {actual*100:>9.1f}% {required*100:>9.1f}%  "
                  f"{status_c}{status}{RESET}")
        elif "Drawdown" in label:
            print(f"  {label:<36} {actual:>9.1f}% {required:>9.1f}%  "
                  f"{status_c}{status}{RESET}")
        elif "Signals" in label:
            print(f"  {label:<36} {actual:>9.2f}  {required:>9.2f}   "
                  f"{status_c}{status}{RESET}")
        else:
            print(f"  {label:<36} {actual:>10.3f} {required:>10.3f}  "
                  f"{status_c}{status}{RESET}")

    print(f"  {'─'*66}")
    if all_pass:
        print(f"\n  {GREEN}{BOLD}READY FOR PAPER TRADING{RESET}")
        print(f"  All 5 targets met.")
    else:
        failed = [lbl for lbl, (_, _, ok) in checks.items() if not ok]
        print(f"\n  {RED}{BOLD}NOT READY FOR PAPER TRADING{RESET}")
        print(f"  Failed targets:")
        for lbl in failed:
            actual, required, _ = checks[lbl]
            if "fill" in lbl.lower():
                print(f"    • {lbl}: got {actual*100:.1f}%, need {required*100:.1f}%")
            elif "Drawdown" in lbl:
                print(f"    • {lbl}: got {actual:.1f}%, need >= {required:.1f}%")
            elif "Signals" in lbl:
                print(f"    • {lbl}: got {actual:.2f}/day, need {required:.2f}/day")
            else:
                print(f"    • {lbl}: got {actual:.3f}, need {required:.3f}")
    print()
    return all_pass


def plot_v3_dashboard(results_ac: dict, results_a: dict, results_c: dict,
                      metrics_ac: dict) -> None:
    eq_ac = results_ac["equity"]
    eq_a  = results_a["equity"]
    eq_c  = results_c["equity"]
    trades = results_ac["trades"]

    fig = plt.figure(figsize=(20, 16))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. Equity curves: A only / C only / A+C ───────────────────────────────
    ax_eq = fig.add_subplot(gs[0, :2])
    ax_eq.plot(eq_ac.index, eq_ac.values, color="#2196F3", linewidth=1.4,
               label=f"A+C  (${results_ac['final_equity']:,.0f})")
    ax_eq.plot(eq_a.index,  eq_a.values,  color="#FF9800", linewidth=1.0,
               label=f"A only (${results_a['final_equity']:,.0f})", alpha=0.85)
    ax_eq.plot(eq_c.index,  eq_c.values,  color="#4CAF50", linewidth=1.0,
               label=f"C only (${results_c['final_equity']:,.0f})", alpha=0.85)
    ax_eq.axhline(results_ac["initial"], color="gray", linestyle="--",
                  linewidth=0.7, alpha=0.5)
    ax_eq.set_ylabel("Equity ($)")
    ax_eq.set_title(
        f"Equity Curves (A only / C only / A+C)  |  "
        f"CAGR: {metrics_ac['cagr_pct']:.1f}%  "
        f"MaxDD: {metrics_ac['max_dd_pct']:.1f}%  "
        f"Sharpe: {metrics_ac['sharpe']:.2f}",
        fontsize=10, fontweight="bold",
    )
    ax_eq.legend(loc="upper left", fontsize=9)

    # ── 2. Drawdown ───────────────────────────────────────────────────────────
    ax_dd = fig.add_subplot(gs[0, 2])
    roll_max = eq_ac.cummax()
    dd       = (eq_ac - roll_max) / (roll_max + 1e-9) * 100
    ax_dd.fill_between(dd.index, dd.values, 0, alpha=0.5, color="#F44336")
    ax_dd.axhline(-25, color="orange", linestyle="--", linewidth=0.8, label="−25% target")
    ax_dd.set_ylabel("Drawdown (%)")
    ax_dd.set_title("Drawdown (A+C)", fontsize=10)
    ax_dd.legend(fontsize=8)

    # ── 3. Hourly win rate ────────────────────────────────────────────────────
    ax_hr = fig.add_subplot(gs[1, :2])
    if not trades.empty and "entry_hour" in trades.columns:
        hourly = trades.groupby("entry_hour").apply(
            lambda x: pd.Series({"n": len(x), "wr": (x["net_pnl"] > 0).mean() * 100})
        )
        hours  = np.arange(24)
        n_arr  = np.array([hourly.loc[h, "n"]  if h in hourly.index else 0 for h in hours])
        wr_arr = np.array([hourly.loc[h, "wr"] if h in hourly.index else 0 for h in hours])
        ax_hr.bar(hours, n_arr, color=plt.cm.RdYlGn(wr_arr / 100), edgecolor="white")
        ax_hr.set_xlabel("UTC Hour")
        ax_hr.set_ylabel("# Trades")
        ax_hr.set_title("Hourly Trade Count & Win Rate", fontsize=10)
        ax_hr.set_xticks(hours)
        ax_wr = ax_hr.twinx()
        ax_wr.plot(hours, wr_arr, "o--", color="#1565C0", markersize=4, linewidth=1)
        ax_wr.axhline(50, color="gray", linestyle=":", linewidth=0.8)
        ax_wr.set_ylabel("Win Rate (%)", color="#1565C0")
        ax_wr.set_ylim(0, 100)

    # ── 4. Exit reasons ───────────────────────────────────────────────────────
    ax_exit = fig.add_subplot(gs[1, 2])
    if not trades.empty:
        ec = trades["exit_reason"].value_counts()
        clr = {"tp": "#4CAF50", "sl": "#F44336", "time": "#FF9800",
               "overnight_close": "#9C27B0", "end_of_data": "#607D8B"}
        ax_exit.bar(ec.index, ec.values,
                    color=[clr.get(k, "#607D8B") for k in ec.index], edgecolor="white")
        ax_exit.set_title("Exit Reason Breakdown", fontsize=10)
        ax_exit.set_ylabel("# Trades")
        for i, (k, v) in enumerate(ec.items()):
            ax_exit.text(i, v + 0.5, str(v), ha="center", va="bottom", fontsize=8)

    # ── 5. Strategy PnL bar ───────────────────────────────────────────────────
    ax_s = fig.add_subplot(gs[2, 0])
    if not trades.empty:
        sg = trades.groupby("strategy")["net_pnl"].sum()
        ax_s.bar(sg.index, sg.values,
                 color=["#4CAF50" if v >= 0 else "#F44336" for v in sg.values],
                 edgecolor="white")
        ax_s.axhline(0, color="gray", linewidth=0.7)
        ax_s.set_title("Net P&L by Strategy ($)", fontsize=10)

    # ── 6. Monthly returns heatmap ────────────────────────────────────────────
    ax_mo = fig.add_subplot(gs[2, 1:])
    if not trades.empty:
        tc = trades.copy()
        tc["year"]  = pd.to_datetime(tc["exit_ts"]).dt.year
        tc["month"] = pd.to_datetime(tc["exit_ts"]).dt.month
        monthly = tc.groupby(["year", "month"])["net_pnl"].sum().unstack(fill_value=0)
        if not monthly.empty:
            vmax = max(abs(monthly.values).max(), 1)
            im = ax_mo.imshow(monthly.values, aspect="auto",
                              cmap="RdYlGn", vmin=-vmax, vmax=vmax)
            ax_mo.set_xticks(range(len(monthly.columns)))
            ax_mo.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"]
                                  [:len(monthly.columns)], fontsize=8)
            ax_mo.set_yticks(range(len(monthly.index)))
            ax_mo.set_yticklabels([str(y) for y in monthly.index], fontsize=8)
            ax_mo.set_title("Monthly P&L Heatmap (A+C)", fontsize=10)
            plt.colorbar(im, ax=ax_mo, fraction=0.03, pad=0.02)

    net_pf = metrics_ac["profit_factor"]
    fig.suptitle(
        f"BTC 15M Scalping — Phase 6 v3 (A+C Only)  |  "
        f"Return: {metrics_ac['total_return_pct']:.1f}%  "
        f"Trades: {metrics_ac['n_trades']:,}  "
        f"Win Rate: {metrics_ac['win_rate']:.1f}%  "
        f"Net PF: {net_pf:.2f}",
        fontsize=12, fontweight="bold", y=0.98,
    )

    out = OUT_DIR / "phase6_v3_dashboard.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Dashboard → results/phase6_v3/phase6_v3_dashboard.png")


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    print("\n" + "═"*78)
    print("  BTC SCALP — PHASE 6 v3: DEFINITIVE BACKTEST (A + C ONLY)")
    print("═"*78)
    print("  Strategies: A (limit, maker costs) + C (regen, limit, maker costs)")
    print("  Removed:    B (PF 1.02 at 1,340 trades) + D (17.1% TP rate)")
    print(f"  Output:     results/phase6_v3/")

    # ── Load and classify ─────────────────────────────────────────────────────
    print("\n  Loading and classifying data …")
    df_15m = load_15m()
    df_1h  = load_1h()
    df     = classify_market_state(df_15m, df_1h)
    df["vwap"] = _compute_vwap(df)
    print(f"  Classified  ({len(df):,} bars)")
    _df_1h = df_1h  # keep reference for gen_c

    n_days = (df.index[-1] - df.index[0]).days

    # ── Signals ───────────────────────────────────────────────────────────────
    print("\n  Loading signals …")
    data_dir  = DATA_CFG["parquet_15m"].parent

    signals_a = _load_a_signals(data_dir)
    print(f"  A (ML-filtered)  : {len(signals_a):,} signals  ({len(signals_a)/n_days:.2f}/day)")

    print("  C: regenerating with ema_touch_pct=0.6%, atr_tp_mult=1.8 …")
    t1 = time.time()
    signals_c = gen_c(df, _df_1h)
    print(f"  C (all, no ML)   : {len(signals_c):,} signals  ({len(signals_c)/n_days:.2f}/day)  "
          f"({time.time()-t1:.1f}s)")

    total_sigs = len(signals_a) + len(signals_c)
    print(f"  TOTAL            : {total_sigs:,}  ({total_sigs/n_days:.2f}/day)")

    # ── Run backtest: A+C combined ────────────────────────────────────────────
    print("\n" + "─"*78)
    print("  Running A+C combined backtest …")
    t1 = time.time()
    results_ac = run_backtest(df, signals_a, signals_c)
    print(f"  Done  ({time.time()-t1:.1f}s)")

    # ── Run A-only and C-only for equity curve comparison ─────────────────────
    print("  Running A-only backtest …")
    results_a  = run_backtest(df, signals_a, pd.DataFrame())
    print("  Running C-only backtest …")
    results_c  = run_backtest(df, pd.DataFrame(), signals_c)

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics_ac = compute_metrics(results_ac)

    # ── Print main table ──────────────────────────────────────────────────────
    print_main_table(results_ac, signals_a, signals_c, n_days)

    # ── Print per-strategy detail ─────────────────────────────────────────────
    print_detail(results_ac, signals_a, signals_c, metrics_ac, n_days)

    # ── Monthly returns table ─────────────────────────────────────────────────
    print_monthly_table(results_ac)

    # ── Overall performance ───────────────────────────────────────────────────
    trades = results_ac["trades"]
    print(f"\n{'═'*78}")
    print(f"  OVERALL PERFORMANCE (A+C)")
    print(f"{'═'*78}")

    def c(val, ok): return GREEN if ok(val) else RED

    print(f"  {'Total Return':<28} {c(metrics_ac['total_return_pct'], lambda v: v>0)}"
          f"{metrics_ac['total_return_pct']:>+8.1f}%{RESET}")
    print(f"  {'CAGR':<28} {c(metrics_ac['cagr_pct'], lambda v: v>0)}"
          f"{metrics_ac['cagr_pct']:>+8.1f}%{RESET}")
    print(f"  {'Sharpe Ratio':<28} {c(metrics_ac['sharpe'], lambda v: v>0.5)}"
          f"{metrics_ac['sharpe']:>8.2f}{RESET}")
    print(f"  {'Sortino Ratio':<28} {c(metrics_ac['sortino'], lambda v: v>0.8)}"
          f"{metrics_ac['sortino']:>8.2f}{RESET}")
    print(f"  {'Max Drawdown':<28} {c(metrics_ac['max_dd_pct'], lambda v: v>-25)}"
          f"{metrics_ac['max_dd_pct']:>+8.1f}%{RESET}")
    print(f"  {'Total Trades':<28} {metrics_ac['n_trades']:>9,}")
    print(f"  {'Win Rate':<28} {c(metrics_ac['win_rate'], lambda v: v>50)}"
          f"{metrics_ac['win_rate']:>8.1f}%{RESET}")
    print(f"  {'Net Profit Factor':<28} {c(metrics_ac['profit_factor'], lambda v: v>1.2)}"
          f"{metrics_ac['profit_factor']:>8.3f}{RESET}")
    print(f"  {'Gross Profit Factor':<28} {c(metrics_ac['gross_profit_factor'], lambda v: v>1.0)}"
          f"{metrics_ac['gross_profit_factor']:>8.3f}{RESET}")
    print(f"  {'Total Gross P&L':<28} ${metrics_ac['total_gross_pnl']:>10,.0f}")
    print(f"  {'Total Costs':<28} ${metrics_ac['total_costs']:>10,.0f}")
    print(f"  {'Daily Halt Days':<28} {metrics_ac['n_halt_days']:>9,}")

    # ── Readiness check ───────────────────────────────────────────────────────
    all_pass = print_readiness(metrics_ac, results_ac, signals_a, signals_c, n_days)

    # ── Save outputs ──────────────────────────────────────────────────────────
    print(f"{'─'*78}")
    print(f"  Saving outputs …")

    if not trades.empty:
        tlog = OUT_DIR / "phase6_v3_trade_log.csv"
        trades.to_csv(tlog, index=False)
        print(f"  Trade log  → results/phase6_v3/phase6_v3_trade_log.csv  ({len(trades):,} rows)")

    # Save equity curves
    eq_path = OUT_DIR / "phase6_v3_equity_curves.csv"
    pd.DataFrame({
        "A_C": results_ac["equity"],
        "A":   results_a["equity"].reindex(results_ac["equity"].index, method="ffill"),
        "C":   results_c["equity"].reindex(results_ac["equity"].index, method="ffill"),
    }).to_csv(eq_path)
    print(f"  Equity curves → results/phase6_v3/phase6_v3_equity_curves.csv")

    # Plot dashboard
    plot_v3_dashboard(results_ac, results_a, results_c, metrics_ac)

    elapsed = time.time() - t_start
    print(f"\n{'═'*78}")
    print(f"  PHASE 6 v3 COMPLETE  ({elapsed:.0f}s)")
    print(f"{'═'*78}")
    print(f"  Final equity : ${results_ac['final_equity']:>12,.2f}  (started $10,000)")
    print(f"  Return       : {metrics_ac['total_return_pct']:>+.1f}%")
    print(f"  Net PF       : {metrics_ac['profit_factor']:>.3f}\n")


if __name__ == "__main__":
    main()
