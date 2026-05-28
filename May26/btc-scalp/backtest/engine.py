"""
backtest/engine.py — Phase 6: full bar-by-bar event-loop backtest.

Design:
  • Iterates every 15M bar in chronological order.
  • At each bar: close expiring positions → open pending signals → update equity.
  • Max 1 active trade per strategy (4 concurrent max across A/B/C/D).
  • 2% daily-loss halt: no new entries for the rest of that calendar day.
  • No overnight: force-close at 23:45 UTC, no new entries before 00:15 UTC.
  • Dynamic spread via backtest/costs.py.
  • All exits are at the next bar's open for TP/SL triggered on previous bar.
  • Time exits close at current bar's close.

Trade lifecycle:
  bar t   : signal at ts=t (bar is closed, all indicators known)
  bar t+1 : entry at open[t+1] + direction × slippage%
  bar t+N : TP or SL checked against high/low of each bar; time exit at close[t+N]

Position sizing: fixed-fractional — each trade risks 1% of current equity.
  units = risk_amount / (entry_price − sl_price)   for longs
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BT, STRAT_A, STRAT_C, SESSIONS, RESULTS
from backtest.costs import CostLedger, get_spread, total_cost_one_side


# ── RISK PARAMETER ────────────────────────────────────────────────────────────
RISK_PER_TRADE = 0.01   # default 1% for B / C / D

# ── PER-STRATEGY COST OVERRIDES (A and C both use limit/maker orders) ─────────
_STRAT_COST_OVERRIDES: dict = {
    "A": {"commission": 0.0002, "slippage": 0.00005, "spread": 0.00005},  # limit entry
    "C": {"commission": 0.0002, "slippage": 0.0001,  "spread": 0.0},      # limit entry
}


# ── TRADE PARAMS BY STRATEGY (B removed: no gross edge; D removed: unreachable TP)
_STRAT_PARAMS = {
    "A": {"tp_mult": STRAT_A["atr_tp_mult"],   "sl_mult": STRAT_A["atr_sl_mult"],
          "max_bars": STRAT_A["time_exit_bars"], "tp_type": "atr"},
    "C": {"tp_mult": STRAT_C["atr_tp_mult"],   "sl_mult": STRAT_C["atr_sl_mult"],
          "max_bars": STRAT_C["time_exit_bars"], "tp_type": "atr"},
}


# ── TRADE COST HELPER ─────────────────────────────────────────────────────────

def _trade_cost(
    strategy:    str,
    entry_price: float,
    units:       float,
    entry_ts:    pd.Timestamp,
    exit_ts:     pd.Timestamp,
    atr_entry:   float,
    atr_exit:    float,
    avg_atr:     float,
) -> float:
    """Round-trip cost in dollars with per-strategy overrides."""
    trade_value = entry_price * units
    ov = _STRAT_COST_OVERRIDES.get(strategy, {})
    comm = ov.get("commission", BT["commission"])
    slip = ov.get("slippage",   BT["slippage"])
    if "spread" in ov:
        e_spread = ex_spread = ov["spread"]
    else:
        e_spread  = get_spread(entry_ts, atr_entry, avg_atr)
        ex_spread = get_spread(exit_ts,  atr_exit,  avg_atr)
    return (comm * 2 + e_spread + ex_spread + slip * 2) * trade_value


# ── OPEN POSITION ─────────────────────────────────────────────────────────────

@dataclass
class Position:
    signal_id:    str
    strategy:     str
    direction:    int          # +1 long / -1 short
    entry_ts:     pd.Timestamp
    entry_price:  float
    tp_price:     float
    sl_price:     float
    units:        float        # position size
    max_bars:     int
    bars_held:    int = 0
    atr_at_entry: float = 0.0
    avg_atr:      float = 0.0
    signal_ts:    pd.Timestamp = field(default_factory=pd.Timestamp.now)


def _compute_tp_sl(
    strategy:    str,
    direction:   int,
    entry:       float,
    atr:         float,
    signal_row:  pd.Series,
) -> tuple[float, float]:
    """Return (tp_price, sl_price) for a given entry."""
    params = _STRAT_PARAMS[strategy]
    sl_mult = params["sl_mult"]
    tp_type = params["tp_type"]

    if tp_type == "fixed":
        tp_price = float(signal_row["tp_price"])
        sl_price = float(signal_row["sl_price"])
    elif tp_type == "vwap_mid":
        # TP = 50% of distance from entry to VWAP (closer target, higher hit rate)
        vwap_val = float(signal_row["vwap"])
        tp_price = entry + 0.5 * (vwap_val - entry)
        sl_price = entry - direction * sl_mult * atr
    else:  # atr
        tp_price = entry + direction * params["tp_mult"] * atr
        sl_price = entry - direction * sl_mult * atr

    return tp_price, sl_price


# ── MAIN BACKTEST ENGINE ───────────────────────────────────────────────────────

def run_backtest(
    df:        pd.DataFrame,    # full classified 15M DataFrame
    signals_a: pd.DataFrame,    # A: Momentum Burst (limit entry at breakout extreme)
    signals_c: pd.DataFrame,    # C: EMA Pullback   (limit entry at EMA20)
) -> dict:
    """
    Run the full backtest. Returns a results dict with equity curve,
    trade log, per-strategy stats, and summary metrics.
    """
    initial_capital = BT["initial_capital"]
    daily_loss_pct  = BT["daily_loss_limit_pct"] / 100.0
    close_hour_utc  = BT["no_overnight_close_utc"]   # 23.75
    open_hour_utc   = BT["no_overnight_open_utc"]    # 0.25

    # ── Merge A and C signals into a single lookup: ts → list of signal rows ──
    all_signals: Dict[pd.Timestamp, List[pd.Series]] = {}
    for strat_key, sig_df in [("A", signals_a), ("C", signals_c)]:
        if sig_df.empty:
            continue
        for ts, row in sig_df.iterrows():
            if ts not in all_signals:
                all_signals[ts] = []
            all_signals[ts].append(row)

    # Pre-build bar high/low lookups for A limit price calculation
    bar_highs_dict: dict = dict(zip(df.index, df["high"].values))
    bar_lows_dict:  dict = dict(zip(df.index, df["low"].values))

    # ── Pre-extract arrays ────────────────────────────────────────────────────
    opens   = df["open"].values
    highs   = df["high"].values
    lows    = df["low"].values
    closes  = df["close"].values
    atrs    = df["atr_14"].values
    atr_avgs = df["atr_avg"].values
    idx     = df.index

    n = len(df)

    # ── State ─────────────────────────────────────────────────────────────────
    equity             = float(initial_capital)
    active: Dict[str, Optional[Position]] = {"A": None, "C": None}
    # Each pending entry: (entry_bar, sig_row, limit_price, cancel_bar)
    # A: limit_price = breakout bar's high (bull) or low (bear); cancel_bar = signal_bar + 3
    # C: limit_price = None (fills at EMA20 on entry_bar); cancel_bar = None
    pending_entries: List[tuple] = []

    # Daily tracking
    day_start_equity: Dict[str, float] = {}  # date_str → equity at day start
    daily_halt: set = set()                   # dates where trading is halted

    trade_log: List[dict]  = []
    equity_curve: List[float] = [equity]
    equity_times: List[pd.Timestamp] = [idx[0]]

    def _get_daily_key(ts: pd.Timestamp) -> str:
        return ts.strftime("%Y-%m-%d")

    # ── Main loop ─────────────────────────────────────────────────────────────
    for bar_i in range(1, n):
        ts       = idx[bar_i]
        bar_high = highs[bar_i]
        bar_low  = lows[bar_i]
        bar_open = opens[bar_i]
        bar_close= closes[bar_i]
        atr_now  = atrs[bar_i]
        avg_atr  = atr_avgs[bar_i]

        day_key  = _get_daily_key(ts)
        hour_dec = ts.hour + ts.minute / 60.0

        # Record start-of-day equity
        if day_key not in day_start_equity:
            day_start_equity[day_key] = equity

        # ── Overnight force-close ─────────────────────────────────────────────
        if hour_dec >= close_hour_utc:
            for strat, pos in list(active.items()):
                if pos is None:
                    continue
                # Force close at this bar's close
                exit_price  = bar_close
                gross_pnl   = pos.direction * (exit_price - pos.entry_price) * pos.units
                cost_amt    = _trade_cost(
                    pos.strategy, pos.entry_price, pos.units,
                    pos.entry_ts, ts, pos.atr_at_entry, atr_now, avg_atr,
                )
                net_pnl = gross_pnl - cost_amt
                equity += net_pnl
                trade_log.append(_make_trade_record(
                    pos, ts, exit_price, gross_pnl, cost_amt, net_pnl,
                    "overnight_close", bar_i,
                ))
                active[strat] = None

        # ── Process pending entries ───────────────────────────────────────────
        still_pending = []
        for entry_bar, sig_row, limit_p, cancel_b in pending_entries:
            strat     = sig_row["strategy"]
            direction = int(sig_row["direction"])
            atr_sig   = float(sig_row["atr_14"])

            # ── A: limit order at breakout extreme ────────────────────────────
            if strat == "A":
                if bar_i > cancel_b:
                    continue  # expired without fill — cancel
                # Don't fill in after-hours
                if hour_dec >= close_hour_utc:
                    continue
                # Check limit fill: fills when price reaches (or opens through) the limit
                if direction == 1:  # bull: buy limit at breakout high
                    a_fills = bar_open <= limit_p or bar_low <= limit_p
                else:               # bear: sell limit at breakdown low
                    a_fills = bar_open >= limit_p or bar_high >= limit_p
                if not a_fills:
                    still_pending.append((entry_bar, sig_row, limit_p, cancel_b))
                    continue  # price hasn't pulled back to limit yet
                entry = limit_p
            # ── C: immediate fill at EMA20 on entry_bar ───────────────────────
            else:
                if bar_i != entry_bar:
                    still_pending.append((entry_bar, sig_row, limit_p, cancel_b))
                    continue
                entry = float(sig_row["ema20"]) if "ema20" in sig_row.index else bar_open

            # ── Gates (after fill determination) ─────────────────────────────
            if active[strat] is not None:
                continue  # slot occupied — limit cancelled for A, skipped for C
            if day_key in daily_halt:
                continue
            if hour_dec < open_hour_utc:
                continue

            tp, sl = _compute_tp_sl(strat, direction, entry, atr_sig, sig_row)

            if direction == 1 and (tp <= entry or sl >= entry):
                continue
            if direction == -1 and (tp >= entry or sl <= entry):
                continue

            sl_dist = abs(entry - sl)
            if sl_dist < 1e-6:
                continue
            risk_amt = equity * RISK_PER_TRADE
            units    = risk_amt / sl_dist

            pos = Position(
                signal_id    = str(sig_row.get("signal_id", "?")),
                strategy     = strat,
                direction    = direction,
                entry_ts     = ts,
                entry_price  = entry,
                tp_price     = tp,
                sl_price     = sl,
                units        = units,
                max_bars     = _STRAT_PARAMS[strat]["max_bars"],
                atr_at_entry = atr_sig,
                avg_atr      = avg_atr,
                signal_ts    = sig_row.name if hasattr(sig_row, "name") else ts,
            )
            active[strat] = pos

        pending_entries = still_pending

        # ── Check active positions for TP / SL / time exit ───────────────────
        for strat, pos in list(active.items()):
            if pos is None:
                continue
            pos.bars_held += 1

            exit_price = None
            exit_reason = None

            if pos.direction == 1:   # long
                if bar_high >= pos.tp_price:
                    exit_price  = pos.tp_price
                    exit_reason = "tp"
                elif bar_low <= pos.sl_price:
                    exit_price  = pos.sl_price
                    exit_reason = "sl"
            else:                     # short
                if bar_low <= pos.tp_price:
                    exit_price  = pos.tp_price
                    exit_reason = "tp"
                elif bar_high >= pos.sl_price:
                    exit_price  = pos.sl_price
                    exit_reason = "sl"

            if exit_price is None and pos.bars_held >= pos.max_bars:
                exit_price  = bar_close
                exit_reason = "time"

            if exit_price is not None:
                gross_pnl  = pos.direction * (exit_price - pos.entry_price) * pos.units
                cost_amt   = _trade_cost(
                    pos.strategy, pos.entry_price, pos.units,
                    pos.entry_ts, ts, pos.atr_at_entry, atr_now, avg_atr,
                )
                net_pnl    = gross_pnl - cost_amt
                equity    += net_pnl

                trade_log.append(_make_trade_record(
                    pos, ts, exit_price, gross_pnl, cost_amt, net_pnl,
                    exit_reason, bar_i,
                ))
                active[strat] = None

                # Check daily loss limit
                day_eq_start = day_start_equity.get(day_key, initial_capital)
                if (equity - day_eq_start) / day_eq_start < -daily_loss_pct:
                    daily_halt.add(day_key)

        # ── Queue signals at this bar ─────────────────────────────────────────
        if ts in all_signals:
            for sig_row in all_signals[ts]:
                strat = sig_row["strategy"]
                if active[strat] is not None:
                    continue  # slot occupied at signal time
                if day_key in daily_halt:
                    continue
                if hour_dec >= close_hour_utc:
                    continue
                direction = int(sig_row["direction"])
                if strat == "A":
                    # Limit at roll_level = the N-bar extreme that was broken
                    # (the exact level price broke above/below, now acts as support/resistance)
                    lp = float(sig_row["roll_level"])
                    pending_entries.append((bar_i + 1, sig_row, lp, bar_i + 3))
                else:  # C — immediate fill at EMA20 next bar
                    pending_entries.append((bar_i + 1, sig_row, None, None))

        equity_curve.append(equity)
        equity_times.append(ts)

    # Force-close any remaining open positions at last bar
    for strat, pos in active.items():
        if pos is None:
            continue
        exit_price = closes[-1]
        gross_pnl  = pos.direction * (exit_price - pos.entry_price) * pos.units
        cost_amt   = _trade_cost(
            pos.strategy, pos.entry_price, pos.units,
            pos.entry_ts, idx[-1], pos.atr_at_entry, atrs[-1], atr_avgs[-1],
        )
        net_pnl = gross_pnl - cost_amt
        equity += net_pnl
        trade_log.append(_make_trade_record(
            pos, idx[-1], exit_price, gross_pnl, cost_amt, net_pnl,
            "end_of_data", n - 1,
        ))

    # ── Build results ─────────────────────────────────────────────────────────
    eq_series = pd.Series(equity_curve, index=equity_times, name="equity")
    trades_df = pd.DataFrame(trade_log)

    return {
        "equity":        eq_series,
        "trades":        trades_df,
        "initial":       initial_capital,
        "final_equity":  equity,
        "n_halt_days":   len(daily_halt),
    }


def _make_trade_record(
    pos:        Position,
    exit_ts:    pd.Timestamp,
    exit_price: float,
    gross_pnl:  float,
    cost_amt:   float,
    net_pnl:    float,
    reason:     str,
    exit_bar:   int,
) -> dict:
    return {
        "signal_id":   pos.signal_id,
        "strategy":    pos.strategy,
        "direction":   pos.direction,
        "entry_ts":    pos.entry_ts,
        "exit_ts":     exit_ts,
        "entry_price": pos.entry_price,
        "exit_price":  exit_price,
        "units":       pos.units,
        "bars_held":   pos.bars_held,
        "tp_price":    pos.tp_price,
        "sl_price":    pos.sl_price,
        "gross_pnl":   gross_pnl,
        "cost":        cost_amt,
        "net_pnl":     net_pnl,
        "exit_reason": reason,
        "entry_hour":  pos.entry_ts.hour,
        "session":     _get_session(pos.entry_ts.hour),
    }


def _get_session(hour: int) -> str:
    if SESSIONS["NY"][0] <= hour < SESSIONS["NY"][1]:
        return "NY"
    if SESSIONS["London"][0] <= hour < SESSIONS["London"][1]:
        return "London"
    return "Asian"


# ── METRICS ───────────────────────────────────────────────────────────────────

def compute_metrics(results: dict) -> dict:
    """Compute Sharpe, Sortino, MaxDD, CAGR and per-strategy stats."""
    eq   = results["equity"]
    init = results["initial"]
    final = results["final_equity"]
    trades = results["trades"]

    n_days      = (eq.index[-1] - eq.index[0]).days
    total_ret   = (final - init) / init
    cagr        = (1 + total_ret) ** (365 / max(n_days, 1)) - 1

    # Resample to daily equity for Sharpe/Sortino
    daily_eq  = eq.resample("D").last().ffill()
    daily_ret = daily_eq.pct_change().dropna()

    rf        = 0.0  # risk-free rate (crypto)
    excess    = daily_ret - rf
    sharpe    = (excess.mean() / (excess.std() + 1e-12)) * np.sqrt(365)

    downside  = excess[excess < 0]
    sortino   = (excess.mean() / (downside.std() + 1e-12)) * np.sqrt(365)

    # Max drawdown
    roll_max  = eq.cummax()
    drawdown  = (eq - roll_max) / (roll_max + 1e-9)
    max_dd    = drawdown.min()

    # Trade stats
    if trades.empty:
        win_rate = tp_rate = 0.0
        avg_win = avg_loss = 0.0
        profit_factor = 0.0
        n_trades = 0
    else:
        n_trades  = len(trades)
        wins      = trades[trades["net_pnl"] > 0]["net_pnl"]
        losses    = trades[trades["net_pnl"] <= 0]["net_pnl"]
        win_rate  = len(wins) / max(n_trades, 1)
        tp_rate   = (trades["exit_reason"] == "tp").sum() / max(n_trades, 1)
        avg_win   = wins.mean()  if len(wins)   > 0 else 0.0
        avg_loss  = losses.mean() if len(losses) > 0 else 0.0
        pf_num    = wins.sum()
        pf_den    = abs(losses.sum())
        profit_factor = pf_num / max(pf_den, 1e-9)

    # Per-strategy breakdown
    per_strat = {}
    if not trades.empty:
        for strat in trades["strategy"].unique():
            t = trades[trades["strategy"] == strat]
            if t.empty:
                per_strat[strat] = {"n": 0, "net_pnl": 0.0, "win_rate": 0.0}
                continue
            gw = t[t["gross_pnl"] > 0]
            gl = t[t["gross_pnl"] < 0]
            w  = t[t["net_pnl"]   > 0]
            per_strat[strat] = {
                "n":          len(t),
                "gross_pnl":  t["gross_pnl"].sum(),
                "total_cost": t["cost"].sum(),
                "net_pnl":    t["net_pnl"].sum(),
                "win_rate":   len(w) / len(t),
                "avg_pnl":    t["net_pnl"].mean(),
                "avg_gross":  t["gross_pnl"].mean(),
                "avg_cost":   t["cost"].mean(),
                "tp_rate":    (t["exit_reason"] == "tp").sum() / len(t),
                "sl_rate":    (t["exit_reason"] == "sl").sum() / len(t),
                "avg_bars":   t["bars_held"].mean(),
                "gross_pf":   gw["gross_pnl"].sum() / max(abs(gl["gross_pnl"].sum()), 1e-9),
            }

    # Session breakdown
    session_stats = {}
    if not trades.empty:
        for sess in ["Asian", "London", "NY"]:
            t = trades[trades["session"] == sess]
            if t.empty:
                session_stats[sess] = {"n": 0, "net_pnl": 0.0, "win_rate": 0.0}
                continue
            w = t[t["net_pnl"] > 0]
            session_stats[sess] = {
                "n":       len(t),
                "net_pnl": t["net_pnl"].sum(),
                "win_rate":len(w) / len(t),
            }

    gross_pf_num = trades[trades["gross_pnl"] > 0]["gross_pnl"].sum() if not trades.empty else 0
    gross_pf_den = abs(trades[trades["gross_pnl"] < 0]["gross_pnl"].sum()) if not trades.empty else 1
    return {
        "total_return_pct": total_ret * 100,
        "cagr_pct":         cagr * 100,
        "sharpe":           sharpe,
        "sortino":          sortino,
        "max_dd_pct":       max_dd * 100,
        "n_trades":         n_trades,
        "win_rate":         win_rate * 100,
        "tp_rate":          tp_rate * 100,
        "avg_win":          avg_win,
        "avg_loss":         avg_loss,
        "profit_factor":    profit_factor,
        "gross_profit_factor": gross_pf_num / max(gross_pf_den, 1e-9),
        "total_gross_pnl":  trades["gross_pnl"].sum() if not trades.empty else 0,
        "total_costs":      trades["cost"].sum()      if not trades.empty else 0,
        "per_strategy":     per_strat,
        "sessions":         session_stats,
        "n_halt_days":      results["n_halt_days"],
    }


# ── CHARTS ────────────────────────────────────────────────────────────────────

def plot_results(results: dict, metrics: dict, out_dir: Optional[Path] = None) -> None:
    """Generate and save the 6-panel results dashboard."""
    eq     = results["equity"]
    trades = results["trades"]

    fig = plt.figure(figsize=(20, 18))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. Equity curve + drawdown ────────────────────────────────────────────
    ax_eq = fig.add_subplot(gs[0, :2])
    ax_dd = ax_eq.twinx()

    roll_max = eq.cummax()
    dd       = (eq - roll_max) / (roll_max + 1e-9) * 100

    ax_eq.plot(eq.index, eq.values, color="#2196F3", linewidth=1.2, label="Equity")
    ax_eq.axhline(results["initial"], color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    ax_eq.set_ylabel("Equity ($)", color="#2196F3")
    ax_eq.tick_params(axis="y", labelcolor="#2196F3")

    ax_dd.fill_between(dd.index, dd.values, 0, alpha=0.25, color="#F44336")
    ax_dd.set_ylabel("Drawdown (%)", color="#F44336")
    ax_dd.tick_params(axis="y", labelcolor="#F44336")

    ax_eq.set_title(
        f"Equity Curve & Drawdown  |  "
        f"CAGR: {metrics['cagr_pct']:.1f}%  "
        f"MaxDD: {metrics['max_dd_pct']:.1f}%  "
        f"Sharpe: {metrics['sharpe']:.2f}",
        fontsize=11, fontweight="bold",
    )
    ax_eq.legend(loc="upper left")

    # ── 2. Strategy PnL pie ───────────────────────────────────────────────────
    ax_pie = fig.add_subplot(gs[0, 2])
    if not trades.empty:
        strat_pnl = trades.groupby("strategy")["net_pnl"].sum()
        pos_pnl   = strat_pnl[strat_pnl > 0]
        if not pos_pnl.empty:
            colors = {"A": "#2196F3", "B": "#4CAF50", "C": "#FF9800", "D": "#9C27B0"}
            ax_pie.pie(
                pos_pnl.values,
                labels=[f"Strat {k}\n${v:,.0f}" for k, v in pos_pnl.items()],
                colors=[colors.get(k, "#607D8B") for k in pos_pnl.index],
                autopct="%1.0f%%",
                startangle=90,
            )
    ax_pie.set_title("Net PnL by Strategy\n(profitable only)", fontsize=10)

    # ── 3. Hourly win rate heatmap ────────────────────────────────────────────
    ax_hr = fig.add_subplot(gs[1, :2])
    if not trades.empty and "entry_hour" in trades.columns:
        hourly = trades.groupby("entry_hour").apply(
            lambda x: pd.Series({
                "n":       len(x),
                "wr":      (x["net_pnl"] > 0).mean() * 100,
                "net_pnl": x["net_pnl"].sum(),
            })
        )
        hours  = np.arange(24)
        n_arr  = np.array([hourly.loc[h, "n"] if h in hourly.index else 0 for h in hours])
        wr_arr = np.array([hourly.loc[h, "wr"] if h in hourly.index else 0 for h in hours])
        colors = plt.cm.RdYlGn(wr_arr / 100)

        bars = ax_hr.bar(hours, n_arr, color=colors, edgecolor="white", linewidth=0.5)
        ax_hr.set_xlabel("UTC Hour")
        ax_hr.set_ylabel("# Trades")
        ax_hr.set_title("Hourly Trade Count & Win Rate", fontsize=10)
        ax_hr.set_xticks(hours)

        ax_wr = ax_hr.twinx()
        ax_wr.plot(hours, wr_arr, "o--", color="#1565C0", markersize=4, linewidth=1)
        ax_wr.axhline(50, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
        ax_wr.set_ylabel("Win Rate (%)", color="#1565C0")
        ax_wr.set_ylim(0, 100)

    # ── 4. Exit reason breakdown ──────────────────────────────────────────────
    ax_exit = fig.add_subplot(gs[1, 2])
    if not trades.empty:
        exit_counts = trades["exit_reason"].value_counts()
        clr = {"tp": "#4CAF50", "sl": "#F44336", "time": "#FF9800",
               "overnight_close": "#9C27B0", "end_of_data": "#607D8B"}
        ax_exit.bar(
            exit_counts.index,
            exit_counts.values,
            color=[clr.get(k, "#607D8B") for k in exit_counts.index],
            edgecolor="white",
        )
        ax_exit.set_title("Exit Reason Breakdown", fontsize=10)
        ax_exit.set_ylabel("# Trades")
        for i, (k, v) in enumerate(exit_counts.items()):
            ax_exit.text(i, v + 0.5, str(v), ha="center", va="bottom", fontsize=8)

    # ── 5. Per-strategy bar chart ─────────────────────────────────────────────
    ax_strat = fig.add_subplot(gs[2, 0])
    if not trades.empty:
        s_groups = trades.groupby("strategy")
        strat_labels = list(s_groups.groups.keys())
        strat_pnls   = [s_groups.get_group(s)["net_pnl"].sum() for s in strat_labels]
        colors       = ["#4CAF50" if p >= 0 else "#F44336" for p in strat_pnls]
        ax_strat.bar(strat_labels, strat_pnls, color=colors, edgecolor="white")
        ax_strat.axhline(0, color="gray", linewidth=0.7)
        ax_strat.set_title("Net PnL by Strategy ($)", fontsize=10)
        ax_strat.set_ylabel("Net PnL ($)")

    # ── 6. Session performance ────────────────────────────────────────────────
    ax_sess = fig.add_subplot(gs[2, 1])
    sess_data = metrics.get("sessions", {})
    if sess_data:
        sess_names = list(sess_data.keys())
        sess_pnls  = [sess_data[s]["net_pnl"] for s in sess_names]
        sess_wrs   = [sess_data[s]["win_rate"] * 100 for s in sess_names]
        colors     = ["#4CAF50" if p >= 0 else "#F44336" for p in sess_pnls]
        ax_sess.bar(sess_names, sess_pnls, color=colors, edgecolor="white")
        ax_sess.axhline(0, color="gray", linewidth=0.7)
        ax_sess.set_title("Net PnL by Session ($)", fontsize=10)

        ax_wr2 = ax_sess.twinx()
        ax_wr2.plot(sess_names, sess_wrs, "o--", color="#1565C0", markersize=8)
        ax_wr2.set_ylabel("Win Rate (%)", color="#1565C0")
        ax_wr2.set_ylim(0, 100)

    # ── 7. Monthly returns heatmap ────────────────────────────────────────────
    ax_mo = fig.add_subplot(gs[2, 2])
    if not trades.empty:
        trades_copy = trades.copy()
        trades_copy["year"]  = trades_copy["exit_ts"].dt.year
        trades_copy["month"] = trades_copy["exit_ts"].dt.month
        monthly = trades_copy.groupby(["year", "month"])["net_pnl"].sum().unstack(fill_value=0)

        if not monthly.empty:
            vmax = max(abs(monthly.values).max(), 1)
            im   = ax_mo.imshow(
                monthly.values,
                aspect="auto",
                cmap="RdYlGn",
                vmin=-vmax, vmax=vmax,
            )
            ax_mo.set_xticks(range(len(monthly.columns)))
            ax_mo.set_xticklabels([str(m) for m in monthly.columns], fontsize=7)
            ax_mo.set_yticks(range(len(monthly.index)))
            ax_mo.set_yticklabels([str(y) for y in monthly.index], fontsize=7)
            ax_mo.set_title("Monthly PnL Heatmap", fontsize=10)
            plt.colorbar(im, ax=ax_mo, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"BTC 15M Scalping — Phase 6 Backtest Results\n"
        f"Return: {metrics['total_return_pct']:.1f}%  |  "
        f"Trades: {metrics['n_trades']:,}  |  "
        f"Win Rate: {metrics['win_rate']:.1f}%  |  "
        f"Profit Factor: {metrics['profit_factor']:.2f}",
        fontsize=13, fontweight="bold", y=0.98,
    )

    out = (out_dir or RESULTS) / "phase6_backtest_dashboard.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Dashboard → {out.relative_to(out.parent.parent)}")
