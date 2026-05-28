"""
backtest/engine.py — Bar-by-bar event-driven backtest engine.

Design principles:
  - Strict no-lookahead: signals are shifted +1 bar before execution.
    Entry at bar t+1 open (approximated by close[t+1] post-slippage).
  - Combined entry filter: signal == 1 AND ml_signal == 1 (ML gate).
  - One position at a time per asset (no pyramiding).
  - SL / TP checked intra-bar using low/high (conservative: assumes worst fill).
  - Commission and slippage applied on every entry and exit fill.
  - Capital updated after every closed trade (compounding).

Metrics returned per asset:
  total_return_pct, annual_return_pct, sharpe_ratio, sortino_ratio,
  max_drawdown_pct, win_rate_pct, profit_factor, total_trades,
  avg_trade_duration_days

Comparison table printed: Strategy vs Buy & Hold for BTC and XAU.
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BT, STRAT, RESULTS, ASSETS
from strategy.sizing import size_trade, apply_slippage, commission_cost

warnings.filterwarnings("ignore")


# ── TRADE RECORD ─────────────────────────────────────────────────────────────

class Trade:
    __slots__ = [
        "entry_date", "exit_date", "direction",
        "entry_price", "exit_price",
        "units", "stop_price", "tp_price",
        "pnl", "pnl_pct", "exit_reason",
        "entry_capital",
        # Improvement 4 — trailing stop fields
        "atr_at_entry", "trail_active", "trail_ref",
        # Signal mode: 1=breakout/breakdown, 2=pullback
        "signal_mode",
    ]

    def __init__(
        self,
        entry_date, direction, entry_price, units,
        stop_price, tp_price, entry_capital,
        atr_at_entry: float = 0.0,
        signal_mode:  int   = 1,
    ):
        self.entry_date    = entry_date
        self.direction     = direction
        self.entry_price   = entry_price
        self.units         = units
        self.stop_price    = stop_price
        self.tp_price      = tp_price
        self.entry_capital = entry_capital
        self.exit_date     = None
        self.exit_price    = None
        self.pnl           = 0.0
        self.pnl_pct       = 0.0
        self.exit_reason   = ""
        # Trailing stop state
        self.atr_at_entry  = atr_at_entry
        self.trail_active  = False
        self.trail_ref     = entry_price    # highest (long) or lowest (short) close seen
        # Entry mode
        self.signal_mode   = signal_mode

    def close(self, exit_date, exit_price: float, reason: str) -> None:
        self.exit_date   = exit_date
        self.exit_price  = exit_price
        self.exit_reason = reason

        raw_pnl  = self.direction * (exit_price - self.entry_price) * self.units
        cost_in  = commission_cost(self.entry_price * self.units)
        cost_out = commission_cost(exit_price * self.units)
        self.pnl     = raw_pnl - cost_in - cost_out
        self.pnl_pct = self.pnl / self.entry_capital * 100.0

    @property
    def duration_days(self) -> int:
        if self.exit_date is None:
            return 0
        return max((self.exit_date - self.entry_date).days, 1)


# ── CORE BACKTEST LOOP ────────────────────────────────────────────────────────

def _run_asset(df: pd.DataFrame, asset_key: str = "BTC") -> Tuple[pd.Series, List[Trade]]:
    """
    Bar-by-bar simulation for one asset.

    Entry rules:
      - signal (shifted +1) == 1 or -1
      - ml_signal (shifted +1) == 1

    Exit rules (checked in order each bar):
      1. Low ≤ stop_price  → stop hit (long)   / High ≥ stop_price → stop hit (short)
      2. High ≥ tp_price   → TP hit   (long)   / Low  ≤ tp_price   → TP hit   (short)
      3. Signal flips or becomes 0             → exit at close

    Returns:
      equity_curve : pd.Series indexed like df, starting at BT["initial_capital"]
      trades       : list of closed Trade objects
    """
    capital      = BT["initial_capital"]
    atr_col      = f"atr_{STRAT['atr_window']}"
    allow        = STRAT["allow_shorting"]
    min_hold     = STRAT.get("min_hold_bars", 0)
    # Improvement 4: trailing stop only for BTC
    use_trail    = (asset_key == "BTC")
    trail_trig   = STRAT.get("trail_trigger_mult", 1.0)
    trail_dist   = STRAT.get("trail_stop_mult",    1.0)

    # Shift signals by 1 bar — enter at next bar's execution price (close used as proxy)
    # Using close as entry proxy (open unavailable for daily data from yfinance/ccxt)
    sig_raw   = df["signal"].shift(1).fillna(0).astype(int)
    ml_sig    = df["ml_signal"].shift(1).fillna(0).astype(int)
    sig_mode  = (df["signal_mode"].shift(1).fillna(0).astype(int)
                 if "signal_mode" in df.columns
                 else pd.Series(1, index=df.index, dtype=int))

    # Combined entry gate
    entry_sig = pd.Series(0, index=df.index, dtype=int)
    long_gate  = (sig_raw == 1) & (ml_sig == 1)
    short_gate = (sig_raw == -1) & (ml_sig == 1) & allow
    entry_sig[long_gate]  = 1
    entry_sig[short_gate] = -1

    equity      = pd.Series(np.nan, index=df.index)
    equity.iloc[0] = capital

    active_trade: Optional[Trade] = None
    bars_held     = 0                # Fix 1: bars elapsed since current trade entry
    session_closed = False           # XAU: after SL/TP in a sticky session, block re-entry
    #                                #  until signal resets to 0 (exit sticky window)
    trades: List[Trade] = []

    bars = df.index.tolist()

    for i, ts in enumerate(bars):
        bar = df.loc[ts]

        # Clear session guard when signal session ends (signal goes back to 0)
        if entry_sig.iloc[i] == 0:
            session_closed = False

        # ── Manage open position ──────────────────────────────────────────────
        if active_trade is not None:
            bars_held += 1
            t  = active_trade
            lo = bar["low"]
            hi = bar["high"]
            cl = bar["close"]

            # Improvement 4: update trailing stop for BTC
            if use_trail and t.atr_at_entry > 0:
                trig_px = t.atr_at_entry * trail_trig
                dist_px = t.atr_at_entry * trail_dist
                if t.direction == 1:
                    t.trail_ref = max(t.trail_ref, cl)
                    if not t.trail_active and (t.trail_ref - t.entry_price) >= trig_px:
                        t.trail_active = True
                    if t.trail_active:
                        new_stop = t.trail_ref - dist_px
                        if new_stop > t.stop_price:      # only ever raise the stop
                            t.stop_price = new_stop
                else:
                    t.trail_ref = min(t.trail_ref, cl)
                    if not t.trail_active and (t.entry_price - t.trail_ref) >= trig_px:
                        t.trail_active = True
                    if t.trail_active:
                        new_stop = t.trail_ref + dist_px
                        if new_stop < t.stop_price:      # only ever lower the stop
                            t.stop_price = new_stop

            exit_price  = None
            exit_reason = ""

            # Hard SL/TP (or trailing stop): always checked
            if t.direction == 1:
                if lo <= t.stop_price:
                    reason      = "TrailSL" if t.trail_active else "SL"
                    exit_price  = apply_slippage(t.stop_price, -1)
                    exit_reason = reason
                elif hi >= t.tp_price:
                    exit_price  = apply_slippage(t.tp_price, -1)
                    exit_reason = "TP"
            else:
                if hi >= t.stop_price:
                    reason      = "TrailSL" if t.trail_active else "SL"
                    exit_price  = apply_slippage(t.stop_price, 1)
                    exit_reason = reason
                elif lo <= t.tp_price:
                    exit_price  = apply_slippage(t.tp_price, 1)
                    exit_reason = "TP"

            # Signal exit: only allowed after min_hold_bars
            if exit_price is None and bars_held >= min_hold:
                cur_sig = entry_sig.iloc[i]
                if cur_sig != t.direction or cur_sig == 0:
                    exit_price  = apply_slippage(cl, -t.direction)
                    exit_reason = "Signal"

            if exit_price is not None:
                t.close(ts, exit_price, exit_reason)
                capital += t.pnl
                trades.append(t)
                active_trade = None
                bars_held    = 0
                # After hard SL/TP exit: lock out re-entry until this sticky
                # signal session fully resets to 0.  Prevents dozens of
                # re-entries when the BB mean-reversion signal stays active
                # while price drifts below the lower band.
                if exit_reason not in ("Signal", "EOD"):
                    session_closed = True

        # ── Open new position (only if flat AND session not closed) ──────────
        if active_trade is None and not session_closed:
            direction = entry_sig.iloc[i]
            if direction != 0:
                # XAU: fill at open — mean-reversion bounce starts at open,
                # not at close; using close overshoots and shrinks avg_win.
                # BTC: fill at close — momentum confirmation requires close price.
                fill_base = bar["open"] if asset_key == "XAU" else bar["close"]
                atr = bar[atr_col]
                if np.isnan(atr) or atr < 1e-9:
                    equity.iloc[i] = capital
                    continue

                fill_price = apply_slippage(fill_base, direction)
                sizing     = size_trade(capital, fill_price, atr, direction, asset_key)

                if sizing["units"] > 0:
                    active_trade = Trade(
                        entry_date    = ts,
                        direction     = direction,
                        entry_price   = fill_price,
                        units         = sizing["units"],
                        stop_price    = sizing["stop_price"],
                        tp_price      = sizing["tp_price"],
                        entry_capital = capital,
                        atr_at_entry  = atr,
                        signal_mode   = int(sig_mode.iloc[i]),
                    )

        # ── Mark-to-market equity ─────────────────────────────────────────────
        if active_trade is not None:
            unrealised = (
                active_trade.direction
                * (bar["close"] - active_trade.entry_price)
                * active_trade.units
            )
            equity.iloc[i] = capital + unrealised
        else:
            equity.iloc[i] = capital

    # Close any open trade at last bar's close
    if active_trade is not None:
        last_ts  = bars[-1]
        last_cl  = df["close"].iloc[-1]
        exit_px  = apply_slippage(last_cl, -active_trade.direction)
        active_trade.close(last_ts, exit_px, "EOD")
        capital += active_trade.pnl
        trades.append(active_trade)
        equity.iloc[-1] = capital

    return equity, trades


# ── METRICS ──────────────────────────────────────────────────────────────────

def _compute_metrics(
    equity:    pd.Series,
    trades:    List[Trade],
    df:        pd.DataFrame,
    asset_key: str,
) -> Dict:
    """Compute all 9 strategy performance metrics."""

    n_days      = (equity.index[-1] - equity.index[0]).days
    n_years     = max(n_days / 365.25, 1 / 365.25)

    start_cap   = BT["initial_capital"]
    end_cap     = equity.iloc[-1]

    total_ret   = (end_cap / start_cap - 1) * 100.0
    annual_ret  = ((end_cap / start_cap) ** (1 / n_years) - 1) * 100.0

    # Daily returns of equity curve
    daily_ret   = equity.pct_change().dropna()
    excess      = daily_ret - 0.0  # risk-free ~ 0 for simplicity

    ann_factor  = np.sqrt(252)
    sharpe      = (excess.mean() / excess.std() * ann_factor) if excess.std() > 0 else 0.0

    downside    = daily_ret[daily_ret < 0]
    sortino_std = downside.std() if len(downside) > 1 else np.nan
    sortino     = (excess.mean() / sortino_std * ann_factor) if (sortino_std and sortino_std > 0) else 0.0

    roll_max    = equity.cummax()
    drawdown    = (equity - roll_max) / roll_max * 100.0
    max_dd      = drawdown.min()

    n_trades    = len(trades)
    wins        = [t for t in trades if t.pnl > 0]
    losses      = [t for t in trades if t.pnl <= 0]
    win_rate    = len(wins) / n_trades * 100.0 if n_trades > 0 else 0.0

    gross_profit = sum(t.pnl for t in wins)
    gross_loss   = abs(sum(t.pnl for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    avg_duration = np.mean([t.duration_days for t in trades]) if trades else 0.0

    return {
        "total_return_pct":       total_ret,
        "annual_return_pct":      annual_ret,
        "sharpe_ratio":           sharpe,
        "sortino_ratio":          sortino,
        "max_drawdown_pct":       max_dd,
        "win_rate_pct":           win_rate,
        "profit_factor":          profit_factor,
        "total_trades":           n_trades,
        "avg_trade_duration_days": avg_duration,
    }


def _buy_hold_metrics(df: pd.DataFrame) -> Dict:
    """Compute Buy & Hold metrics for the same date range."""
    start = df["close"].iloc[0]
    end   = df["close"].iloc[-1]

    eq          = df["close"] / start * BT["initial_capital"]
    daily_ret   = eq.pct_change().dropna()
    n_years     = (df.index[-1] - df.index[0]).days / 365.25
    ann_factor  = np.sqrt(252)

    total_ret   = (end / start - 1) * 100.0
    annual_ret  = ((end / start) ** (1 / max(n_years, 1e-9)) - 1) * 100.0
    sharpe      = (daily_ret.mean() / daily_ret.std() * ann_factor) if daily_ret.std() > 0 else 0.0

    downside    = daily_ret[daily_ret < 0]
    sortino_std = downside.std() if len(downside) > 1 else np.nan
    sortino     = (daily_ret.mean() / sortino_std * ann_factor) if (sortino_std and sortino_std > 0) else 0.0

    roll_max    = eq.cummax()
    drawdown    = (eq - roll_max) / roll_max * 100.0
    max_dd      = drawdown.min()

    return {
        "total_return_pct":        total_ret,
        "annual_return_pct":       annual_ret,
        "sharpe_ratio":            sharpe,
        "sortino_ratio":           sortino,
        "max_drawdown_pct":        max_dd,
        "win_rate_pct":            np.nan,
        "profit_factor":           np.nan,
        "total_trades":            1,
        "avg_trade_duration_days": (df.index[-1] - df.index[0]).days,
    }


# ── TRADE LOG ────────────────────────────────────────────────────────────────

def _save_trade_log(trades: List[Trade], asset_key: str) -> None:
    if not trades:
        return
    rows = []
    for t in trades:
        rows.append({
            "asset":         asset_key,
            "entry_date":    t.entry_date.date(),
            "exit_date":     t.exit_date.date() if t.exit_date else "",
            "direction":     "LONG" if t.direction == 1 else "SHORT",
            "signal_mode":   t.signal_mode,
            "entry_price":   round(t.entry_price, 4),
            "exit_price":    round(t.exit_price, 4) if t.exit_price else "",
            "units":         round(t.units, 6),
            "pnl":           round(t.pnl, 2),
            "pnl_pct":       round(t.pnl_pct, 3),
            "exit_reason":   t.exit_reason,
            "duration_days": t.duration_days,
        })
    path = RESULTS / f"{asset_key.lower()}_trade_log.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Trade log → results/{path.name}  ({len(rows)} trades)")


# ── EQUITY CHART ─────────────────────────────────────────────────────────────

def _plot_equity(
    equity:    pd.Series,
    df:        pd.DataFrame,
    asset_key: str,
    metrics:   Dict,
    bh_metrics: Dict,
) -> None:
    cfg     = ASSETS[asset_key]
    bh_eq   = df["close"] / df["close"].iloc[0] * BT["initial_capital"]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    fig.suptitle(
        f"{cfg['label']} — Strategy vs Buy & Hold\n"
        f"Sharpe {metrics['sharpe_ratio']:.2f}  |  "
        f"Ann.Return {metrics['annual_return_pct']:.1f}%  |  "
        f"MaxDD {metrics['max_drawdown_pct']:.1f}%  |  "
        f"Trades {metrics['total_trades']}",
        fontsize=12, fontweight="bold",
    )

    ax1.plot(equity.index, equity.values, color=cfg["color"],
             linewidth=1.2, label="Strategy")
    ax1.plot(bh_eq.index, bh_eq.values, color="#888888",
             linewidth=0.8, linestyle="--", alpha=0.7, label="Buy & Hold")
    ax1.axhline(BT["initial_capital"], color="#ccc", linewidth=0.5, linestyle=":")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.set_ylabel("Portfolio Value (USD)", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.25)

    # Drawdown panel
    roll_max = equity.cummax()
    dd       = (equity - roll_max) / roll_max * 100.0
    ax2.fill_between(dd.index, dd.values, 0, color="#E53935", alpha=0.5)
    ax2.set_ylabel("Drawdown %", fontsize=9)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    out = RESULTS / f"{asset_key.lower()}_equity.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Equity chart → results/{out.name}")


# ── COMPARISON TABLE ─────────────────────────────────────────────────────────

def _print_comparison(
    asset_key:  str,
    strat_m:    Dict,
    bh_m:       Dict,
    trades:     List[Trade],
) -> None:
    """Print Strategy vs Buy & Hold side-by-side table."""
    metrics_labels = [
        ("Total Return %",          "total_return_pct",        "{:>10.2f}%"),
        ("Annual Return %",         "annual_return_pct",       "{:>10.2f}%"),
        ("Sharpe Ratio",            "sharpe_ratio",            "{:>10.3f}"),
        ("Sortino Ratio",           "sortino_ratio",           "{:>10.3f}"),
        ("Max Drawdown %",          "max_drawdown_pct",        "{:>10.2f}%"),
        ("Win Rate %",              "win_rate_pct",            "{:>10.1f}%"),
        ("Profit Factor",           "profit_factor",           "{:>10.3f}"),
        ("Total Trades",            "total_trades",            "{:>10d}"),
        ("Avg Trade Duration (d)",  "avg_trade_duration_days", "{:>10.1f}"),
    ]

    print(f"\n  {'─'*64}")
    print(f"  {asset_key:^64}")
    print(f"  {'─'*64}")
    print(f"  {'Metric':<30} {'Strategy':>15} {'Buy&Hold':>15}")
    print(f"  {'─'*64}")

    for label, key, fmt in metrics_labels:
        sv = strat_m[key]
        bv = bh_m[key]

        def _fmt(v, f):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return f"{'—':>10}"
            if "d}" in f:
                return f.format(int(v))
            return f.format(v)

        print(f"  {label:<30} {_fmt(sv, fmt):>15} {_fmt(bv, fmt):>15}")

    print(f"  {'─'*64}")

    # Exit reason breakdown
    if trades:
        reasons = pd.Series([t.exit_reason for t in trades]).value_counts()
        print(f"\n  Exit reasons:")
        for reason, count in reasons.items():
            print(f"    {reason:<12} {count:>6,}  ({count/len(trades)*100:.1f}%)")

    # Mode 1 vs Mode 2 trade attribution (BTC only)
    if trades and asset_key == "BTC":
        m1 = [t for t in trades if t.signal_mode == 1]
        m2 = [t for t in trades if t.signal_mode == 2]
        print(f"\n  Trade attribution:")
        print(f"    Mode 1 — Breakout/Breakdown : {len(m1):>3} trades"
              f"  WR {len([t for t in m1 if t.pnl>0])/max(len(m1),1)*100:.0f}%"
              f"  avg P&L ${np.mean([t.pnl for t in m1]):.2f}" if m1 else
              f"    Mode 1 — Breakout/Breakdown :   0 trades")
        print(f"    Mode 2 — Pullback           : {len(m2):>3} trades"
              f"  WR {len([t for t in m2 if t.pnl>0])/max(len(m2),1)*100:.0f}%"
              f"  avg P&L ${np.mean([t.pnl for t in m2]):.2f}" if m2 else
              f"    Mode 2 — Pullback           :   0 trades")

    # Avg win vs avg loss comparison (key diagnostic for fill-lag fix)
    if trades:
        wins   = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl < 0]
        avg_w  = np.mean([t.pnl for t in wins])   if wins   else 0.0
        avg_l  = np.mean([t.pnl for t in losses]) if losses else 0.0
        ratio  = avg_w / abs(avg_l) if avg_l != 0 else float("inf")
        print(f"\n  {'─'*40}")
        print(f"  Avg win  : ${avg_w:>8.2f}  ({len(wins):>3} trades)")
        print(f"  Avg loss : ${avg_l:>8.2f}  ({len(losses):>3} trades)")
        print(f"  Win/Loss ratio: {ratio:.3f}  (need >1.0 for PF improvement)")
        if asset_key == "XAU":
            print(f"  v3 baseline  : avg_win=$63.75  avg_loss=$-89.50  ratio=0.712")
            delta_w = avg_w - 63.75
            delta_l = avg_l - (-89.50)
            print(f"  Δ vs v3      : avg_win {delta_w:+.2f}  avg_loss {delta_l:+.2f}")
        print(f"  {'─'*40}")


# ── MAIN ENTRY POINT ─────────────────────────────────────────────────────────

def run_backtest(ml_results: Dict[str, pd.DataFrame]) -> Dict:
    """
    Run the bar-by-bar backtest for BTC and XAU.

    Parameters
    ----------
    ml_results : dict  {"BTC": df, "XAU": df}
        DataFrames from Phase 6, containing all feature, signal,
        ml_confidence, and ml_signal columns.

    Returns
    -------
    dict with keys "BTC" and "XAU", each containing:
        {"equity": pd.Series, "trades": list, "metrics": dict, "bh_metrics": dict}
    """
    results = {}

    for asset_key, df in ml_results.items():
        print(f"\n  {'─'*58}")
        print(f"  Running {asset_key} backtest …")
        print(f"  {'─'*58}")

        # Ensure ml_signal exists (fallback if Phase 6 was skipped)
        if "ml_signal" not in df.columns:
            df = df.copy()
            df["ml_signal"] = 1   # no ML filter — pass everything

        equity, trades = _run_asset(df, asset_key)

        strat_m  = _compute_metrics(equity, trades, df, asset_key)
        bh_m     = _buy_hold_metrics(df)

        _print_comparison(asset_key, strat_m, bh_m, trades)
        _save_trade_log(trades, asset_key)
        _plot_equity(equity, df, asset_key, strat_m, bh_m)

        results[asset_key] = {
            "equity":    equity,
            "trades":    trades,
            "metrics":   strat_m,
            "bh_metrics": bh_m,
            "df":        df,
        }

    # ── Cross-asset combined summary ──────────────────────────────────────────
    print(f"\n  {'═'*64}")
    print(f"  {'COMBINED SUMMARY':^64}")
    print(f"  {'═'*64}")
    print(f"  {'Asset':<10} {'Ann.Ret%':>10} {'Sharpe':>8} {'MaxDD%':>8} "
          f"{'WinRate%':>10} {'Trades':>8}")
    print(f"  {'─'*64}")
    for key, r in results.items():
        m = r["metrics"]
        print(f"  {key:<10} {m['annual_return_pct']:>9.1f}% {m['sharpe_ratio']:>8.3f} "
              f"{m['max_drawdown_pct']:>7.1f}% {m['win_rate_pct']:>9.1f}% "
              f"{m['total_trades']:>8,}")
    print(f"  {'═'*64}")

    return results
