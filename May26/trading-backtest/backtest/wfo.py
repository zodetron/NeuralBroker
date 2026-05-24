"""
backtest/wfo.py — Walk-Forward Optimization (Phase 8).

Design:
  - Sliding windows: 6-month train → 2-month OOS test (from WFO config).
  - Parameter grid: atr_stop_mult × momentum_roc_period (5×5 = 25 combos).
  - In-sample selection: best Sharpe ratio on training window.
  - OOS application: apply IS-best params to test window (no peeking).
  - Output: per-window table (IS best params + OOS Sharpe), OOS-only equity curve.

Efficiency: signals are precomputed once per momentum_roc_period variant,
then reused across windows and atr_stop_mult sweeps.
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
from config import BT, STRAT, WFO, RESULTS, ASSETS
from strategy.sizing import apply_slippage, commission_cost, size_trade

warnings.filterwarnings("ignore")


# ── SIGNAL COMPUTATION WITH PARAM OVERRIDE ────────────────────────────────────

def _compute_roc(close: pd.Series, period: int) -> pd.Series:
    """Rate of change over `period` bars (uses log returns × 100)."""
    return np.log(close / close.shift(period)) * 100.0


def _compute_signals_for_period(df: pd.DataFrame, roc_period: int) -> pd.Series:
    """
    Recompute the strategy signal using a specific momentum_roc_period.
    Mirrors strategy/signals.py logic but is self-contained for WFO speed.
    Returns a signal Series (+1 / 0 / -1).
    """
    allow = STRAT["allow_shorting"]

    # Get (or compute) the ROC column for this period
    roc_col = f"roc_{roc_period}"
    if roc_col in df.columns:
        roc = df[roc_col]
    else:
        roc = _compute_roc(df["close"], roc_period)

    roc_std  = roc.rolling(252, min_periods=60).std().replace(0, np.nan)
    roc_norm = roc / roc_std
    rsi_norm = (df["rsi_14"] - 50.0) / 50.0
    mom      = (roc_norm + rsi_norm) / 2.0

    sig = pd.Series(0, index=df.index, dtype=int)

    bull = df["regime"] == "Bull"
    sig[bull & (mom > 0) & (df["rsi_14"] > STRAT["momentum_rsi_min"])] = 1

    side = df["regime"] == "Sideways"
    sig[side & (df["rsi_14"] < STRAT["rsi_oversold"])]  = 1
    if allow:
        sig[side & (df["rsi_14"] > STRAT["rsi_overbought"])] = -1

    bear = df["regime"] == "Bear"
    if allow:
        sig[bear] = -1

    bad_mask = df["regime"].isna() | df["rsi_14"].isna() | mom.isna()
    sig[bad_mask] = 0

    return sig


# ── MINI BACKTEST (one window, configurable params) ───────────────────────────

def _mini_backtest(
    df:            pd.DataFrame,
    signal:        pd.Series,
    atr_stop_mult: float,
    atr_tp_mult:   float,
    capital_start: float,
) -> Tuple[pd.Series, float]:
    """
    Lightweight bar-by-bar backtest for one window.

    Parameters
    ----------
    df            : OHLCV + feature DataFrame (any slice)
    signal        : pre-shifted entry signal series (same index as df)
    atr_stop_mult : stop-loss ATR multiplier (param being optimized)
    atr_tp_mult   : TP ATR multiplier (kept at STRAT default × scale)
    capital_start : starting capital for this window

    Returns
    -------
    equity   : portfolio value series
    sharpe   : annualised Sharpe of daily equity returns
    """
    atr_col   = f"atr_{STRAT['atr_window']}"
    allow     = STRAT["allow_shorting"]
    min_hold  = STRAT.get("min_hold_bars", 0)   # Fix 1
    capital   = capital_start

    equity    = pd.Series(np.nan, index=df.index)
    equity.iloc[0] = capital

    active        = None   # dict holding open position state
    bars_held_w   = 0      # Fix 1: bars held counter for WFO mini-backtest
    atr_tp_mult_  = atr_tp_mult   # may differ from STRAT

    for i, ts in enumerate(df.index):
        bar = df.loc[ts]
        lo, hi, cl = bar["low"], bar["high"], bar["close"]
        atr         = bar[atr_col]

        # ── Manage open position ──────────────────────────────────────────────
        if active is not None:
            bars_held_w += 1
            exit_price  = None
            exit_reason = ""
            d           = active["direction"]

            # Hard SL/TP always allowed
            if d == 1:
                if lo <= active["stop"]:
                    exit_price  = apply_slippage(active["stop"], -1)
                    exit_reason = "SL"
                elif hi >= active["tp"]:
                    exit_price  = apply_slippage(active["tp"], -1)
                    exit_reason = "TP"
            else:
                if hi >= active["stop"]:
                    exit_price  = apply_slippage(active["stop"], 1)
                    exit_reason = "SL"
                elif lo <= active["tp"]:
                    exit_price  = apply_slippage(active["tp"], 1)
                    exit_reason = "TP"

            # Signal exit only after min_hold_bars (Fix 1)
            cur_sig = signal.iloc[i] if i < len(signal) else 0
            if exit_price is None and bars_held_w >= min_hold and (cur_sig != d or cur_sig == 0):
                exit_price  = apply_slippage(cl, -d)
                exit_reason = "Signal"

            if exit_price is not None:
                raw_pnl  = d * (exit_price - active["entry"]) * active["units"]
                cost_in  = commission_cost(active["entry"] * active["units"])
                cost_out = commission_cost(exit_price * active["units"])
                capital += raw_pnl - cost_in - cost_out
                active      = None
                bars_held_w = 0

        # ── Open new position ─────────────────────────────────────────────────
        if active is None and not (np.isnan(atr) or atr < 1e-9):
            direction = signal.iloc[i] if i < len(signal) else 0
            if direction != 0:
                fill    = apply_slippage(cl, direction)
                stop_d  = atr_stop_mult * atr
                tp_d    = atr_tp_mult_ * atr

                if stop_d > 1e-9:
                    risk_usd = capital * STRAT["risk_per_trade"]
                    units    = risk_usd / stop_d
                    max_val  = capital * STRAT["max_position_pct"]
                    value    = units * fill
                    if value > max_val:
                        value = max_val
                        units = value / fill

                    if units > 0:
                        active = {
                            "direction": direction,
                            "entry":     fill,
                            "units":     units,
                            "stop":      fill - direction * stop_d,
                            "tp":        fill + direction * tp_d,
                        }

        # Mark-to-market
        if active is not None:
            equity.iloc[i] = capital + active["direction"] * (cl - active["entry"]) * active["units"]
        else:
            equity.iloc[i] = capital

    # Close any open trade at last bar
    if active is not None:
        last_cl = df["close"].iloc[-1]
        d       = active["direction"]
        exit_px = apply_slippage(last_cl, -d)
        raw_pnl = d * (exit_px - active["entry"]) * active["units"]
        cost_in = commission_cost(active["entry"] * active["units"])
        cost_out= commission_cost(exit_px * active["units"])
        capital += raw_pnl - cost_in - cost_out
        equity.iloc[-1] = capital

    daily_ret = equity.pct_change().dropna()
    if daily_ret.std() > 0 and len(daily_ret) > 1:
        sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252)
    else:
        sharpe = 0.0

    return equity, sharpe


# ── WINDOW GENERATION ────────────────────────────────────────────────────────

def _make_wfo_windows(index: pd.DatetimeIndex) -> List[Tuple]:
    """Generate (train_start, train_end, test_start, test_end) tuples."""
    train_off = pd.DateOffset(months=WFO["train_months"])
    test_off  = pd.DateOffset(months=WFO["test_months"])

    windows   = []
    t_start   = index[0]

    while True:
        t_end   = t_start + train_off
        v_start = t_end
        v_end   = v_start + test_off
        if v_end > index[-1]:
            break
        windows.append((t_start, t_end, v_start, v_end))
        t_start = t_start + test_off

    return windows


# ── FULL WFO PIPELINE ────────────────────────────────────────────────────────

def run_wfo(ml_results: Dict[str, pd.DataFrame]) -> Dict:
    """
    Walk-Forward Optimization for BTC and XAU.

    Returns dict {"BTC": {...}, "XAU": {...}} with:
      oos_equity   : pd.Series — OOS-only equity curve (chained across windows)
      window_df    : pd.DataFrame — per-window results
      best_params  : most-frequent OOS param combo
    """
    results = {}

    stop_grid  = WFO["param_grid"]["atr_stop_mult"]
    roc_grid   = WFO["param_grid"]["momentum_roc_period"]
    # TP is scaled proportionally with stop to keep R:R constant
    base_rr    = STRAT["atr_tp_mult"] / STRAT["atr_stop_mult"]

    for asset_key, df in ml_results.items():
        print(f"\n  {'─'*60}")
        print(f"  {asset_key} Walk-Forward Optimization")
        print(f"  {'─'*60}")
        print(f"  Grid: {len(stop_grid)} stop_mults × {len(roc_grid)} roc_periods "
              f"= {len(stop_grid)*len(roc_grid)} combos/window")

        # ── Precompute signals for every roc_period variant ──────────────────
        all_signals: Dict[int, pd.Series] = {}
        for rp in roc_grid:
            raw = _compute_signals_for_period(df, rp)
            # Apply ML gate
            if "ml_signal" in df.columns:
                ml_gate = df["ml_signal"].fillna(0).astype(int)
                gated   = raw.where((raw == 1) & (ml_gate == 1), 0)
                # Shorts pass without ML gate (conservative)
                if STRAT["allow_shorting"]:
                    gated = gated.where(raw != -1, raw)
                all_signals[rp] = gated
            else:
                all_signals[rp] = raw

        windows      = _make_wfo_windows(df.index)
        if not windows:
            print(f"  ⚠  Not enough data for WFO.")
            results[asset_key] = None
            continue

        print(f"  {len(windows)} WFO windows  "
              f"(train={WFO['train_months']}mo / test={WFO['test_months']}mo)")

        oos_equity_segments: List[pd.Series] = []
        window_records: List[Dict] = []
        capital_running = BT["initial_capital"]

        for win_i, (tr_s, tr_e, te_s, te_e) in enumerate(windows, 1):
            train_mask = (df.index >= tr_s) & (df.index <  tr_e)
            test_mask  = (df.index >= te_s) & (df.index <  te_e)

            df_train = df[train_mask]
            df_test  = df[test_mask]

            if len(df_train) < 40 or len(df_test) < 5:
                continue

            # ── In-sample: find best combo ────────────────────────────────────
            best_sharpe  = -np.inf
            best_stop    = STRAT["atr_stop_mult"]
            best_roc     = STRAT["momentum_roc_period"]

            for stop_m in stop_grid:
                tp_m = stop_m * base_rr
                for roc_p in roc_grid:
                    sig_train = all_signals[roc_p][train_mask].shift(1).fillna(0).astype(int)
                    _, is_sharpe = _mini_backtest(
                        df_train, sig_train, stop_m, tp_m, capital_running
                    )
                    if is_sharpe > best_sharpe:
                        best_sharpe = is_sharpe
                        best_stop   = stop_m
                        best_roc    = roc_p

            # ── OOS: apply IS-best params ─────────────────────────────────────
            tp_m_oos    = best_stop * base_rr
            sig_test    = all_signals[best_roc][test_mask].shift(1).fillna(0).astype(int)
            oos_eq, oos_sharpe = _mini_backtest(
                df_test, sig_test, best_stop, tp_m_oos, capital_running
            )

            oos_ret = (oos_eq.iloc[-1] / capital_running - 1) * 100.0
            capital_running = oos_eq.iloc[-1]   # compound forward

            oos_equity_segments.append(oos_eq)
            window_records.append({
                "window":      win_i,
                "train_start": tr_s.date(),
                "train_end":   tr_e.date(),
                "test_start":  te_s.date(),
                "test_end":    te_e.date(),
                "best_stop":   best_stop,
                "best_roc":    best_roc,
                "is_sharpe":   round(best_sharpe, 3),
                "oos_sharpe":  round(oos_sharpe, 3),
                "oos_ret_pct": round(oos_ret, 2),
                "n_train":     len(df_train),
                "n_test":      len(df_test),
            })

            if win_i % 5 == 0 or win_i == len(windows):
                print(f"    [{win_i:3d}/{len(windows)}]  "
                      f"test {te_s.date()}→{te_e.date()}  "
                      f"best=({best_stop}×atr, roc{best_roc:>2}d)  "
                      f"IS={best_sharpe:+.3f}  OOS={oos_sharpe:+.3f}  "
                      f"ret={oos_ret:+.2f}%", flush=True)

        # ── Aggregate OOS equity curve ─────────────────────────────────────────
        if oos_equity_segments:
            oos_equity = pd.concat(oos_equity_segments)
        else:
            oos_equity = pd.Series(dtype=float)

        pw_df = pd.DataFrame(window_records)

        _print_wfo_table(asset_key, pw_df)
        _plot_wfo(oos_equity, df, pw_df, asset_key)

        best_stop_mode = pw_df["best_stop"].mode().iloc[0] if len(pw_df) else STRAT["atr_stop_mult"]
        best_roc_mode  = pw_df["best_roc"].mode().iloc[0]  if len(pw_df) else STRAT["momentum_roc_period"]

        results[asset_key] = {
            "oos_equity":  oos_equity,
            "window_df":   pw_df,
            "best_params": {"atr_stop_mult": best_stop_mode, "momentum_roc_period": best_roc_mode},
        }

    return results


# ── REPORTING ─────────────────────────────────────────────────────────────────

def _print_wfo_table(asset_key: str, pw_df: pd.DataFrame) -> None:
    if pw_df.empty:
        print(f"  {asset_key}: no completed windows.")
        return

    print(f"\n  {'─'*80}")
    print(f"  {asset_key} WFO Results  ({len(pw_df)} OOS windows)")
    print(f"  {'─'*80}")
    print(f"  {'Win':>4} {'Test period':>26}  {'stop':>5} {'roc':>4}  "
          f"{'IS_Sh':>7} {'OOS_Sh':>7} {'OOS_Ret%':>9}  {'N_tr':>5} {'N_te':>5}")
    print(f"  {'─'*80}")

    for _, r in pw_df.iterrows():
        print(
            f"  {int(r['window']):>4}  "
            f"{str(r['test_start'])}→{str(r['test_end']):>12}  "
            f"{r['best_stop']:>5.1f} {int(r['best_roc']):>4}  "
            f"{r['is_sharpe']:>+7.3f} {r['oos_sharpe']:>+7.3f} "
            f"{r['oos_ret_pct']:>+9.2f}%  "
            f"{int(r['n_train']):>5} {int(r['n_test']):>5}"
        )
    print(f"  {'─'*80}")
    print(f"  {'MEAN':>4}  {'':>26}  "
          f"{'':>5} {'':>4}  "
          f"{pw_df['is_sharpe'].mean():>+7.3f} "
          f"{pw_df['oos_sharpe'].mean():>+7.3f} "
          f"{pw_df['oos_ret_pct'].mean():>+9.2f}%")

    positive_oos = (pw_df["oos_sharpe"] > 0).sum()
    print(f"\n  OOS windows with positive Sharpe: {positive_oos}/{len(pw_df)} "
          f"({positive_oos/len(pw_df)*100:.0f}%)")

    # Most-common params
    print(f"\n  Most-selected params (mode across OOS windows):")
    for col, label in [("best_stop", "atr_stop_mult"), ("best_roc", "momentum_roc_period")]:
        vc = pw_df[col].value_counts()
        top_val  = vc.index[0]
        top_cnt  = vc.iloc[0]
        print(f"    {label:<25}: {top_val}  (chosen {top_cnt}/{len(pw_df)} windows)")


def _plot_wfo(
    oos_equity: pd.Series,
    full_df:    pd.DataFrame,
    pw_df:      pd.DataFrame,
    asset_key:  str,
) -> None:
    cfg = ASSETS[asset_key]

    if oos_equity.empty:
        print(f"  ⚠  {asset_key}: no OOS equity to plot.")
        return

    bh_start = full_df["close"].iloc[0]
    bh_eq    = (full_df["close"] / bh_start * BT["initial_capital"]).reindex(oos_equity.index)

    fig, axes = plt.subplots(3, 1, figsize=(16, 12),
                             gridspec_kw={"height_ratios": [3, 1, 1]}, sharex=True)
    fig.suptitle(
        f"{cfg['label']} — Walk-Forward Optimization (OOS Results)\n"
        f"OOS Sharpe {pw_df['oos_sharpe'].mean():+.3f}  |  "
        f"OOS Return {(oos_equity.iloc[-1]/BT['initial_capital']-1)*100:+.1f}%  |  "
        f"{len(pw_df)} windows",
        fontsize=12, fontweight="bold",
    )

    # ── Panel 1: OOS equity vs B&H ────────────────────────────────────────────
    ax = axes[0]
    ax.plot(oos_equity.index, oos_equity.values, color=cfg["color"],
            linewidth=1.3, label="WFO OOS Strategy")
    if bh_eq is not None and not bh_eq.isna().all():
        ax.plot(bh_eq.index, bh_eq.values, color="#888888",
                linewidth=0.8, linestyle="--", alpha=0.7, label="Buy & Hold")
    ax.axhline(BT["initial_capital"], color="#ccc", linewidth=0.5, linestyle=":")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_ylabel("Portfolio Value (USD)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # ── Panel 2: OOS drawdown ─────────────────────────────────────────────────
    ax2 = axes[1]
    roll_max = oos_equity.cummax()
    dd       = (oos_equity - roll_max) / roll_max * 100.0
    ax2.fill_between(dd.index, dd.values, 0, color="#E53935", alpha=0.5)
    ax2.set_ylabel("OOS DD %", fontsize=9)
    ax2.grid(True, alpha=0.2)

    # ── Panel 3: OOS Sharpe per window ────────────────────────────────────────
    ax3 = axes[2]
    test_mids = pd.to_datetime(
        [(r["test_start"].isoformat() if hasattr(r["test_start"], "isoformat")
          else str(r["test_start"])) for _, r in pw_df.iterrows()]
    )
    colors = ["#43A047" if s > 0 else "#E53935" for s in pw_df["oos_sharpe"]]
    ax3.bar(test_mids, pw_df["oos_sharpe"].values, color=colors, width=45, alpha=0.8)
    ax3.axhline(0, color="#555", linewidth=0.7)
    ax3.set_ylabel("OOS Sharpe", fontsize=9)
    ax3.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    out = RESULTS / f"{asset_key.lower()}_wfo.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  WFO chart → results/{out.name}")
