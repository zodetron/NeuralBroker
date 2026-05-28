"""
models/ml_filter.py — Per-strategy XGBoost walk-forward ML filter.

Three separate models: one per strategy (A / B / C).
Each model trains only on signals from its own strategy and uses
feature patterns relevant to that strategy's market conditions.

Label construction:
  Simulate each signal's trade outcome using subsequent OHLCV bars.
  Label = 1 if TP reached before SL within time_exit_bars, else 0.
  - Strategy A/C: TP/SL are ATR multiples from entry.
  - Strategy B  : TP = VWAP at signal bar (full mean reversion); SL = 0.5×ATR.

Walk-forward:
  Rolling train window (per-strategy width) → 1-week OOS test.
  Slides forward 1 week at a time.
  OOS predictions accumulated; signals without OOS coverage are excluded.

Zero-lookahead contract:
  All features extracted at bar t (the signal bar).
  Label constructed using bars t+1 … t+N (strictly future, used only for training).
  At prediction time, only bar-t features are consumed.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, classification_report,
)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ML_PER_STRAT, RESULTS

warnings.filterwarnings("ignore")

# ── PER-STRATEGY TRADE SIMULATION PARAMS ─────────────────────────────────────
# (mirrors the backtest engine params for each strategy)
_TRADE_CFG: Dict[str, dict] = {
    "A": {"tp_mult": 1.2,  "sl_mult": 0.6, "time_exit": 8,  "tp_type": "atr"},
    "B": {"tp_mult": None, "sl_mult": 0.5, "time_exit": 6,  "tp_type": "vwap"},
    "C": {"tp_mult": 1.5,  "sl_mult": 0.8, "time_exit": 12, "tp_type": "atr"},
}

# ── FEATURE COLUMNS ───────────────────────────────────────────────────────────
_COMMON_FEATS = [
    "direction",
    "state_code",   "trend_1h",      "align_bonus",
    "adx_14",       "ema_slope",     "atr_ratio",
    "vol_ratio",    "ret_4",         "ret_8",      "ret_16",  "rvol_20",
    "hour_sin",     "hour_cos",      "dow_sin",    "dow_cos",
]
_STRAT_FEATS: Dict[str, List[str]] = {
    "A": _COMMON_FEATS + ["vol_ratio_sig", "conf_bonus",    "breakout_strength"],
    "B": _COMMON_FEATS + ["rsi14",         "vwap_dev_pct",  "vwap_atr_ratio"  ],
    "C": _COMMON_FEATS + ["rsi14",         "ema_dist_pct",  "adx_1h"          ],
}


# ── LABEL CONSTRUCTION ────────────────────────────────────────────────────────

def build_labels(
    signals:      pd.DataFrame,
    df:           pd.DataFrame,
    strategy_key: str,
) -> pd.Series:
    """
    Simulate each trade and return binary outcome labels.

    Uses open[t+1] as entry price.
    Scans subsequent bar highs/lows for TP or SL hit.
    Label = 1 if TP hit first, 0 if SL hit or time expires.
    """
    cfg        = _TRADE_CFG[strategy_key]
    max_bars   = cfg["time_exit"]
    sl_mult    = cfg["sl_mult"]
    tp_mult    = cfg["tp_mult"]
    tp_type    = cfg["tp_type"]

    open_arr   = df["open"].values
    high_arr   = df["high"].values
    low_arr    = df["low"].values
    has_vwap   = "vwap" in signals.columns

    df_index   = df.index
    labels     = np.zeros(len(signals), dtype=np.int8)

    for i, (ts, row) in enumerate(signals.iterrows()):
        try:
            bar_pos   = df_index.get_loc(ts)
        except KeyError:
            continue

        entry_pos = bar_pos + 1
        if entry_pos >= len(df):
            continue

        entry     = open_arr[entry_pos]
        direction = int(row["direction"])
        atr       = float(row["atr_14"])

        # TP target
        if tp_type == "vwap" and has_vwap:
            tp_price = float(row["vwap"])          # mean-reversion to VWAP
        else:
            tp_price = entry + direction * tp_mult * atr

        sl_price = entry - direction * sl_mult * atr

        hit = 0   # default: time exit (loss)
        for k in range(1, max_bars + 1):
            pos = entry_pos + k
            if pos >= len(df):
                break
            h = high_arr[pos]
            l = low_arr[pos]
            if direction == 1:          # long
                if h >= tp_price: hit = 1; break
                if l <= sl_price: hit = 0; break
            else:                       # short
                if l <= tp_price: hit = 1; break
                if h >= sl_price: hit = 0; break

        labels[i] = hit

    return pd.Series(labels, index=signals.index, name="label")


# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────

def _build_bar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute bar-level features on the full 15M DataFrame.
    All features are strictly causal (no lookahead).
    """
    feat = pd.DataFrame(index=df.index)
    c    = df["close"]
    lr   = np.log(c / c.shift(1)).fillna(0).clip(-0.2, 0.2)

    # Recent log returns
    feat["ret_4"]   = lr.rolling(4).sum()
    feat["ret_8"]   = lr.rolling(8).sum()
    feat["ret_16"]  = lr.rolling(16).sum()

    # Annualised realised vol  (sqrt(2190) for 15M 24/7)
    feat["rvol_20"] = lr.rolling(20).std(ddof=1) * np.sqrt(2190)

    # Volume ratio
    vol_ma = df["volume"].rolling(20).mean()
    feat["vol_ratio"] = df["volume"] / (vol_ma + 1e-9)

    # ATR regime ratio
    feat["atr_ratio"] = df["atr_14"] / (df["atr_avg"] + 1e-9)

    # Trend / ADX
    feat["adx_14"]   = df["adx_14"]
    feat["ema_slope"] = df["ema20_slope"]

    # Market state code
    _state_map = {"TRENDING_UP": 0, "TRENDING_DOWN": 1,
                  "HIGH_VOL_CHOP": 2, "LOW_VOL_RANGE": 3}
    feat["state_code"] = df["market_state"].map(_state_map).fillna(-1).astype(int)

    # 1H context
    feat["trend_1h"]   = df["trend_1h"].fillna(0).astype(int)
    feat["align_bonus"] = df["align_bonus"].fillna(0).astype(np.int8)

    # Cyclical time encoding
    hour_dec = df.index.hour + df.index.minute / 60.0
    feat["hour_sin"] = np.sin(2 * np.pi * hour_dec / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * hour_dec / 24)
    dow = df.index.dayofweek.values.astype(float)
    feat["dow_sin"]  = np.sin(2 * np.pi * dow / 7)
    feat["dow_cos"]  = np.cos(2 * np.pi * dow / 7)

    return feat


def build_features(
    signals:      pd.DataFrame,
    df:           pd.DataFrame,
    strategy_key: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Return (feature_matrix, feature_col_names) for the given strategy's signals.

    Rows correspond to signals (indexed by bar timestamp).
    All features are measured at bar t (the signal bar, strictly no-lookahead).
    """
    bar_feats = _build_bar_features(df)

    # Align to signal timestamps
    X = bar_feats.reindex(signals.index)

    # Direction of trade
    X["direction"] = signals["direction"].values

    # Strategy-specific columns from the signal DataFrame
    for col in ["rsi14", "vwap_dev_pct", "ema_dist_pct", "vol_ratio",
                "adx_1h", "conf_bonus"]:
        X[col] = signals[col].values if col in signals.columns else 0.0

    # Derived strategy-specific features
    if strategy_key == "A":
        X["vol_ratio_sig"]       = signals["vol_ratio"].values if "vol_ratio" in signals.columns else X["vol_ratio"].values
        X["breakout_strength"]   = (
            (signals["close"] - signals["roll_level"]).abs()
            / (signals["atr_14"] + 1e-9)
        ).values if "roll_level" in signals.columns else 0.0

    elif strategy_key == "B":
        X["vwap_atr_ratio"] = (
            signals["vwap_dev_pct"].abs() / (signals["atr_14"] / signals["close"] * 100 + 1e-9)
        ).values if "vwap_dev_pct" in signals.columns else 0.0

    feat_cols = _STRAT_FEATS[strategy_key]
    # Ensure all feature columns exist
    for c in feat_cols:
        if c not in X.columns:
            X[c] = 0.0

    return X[feat_cols].copy(), feat_cols


# ── WALK-FORWARD WINDOWS ──────────────────────────────────────────────────────

def _make_wf_windows(
    signal_index: pd.DatetimeIndex,
    train_weeks:  int,
    test_weeks:   int,
) -> List[Tuple]:
    """
    Generate (train_start, train_end, test_start, test_end) tuples.
    Slides forward by test_weeks each step.
    """
    if len(signal_index) == 0:
        return []

    train_off = pd.Timedelta(weeks=train_weeks)
    test_off  = pd.Timedelta(weeks=test_weeks)

    windows  = []
    t_start  = signal_index[0]

    while True:
        t_end   = t_start + train_off
        v_start = t_end
        v_end   = v_start + test_off
        if v_end > signal_index[-1]:
            break
        windows.append((t_start, t_end, v_start, v_end))
        t_start = t_start + test_off

    return windows


# ── XGB SINGLE WINDOW ────────────────────────────────────────────────────────

def _fit_window(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, object]:
    from xgboost import XGBClassifier

    pos = y_tr.sum()
    neg = len(y_tr) - pos
    spw = neg / max(pos, 1)

    model = XGBClassifier(
        n_estimators    = 200,
        max_depth       = 3,
        learning_rate   = 0.05,
        subsample       = 0.8,
        colsample_bytree= 0.8,
        scale_pos_weight= spw,
        eval_metric     = "logloss",
        verbosity       = 0,
        random_state    = 42,
        use_label_encoder=False,
    )
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    return pred, proba, model


# ── MAIN WALK-FORWARD PIPELINE ────────────────────────────────────────────────

def run_walk_forward(
    signals:      pd.DataFrame,
    df:           pd.DataFrame,
    strategy_key: str,
) -> pd.DataFrame:
    """
    Full walk-forward ML filter for one strategy.

    Returns signals with two new columns:
      ml_confidence : P(TP hit before SL), NaN for warmup signals
      ml_signal     : 1 if confidence >= threshold, else 0 (0 for NaN)
    """
    cfg         = ML_PER_STRAT[strategy_key]
    threshold   = cfg["threshold"]
    train_weeks = cfg["train_weeks"]
    test_weeks  = cfg["test_weeks"]
    min_samples = cfg["min_samples"]

    if signals.empty:
        print(f"  {strategy_key}: no signals — skipping.")
        sigs = signals.copy()
        sigs["ml_confidence"] = np.nan
        sigs["ml_signal"]     = 0
        return sigs

    print(f"\n  {'─'*62}")
    print(f"  Strategy {strategy_key} — XGBoost Walk-Forward")
    print(f"  Train: {train_weeks}w  |  Test: {test_weeks}w  |  "
          f"Threshold: {threshold}  |  Min train samples: {min_samples}")
    print(f"  {'─'*62}")

    # ── Labels ───────────────────────────────────────────────────────────────
    labels = build_labels(signals, df, strategy_key)
    pos_rate = labels.mean() * 100
    print(f"  Labels: {len(labels):,} total  |  "
          f"TP rate: {pos_rate:.1f}%  |  SL rate: {100-pos_rate:.1f}%")

    # ── Features ─────────────────────────────────────────────────────────────
    X_all, feat_cols = build_features(signals, df, strategy_key)
    y_all = labels.values.astype(int)

    # Drop rows with NaN features
    valid    = ~X_all.isna().any(axis=1)
    X_valid  = X_all[valid].values
    y_valid  = y_all[valid]
    idx      = X_all.index[valid]

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)

    # ── WFO windows ──────────────────────────────────────────────────────────
    windows = _make_wf_windows(idx, train_weeks, test_weeks)
    print(f"  WFO windows: {len(windows)}  "
          f"(signal range {idx[0].date()} → {idx[-1].date()})")

    oos_records  = {}   # ts → confidence
    per_window   = []
    importances  = []
    all_actual, all_pred, all_prob = [], [], []

    skipped = 0
    for win_i, (tr_s, tr_e, te_s, te_e) in enumerate(windows, 1):
        tr_mask  = (idx >= tr_s) & (idx < tr_e)
        te_mask  = (idx >= te_s) & (idx < te_e)
        X_tr, y_tr = X_scaled[tr_mask], y_valid[tr_mask]
        X_te, y_te = X_scaled[te_mask], y_valid[te_mask]
        idx_te     = idx[te_mask]

        if len(X_tr) < min_samples or len(X_te) == 0:
            skipped += 1
            continue
        if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
            skipped += 1
            continue

        pred, proba, model = _fit_window(X_tr, y_tr, X_te)

        prec = precision_score(y_te, pred, zero_division=0)
        rec  = recall_score(y_te, pred, zero_division=0)
        f1   = f1_score(y_te, pred, zero_division=0)
        try:
            auc = roc_auc_score(y_te, proba)
        except Exception:
            auc = np.nan

        per_window.append({
            "window": win_i, "train_start": tr_s.date(), "test_start": te_s.date(),
            "test_end": te_e.date(), "n_train": len(X_tr), "n_test": len(X_te),
            "precision": prec, "recall": rec, "f1": f1, "auc": auc,
        })
        importances.append(model.feature_importances_)
        all_actual.extend(y_te.tolist())
        all_pred.extend(pred.tolist())
        all_prob.extend(proba.tolist())

        for ts, p in zip(idx_te, proba):
            oos_records[ts] = float(p)

    if skipped:
        print(f"  Skipped {skipped} windows (insufficient samples or degenerate labels)")

    # ── Attach confidence back to original signals ────────────────────────────
    sigs = signals.copy()
    sigs["ml_confidence"] = pd.Series(oos_records).reindex(sigs.index)
    sigs["ml_signal"]     = (sigs["ml_confidence"] >= threshold).astype(int)
    sigs.loc[sigs["ml_confidence"].isna(), "ml_signal"] = 0

    # ── Report ────────────────────────────────────────────────────────────────
    pw_df = pd.DataFrame(per_window)
    _print_wf_report(
        strategy_key, pw_df, threshold,
        np.array(all_actual), np.array(all_pred), np.array(all_prob),
        sigs,
    )

    if importances:
        avg_imp = np.array(importances).mean(axis=0)
        _plot_feature_importance(avg_imp, feat_cols, strategy_key)

    return sigs


# ── REPORTING ─────────────────────────────────────────────────────────────────

def _print_wf_report(
    strategy_key: str,
    pw_df:        pd.DataFrame,
    threshold:    float,
    actual:       np.ndarray,
    pred:         np.ndarray,
    prob:         np.ndarray,
    sigs_out:     pd.DataFrame,
) -> None:
    if pw_df.empty:
        print(f"  Strategy {strategy_key}: no OOS windows completed.")
        return

    # Print every Nth window to keep output clean
    n_win    = len(pw_df)
    stride   = max(1, n_win // 12)    # print ~12 rows

    print(f"\n  {'Win':>4}  {'Train start':>12}  {'Test period':>24}  "
          f"{'N_tr':>5} {'N_te':>4}  {'Prec':>5} {'Rec':>5} {'F1':>5} {'AUC':>5}")
    print(f"  {'─'*74}")
    for _, r in pw_df.iloc[::stride].iterrows():
        print(f"  {int(r['window']):>4}  {str(r['train_start']):>12}  "
              f"{str(r['test_start'])} → {str(r['test_end']):<12}  "
              f"{int(r['n_train']):>5} {int(r['n_test']):>4}  "
              f"{r['precision']:>5.3f} {r['recall']:>5.3f} "
              f"{r['f1']:>5.3f} {r['auc']:>5.3f}")

    print(f"  {'─'*74}")
    print(f"  {'MEAN':>4}  {'':>12}  {'':>24}  "
          f"{'':>5} {'':>4}  "
          f"{pw_df['precision'].mean():>5.3f} "
          f"{pw_df['recall'].mean():>5.3f} "
          f"{pw_df['f1'].mean():>5.3f} "
          f"{pw_df['auc'].mean():>5.3f}")

    # Pooled OOS classification report
    if len(actual) > 0:
        print(f"\n  Pooled OOS report  (N={len(actual):,}, threshold=0.5 for stats):")
        rep = classification_report(actual, pred, target_names=["SL/Timeout", "TP"],
                                    zero_division=0)
        for line in rep.strip().split("\n"):
            print(f"    {line}")

    # Filter stats
    n_oos   = sigs_out["ml_confidence"].notna().sum()
    n_pass  = sigs_out["ml_signal"].sum()
    n_raw   = len(sigs_out)
    n_days  = (sigs_out.index[-1] - sigs_out.index[0]).days if len(sigs_out) > 1 else 1

    print(f"\n  Filter @ confidence ≥ {threshold}:")
    print(f"    Raw signals (all time)      : {n_raw:>6,}")
    print(f"    With OOS confidence         : {n_oos:>6,}  ({n_oos/max(n_raw,1)*100:.1f}%)")
    print(f"    Passing threshold           : {n_pass:>6,}  ({n_pass/max(n_oos,1)*100:.1f}% of OOS signals)")
    print(f"    After-filter rate           : {n_pass/max(n_days,1):>6.2f} / day")


def _plot_feature_importance(
    avg_imp:      np.ndarray,
    feat_cols:    List[str],
    strategy_key: str,
) -> None:
    imp_s = pd.Series(avg_imp, index=feat_cols).sort_values(ascending=True)
    top   = imp_s.tail(min(15, len(imp_s)))

    _group_color = {
        "direction": "#E91E63", "state_code": "#9C27B0", "trend_1h": "#9C27B0",
        "align_bonus": "#9C27B0", "adx_14": "#2196F3", "ema_slope": "#2196F3",
        "atr_ratio": "#FF9800", "vol_ratio": "#FF9800", "vol_ratio_sig": "#FF9800",
        "ret_4": "#4CAF50", "ret_8": "#4CAF50", "ret_16": "#4CAF50",
        "rvol_20": "#4CAF50",
    }
    colors = [_group_color.get(c, "#607D8B") for c in top.index]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top.index, top.values, color=colors, edgecolor="white", height=0.7)
    ax.set_title(
        f"Strategy {strategy_key} — XGBoost Feature Importance\n"
        f"(avg across WFO windows)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Mean Importance (gain)", fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{w:.4f}", va="center", ha="left", fontsize=7)

    plt.tight_layout()
    out = RESULTS / f"strategy_{strategy_key.lower()}_feature_importance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Feature importance chart → results/strategy_{strategy_key.lower()}_feature_importance.png")
