"""
models/ml_filter.py — XGBoost walk-forward trade signal filter.

Pipeline:
  1. Build binary labels: 1 if fwd_{n}d return > 0, else 0.
  2. Walk-forward CV: rolling train window (12mo) → OOS test (3mo).
     - Never touches test-period labels during training.
  3. For each window: fit XGBoostClassifier, predict OOS probabilities.
  4. Aggregate: pool OOS predictions, compute classification report.
  5. Average feature importances across all windows → bar chart.
  6. Return ml_confidence column: P(up) per bar, NaN outside OOS windows.

Lookahead contract:
  Labels are constructed with shift(-forward_days) and are only used to
  TRAIN the model on past windows.  When predicting OOS, only features[t]
  (no future close) are consumed → zero lookahead on the signal side.
"""

import sys
import warnings
from pathlib import Path
from typing import List, Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ML, RESULTS
from features.engineer import FEATURE_COLS

warnings.filterwarnings("ignore")


# ── LABEL CONSTRUCTION ────────────────────────────────────────────────────────

def build_labels(df: pd.DataFrame) -> pd.Series:
    """
    Binary label: 1 if close rises over the next forward_days bars, else 0.
    Last forward_days rows are NaN (future not yet observable).

    No lookahead: labels are only used on the TRAINING portion of each
    walk-forward window.  The OOS predictions never use labels.
    """
    fwd = ML["forward_days"]
    fwd_close = df["close"].shift(-fwd)
    label = (fwd_close > df["close"]).astype(float)
    label.iloc[-fwd:] = np.nan   # no label for the final rows
    return label.rename("label")


# ── WALK-FORWARD ENGINE ───────────────────────────────────────────────────────

def _make_windows(index: pd.DatetimeIndex) -> List[Tuple]:
    """
    Generate (train_start, train_end, test_start, test_end) date tuples.
    Rolling train window of train_months; test window of test_months;
    slides forward by test_months each step.
    """
    train_off = pd.DateOffset(months=ML["wf_train_months"])
    test_off  = pd.DateOffset(months=ML["wf_test_months"])

    windows   = []
    t_start   = index[0]

    while True:
        t_end   = t_start + train_off
        v_start = t_end
        v_end   = v_start + test_off

        if v_end > index[-1]:
            break

        windows.append((t_start, t_end, v_start, v_end))
        t_start = t_start + test_off   # rolling: slide start by test_months

    return windows


def _fit_window(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, object]:
    """
    Fit XGBoost on one window, return OOS predictions and model.
    Uses scale_pos_weight to handle class imbalance.
    """
    from xgboost import XGBClassifier

    pos = y_train.sum()
    neg = len(y_train) - pos
    spw = neg / pos if pos > 0 else 1.0

    params = {**ML["xgb"], "scale_pos_weight": spw}
    model  = XGBClassifier(**params)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]   # P(label=1)
    pred  = (proba >= 0.5).astype(int)           # standard 0.5 threshold for metrics

    return pred, proba, model


def run_walk_forward(
    df:        pd.DataFrame,
    asset_key: str,
) -> pd.DataFrame:
    """
    Full walk-forward ML filter pipeline.

    Returns:
        df with two new columns:
          ml_confidence : P(fwd return > 0), NaN outside OOS periods
          ml_signal     : 1 if ml_confidence > ML["min_confidence"], else 0
    """
    fwd = ML["forward_days"]

    # ── Prepare labels + feature matrix ──────────────────────────────────────
    df       = df.copy()
    df["label"] = build_labels(df)

    # Only rows with complete features AND a label
    valid_mask = df[FEATURE_COLS].notna().all(axis=1) & df["label"].notna()
    full_df    = df[valid_mask].copy()

    X_all  = full_df[FEATURE_COLS].values
    y_all  = full_df["label"].values.astype(int)
    idx    = full_df.index

    scaler = StandardScaler()
    X_all  = scaler.fit_transform(X_all)   # standardise once across full history

    windows = _make_windows(idx)
    if not windows:
        print(f"  ⚠  {asset_key}: not enough data for walk-forward (need ≥"
              f" {ML['wf_train_months']+ML['wf_test_months']} months).")
        df["ml_confidence"] = np.nan
        df["ml_signal"]     = 0
        return df

    print(f"  {asset_key}: {len(windows)} walk-forward windows  "
          f"(train={ML['wf_train_months']}mo, test={ML['wf_test_months']}mo)")

    oos_records   = []     # {date → {actual, predicted, confidence}}
    per_window    = []     # per-window metrics
    importances   = []     # per-window feature importances
    all_actual    = []
    all_pred      = []
    all_prob      = []

    for win_i, (tr_s, tr_e, te_s, te_e) in enumerate(windows, 1):
        # Locate rows in the index
        train_mask = (idx >= tr_s) & (idx <  tr_e)
        test_mask  = (idx >= te_s) & (idx <  te_e)

        X_tr, y_tr = X_all[train_mask], y_all[train_mask]
        X_te, y_te = X_all[test_mask],  y_all[test_mask]
        idx_te     = idx[test_mask]

        if len(X_tr) < ML["min_train_samples"] or len(X_te) == 0:
            continue
        if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
            continue   # degenerate labels (all one class)

        pred, proba, model = _fit_window(X_tr, y_tr, X_te)

        # Per-window metrics
        prec  = precision_score(y_te, pred, zero_division=0)
        rec   = recall_score(y_te, pred, zero_division=0)
        f1    = f1_score(y_te, pred, zero_division=0)
        try:
            auc = roc_auc_score(y_te, proba)
        except Exception:
            auc = np.nan

        per_window.append({
            "window":    win_i,
            "train_start": tr_s.date(),
            "test_start":  te_s.date(),
            "test_end":    te_e.date(),
            "n_train":   len(X_tr),
            "n_test":    len(X_te),
            "precision": prec,
            "recall":    rec,
            "f1":        f1,
            "auc":       auc,
        })

        importances.append(model.feature_importances_)
        all_actual.extend(y_te.tolist())
        all_pred.extend(pred.tolist())
        all_prob.extend(proba.tolist())

        for date, act, pr, pb in zip(idx_te, y_te, pred, proba):
            oos_records.append({
                "date":          date,
                "actual":        int(act),
                "predicted":     int(pr),
                "ml_confidence": float(pb),
            })

        if win_i % 10 == 0 or win_i == len(windows):
            print(f"    [{win_i:3d}/{len(windows)}] "
                  f"train {tr_s.date()}→{tr_e.date()}  "
                  f"test {te_s.date()}→{te_e.date()}  "
                  f"F1={f1:.3f}  AUC={auc:.3f}", flush=True)

    # ── Attach OOS confidence back to original df ─────────────────────────────
    if oos_records:
        oos_df = pd.DataFrame(oos_records).set_index("date")
        df["ml_confidence"] = oos_df["ml_confidence"]
        df["ml_signal"]     = (df["ml_confidence"] >= ML["min_confidence"]).astype(int)
    else:
        df["ml_confidence"] = np.nan
        df["ml_signal"]     = 0

    # ── Aggregate results ─────────────────────────────────────────────────────
    pw_df = pd.DataFrame(per_window)

    _print_oos_report(
        asset_key, pw_df,
        np.array(all_actual), np.array(all_pred), np.array(all_prob),
        df,
    )

    # ── Feature importance chart ──────────────────────────────────────────────
    if importances:
        avg_imp = np.array(importances).mean(axis=0)
        _plot_feature_importance(avg_imp, asset_key)

    return df


# ── REPORTING ─────────────────────────────────────────────────────────────────

def _print_oos_report(
    asset_key: str,
    pw_df:     pd.DataFrame,
    actual:    np.ndarray,
    pred:      np.ndarray,
    prob:      np.ndarray,
    full_df:   pd.DataFrame,
) -> None:
    """Print per-window table + pooled OOS classification report."""

    print(f"\n  {'─'*68}")
    print(f"  {asset_key} Walk-Forward Results  ({len(pw_df)} windows)")
    print(f"  {'─'*68}")
    print(f"  {'Win':>4} {'Train start':>12} {'Test period':>22}  "
          f"{'N_tr':>6} {'N_te':>5} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}")
    print(f"  {'─'*68}")

    for _, r in pw_df.iterrows():
        print(
            f"  {int(r['window']):>4}  {str(r['train_start']):>12}  "
            f"{str(r['test_start'])}→{str(r['test_end']):>12}  "
            f"{int(r['n_train']):>6} {int(r['n_test']):>5} "
            f"{r['precision']:>6.3f} {r['recall']:>6.3f} "
            f"{r['f1']:>6.3f} {r['auc']:>6.3f}"
        )
    print(f"  {'─'*68}")
    print(f"  {'MEAN':>4}  {'':>12}  {'':>22}  "
          f"{'':>6} {'':>5} "
          f"{pw_df['precision'].mean():>6.3f} "
          f"{pw_df['recall'].mean():>6.3f} "
          f"{pw_df['f1'].mean():>6.3f} "
          f"{pw_df['auc'].mean():>6.3f}")

    # Pooled OOS classification report
    if len(actual) > 0:
        print(f"\n  Pooled OOS Classification Report  "
              f"(N={len(actual):,}, threshold=0.5):")
        print(f"  {'─'*44}")
        report_str = classification_report(
            actual, pred,
            target_names=["Down/Flat", "Up"],
            zero_division=0,
        )
        for line in report_str.strip().split("\n"):
            print(f"    {line}")

        # Signal filter stats
        oos_conf = full_df["ml_confidence"].dropna()
        n_above  = (oos_conf >= ML["min_confidence"]).sum()
        n_total  = len(oos_conf)
        strategy_sigs = full_df["signal"].reindex(oos_conf.index).fillna(0)
        both_pass = (
            (full_df["signal"].reindex(oos_conf.index).fillna(0) == 1) &
            (oos_conf >= ML["min_confidence"])
        ).sum()
        original  = (strategy_sigs == 1).sum()

        print(f"\n  Filter @ confidence > {ML['min_confidence']}:")
        print(f"    OOS bars with confidence     : {n_total:,}")
        print(f"    Bars passing ML gate         : {n_above:,}  ({n_above/n_total*100:.1f}%)")
        print(f"    Strategy signals (OOS period): {int(original):,}")
        print(f"    Signals surviving ML filter  : {int(both_pass):,}  "
              f"({int(both_pass)/max(int(original),1)*100:.1f}% of strategy signals)")


# ── FEATURE IMPORTANCE CHART ──────────────────────────────────────────────────

_GROUP_COLORS = {
    "Momentum": "#2196F3",   # blue
    "Regime":   "#4CAF50",   # green
    "Price":    "#FF9800",   # orange
}

def _feature_group(col: str) -> str:
    if col.startswith("roc_") or col.startswith("rsi_") or col == "macd_hist":
        return "Momentum"
    if col in ["hurst", "sharpe_20", "sharpe_60", "vol_ratio"]:
        return "Regime"
    return "Price"


def _plot_feature_importance(avg_imp: np.ndarray, asset_key: str) -> None:
    """Horizontal bar chart of avg XGBoost feature importances, coloured by group."""
    imp_series = pd.Series(avg_imp, index=FEATURE_COLS).sort_values(ascending=True)
    top = imp_series.tail(15)   # top 15

    colors = [_GROUP_COLORS[_feature_group(c)] for c in top.index]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top.index, top.values, color=colors, edgecolor="white", height=0.7)

    # Legend patches
    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(color=c, label=g)
        for g, c in _GROUP_COLORS.items()
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)

    ax.set_title(
        f"{asset_key} — XGBoost Feature Importance\n"
        f"(avg across {len(FEATURE_COLS)} features, walk-forward windows)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Mean Importance (gain)", fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)
    ax.tick_params(axis="y", labelsize=9)

    # Value labels on bars
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{w:.4f}", va="center", ha="left", fontsize=7)

    plt.tight_layout()
    out = RESULTS / f"{asset_key.lower()}_feature_importance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Feature importance chart → results/{out.name}")
