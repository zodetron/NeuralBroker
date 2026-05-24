"""
models/regime.py — Gaussian HMM regime detection.

Trains a 3-state Gaussian HMM on [log_ret, rvol_20] and maps hidden
states to Bull / Sideways / Bear by sorting on state mean return.

Lookahead note:
  Training on the full history is acceptable here for *analysis* purposes
  (labelling regimes post-hoc to understand market structure).
  In Phase 7 / Phase 8 the strategy uses a walk-forward expanding-window
  HMM so no future data leaks into live signals.

Outputs:
  - regime column added to the feature DataFrame
  - results/{asset}_regimes.png  — price chart with regime background
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import HMM as HMM_CFG, ASSETS, RESULTS

warnings.filterwarnings("ignore")


# ── HMM TRAINING ──────────────────────────────────────────────────────────────

def _fit_hmm(obs_scaled: np.ndarray):
    """
    Fit a GaussianHMM. Tries multiple random seeds and picks the model
    with the highest log-likelihood (avoids bad local optima).
    """
    from hmmlearn.hmm import GaussianHMM

    best_model  = None
    best_score  = -np.inf
    seeds       = [HMM_CFG["random_state"], 0, 7, 13, 99]

    for seed in seeds:
        try:
            m = GaussianHMM(
                n_components     = HMM_CFG["n_states"],
                covariance_type  = HMM_CFG["covariance_type"],
                n_iter           = HMM_CFG["n_iter"],
                random_state     = seed,
            )
            m.fit(obs_scaled)
            score = m.score(obs_scaled)
            if score > best_score:
                best_score  = score
                best_model  = m
        except Exception:
            continue

    if best_model is None:
        raise RuntimeError("HMM failed to converge on all seeds.")

    converged = best_model.monitor_.converged
    print(f"  HMM log-likelihood: {best_score:,.2f}  |  converged: {converged}")
    return best_model


# ── STATE → REGIME LABELLING ──────────────────────────────────────────────────

def _label_states(states: np.ndarray, obs_raw: np.ndarray) -> Dict[int, str]:
    """
    Map integer HMM state IDs → Bull / Sideways / Bear.
    Sorting criterion: mean log return of each state.
      Highest  → Bull
      Middle   → Sideways
      Lowest   → Bear
    """
    n_states   = HMM_CFG["n_states"]
    mean_rets  = {}
    for s in range(n_states):
        mask           = states == s
        mean_rets[s]   = obs_raw[mask, 0].mean() if mask.sum() > 0 else 0.0

    sorted_states = sorted(mean_rets, key=lambda s: mean_rets[s])
    regime_names  = ["Bear", "Sideways", "Bull"]   # low → high return order

    mapping = {state: name for state, name in zip(sorted_states, regime_names)}
    return mapping


# ── REGIME CHART ──────────────────────────────────────────────────────────────

def _plot_regimes(df: pd.DataFrame, asset_key: str) -> None:
    """
    Two-panel chart:
      Top  : price (log scale for BTC) with regime-coloured background bands
      Bottom: regime indicator bar
    Saved to results/{asset_key}_regimes.png.
    """
    cfg        = ASSETS[asset_key]
    color_map  = HMM_CFG["regime_colors"]   # {"Bull": "#90EE90", ...}
    use_log    = asset_key == "BTC"

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(18, 9),
        gridspec_kw={"height_ratios": [4, 1]},
        sharex=True,
    )
    fig.suptitle(
        f"{cfg['label']} — HMM 3-State Regime Detection",
        fontsize=14, fontweight="bold",
    )

    # ── Panel 1: price + regime background ───────────────────────────────────
    regime_numeric = df["regime"].map({"Bear": -1, "Sideways": 0, "Bull": 1})

    # Draw regime background bands (find contiguous blocks for efficiency)
    prev_regime = None
    band_start  = df.index[0]

    for ts, regime in df["regime"].items():
        if regime != prev_regime:
            if prev_regime is not None:
                ax1.axvspan(
                    band_start, ts,
                    alpha=0.25,
                    color=color_map.get(prev_regime, "white"),
                    linewidth=0,
                )
            band_start  = ts
            prev_regime = regime
    # Close last band
    ax1.axvspan(
        band_start, df.index[-1],
        alpha=0.25,
        color=color_map.get(prev_regime, "white"),
        linewidth=0,
    )

    ax1.plot(df.index, df["close"], color=cfg["color"],
             linewidth=0.9, label=f"{cfg['label']} close")
    if use_log:
        ax1.set_yscale("log")
        ax1.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
        )
    else:
        ax1.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
        )
    ax1.set_ylabel(f"Price (USD){' — log scale' if use_log else ''}", fontsize=10)
    ax1.grid(True, alpha=0.25)

    # Legend patches
    legend_patches = [
        mpatches.Patch(color=color_map["Bull"],     alpha=0.6, label="Bull"),
        mpatches.Patch(color=color_map["Sideways"], alpha=0.6, label="Sideways"),
        mpatches.Patch(color=color_map["Bear"],     alpha=0.6, label="Bear"),
    ]
    ax1.legend(handles=legend_patches, loc="upper left", fontsize=9)

    # ── Panel 2: regime indicator ─────────────────────────────────────────────
    colors_series = df["regime"].map(color_map)
    ax2.bar(df.index, regime_numeric.values, color=colors_series.values,
            width=2.0, linewidth=0)
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(["Bear", "Sideways", "Bull"], fontsize=8)
    ax2.set_ylabel("Regime", fontsize=9)
    ax2.set_ylim(-1.5, 1.5)
    ax2.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    out = RESULTS / f"{asset_key.lower()}_regimes.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved → results/{out.name}")


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────

def detect_regimes(feat_df: pd.DataFrame, asset_key: str) -> pd.DataFrame:
    """
    Full regime-detection pipeline for one asset.

    Steps:
      1. Extract [log_ret, rvol_20] from feat_df, drop leading NaNs.
      2. Standardise observations (zero-mean, unit-variance).
      3. Fit GaussianHMM (3 states, best of 5 random seeds).
      4. Map states → Bull / Sideways / Bear by mean return.
      5. Align regime labels back to original feat_df index.
      6. Print regime distribution + per-regime statistics.
      7. Generate and save price-regime chart.

    Returns:
      feat_df with a new 'regime' column (str: Bull / Sideways / Bear).
      Rows before the HMM warmup period are labelled NaN.
    """
    OBS_COLS = ["log_ret", "rvol_20"]

    # ── 1. Prepare observations ───────────────────────────────────────────────
    obs_df     = feat_df[OBS_COLS].dropna()
    obs_raw    = obs_df.values                     # shape (T, 2)

    print(f"  Training HMM on {len(obs_df):,} observations "
          f"({obs_df.index[0].date()} → {obs_df.index[-1].date()}) …")

    # ── 2. Standardise ────────────────────────────────────────────────────────
    scaler     = StandardScaler()
    obs_scaled = scaler.fit_transform(obs_raw)

    # ── 3. Fit HMM ────────────────────────────────────────────────────────────
    model      = _fit_hmm(obs_scaled)
    states     = model.predict(obs_scaled)         # integer state per row

    # ── 4. Label states ───────────────────────────────────────────────────────
    state_map  = _label_states(states, obs_raw)    # {int → "Bull"/"Bear"/...}
    regime_arr = np.array([state_map[s] for s in states])

    # ── 5. Align to original index ────────────────────────────────────────────
    regime_series = pd.Series(regime_arr, index=obs_df.index, name="regime")
    df_out        = feat_df.copy()
    df_out["regime"] = regime_series                # NaN for warm-up rows

    # ── 6. Print statistics ───────────────────────────────────────────────────
    valid = df_out.dropna(subset=["regime", "log_ret", "rvol_20"])

    print(f"\n  {'─'*58}")
    print(f"  {asset_key} Regime Distribution")
    print(f"  {'─'*58}")
    print(f"  {'Regime':<12} {'Days':>6} {'% Time':>8} "
          f"{'Avg DailyRet':>14} {'Ann.Ret%':>10} {'Ann.Vol%':>10}")
    print(f"  {'─'*58}")

    for regime in ["Bull", "Sideways", "Bear"]:
        mask  = valid["regime"] == regime
        sub   = valid.loc[mask]
        n     = mask.sum()
        pct   = n / len(valid) * 100
        avg_r = sub["log_ret"].mean()
        ann_r = avg_r * 252 * 100
        ann_v = sub["rvol_20"].mean() * 100
        print(f"  {regime:<12} {n:>6,} {pct:>7.1f}% "
              f"{avg_r:>14.5f} {ann_r:>9.1f}% {ann_v:>9.1f}%")

    print(f"  {'─'*58}")
    print(f"  Total labelled rows: {len(valid):,}")

    # Regime transition stats
    transitions = (valid["regime"] != valid["regime"].shift()).sum()
    print(f"  Regime transitions : {transitions:,}")
    avg_duration = len(valid) / max(transitions, 1)
    print(f"  Avg regime duration: {avg_duration:.1f} days")

    # ── 7. Plot ───────────────────────────────────────────────────────────────
    plot_df = valid[["close", "regime"]].copy()
    _plot_regimes(plot_df, asset_key)

    return df_out
