"""
features/engineer.py — Feature engineering pipeline.

ZERO LOOKAHEAD BIAS CONTRACT:
  All features at row t use only data from rows [t-window ... t].
  close_t (end-of-day price) is available at bar close, so it is
  legitimate to use it in feature computation.
  The strategy layer will shift features by 1 bar before generating
  signals, ensuring we never act on information from the future.

Feature groups:
  Price-based    : log_ret, rvol_20, rvol_60, atr_14, atr_30,
                   bb_width, sma50_dist, sma200_dist
  Momentum       : roc_5, roc_10, roc_20, roc_60, rsi_14, rsi_28,
                   macd_hist
  Regime proxies : sharpe_20, sharpe_60, hurst, vol_ratio
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FEAT, ASSETS, STRAT

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _true_range(df: pd.DataFrame) -> pd.Series:
    """True Range = max(H-L, |H-Cprev|, |L-Cprev|). No lookahead."""
    c_prev = df["close"].shift(1)
    return pd.concat([
        df["high"] - df["low"],
        (df["high"] - c_prev).abs(),
        (df["low"]  - c_prev).abs(),
    ], axis=1).max(axis=1)


def _rsi(series: pd.Series, window: int) -> pd.Series:
    """Wilder-smoothed RSI using EMA(com = window-1). Standard definition."""
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=window - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=window - 1, adjust=False).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - 100 / (1 + rs)


def _adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Average Directional Index (Wilder smoothing).
    ADX > 25 → trending; ADX < 20 → ranging/flat.
    No lookahead: uses only past OHLC data.
    """
    high  = df["high"]
    low   = df["low"]
    tr    = _true_range(df)

    up_move   = (high - high.shift(1)).clip(lower=0)
    down_move = (low.shift(1) - low).clip(lower=0)

    plus_dm  = np.where((up_move > down_move), up_move,  0.0)
    minus_dm = np.where((down_move > up_move), down_move, 0.0)

    plus_dm_s  = pd.Series(plus_dm,  index=df.index)
    minus_dm_s = pd.Series(minus_dm, index=df.index)

    com = window - 1
    tr14       = tr.ewm(com=com, adjust=False).mean()
    plus_di    = 100 * plus_dm_s.ewm(com=com, adjust=False).mean()  / (tr14 + 1e-9)
    minus_di   = 100 * minus_dm_s.ewm(com=com, adjust=False).mean() / (tr14 + 1e-9)

    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    adx = dx.ewm(com=com, adjust=False).mean()
    return adx


def _hurst_rs(arr: np.ndarray) -> float:
    """
    R/S Hurst exponent for a window of log returns.
      H > 0.5  → trending (persistent)
      H ≈ 0.5  → random walk
      H < 0.5  → mean-reverting (anti-persistent)
    Uses classic Hurst-Mandelbrot R/S rescaled range analysis.
    """
    n = len(arr)
    if n < 20:
        return np.nan
    arr = arr - arr.mean()           # demean
    z   = np.cumsum(arr)             # cumulative deviation
    r   = z.max() - z.min()          # range
    s   = arr.std(ddof=1)            # standard deviation
    if s < 1e-9:
        return np.nan
    return float(np.log(r / s) / np.log(n))


# ── MAIN FEATURE FUNCTION ─────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all features for a single-asset OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: open, high, low, close, volume.
        Index must be a DatetimeIndex (UTC).

    Returns
    -------
    pd.DataFrame
        Original OHLCV columns retained, all feature columns appended.
        Rows with insufficient history (leading NaNs) are NOT dropped here —
        caller decides how many to keep.
    """
    d  = df.copy()
    c  = d["close"]
    lr = np.log(c / c.shift(1)).clip(
            -FEAT["log_return_clip"], FEAT["log_return_clip"]
         )

    # ── 1. LOG RETURNS ────────────────────────────────────────────────────────
    d["log_ret"] = lr

    # ── 2. REALIZED VOLATILITY (annualised) ──────────────────────────────────
    for w in FEAT["vol_windows"]:
        # ddof=1 for sample std; *sqrt(252) to annualise daily vol
        d[f"rvol_{w}"] = lr.rolling(w).std(ddof=1) * np.sqrt(252)

    # ── 3. ATR (Wilder EMA smoothing) ────────────────────────────────────────
    tr = _true_range(d)
    for w in FEAT["atr_windows"]:
        # ewm(span=w) = Wilder's smoothing (com = w-1)
        d[f"atr_{w}"] = tr.ewm(span=w, adjust=False).mean()

    # ── 4. BOLLINGER BANDS ────────────────────────────────────────────────────
    bb_w   = FEAT["bb_window"]
    bb_mid = c.rolling(bb_w).mean()
    bb_std = c.rolling(bb_w).std(ddof=1)
    bb_upper = bb_mid + FEAT["bb_std"] * bb_std
    bb_lower = bb_mid - FEAT["bb_std"] * bb_std
    d["bb_width"] = (2 * FEAT["bb_std"] * bb_std) / (bb_mid + 1e-9)
    d["bb_upper"] = bb_upper   # Improvement 2: raw upper band for XAU signals
    d["bb_lower"] = bb_lower   # Improvement 2: raw lower band for XAU signals
    d["bb_mid"]   = bb_mid     # 20-day SMA midpoint — XAU sticky exit target
    # Normalised position within band: 0 = at lower, 1 = at upper
    d["bb_pct"]   = (c - bb_lower) / (bb_upper - bb_lower + 1e-9)

    # ── 5. SMA DISTANCE (% above/below SMA) ──────────────────────────────────
    for w in FEAT["sma_windows"]:
        sma = c.rolling(w).mean()
        d[f"sma{w}_dist"] = (c - sma) / (sma + 1e-9) * 100

    # ── 6. RATE OF CHANGE ─────────────────────────────────────────────────────
    for w in FEAT["roc_windows"]:
        c_prev        = c.shift(w)
        d[f"roc_{w}"] = (c - c_prev) / (c_prev + 1e-9) * 100

    # ── 7. RSI ────────────────────────────────────────────────────────────────
    for w in FEAT["rsi_windows"]:
        d[f"rsi_{w}"] = _rsi(c, w)

    # ── 8. MACD HISTOGRAM ─────────────────────────────────────────────────────
    ema12          = c.ewm(span=12, adjust=False).mean()
    ema26          = c.ewm(span=26, adjust=False).mean()
    macd           = ema12 - ema26
    macd_signal    = macd.ewm(span=9, adjust=False).mean()
    d["macd_hist"] = macd - macd_signal

    # ── 9. ROLLING SHARPE RATIO ───────────────────────────────────────────────
    for w in FEAT["sharpe_windows"]:
        roll             = lr.rolling(w)
        d[f"sharpe_{w}"] = (roll.mean() / (roll.std(ddof=1) + 1e-9)) * np.sqrt(252)

    # ── 10. HURST EXPONENT (rolling R/S) ──────────────────────────────────────
    # Computed on a rolling window of log returns.
    # Takes ~5 seconds on 6 k rows — acceptable for daily data.
    d["hurst"] = (
        lr.rolling(FEAT["hurst_window"])
          .apply(_hurst_rs, raw=True)
    )

    # ── 11. VOLATILITY RATIO (short-term / long-term vol) ─────────────────────
    short_vol = d[f"rvol_{FEAT['vol_windows'][0]}"]   # 20-day
    long_vol  = d[f"rvol_{FEAT['vol_windows'][1]}"]   # 60-day
    d["vol_ratio"] = short_vol / (long_vol + 1e-9)

    # ── 12. BREAKOUT LEVELS (Improvement 1 — BTC volume-confirmed breakout) ──
    bw = FEAT["breakout_window"]
    # Shift by 1 so today's close is compared to the PRIOR N-day range (no lookahead)
    d["roll_high_20"]  = c.shift(1).rolling(bw).max()
    d["roll_low_20"]   = c.shift(1).rolling(bw).min()
    # Normalised distances: positive = above, negative = below
    d["roll_high_dist"] = (c / (d["roll_high_20"] + 1e-9) - 1) * 100
    d["roll_low_dist"]  = (c / (d["roll_low_20"]  + 1e-9) - 1) * 100

    # ── 13. RELATIVE VOLUME (Improvement 1 — volume confirmation) ─────────────
    d["vol_ma_20"]    = d["volume"].rolling(bw).mean()
    d["vol_ratio_20"] = d["volume"] / (d["vol_ma_20"] + 1e-9)

    # ── 14. ADX — Average Directional Index (Improvement 3) ───────────────────
    d["adx_14"] = _adx(d, window=FEAT["adx_window"])

    return d


# ── FEATURE COLUMNS (in order) ────────────────────────────────────────────────
FEATURE_COLS = (
    ["log_ret"]
    + [f"rvol_{w}"     for w in FEAT["vol_windows"]]
    + [f"atr_{w}"      for w in FEAT["atr_windows"]]
    + ["bb_width", "bb_pct"]
    + [f"sma{w}_dist"  for w in FEAT["sma_windows"]]
    + [f"roc_{w}"      for w in FEAT["roc_windows"]]
    + [f"rsi_{w}"      for w in FEAT["rsi_windows"]]
    + ["macd_hist"]
    + [f"sharpe_{w}"   for w in FEAT["sharpe_windows"]]
    + ["hurst", "vol_ratio"]
    + ["roll_high_dist", "roll_low_dist", "vol_ratio_20", "adx_14"]
)


# ── PIPELINE ENTRY POINT ──────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, asset_key: str, save: bool = True) -> pd.DataFrame:
    """
    Full pipeline: compute features, optionally save to /data, print summary.

    Parameters
    ----------
    df        : raw OHLCV DataFrame from data.pipeline
    asset_key : 'BTC' or 'XAU' (used for CSV filename)
    save      : write CSV to data/{asset_key}_features.csv

    Returns
    -------
    pd.DataFrame with OHLCV + all feature columns, NaN rows at the front retained.
    """
    print(f"  Computing features for {asset_key} ({len(df):,} rows) …", end="", flush=True)
    feat_df = compute_features(df)
    print(" done.")

    # ── SAVE ──────────────────────────────────────────────────────────────────
    if save:
        out_path = ASSETS[asset_key]["csv"].parent / f"{asset_key.lower()}_features.csv"
        feat_df.to_csv(out_path)
        print(f"  Saved → {out_path.name}")

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    feat_only = feat_df[FEATURE_COLS]
    nan_counts = feat_only.isna().sum()
    has_nans   = nan_counts[nan_counts > 0]

    print(f"\n  {'─'*60}")
    print(f"  {asset_key} Feature Summary  ({len(FEATURE_COLS)} features computed)")
    print(f"  {'─'*60}")
    print(f"  {'Feature':<18}  {'Non-NaN':>8}  {'NaN':>6}  {'Min':>10}  {'Mean':>10}  {'Max':>10}")
    print(f"  {'─'*60}")

    for col in FEATURE_COLS:
        s      = feat_df[col].dropna()
        n_nan  = feat_df[col].isna().sum()
        if len(s) == 0:
            print(f"  {col:<18}  {'—':>8}  {n_nan:>6}  {'all NaN':>32}")
        else:
            print(
                f"  {col:<18}  {len(s):>8,}  {n_nan:>6}  "
                f"{s.min():>10.4f}  {s.mean():>10.4f}  {s.max():>10.4f}"
            )

    print(f"  {'─'*60}")
    if len(has_nans) == 0:
        print(f"  NaN check: ✓ no NaNs in any feature column")
    else:
        print(f"  NaN check: {len(has_nans)} columns with NaNs "
              f"(all leading — caused by rolling-window warmup, expected):")
        for col, cnt in has_nans.items():
            print(f"    {col:<18}  {cnt:>5} NaNs  "
                  f"(first valid: row {feat_df[col].first_valid_index().date() if feat_df[col].first_valid_index() else '—'})")

    clean_rows = feat_df.dropna(subset=FEATURE_COLS)
    print(f"\n  Rows with ALL features valid : {len(clean_rows):,} "
          f"(of {len(feat_df):,} total)")

    return feat_df
