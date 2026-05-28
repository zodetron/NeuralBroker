"""
config.py — All tunable parameters for the BTC 15M scalping strategy.
"""

from pathlib import Path

ROOT    = Path(__file__).parent
DATA    = ROOT / "data"
RESULTS = ROOT / "results"

DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

# ── DATA ──────────────────────────────────────────────────────────────────────
DATA_CFG = {
    "symbol_15m": "BTC/USDT",
    "symbol_1h":  "BTC/USDT",
    "tf_15m":     "15m",
    "tf_1h":      "1h",
    "start":      "2019-09-01",    # earliest reliable Binance 15M data
    "exchange":   "binance",
    "parquet_15m": DATA / "btc_15m.parquet",
    "parquet_1h":  DATA / "btc_1h.parquet",
}

# ── MARKET STATE DETECTOR ────────────────────────────────────────────────────
MS = {
    "ema_period":       20,
    "adx_period":       14,
    "adx_trend_thresh": 25,          # ADX > 25 → trending
    "atr_period":       20,
    "atr_high_mult":    1.5,         # ATR > 1.5× avg → HIGH_VOL_CHOP
    "atr_low_mult":     1.0,         # ATR < 1.0× avg → LOW_VOL_RANGE
    "slope_bars":        5,          # bars to measure EMA slope direction
}

# ── STRATEGY A — Momentum Burst ───────────────────────────────────────────────
STRAT_A = {
    "lookback_bars":     10,         # highest high / lowest low of last N bars
    "vol_mult":           2.0,       # volume > N× 20-bar avg
    "atr_tp_mult":        1.2,
    "atr_sl_mult":        0.6,
    "time_exit_bars":     8,         # 8 × 15M = 2 hours
    "trend_align_bonus":  0.05,      # added to ML confidence if 1H aligned
}

# ── STRATEGY B — VWAP Reversion ───────────────────────────────────────────────
STRAT_B = {
    "vwap_reset_hour":    0,         # reset VWAP at UTC midnight
    "vwap_band_pct":      0.0050,    # 0.50% from VWAP (tightened from 0.35%)
    "rsi_period":         14,
    "rsi_long_max":       42,
    "rsi_short_min":      58,
    "vol_spike_mult":     1.5,       # block if volume > 1.5× avg (news spike)
    "atr_sl_mult":        0.5,
    "time_exit_bars":     6,         # 6 × 15M = 90 minutes
}

# ── STRATEGY C — EMA Pullback ─────────────────────────────────────────────────
STRAT_C = {
    "ema_period":         20,        # same as market state EMA
    "rsi_period":         14,
    "rsi_pb_lo":          38,
    "rsi_pb_hi":          52,
    "ema_touch_pct":       0.3,       # ±0.3% of EMA20 counts as a "touch" (widened from 0.1%)
    "red_bars_required":   2,        # at least 2 of last 3 bars bearish (for long)
    "atr_tp_mult":        1.5,
    "atr_sl_mult":        0.8,
    "time_exit_bars":    12,         # 12 × 15M = 3 hours
}

# ── ML FILTER ─────────────────────────────────────────────────────────────────
# Per-strategy tiered thresholds — separate model per strategy.
# Rationale: each strategy fires in different market conditions;
# a single model cannot learn trend vs reversion vs burst patterns together.
ML_PER_STRAT = {
    "A": {
        "threshold":    0.56,   # higher bar — breakout false positives are costly
        "train_weeks":  8,      # longer window: A fires ~0.9/day → need 8w for ~50 samples
        "test_weeks":   1,
        "min_samples":  40,
    },
    "B": {
        "threshold":    0.51,   # lower bar — natural reversion edge, vol already filters
        "train_weeks":  4,
        "test_weeks":   1,
        "min_samples":  120,
    },
    "C": {
        "threshold":    0.52,   # secondary check; 1H alignment already gates heavily
        "train_weeks": 16,      # wide window: C fires rarely, need history for samples
        "test_weeks":   1,
        "min_samples":  20,
    },
}

ML = {
    "min_confidence":  0.54,   # legacy default — not used in Phase 5 (per-strat instead)
    "wf_train_weeks":   4,
    "wf_test_weeks":    1,
    "forward_bars":     4,           # label: does trade hit TP within 4 bars?
    "min_train_samples": 200,
    "xgb": {
        "n_estimators":     200,
        "max_depth":          4,
        "learning_rate":    0.05,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "use_label_encoder": False,
        "eval_metric":      "logloss",
        "verbosity":          0,
        "random_state":      42,
    },
}

# ── BACKTEST ──────────────────────────────────────────────────────────────────
BT = {
    "initial_capital":      10_000,
    "commission":           0.0005,  # 0.05% per side (maker fee)
    "slippage":             0.0002,  # 0.02% adverse fill
    "max_concurrent":            3,  # max 1 per strategy simultaneously
    "daily_loss_limit_pct":   2.0,   # halt trading day if down 2%
    "no_overnight_close_utc": 23.75, # 23:45 UTC decimal
    "no_overnight_open_utc":  0.25,  # 00:15 UTC decimal

    # ── Dynamic spread ────────────────────────────────────────────────────────
    # BTC 15M spreads vary by session liquidity and volatility regime.
    # Applied once at trade entry (half-spread cost, representing crossing the book).
    "spread_normal":         0.0001,  # 0.01%  — normal market hours
    "spread_low_liq":        0.0003,  # 0.03%  — 02:00–06:00 UTC (Asian off-hours)
    "spread_high_vol":       0.0005,  # 0.05%  — ATR spike > 3× avg
    "spread_atr_mult_thresh": 3.0,    # ATR ratio above which high-vol spread applies
    "spread_low_liq_start":    2,     # UTC hour range start (inclusive)
    "spread_low_liq_end":      6,     # UTC hour range end   (inclusive)

    # Round-trip cost summary (reference only — computed dynamically):
    #   Normal:       commission 0.10% + spread 0.01% + slippage 0.02% = 0.13%
    #   Low liquidity: 0.10% + 0.03% + 0.02% = 0.15%
    #   High vol:      0.10% + 0.05% + 0.02% = 0.17%
}

# ── WFO ───────────────────────────────────────────────────────────────────────
WFO = {
    "train_weeks":  8,
    "test_weeks":   2,
}

# ── SESSION HOURS (UTC) ───────────────────────────────────────────────────────
SESSIONS = {
    "Asian":  (0,  8),    # 00:00 – 08:00 UTC
    "London": (7, 16),    # 07:00 – 16:00 UTC
    "NY":    (13, 22),    # 13:00 – 22:00 UTC
}
