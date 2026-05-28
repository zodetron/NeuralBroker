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
    "vwap_band_pct":      0.0035,    # 0.35% from VWAP
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
    "red_bars_required":   3,        # previous N bars must be bearish (for long)
    "atr_tp_mult":        1.5,
    "atr_sl_mult":        0.8,
    "time_exit_bars":    12,         # 12 × 15M = 3 hours
}

# ── ML FILTER ─────────────────────────────────────────────────────────────────
ML = {
    "min_confidence":  0.54,
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
    "commission":           0.0005,  # 0.05% maker fee per side
    "slippage":             0.0002,  # 0.02% adverse
    "max_concurrent":            3,  # max 1 per strategy at a time
    "daily_loss_limit_pct":   2.0,   # stop trading day if down 2%
    "no_overnight_close_utc": 23.75, # 23:45 UTC in decimal hours
    "no_overnight_open_utc":  0.25,  # 00:15 UTC in decimal hours
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
