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
    "atr_sl_mult":        0.5,        # reverted: 1.0 → 0.5 (tighter SL, midpoint TP)
    "time_exit_bars":    12,         # extended: 6 → 12 bars = 3 hours
}

# ── STRATEGY C — EMA Pullback ─────────────────────────────────────────────────
STRAT_C = {
    "ema_period":         20,        # same as market state EMA
    "rsi_period":         14,
    "rsi_pb_lo":          38,
    "rsi_pb_hi":          52,
    "ema_touch_pct":       0.6,       # ±0.6% of EMA20 counts as a "touch" (widened)
    "red_bars_required":   1,        # at least 1 of last 3 bars bearish (for long)
    "atr_tp_mult":        1.8,       # raised: 1.5 → 1.8
    "atr_sl_mult":        0.8,
    "time_exit_bars":    12,         # 12 × 15M = 3 hours
}

# ── STRATEGY D — Manipulation Fade (ICT Fib 2.5/4.0) ─────────────────────────
STRAT_D = {
    # Setup candle detection
    "big_body_mult":     1.5,   # body > 1.5× avg body → "big" candle
    "key_level_pct":     0.015, # setup candle high/low within 1.5% of N-bar extreme
    "atr_spike_thresh":  2.5,   # skip if ATR > 2.5× avg at setup (news candle)
    "atr_news_thresh":   3.0,   # skip if any of last 4 bars had ATR > 3.0× avg

    # Fib extension levels
    "fib_entry":         2.5,   # entry at 2.5 extension of manipulation leg
    "fib_stop":          4.0,   # stop loss at 4.0 extension (invalidation)

    # Displacement confirmation
    "disp_body_mult":    0.5,   # displacement bar body > 0.5× avg body (relaxed)

    # Trade management
    "stop_buffer_atr":   0.1,   # ATR buffer added to fib_stop for actual SL
    "min_rr":            1.5,   # skip signal if fib TP/SL ratio < 1.5
    "time_exit_bars":   16,     # 16 × 15M = 4 hours max hold

    # Staleness
    "staleness_bars":   20,     # invalidate setup if manipulation leg > 20 bars old

    # Confluence scores (HIGH_CONFLUENCE flag)
    "conf_roll50_pct":   0.003, # near 50-bar H/L within 0.3% → +1 confluence
    "conf_vwap_dev":     0.004, # near VWAP within 0.4% → +1 confluence
    "conf_rsi_bull":    35,     # RSI < 35 for bull fade → +1 confluence
    "conf_rsi_bear":    65,     # RSI > 65 for bear fade → +1 confluence
}

# ── ML FILTER ─────────────────────────────────────────────────────────────────
# Per-strategy tiered thresholds — separate model per strategy.
# Rationale: each strategy fires in different market conditions;
# a single model cannot learn trend vs reversion vs burst patterns together.
ML_PER_STRAT = {
    "A": {
        "threshold":    0.54,   # lowered: 0.56 → 0.54 to recover filtered signals
        "train_weeks":  8,      # longer window: A fires ~0.9/day → need 8w for ~50 samples
        "test_weeks":   1,
        "min_samples":  40,
    },
    "B": {
        "threshold":    0.51,   # lower bar — natural reversion edge, vol already filters
        "train_weeks":  4,
        "test_weeks":   1,
        "min_samples":  60,     # lowered: 120 → 60 (was skipping 83% of windows)
    },
    "C": {
        "threshold":    0.52,   # not used — C skips ML entirely (too few signals)
        "train_weeks": 16,
        "test_weeks":   1,
        "min_samples":  20,
    },
    "D": {
        "threshold":    0.52,   # 0.50 for HIGH_CONFLUENCE signals
        "train_weeks": 12,
        "test_weeks":   1,
        "min_samples":  40,
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
    "commission":           0.0005,  # 0.05% per side (taker fee)
    "slippage":             0.0002,  # 0.02% adverse fill
    "max_concurrent":            3,  # max 1 per strategy simultaneously
    "daily_loss_limit_pct":   2.0,   # halt trading day if down 2%
    # Strategy A variable risk sizing
    "MAX_RISK_PCT_HIGH_VOL": 0.01,   # 1% in HIGH_VOL_CHOP (costs eat edge)
    "MAX_RISK_PCT_NORMAL":   0.02,   # 2% in all other states
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
