"""
config.py — Single source of truth for all parameters.
Never hardcode values in logic files — change them here only.
"""

from pathlib import Path

# ── PATHS ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent
DATA    = ROOT / "data"
RESULTS = ROOT / "results"

# ── ASSETS ────────────────────────────────────────────────────────────────────
ASSETS = {
    "BTC": {
        "source":   "ccxt",
        "exchange": "binance",
        "symbol":   "BTC/USDT",
        "csv":      DATA / "btc_daily.csv",
        "start":    "2017-09-01",   # Binance BTC/USDT launch
        "label":    "BTC/USD",
        "color":    "#F7931A",
    },
    "XAU": {
        "source":   "yfinance",
        "ticker":   "GC=F",
        "csv":      DATA / "xau_daily.csv",
        "start":    "2010-01-01",   # long history for feature warmup
        "label":    "XAU/USD (Gold)",
        "color":    "#FFD700",
    },
}

TIMEFRAME = "1d"

# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────
FEAT = {
    "vol_windows":    [20, 60],        # realized-volatility rolling windows
    "atr_windows":    [14, 30],        # ATR windows
    "sma_windows":    [50, 200],       # SMA distance windows
    "roc_windows":    [5, 10, 20, 60], # rate-of-change lookbacks
    "rsi_windows":    [14, 28],        # RSI windows
    "bb_window":      20,              # Bollinger Band window
    "bb_std":         2.0,
    "hurst_window":   60,              # Hurst exponent rolling window
    "sharpe_windows": [20, 60],        # rolling Sharpe windows
    "log_return_clip":0.2,             # clip extreme log returns for stability
}

# ── REGIME DETECTION (HMM) ────────────────────────────────────────────────────
HMM = {
    "n_states":         3,
    "n_iter":           1000,
    "random_state":     42,
    "covariance_type":  "full",
    "regime_names":     {0: "Bull", 1: "Sideways", 2: "Bear"},
    "regime_colors":    {"Bull": "#90EE90", "Sideways": "#D3D3D3", "Bear": "#FFB6C1"},
}

# ── STRATEGY ──────────────────────────────────────────────────────────────────
STRAT = {
    # Position sizing
    "risk_per_trade":    0.02,    # 2% of capital at risk per trade
    "max_position_pct":  0.25,    # never commit >25% of capital to one trade

    # Stop loss / take profit (ATR multiples)
    "atr_stop_mult":     1.5,     # SL = entry ± 1.5 × ATR14
    "atr_tp_mult":       3.0,     # TP = entry ± 3.0 × ATR14  (2:1 R:R minimum)

    # Momentum signal (Bull regime)
    "momentum_roc_period": 10,    # primary ROC window for bull momentum score
    "momentum_rsi_min":    45,    # RSI must be above this to confirm momentum

    # Mean-reversion signal (Sideways regime)
    "rsi_oversold":     35,       # Sideways: buy when RSI14 < 35
    "rsi_overbought":   65,       # Sideways: sell when RSI14 > 65

    # Bear regime
    "allow_shorting":   False,    # set True to allow short positions in Bear

    # ATR window used for sizing and stops
    "atr_window":       14,
}

# ── ML FILTER (XGBoost) ───────────────────────────────────────────────────────
ML = {
    "forward_days":           5,      # label = 1 if fwd return > 0 over next 5d
    "min_confidence":         0.55,   # only trade if P(positive) > 0.55
    "wf_train_months":        12,     # walk-forward: train window
    "wf_test_months":         3,      # walk-forward: test window
    "min_train_samples":      200,    # skip window if fewer training samples
    "xgb": {
        "n_estimators":       200,
        "max_depth":          4,
        "learning_rate":      0.05,
        "subsample":          0.8,
        "colsample_bytree":   0.8,
        "min_child_weight":   5,
        "random_state":       42,
        "eval_metric":        "logloss",
        "verbosity":          0,
    },
}

# ── BACKTEST ENGINE ───────────────────────────────────────────────────────────
BT = {
    "initial_capital": 10_000.0,
    "commission":      0.001,    # 0.1% per trade (round-trip = 0.2%)
    "slippage":        0.0005,   # 0.05% adverse slippage on fill
}

# ── WALK-FORWARD OPTIMISATION ─────────────────────────────────────────────────
WFO = {
    "train_months": 6,
    "test_months":  2,
    "param_grid": {
        "atr_stop_mult":       [1.0, 1.5, 2.0, 2.5, 3.0],
        "momentum_roc_period": [5, 10, 15, 20, 30],
    },
}
