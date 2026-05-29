"""
config.py — All tunable parameters for the MTF Scalping live system.
Single source of truth: every module imports from here, nothing is hardcoded elsewhere.
"""

from pathlib import Path

ROOT = Path(__file__).parent

# ── Data feed ──────────────────────────────────────────────────────────────────
BINANCE_WS_URL      = "wss://stream.binance.com:9443/ws/btcusdt@kline_1m"
BINANCE_REST_BASE   = "https://api.binance.com"
SYMBOL              = "BTCUSDT"
HIST_START          = "2019-01-01"          # earliest date for historical fill

# ── Storage paths ──────────────────────────────────────────────────────────────
DB_PATH             = str(ROOT / "data" / "candles.duckdb")
TRADES_DB_PATH      = str(ROOT / "results" / "paper_trades.db")
MODELS_DIR          = str(ROOT / "models")

# ── Paper trading ──────────────────────────────────────────────────────────────
STARTING_BALANCE        = 10_000.0
RISK_PCT_STANDARD       = 1.5          # % of balance per standard signal
RISK_PCT_CONFLUENCE     = 2.0          # % of balance for confluence signals
MAX_TRADES_PER_DAY      = 4
DAILY_LOSS_LIMIT_PCT    = 2.0          # halt if down this % intraday
COMMISSION_RT           = 0.001        # 0.1% round-trip commission

# ── Signal filtering ───────────────────────────────────────────────────────────
MIN_DAILY_RANGE_PCT     = 0.3          # skip dead markets
ML_THRESHOLD_STANDARD   = 0.54
ML_THRESHOLD_CONFLUENCE = 0.50
MIN_RR_RATIO            = 1.2
SIGNAL_COOLDOWN_MINS    = 15           # min minutes between signals
TIME_EXIT_BARS          = 12           # 12 × 5M = 60 min max hold

# ── ATR-based TP/SL multipliers ────────────────────────────────────────────────
ATR_TP_MULT_1           = 1.5          # first target
ATR_TP_MULT_2           = 3.0          # second target (trail)
ATR_SL_MULT             = 1.0

# ── Swappy ICT parameters ──────────────────────────────────────────────────────
SWAPPY_MIN_RR           = 1.3
SWAPPY_MAX_BARS         = 20           # invalidate setup after 20 15M bars
SWAPPY_TP_PCT           = 0.5          # take 50% of distance back to origin
SWAPPY_TOUCH_PCT        = 0.0005       # ±0.05% fib_2_5 touch tolerance
SWAPPY_SLIP_GUARD_PCT   = 0.003        # skip if entry > fib_2_5 + 0.3%
SWAPPY_BIG_BODY_MULT    = 1.5          # setup candle body > 1.5× avg
SWAPPY_SUPPORT_PCT      = 0.005        # within 0.5% of nearest support
SWAPPY_ATR_SPIKE        = 2.0          # skip if ATR > 2.0× avg at setup
SWAPPY_DISP_BODY_MULT   = 0.8          # displacement body > 0.8× avg
SWAPPY_DISP_TOP_PCT     = 0.30         # closes in top/bottom 30% of range
SWAPPY_VOL_MULT         = 1.3          # volume > 1.3× 20-bar avg
SWAPPY_RSI_LONG_MAX     = 45           # RSI < 45 for long entry
SWAPPY_RSI_SHORT_MIN    = 55           # RSI > 55 for short entry
SWAPPY_CONF_RSI_LONG    = 35           # RSI < 35 → high confluence long
SWAPPY_CONF_RSI_SHORT   = 65           # RSI > 65 → high confluence short
SWAPPY_CONF_ZONE_PCT    = 0.003        # fib within 0.3% of 4H level

# ── Market structure thresholds ────────────────────────────────────────────────
ADX_TREND_THRESH        = 25
ADX_RANGE_THRESH        = 20
EMA_TOUCH_PCT           = 0.002        # ±0.2% of EMA20 for pullback
ZONE_RANGE_PCT          = 0.003        # within 0.3% of 4H high/low
OB_RALLY_PCT            = 0.01         # 1%+ 3-bar move to qualify order block
BREAKOUT_VOL_MULT       = 1.5
ENGULF_BODY_MULT        = 0.8          # engulfing body > 0.8× avg
BREAKOUT_VOL_5M         = 1.3

# ── API ────────────────────────────────────────────────────────────────────────
API_PORT                = 8000
CORS_ORIGINS            = ["http://localhost:5173", "http://127.0.0.1:5173"]

# ── Hardcoded backtest reference (verified from Phase 6 v3 analysis) ───────────
BACKTEST_NET_PF         = 1.439
BACKTEST_SHARPE         = 0.72
BACKTEST_CAGR_PCT       = 15.0         # annualised from 6.7-year test
BACKTEST_MAX_DD         = -15.6        # peak-to-trough % (A-only subsystem)
