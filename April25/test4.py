"""
BTC/USD 1H Backtest — All 4 Strategies
Capital: $100 | Leverage: 1:1000 | Period: last ~2 years
Strategies: EMA Cross+RSI, RSI Reversal, BB Breakout, MACD+EMA

HOW LEVERAGE WORKS HERE:
  - You have $100 capital
  - Risk per trade = 1% of current capital (so $1 on first trade)
  - With 1:1000 leverage, you control $1000 worth of BTC per $1 risked
  - SL is set so max loss = risk amount ($1)
  - TP is set at 2x risk = $2 profit per win
  - This is realistic forex/crypto CFD sizing
"""

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
STARTING_CAPITAL = 100.0     # USD
LEVERAGE         = 1000      # 1:1000
RISK_PCT         = 0.01      # Risk 1% of capital per trade
SL_PCT           = 0.005     # 0.5% price move = SL (tight due to leverage)
TP_MULT          = 2.0       # TP = 2x SL distance (2:1 RR)
TICKER           = "BTC-USD"
INTERVAL         = "1h"

# ──────────────────────────────────────────────
# FETCH DATA
# ──────────────────────────────────────────────
print("📥 Fetching BTC/USD 1H data (last ~2 years in chunks)...")

end_date   = datetime.today()
start_date = end_date - timedelta(days=720)

chunks = []
chunk_start = start_date
while chunk_start < end_date:
    chunk_end = min(chunk_start + timedelta(days=59), end_date)
    chunk = yf.download(
        TICKER,
        start=chunk_start.strftime("%Y-%m-%d"),
        end=chunk_end.strftime("%Y-%m-%d"),
        interval=INTERVAL,
        auto_adjust=True,
        progress=False
    )
    if len(chunk) > 0:
        chunks.append(chunk)
    chunk_start = chunk_end

df = pd.concat(chunks)
df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
df = df[~df.index.duplicated(keep="first")]
df.sort_index(inplace=True)
df.dropna(inplace=True)
print(f"✅ {len(df)} candles loaded | {df.index[0].date()} → {df.index[-1].date()}\n")

# ──────────────────────────────────────────────
# INDICATORS
# ──────────────────────────────────────────────
def add_indicators(df):
    d = df.copy()
    d["ema20"]   = d["Close"].ewm(span=20,  adjust=False).mean()
    d["ema50"]   = d["Close"].ewm(span=50,  adjust=False).mean()
    d["ema200"]  = d["Close"].ewm(span=200, adjust=False).mean()

    delta        = d["Close"].diff()
    gain         = delta.clip(lower=0).rolling(14).mean()
    loss         = (-delta.clip(upper=0)).rolling(14).mean()
    d["rsi"]     = 100 - (100 / (1 + gain / loss))

    d["bb_mid"]  = d["Close"].rolling(20).mean()
    bb_std       = d["Close"].rolling(20).std()
    d["bb_upper"]= d["bb_mid"] + 2 * bb_std
    d["bb_lower"]= d["bb_mid"] - 2 * bb_std

    ema12        = d["Close"].ewm(span=12, adjust=False).mean()
    ema26        = d["Close"].ewm(span=26, adjust=False).mean()
    d["macd"]    = ema12 - ema26
    d["macd_sig"]= d["macd"].ewm(span=9, adjust=False).mean()
    return d

df = add_indicators(df)
df.dropna(inplace=True)

# ──────────────────────────────────────────────
# BACKTEST ENGINE  (fixed PnL math)
# ──────────────────────────────────────────────
def backtest(df, signal_func, strategy_name, sl_pct=SL_PCT, rr=TP_MULT):
    """
    PnL logic:
      risk_usd  = capital * RISK_PCT       e.g. $1
      pnl on TP = +risk_usd * rr          e.g. +$2
      pnl on SL = -risk_usd               e.g. -$1
    Leverage scales notional but risk is always capped at risk_usd.
    """
    tp_pct   = sl_pct * rr
    capital  = STARTING_CAPITAL
    position = None
    trades   = []

    entry_price = sl_price = tp_price = 0.0
    entry_time  = None
    direction   = None
    risk_usd    = 0.0

    signals = signal_func(df)

    for i in range(1, len(df)):
        if capital <= 0:
            break

        price = float(df["Close"].iloc[i])
        ts    = df.index[i]
        sig   = int(signals.iloc[i])

        # ── Check open position ──
        if position is not None:
            hit_tp = hit_sl = False

            if position == "long":
                hit_tp = price >= tp_price
                hit_sl = price <= sl_price
            else:
                hit_tp = price <= tp_price
                hit_sl = price >= sl_price

            if hit_tp or hit_sl:
                pnl     = risk_usd * rr if hit_tp else -risk_usd
                capital = max(0.0, capital + pnl)

                trades.append({
                    "strategy":    strategy_name,
                    "entry_time":  entry_time,
                    "exit_time":   ts,
                    "direction":   position,
                    "entry_price": round(entry_price, 2),
                    "exit_price":  round(price, 2),
                    "sl_price":    round(sl_price, 2),
                    "tp_price":    round(tp_price, 2),
                    "result":      "TP" if hit_tp else "SL",
                    "pnl_usd":     round(pnl, 4),
                    "capital":     round(capital, 4),
                })
                position = None
                continue

        # ── Open new position ──
        if position is None and sig != 0 and capital > 0:
            risk_usd    = round(capital * RISK_PCT, 6)
            entry_price = price
            entry_time  = ts

            if sig == 1:
                position = "long"
                sl_price = entry_price * (1 - sl_pct)
                tp_price = entry_price * (1 + tp_pct)
            else:
                position = "short"
                sl_price = entry_price * (1 + sl_pct)
                tp_price = entry_price * (1 - tp_pct)

    return pd.DataFrame(trades)

# ──────────────────────────────────────────────
# SIGNAL FUNCTIONS
# ──────────────────────────────────────────────
def sig_ema_rsi(df):
    s = pd.Series(0, index=df.index)
    up   = (df["ema20"] > df["ema50"]) & (df["ema20"].shift(1) <= df["ema50"].shift(1)) & (df["rsi"] > 50)
    down = (df["ema20"] < df["ema50"]) & (df["ema20"].shift(1) >= df["ema50"].shift(1)) & (df["rsi"] < 50)
    s[up] = 1
    s[down] = -1
    return s

def sig_rsi_reversal(df):
    s = pd.Series(0, index=df.index)
    s[(df["rsi"] > 30) & (df["rsi"].shift(1) <= 30)] = 1
    s[(df["rsi"] < 70) & (df["rsi"].shift(1) >= 70)] = -1
    return s

def sig_bb_breakout(df):
    s = pd.Series(0, index=df.index)
    s[(df["Close"] > df["bb_upper"]) & (df["Close"].shift(1) <= df["bb_upper"].shift(1))] = 1
    s[(df["Close"] < df["bb_lower"]) & (df["Close"].shift(1) >= df["bb_lower"].shift(1))] = -1
    return s

def sig_macd_ema(df):
    s = pd.Series(0, index=df.index)
    up   = (df["macd"] > df["macd_sig"]) & (df["macd"].shift(1) <= df["macd_sig"].shift(1)) & (df["Close"] > df["ema200"])
    down = (df["macd"] < df["macd_sig"]) & (df["macd"].shift(1) >= df["macd_sig"].shift(1)) & (df["Close"] < df["ema200"])
    s[up] = 1
    s[down] = -1
    return s

# ──────────────────────────────────────────────
# RUN ALL 4 STRATEGIES
# ──────────────────────────────────────────────
strategies = [
    ("1. EMA Cross + RSI",  sig_ema_rsi,      0.005),
    ("2. RSI Reversal",     sig_rsi_reversal, 0.005),
    ("3. BB Breakout",      sig_bb_breakout,  0.005),
    ("4. MACD + EMA200",    sig_macd_ema,     0.005),
]

all_trades = []
summary    = []

for name, sig_fn, sl in strategies:
    trades = backtest(df, sig_fn, name, sl_pct=sl)
    all_trades.append(trades)

    if len(trades) == 0:
        summary.append({
            "Strategy":      name,
            "Trades":        0,
            "Wins":          0,
            "Losses":        0,
            "Win Rate":      "0.0%",
            "Total PnL ($)": 0.0,
            "Final Cap ($)": STARTING_CAPITAL,
            "Return %":      "0.0%",
            "Max Drawdown":  "0.0%",
        })
        continue

    wins      = (trades["result"] == "TP").sum()
    losses    = (trades["result"] == "SL").sum()
    total_pnl = trades["pnl_usd"].sum()
    final_cap = trades["capital"].iloc[-1]
    win_rate  = wins / len(trades) * 100
    ret_pct   = (final_cap - STARTING_CAPITAL) / STARTING_CAPITAL * 100

    cap_series  = pd.concat([pd.Series([STARTING_CAPITAL]), trades["capital"].reset_index(drop=True)])
    rolling_max = cap_series.cummax()
    drawdown    = (cap_series - rolling_max) / rolling_max * 100
    max_dd      = drawdown.min()

    summary.append({
        "Strategy":      name,
        "Trades":        len(trades),
        "Wins":          int(wins),
        "Losses":        int(losses),
        "Win Rate":      f"{win_rate:.1f}%",
        "Total PnL ($)": round(total_pnl, 2),
        "Final Cap ($)": round(final_cap, 2),
        "Return %":      f"{ret_pct:.1f}%",
        "Max Drawdown":  f"{max_dd:.1f}%",
    })

# ──────────────────────────────────────────────
# PRINT RESULTS
# ──────────────────────────────────────────────
print("=" * 80)
print("           BTC/USD 1H BACKTEST — LAST 2 YEARS")
print(f"           Capital: ${STARTING_CAPITAL}  |  Leverage: 1:{LEVERAGE}  |  Risk/Trade: {RISK_PCT*100}%")
print(f"           SL: {SL_PCT*100}% move  |  TP: {TP_MULT}:1 RR  |  Max loss/trade: ${STARTING_CAPITAL*RISK_PCT}")
print("=" * 80)

summary_df = pd.DataFrame(summary)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)
print(summary_df.to_string(index=False))
print("=" * 80)

# Per-strategy last 10 trades
for i, (name, _, _) in enumerate(strategies):
    trades = all_trades[i]
    if len(trades) == 0:
        print(f"\n{name}: No trades.")
        continue
    print(f"\n{'─'*80}")
    print(f"  {name} — Last 10 Trades")
    print(f"{'─'*80}")
    cols = ["entry_time","direction","entry_price","exit_price","result","pnl_usd","capital"]
    print(trades[cols].tail(10).to_string(index=False))

# Best strategy
print("\n" + "=" * 80)
best_idx = summary_df["Total PnL ($)"].idxmax()
best     = summary_df.iloc[best_idx]
print(f"  🏆  BEST STRATEGY : {best['Strategy']}")
print(f"      PnL           : ${best['Total PnL ($)']}")
print(f"      Return        : {best['Return %']}")
print(f"      Win Rate      : {best['Win Rate']}")
print(f"      Max Drawdown  : {best['Max Drawdown']}")
print("=" * 80)

# Save CSV
valid = [t for t in all_trades if len(t) > 0]
if valid:
    all_df = pd.concat(valid, ignore_index=True)
    all_df.to_csv("btc_backtest_trades.csv", index=False)
    print(f"\n📄 Full trade log → btc_backtest_trades.csv")
print("✅ Done!\n")