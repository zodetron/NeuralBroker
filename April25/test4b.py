"""
BTC/USD 1H — BB Breakout v1 vs v2 Head-to-Head
Capital: $100 | Leverage: 1:1000 | Last ~2 years

v1 (original): Raw BB breakout, 0.5% SL, 2:1 RR
v2 (enhanced): BB + EMA200 trend filter + RSI confirm + 0.8% SL + 2:1 RR

Goal: Fewer but higher-quality trades → better win rate → higher PnL
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
STARTING_CAPITAL = 100.0
LEVERAGE         = 1000
RISK_PCT         = 0.01      # 1% risk per trade
TICKER           = "BTC-USD"
INTERVAL         = "1h"

# ──────────────────────────────────────────────
# FETCH DATA
# ──────────────────────────────────────────────
print("📥 Fetching BTC/USD 1H data (chunked)...")
end_date   = datetime.today()
start_date = end_date - timedelta(days=720)

chunks, chunk_start = [], start_date
while chunk_start < end_date:
    chunk_end = min(chunk_start + timedelta(days=59), end_date)
    chunk = yf.download(TICKER,
                        start=chunk_start.strftime("%Y-%m-%d"),
                        end=chunk_end.strftime("%Y-%m-%d"),
                        interval=INTERVAL, auto_adjust=True, progress=False)
    if len(chunk) > 0:
        chunks.append(chunk)
    chunk_start = chunk_end

df = pd.concat(chunks)
df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
df = df[~df.index.duplicated(keep="first")]
df.sort_index(inplace=True)
df.dropna(inplace=True)
print(f"✅ {len(df)} candles | {df.index[0].date()} → {df.index[-1].date()}\n")

# ──────────────────────────────────────────────
# INDICATORS
# ──────────────────────────────────────────────
def add_indicators(df):
    d = df.copy()

    # Bollinger Bands
    d["bb_mid"]   = d["Close"].rolling(20).mean()
    bb_std        = d["Close"].rolling(20).std()
    d["bb_upper"] = d["bb_mid"] + 2 * bb_std
    d["bb_lower"] = d["bb_mid"] - 2 * bb_std
    d["bb_width"] = d["bb_upper"] - d["bb_lower"]  # band width for squeeze filter

    # EMAs
    d["ema50"]  = d["Close"].ewm(span=50,  adjust=False).mean()
    d["ema200"] = d["Close"].ewm(span=200, adjust=False).mean()

    # RSI
    delta      = d["Close"].diff()
    gain       = delta.clip(lower=0).rolling(14).mean()
    loss       = (-delta.clip(upper=0)).rolling(14).mean()
    d["rsi"]   = 100 - (100 / (1 + gain / loss))

    # ATR (for smarter SL in v2)
    high_low   = d["High"] - d["Low"]
    high_close = (d["High"] - d["Close"].shift()).abs()
    low_close  = (d["Low"]  - d["Close"].shift()).abs()
    tr         = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    d["atr14"] = tr.rolling(14).mean()

    return d

df = add_indicators(df)
df.dropna(inplace=True)

# ──────────────────────────────────────────────
# BACKTEST ENGINE
# ──────────────────────────────────────────────
def backtest(df, signal_func, strategy_name, sl_pct, rr=2.0):
    tp_pct   = sl_pct * rr
    capital  = STARTING_CAPITAL
    position = None
    trades   = []
    entry_price = sl_price = tp_price = risk_usd = 0.0
    entry_time = direction = None

    signals = signal_func(df)

    for i in range(1, len(df)):
        if capital <= 0:
            break

        price = float(df["Close"].iloc[i])
        ts    = df.index[i]
        sig   = int(signals.iloc[i])

        # Manage open position
        if position is not None:
            hit_tp = (position == "long"  and price >= tp_price) or \
                     (position == "short" and price <= tp_price)
            hit_sl = (position == "long"  and price <= sl_price) or \
                     (position == "short" and price >= sl_price)

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
                    "result":      "TP" if hit_tp else "SL",
                    "pnl_usd":     round(pnl, 4),
                    "capital":     round(capital, 4),
                })
                position = None
                continue

        # Open new position
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
# STRATEGY v1 — Original BB Breakout
# ──────────────────────────────────────────────
def sig_bb_v1(df):
    s = pd.Series(0, index=df.index)
    break_up   = (df["Close"] > df["bb_upper"]) & (df["Close"].shift(1) <= df["bb_upper"].shift(1))
    break_down = (df["Close"] < df["bb_lower"]) & (df["Close"].shift(1) >= df["bb_lower"].shift(1))
    s[break_up]   = 1
    s[break_down] = -1
    return s

# ──────────────────────────────────────────────
# STRATEGY v2 — Enhanced BB Breakout
# Improvements:
#   1. EMA200 trend filter  → only long above EMA200, only short below EMA200
#   2. RSI confirmation     → long only if RSI > 50, short only if RSI < 50
#   3. No trades in squeeze → only trade when BB is wide (volatile market)
#   4. Wider SL (0.8%)      → fewer false stops
# ──────────────────────────────────────────────
def sig_bb_v2(df):
    s = pd.Series(0, index=df.index)

    # Band squeeze filter: only trade when BB width > 20-period median width
    bb_wide = df["bb_width"] > df["bb_width"].rolling(20).median()

    # Trend filter
    above_ema200 = df["Close"] > df["ema200"]
    below_ema200 = df["Close"] < df["ema200"]

    # RSI filter
    rsi_bull = df["rsi"] > 50
    rsi_bear = df["rsi"] < 50

    # Breakout signals
    break_up   = (df["Close"] > df["bb_upper"]) & (df["Close"].shift(1) <= df["bb_upper"].shift(1))
    break_down = (df["Close"] < df["bb_lower"]) & (df["Close"].shift(1) >= df["bb_lower"].shift(1))

    # Combined: breakout + trend + RSI + no squeeze
    long_sig  = break_up   & above_ema200 & rsi_bull & bb_wide
    short_sig = break_down & below_ema200 & rsi_bear & bb_wide

    s[long_sig]  = 1
    s[short_sig] = -1
    return s

# ──────────────────────────────────────────────
# STRATEGY v3 — BB Breakout + EMA50 Pullback
# Extra filter: only enter on slight pullback after breakout
# Entry: price breaks band, then pulls back to EMA50 direction
# ──────────────────────────────────────────────
def sig_bb_v3(df):
    s = pd.Series(0, index=df.index)

    above_ema200 = df["Close"] > df["ema200"]
    below_ema200 = df["Close"] < df["ema200"]
    rsi_bull = df["rsi"] > 45
    rsi_bear = df["rsi"] < 55
    bb_wide  = df["bb_width"] > df["bb_width"].rolling(20).median()

    # Price near EMA50 (within 0.3%) after a breakout candle
    near_ema50 = (df["Close"] - df["ema50"]).abs() / df["ema50"] < 0.003

    # Previous candle broke the band
    prev_broke_up   = df["Close"].shift(1) > df["bb_upper"].shift(1)
    prev_broke_down = df["Close"].shift(1) < df["bb_lower"].shift(1)

    long_sig  = prev_broke_up   & near_ema50 & above_ema200 & rsi_bull & bb_wide
    short_sig = prev_broke_down & near_ema50 & below_ema200 & rsi_bear & bb_wide

    s[long_sig]  = 1
    s[short_sig] = -1
    return s

# ──────────────────────────────────────────────
# RUN ALL VERSIONS
# ──────────────────────────────────────────────
versions = [
    ("BB v1 (Original)",         sig_bb_v1, 0.005),
    ("BB v2 (Trend+RSI+Wide)",   sig_bb_v2, 0.008),
    ("BB v3 (Pullback Entry)",   sig_bb_v3, 0.008),
]

all_trades = []
summary    = []

for name, sig_fn, sl in versions:
    trades = backtest(df, sig_fn, name, sl_pct=sl)
    all_trades.append(trades)

    if len(trades) == 0:
        summary.append({
            "Version":       name,
            "Trades":        0,
            "Wins":          0,
            "Losses":        0,
            "Win Rate":      "0.0%",
            "Total PnL ($)": 0.0,
            "Final Cap ($)": STARTING_CAPITAL,
            "Return %":      "0.0%",
            "Max Drawdown":  "0.0%",
            "Avg Win ($)":   0.0,
            "Avg Loss ($)":  0.0,
        })
        continue

    wins       = (trades["result"] == "TP").sum()
    losses     = (trades["result"] == "SL").sum()
    total_pnl  = trades["pnl_usd"].sum()
    final_cap  = trades["capital"].iloc[-1]
    win_rate   = wins / len(trades) * 100
    ret_pct    = (final_cap - STARTING_CAPITAL) / STARTING_CAPITAL * 100
    avg_win    = trades.loc[trades["result"] == "TP", "pnl_usd"].mean() if wins > 0 else 0
    avg_loss   = trades.loc[trades["result"] == "SL", "pnl_usd"].mean() if losses > 0 else 0

    cap_series  = pd.concat([pd.Series([STARTING_CAPITAL]), trades["capital"].reset_index(drop=True)])
    rolling_max = cap_series.cummax()
    drawdown    = (cap_series - rolling_max) / rolling_max * 100
    max_dd      = drawdown.min()

    summary.append({
        "Version":       name,
        "Trades":        len(trades),
        "Wins":          int(wins),
        "Losses":        int(losses),
        "Win Rate":      f"{win_rate:.1f}%",
        "Total PnL ($)": round(total_pnl, 2),
        "Final Cap ($)": round(final_cap, 2),
        "Return %":      f"{ret_pct:.1f}%",
        "Max Drawdown":  f"{max_dd:.1f}%",
        "Avg Win ($)":   round(avg_win, 3),
        "Avg Loss ($)":  round(avg_loss, 3),
    })

# ──────────────────────────────────────────────
# PRINT RESULTS
# ──────────────────────────────────────────────
print("=" * 90)
print("     BTC/USD 1H — BB BREAKOUT: v1 vs v2 vs v3")
print(f"     Capital: ${STARTING_CAPITAL}  |  Leverage: 1:{LEVERAGE}  |  Risk/Trade: {RISK_PCT*100}%")
print("=" * 90)

summary_df = pd.DataFrame(summary)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)
print(summary_df.to_string(index=False))
print("=" * 90)

# Improvement callout
v1_pnl = summary_df.iloc[0]["Total PnL ($)"]
v2_pnl = summary_df.iloc[1]["Total PnL ($)"]
v1_wr  = summary_df.iloc[0]["Win Rate"]
v2_wr  = summary_df.iloc[1]["Win Rate"]
print(f"\n  📊 v1 → v2 improvement:")
print(f"     Win Rate  : {v1_wr} → {v2_wr}")
print(f"     PnL       : ${v1_pnl} → ${v2_pnl}")
diff = round(v2_pnl - v1_pnl, 2)
print(f"     Delta     : {'+'if diff>=0 else ''}{diff}")

# Per-version last 10 trades
for i, (name, _, _) in enumerate(versions):
    trades = all_trades[i]
    if len(trades) == 0:
        print(f"\n{name}: No trades.")
        continue
    print(f"\n{'─'*90}")
    print(f"  {name} — Last 10 Trades")
    print(f"{'─'*90}")
    cols = ["entry_time","direction","entry_price","exit_price","result","pnl_usd","capital"]
    print(trades[cols].tail(10).to_string(index=False))

# Winner
print("\n" + "=" * 90)
best_idx = summary_df["Total PnL ($)"].idxmax()
best     = summary_df.iloc[best_idx]
print(f"  🏆  WINNER        : {best['Version']}")
print(f"      PnL           : ${best['Total PnL ($)']}")
print(f"      Return        : {best['Return %']}")
print(f"      Win Rate      : {best['Win Rate']}")
print(f"      Max Drawdown  : {best['Max Drawdown']}")
print(f"      Avg Win       : ${best['Avg Win ($)']}")
print(f"      Avg Loss      : ${best['Avg Loss ($)']}")
print("=" * 90)

# Save
valid = [t for t in all_trades if len(t) > 0]
if valid:
    pd.concat(valid, ignore_index=True).to_csv("bb_comparison_trades.csv", index=False)
    print(f"\n📄 Full trade log → bb_comparison_trades.csv")
print("✅ Done!\n")