# Add daily max loss / trade limit

# ICT + NY + ATR SL + PARTIAL TP + BE

import pandas as pd
import numpy as np
from datetime import timedelta, time

# ================= CONFIG =================
CSV_FILE = "btcusd.csv"
LOT_SIZE = 0.05

TP1_R = 1.0
TP2_R = 2.0
PARTIAL_SIZE = 0.5

ATR_SL_MULT = 1.5

HTF_LOOKBACK = 48
LIQ_LOOKBACK = 20
ATR_PERIOD = 14
SWING_LOOKBACK = 5

BIAS_STATE_BARS = 20
EVENT_STATE_BARS = 5

NY_START = time(13, 0)
NY_END   = time(17, 0)

# ================= LOAD DATA =================
df = pd.read_csv(CSV_FILE)
df["open_time"] = pd.to_datetime(df["open_time"])
df.set_index("open_time", inplace=True)
df.sort_index(inplace=True)

# ================= LAST 6 MONTHS =================
end_date = df.index.max()
start_date = end_date - timedelta(days=182)
df = df.loc[start_date:end_date]

print(f"Backtest period: {df.index.min()} → {df.index.max()}")
print("Candles:", len(df))

# ================= NY KILL ZONE =================
df["in_ny"] = (df.index.time >= NY_START) & (df.index.time <= NY_END)

# ================= HTF BIAS =================
df["range_high"] = df["high"].rolling(HTF_LOOKBACK).max()
df["range_low"] = df["low"].rolling(HTF_LOOKBACK).min()
df["equilibrium"] = (df["range_high"] + df["range_low"]) / 2

df["bull_bias"] = (df["close"] > df["equilibrium"]).rolling(BIAS_STATE_BARS).max().astype(bool)
df["bear_bias"] = (df["close"] < df["equilibrium"]).rolling(BIAS_STATE_BARS).max().astype(bool)

# ================= LIQUIDITY =================
df["liq_low"] = df["low"] <= df["low"].rolling(LIQ_LOOKBACK).min() * 1.0002
df["liq_high"] = df["high"] >= df["high"].rolling(LIQ_LOOKBACK).max() * 0.9998

df["liq_long"] = df["liq_low"].rolling(EVENT_STATE_BARS).max().astype(bool)
df["liq_short"] = df["liq_high"].rolling(EVENT_STATE_BARS).max().astype(bool)

# ================= ATR =================
df["atr"] = (df["high"] - df["low"]).rolling(ATR_PERIOD).mean()

# ================= DISPLACEMENT + FVG =================
df["bull_disp"] = (df["close"] > df["open"]) & ((df["close"] - df["open"]) > df["atr"])
df["bear_disp"] = (df["open"] > df["close"]) & ((df["open"] - df["close"]) > df["atr"])

df["bull_fvg"] = df["low"] > df["high"].shift(2)
df["bear_fvg"] = df["high"] < df["low"].shift(2)

df["fvg_long"] = (df["bull_disp"] & df["bull_fvg"]).rolling(EVENT_STATE_BARS).max().astype(bool)
df["fvg_short"] = (df["bear_disp"] & df["bear_fvg"]).rolling(EVENT_STATE_BARS).max().astype(bool)

# ================= MSS =================
df["swing_high"] = df["high"].rolling(SWING_LOOKBACK).max()
df["swing_low"] = df["low"].rolling(SWING_LOOKBACK).min()

df["bull_mss"] = (df["close"] > df["swing_high"].shift(1)).rolling(EVENT_STATE_BARS).max().astype(bool)
df["bear_mss"] = (df["close"] < df["swing_low"].shift(1)).rolling(EVENT_STATE_BARS).max().astype(bool)

# ================= ENTRY ZONE =================
df["discount"] = df["close"] < df["equilibrium"]
df["premium"] = df["close"] > df["equilibrium"]

# ================= ENTRY SIGNALS =================
df["long_signal"] = (
    df["in_ny"] & df["bull_bias"] & df["liq_long"] &
    df["fvg_long"] & df["bull_mss"] & df["discount"]
)

df["short_signal"] = (
    df["in_ny"] & df["bear_bias"] & df["liq_short"] &
    df["fvg_short"] & df["bear_mss"] & df["premium"]
)

# ================= BACKTEST ENGINE =================
position = None
entry = sl = tp1 = tp2 = 0.0
size_remaining = 0.0
trades = []
current_day = None
daily_pnl = 0.0
daily_trades = 0

MAX_DAILY_LOSS_R = 1.0      # stop trading after -1R in a day
MAX_TRADES_PER_DAY = 2     # max 2 trades per day


for timestamp, row in df.iterrows():

    day = timestamp.date()

    # Reset counters at new day
    if day != current_day:
        current_day = day
        daily_pnl = 0.0
        daily_trades = 0

    # Block trading if daily limits hit
    if daily_pnl <= -MAX_DAILY_LOSS_R or daily_trades >= MAX_TRADES_PER_DAY:
        continue

    if position is None:
        if row["long_signal"]:
            position = "LONG"
            entry = row["close"]
            sl = entry - row["atr"] * ATR_SL_MULT
            tp1 = entry + (entry - sl) * TP1_R
            tp2 = entry + (entry - sl) * TP2_R
            size_remaining = LOT_SIZE
            daily_trades += 1

        elif row["short_signal"]:
            position = "SHORT"
            entry = row["close"]
            sl = entry + row["atr"] * ATR_SL_MULT
            tp1 = entry - (sl - entry) * TP1_R
            tp2 = entry - (sl - entry) * TP2_R
            size_remaining = LOT_SIZE
            daily_trades += 1

    else:
        if position == "LONG":
            if row["low"] <= sl:
                pnl = (sl - entry) * size_remaining
                trades.append(pnl)
                daily_pnl += pnl
                position = None

            elif row["high"] >= tp1 and size_remaining == LOT_SIZE:
                pnl = (tp1 - entry) * (LOT_SIZE * PARTIAL_SIZE)
                trades.append(pnl)
                daily_pnl += pnl
                size_remaining *= (1 - PARTIAL_SIZE)
                sl = entry  # BE

            elif row["high"] >= tp2:
                pnl = (tp2 - entry) * size_remaining
                trades.append(pnl)
                daily_pnl += pnl
                position = None

        elif position == "SHORT":
            if row["high"] >= sl:
                pnl = (entry - sl) * size_remaining
                trades.append(pnl)
                daily_pnl += pnl
                position = None

            elif row["low"] <= tp1 and size_remaining == LOT_SIZE:
                pnl = (entry - tp1) * (LOT_SIZE * PARTIAL_SIZE)
                trades.append(pnl)
                daily_pnl += pnl
                size_remaining *= (1 - PARTIAL_SIZE)
                sl = entry

            elif row["low"] <= tp2:
                pnl = (entry - tp2) * size_remaining
                trades.append(pnl)
                daily_pnl += pnl
                position = None


# ================= RESULTS =================
trades = np.array(trades)
wins = trades[trades > 0]
losses = trades[trades < 0]

print("\n===== ICT + NY + ATR + PARTIAL TP + BE =====")
print("Total Trades:", len(trades))
print("Win Rate:", round(len(wins) / len(trades) * 100, 2) if len(trades) else 0, "%")
print("Net PnL:", round(trades.sum(), 4))
print("Avg Win:", round(wins.mean(), 4) if len(wins) else 0)
print("Avg Loss:", round(losses.mean(), 4) if len(losses) else 0)
print("Worst Trade:", round(trades.min(), 4) if len(trades) else 0)


import matplotlib.pyplot as plt

# ================= EQUITY CURVE =================
equity = np.cumsum(trades)

# ================= DRAWDOWN =================
equity_peak = np.maximum.accumulate(equity)
drawdown = equity - equity_peak

# ================= METRICS =================
max_dd = drawdown.min()
final_pnl = equity[-1] if len(equity) else 0

print("\n===== EQUITY METRICS =====")
print("Final PnL:", round(final_pnl, 2))
print("Max Drawdown:", round(max_dd, 2))
print("Return / Max DD:", round(final_pnl / abs(max_dd), 2) if max_dd != 0 else 0)

# ================= PLOT =================
plt.figure(figsize=(12, 6))
plt.plot(equity, label="Equity Curve")
plt.fill_between(range(len(drawdown)), drawdown, color="red", alpha=0.3, label="Drawdown")
plt.axhline(0, color="black", linewidth=0.8)
plt.title("BTCUSD ICT Strategy – Equity Curve & Drawdown")
plt.xlabel("Trade Number")
plt.ylabel("PnL")
plt.legend()
plt.grid(True)
plt.show()
