import pandas as pd
import numpy as np
from datetime import timedelta, time
import matplotlib.pyplot as plt

# ================= CONFIG =================
CSV_FILE = "btcusd.csv"
LOT_SIZE = 0.05

TP1_R = 1.0
TP2_R = 2.0
PARTIAL_SIZE = 0.33

ATR_SL_MULT = 1.5

MAX_DAILY_LOSS_R = 1.0
MAX_TRADES_PER_DAY = 2

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

# ================= VOLATILITY FILTER =================
df["atr_median"] = df["atr"].rolling(50).median()
df["vol_ok"] = df["atr"] > df["atr_median"]

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
    df["in_ny"] & df["vol_ok"] &
    df["bull_bias"] & df["liq_long"] &
    df["fvg_long"] & df["bull_mss"] & df["discount"]
)

df["short_signal"] = (
    df["in_ny"] & df["vol_ok"] &
    df["bear_bias"] & df["liq_short"] &
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

for timestamp, row in df.iterrows():

    day = timestamp.date()

    if day != current_day:
        current_day = day
        daily_pnl = 0.0
        daily_trades = 0

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
                sl = entry

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
equity = np.cumsum(trades)

wins = trades[trades > 0]
losses = trades[trades < 0]

peak = np.maximum.accumulate(equity)
drawdown = equity - peak

print("\n===== FINAL SYSTEM RESULTS =====")
print("Total Trades:", len(trades))
print("Win Rate:", round(len(wins) / len(trades) * 100, 2), "%")
print("Net PnL:", round(trades.sum(), 2))
print("Max Drawdown:", round(drawdown.min(), 2))
print("Return / Max DD:", round(equity[-1] / abs(drawdown.min()), 2))

# ================= MONTE CARLO =================
NUM_SIMULATIONS = 1000
mc_pnl = []
mc_dd = []

for _ in range(NUM_SIMULATIONS):
    shuffled = np.random.permutation(trades)
    eq = np.cumsum(shuffled)
    pk = np.maximum.accumulate(eq)
    dd = eq - pk
    mc_pnl.append(eq[-1])
    mc_dd.append(dd.min())

mc_pnl = np.array(mc_pnl)
mc_dd = np.array(mc_dd)

print("\n===== MONTE CARLO RESULTS =====")
print("Final PnL 5% / 50% / 95%:",
      round(np.percentile(mc_pnl, 5), 2),
      round(np.percentile(mc_pnl, 50), 2),
      round(np.percentile(mc_pnl, 95), 2))

print("Max DD 5% / 50% / 95%:",
      round(np.percentile(mc_dd, 5), 2),
      round(np.percentile(mc_dd, 50), 2),
      round(np.percentile(mc_dd, 95), 2))

# ================= PLOTS =================
plt.figure(figsize=(12, 5))
plt.plot(equity)
plt.title("Equity Curve")
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 5))
plt.hist(mc_pnl, bins=40)
plt.title("Monte Carlo Final PnL Distribution")
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 5))
plt.hist(mc_dd, bins=40)
plt.title("Monte Carlo Max Drawdown Distribution")
plt.grid(True)
plt.show()
