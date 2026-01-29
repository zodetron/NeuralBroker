import pandas as pd
import numpy as np
from datetime import timedelta

# ================= CONFIG =================
CSV_FILE = "btcusd.csv"
LOT_SIZE = 0.05
RR = 2.0

HTF_LOOKBACK = 48
LIQ_LOOKBACK = 20
ATR_PERIOD = 14
SWING_LOOKBACK = 5

BIAS_STATE_BARS = 20
EVENT_STATE_BARS = 5

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

# ================= HTF RANGE & BIAS =================
df["range_high"] = df["high"].rolling(HTF_LOOKBACK).max()
df["range_low"] = df["low"].rolling(HTF_LOOKBACK).min()
df["equilibrium"] = (df["range_high"] + df["range_low"]) / 2

df["bull_bias_raw"] = df["close"] > df["equilibrium"]
df["bear_bias_raw"] = df["close"] < df["equilibrium"]

df["bull_bias"] = df["bull_bias_raw"].rolling(BIAS_STATE_BARS).max().astype(bool)
df["bear_bias"] = df["bear_bias_raw"].rolling(BIAS_STATE_BARS).max().astype(bool)

# ================= LIQUIDITY SWEEP =================
df["liq_low"] = (df["low"] <= df["low"].rolling(LIQ_LOOKBACK).min() * 1.0002)
df["liq_high"] = (df["high"] >= df["high"].rolling(LIQ_LOOKBACK).max() * 0.9998)

df["liq_swept_long"] = df["liq_low"].rolling(EVENT_STATE_BARS).max().astype(bool)
df["liq_swept_short"] = df["liq_high"].rolling(EVENT_STATE_BARS).max().astype(bool)

# ================= ATR & DISPLACEMENT =================
df["atr"] = (df["high"] - df["low"]).rolling(ATR_PERIOD).mean()

df["bull_displace"] = (
    (df["close"] > df["open"]) &
    ((df["close"] - df["open"]) > df["atr"])
)

df["bear_displace"] = (
    (df["open"] > df["close"]) &
    ((df["open"] - df["close"]) > df["atr"])
)

# ================= FVG =================
df["bull_fvg"] = df["low"] > df["high"].shift(2)
df["bear_fvg"] = df["high"] < df["low"].shift(2)

df["fvg_long"] = (df["bull_displace"] & df["bull_fvg"]).rolling(EVENT_STATE_BARS).max().astype(bool)
df["fvg_short"] = (df["bear_displace"] & df["bear_fvg"]).rolling(EVENT_STATE_BARS).max().astype(bool)

# ================= MARKET STRUCTURE SHIFT =================
df["swing_high"] = df["high"].rolling(SWING_LOOKBACK).max()
df["swing_low"] = df["low"].rolling(SWING_LOOKBACK).min()

df["bull_mss_raw"] = df["close"] > df["swing_high"].shift(1)
df["bear_mss_raw"] = df["close"] < df["swing_low"].shift(1)

df["bull_mss"] = df["bull_mss_raw"].rolling(EVENT_STATE_BARS).max().astype(bool)
df["bear_mss"] = df["bear_mss_raw"].rolling(EVENT_STATE_BARS).max().astype(bool)

# ================= ENTRY ZONE =================
df["discount"] = (df["close"] < df["equilibrium"])
df["premium"] = (df["close"] > df["equilibrium"])

# ================= FINAL ENTRY SIGNALS =================
df["long_signal"] = (
    df["bull_bias"] &
    df["liq_swept_long"] &
    df["fvg_long"] &
    df["bull_mss"] &
    df["discount"]
)

df["short_signal"] = (
    df["bear_bias"] &
    df["liq_swept_short"] &
    df["fvg_short"] &
    df["bear_mss"] &
    df["premium"]
)

# ================= BACKTEST ENGINE =================
position = None
entry = sl = tp = 0.0
trades = []

for _, row in df.iterrows():
    if position is None:
        if row["long_signal"]:
            position = "LONG"
            entry = row["close"]
            sl = row["range_low"]
            tp = entry + (entry - sl) * RR

        elif row["short_signal"]:
            position = "SHORT"
            entry = row["close"]
            sl = row["range_high"]
            tp = entry - (sl - entry) * RR

    else:
        if position == "LONG":
            if row["low"] <= sl:
                trades.append((sl - entry) * LOT_SIZE)
                position = None
            elif row["high"] >= tp:
                trades.append((tp - entry) * LOT_SIZE)
                position = None

        elif position == "SHORT":
            if row["high"] >= sl:
                trades.append((entry - sl) * LOT_SIZE)
                position = None
            elif row["low"] <= tp:
                trades.append((entry - tp) * LOT_SIZE)
                position = None

# ================= RESULTS =================
trades = np.array(trades)

wins = trades[trades > 0]
losses = trades[trades < 0]

print("\n===== ICT STATE-BASED RESULTS (FIXED) =====")
print("Total Trades:", len(trades))
print("Win Rate:", round(len(wins) / len(trades) * 100, 2) if len(trades) else 0, "%")
print("Net PnL (BTC):", round(trades.sum(), 4))
print("Avg Win:", round(wins.mean(), 4) if len(wins) else 0)
print("Avg Loss:", round(losses.mean(), 4) if len(losses) else 0)
print("Worst Trade:", round(trades.min(), 4) if len(trades) else 0)
