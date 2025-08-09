import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ---- CONFIG ----
symbol = "BTC-USD"
end_date = datetime.now()
start_date = end_date - timedelta(days=60)  # last 60 days only

ema_fast_len = 8
ema_slow_len = 21
stop_loss_pct = 0.8
initial_balance = 10000

print(f"Downloading data for {symbol} from {start_date.date()} to {end_date.date()} ...")
df = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), interval="15m")

if df.empty:
    raise ValueError("No data downloaded. Check symbol, start date, and interval.")


# Continue as before...


# Flatten MultiIndex columns to single-level (e.g. ('Close', 'BTC-USD') -> 'Close')
df.columns = df.columns.get_level_values(0)

print(f"Columns after flattening: {df.columns.tolist()}")
print(f"Data sample:\n{df.head()}")

# ---- CALCULATE EMAs ----
df[f"EMA{ema_fast_len}"] = df["Close"].ewm(span=ema_fast_len, adjust=False).mean()
df[f"EMA{ema_slow_len}"] = df["Close"].ewm(span=ema_slow_len, adjust=False).mean()

# ---- DROP ROWS with NaNs in EMA columns ONLY ----
df.dropna(subset=[f"EMA{ema_fast_len}", f"EMA{ema_slow_len}"], inplace=True)

print(f"Data shape after dropping NaNs in EMA columns: {df.shape}")

# ---- SIGNALS without EMA 200 trend filter ----
df["Signal"] = 0
df.loc[
    (df[f"EMA{ema_fast_len}"] > df[f"EMA{ema_slow_len}"]),
    "Signal"
] = 1
df.loc[
    (df[f"EMA{ema_fast_len}"] < df[f"EMA{ema_slow_len}"]),
    "Signal"
] = -1

print(f"Signal value counts:\n{df['Signal'].value_counts()}")

# ---- BACKTEST ----
position = 0
balance = initial_balance
equity_curve = []
entry_price = 0

for i in range(1, len(df)):
    price = df["Close"].iloc[i]
    signal = df["Signal"].iloc[i]

    if position == 0:
        if signal == 1:
            position = 1
            entry_price = price
        elif signal == -1:
            position = -1
            entry_price = price

    elif position == 1:
        if price <= entry_price * (1 - stop_loss_pct):
            balance *= price / entry_price
            position = 0
        elif signal == -1:
            balance *= price / entry_price
            position = -1
            entry_price = price

    elif position == -1:
        if price >= entry_price * (1 + stop_loss_pct):
            balance *= (2 - price / entry_price)
            position = 0
        elif signal == 1:
            balance *= (2 - price / entry_price)
            position = 1
            entry_price = price

    equity_curve.append(balance)

# ---- PLOT ----
plt.figure(figsize=(12, 6))
plt.plot(df.index[1:], equity_curve, label="Equity Curve", color="blue")
plt.title(f"{symbol} EMA {ema_fast_len}/{ema_slow_len} with {stop_loss_pct*100:.1f}% SL (15-min timeframe)")
plt.xlabel("Date")
plt.ylabel("Balance ($)")
plt.grid(True)
plt.legend()
plt.show()
