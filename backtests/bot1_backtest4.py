import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# ---- CONFIG ----
symbol = "BTC-USD"
start_date = "2024-01-01"
ema_fast_len = 8
ema_slow_len = 21
ema_trend_len = 200
stop_loss_pct = 0.02  # 2% stop-loss
initial_balance = 10000

# ---- FETCH DATA ----
df = yf.download(symbol, start=start_date, interval="1h")

# ---- CALCULATE EMAs ----
df[f"EMA{ema_fast_len}"] = df["Close"].ewm(span=ema_fast_len, adjust=False).mean()
df[f"EMA{ema_slow_len}"] = df["Close"].ewm(span=ema_slow_len, adjust=False).mean()
df[f"EMA{ema_trend_len}"] = df["Close"].ewm(span=ema_trend_len, adjust=False).mean()

# ---- DROP ROWS with NaNs ONLY in EMA columns to keep index aligned ----
df.dropna(subset=[f"EMA{ema_fast_len}", f"EMA{ema_slow_len}", f"EMA{ema_trend_len}"], inplace=True)

# ---- SIGNALS with Trend Filter ----
df["Signal"] = 0
df.loc[
    (df[f"EMA{ema_fast_len}"] > df[f"EMA{ema_slow_len}"]) &
    (df["Close"] > df[f"EMA{ema_trend_len}"]),
    "Signal"
] = 1
df.loc[
    (df[f"EMA{ema_fast_len}"] < df[f"EMA{ema_slow_len}"]) &
    (df["Close"] < df[f"EMA{ema_trend_len}"]),
    "Signal"
] = -1

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
plt.title(f"{symbol} EMA {ema_fast_len}/{ema_slow_len} with EMA{ema_trend_len} Trend Filter & {stop_loss_pct*100:.1f}% SL")
plt.xlabel("Date")
plt.ylabel("Balance ($)")
plt.grid(True)
plt.legend()
plt.show()
