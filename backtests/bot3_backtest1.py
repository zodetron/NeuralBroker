# =============================
# EMA + Volume Spike Strategy Backtest
# =============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta  # pip install ta

# === CONFIG ===
EMA_SHORT = 12
EMA_LONG = 20
VOLUME_SPIKE_RATIO = 1.5
STOP_LOSS_PCT = 0.01  # 1%
TRADE_SIZE = 1  # assume 1 unit for simplicity

# === LOAD DATA ===
# CSV must have columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
# Example: BTC_USD_5m.csv
df = pd.read_csv("BTC_USD_5m.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# === INDICATORS ===
df["ema12"] = ta.trend.EMAIndicator(df["close"], window=EMA_SHORT).ema_indicator()
df["ema20"] = ta.trend.EMAIndicator(df["close"], window=EMA_LONG).ema_indicator()
df["avg_vol"] = df["volume"].rolling(window=10).mean()
df["vol_ratio"] = df["volume"] / df["avg_vol"]

# === SIGNALS ===
df["signal"] = 0
df.loc[(df["ema12"] > df["ema20"]) & (df["vol_ratio"] > VOLUME_SPIKE_RATIO), "signal"] = 1   # BUY
df.loc[(df["ema12"] < df["ema20"]) & (df["vol_ratio"] > VOLUME_SPIKE_RATIO), "signal"] = -1  # SELL

# === BACKTEST ===
positions = []
balance = 10000  # starting balance in USD
position = None  # {'side': 'buy'/'sell', 'entry': price}
equity_curve = []

for i in range(len(df)):
    price = df.loc[i, "close"]
    signal = df.loc[i, "signal"]

    # Stop-loss check
    if position:
        if position["side"] == "buy" and price <= position["entry"] * (1 - STOP_LOSS_PCT):
            pnl = (price - position["entry"]) * TRADE_SIZE
            balance += pnl
            positions.append({"exit_time": df.loc[i, "timestamp"], "exit_price": price, "pnl": pnl})
            position = None
        elif position["side"] == "sell" and price >= position["entry"] * (1 + STOP_LOSS_PCT):
            pnl = (position["entry"] - price) * TRADE_SIZE
            balance += pnl
            positions.append({"exit_time": df.loc[i, "timestamp"], "exit_price": price, "pnl": pnl})
            position = None

    # Entry signals
    if not position:
        if signal == 1:
            position = {"side": "buy", "entry": price, "entry_time": df.loc[i, "timestamp"]}
        elif signal == -1:
            position = {"side": "sell", "entry": price, "entry_time": df.loc[i, "timestamp"]}
    else:
        # Opposite crossover → close & reverse
        if (position["side"] == "buy" and signal == -1) or (position["side"] == "sell" and signal == 1):
            pnl = (price - position["entry"]) * TRADE_SIZE if position["side"] == "buy" else (position["entry"] - price) * TRADE_SIZE
            balance += pnl
            positions.append({"exit_time": df.loc[i, "timestamp"], "exit_price": price, "pnl": pnl})
            position = None
            # open new
            if signal == 1:
                position = {"side": "buy", "entry": price, "entry_time": df.loc[i, "timestamp"]}
            elif signal == -1:
                position = {"side": "sell", "entry": price, "entry_time": df.loc[i, "timestamp"]}

    equity_curve.append(balance)

# === RESULTS ===
trades_df = pd.DataFrame(positions)
total_pnl = trades_df["pnl"].sum() if not trades_df.empty else 0
win_rate = (trades_df["pnl"] > 0).mean() * 100 if not trades_df.empty else 0

print(f"\n💰 Final Balance: ${balance:.2f}")
print(f"📊 Total PnL: {total_pnl:.2f}")
print(f"✅ Win Rate: {win_rate:.2f}%")
print(f"📈 Total Trades: {len(trades_df)}")

# === PLOT ===
plt.figure(figsize=(12,6))
plt.plot(df["timestamp"], equity_curve, label="Equity Curve")
plt.title("EMA + Volume Spike Strategy Backtest")
plt.xlabel("Time")
plt.ylabel("Balance (USD)")
plt.legend()
plt.show()
