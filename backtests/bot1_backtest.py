import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
# on 5 min time frame full blown
# --------------------------
# PARAMETERS
# --------------------------
symbol = "BTC/USDT"
timeframe = "5m"
since = ccxt.binance().parse8601("2018-01-01T00:00:00Z")
ema_fast = 12
ema_slow = 20
initial_capital = 1000  # USD
fee_rate = 0.00075  # 0.075% per side
slippage_rate = 0.0002  # 0.02% per trade

# --------------------------
# FETCH HISTORICAL DATA
# --------------------------
exchange = ccxt.binance()
ohlcv = []
limit = 1000
now = exchange.milliseconds()

print("Downloading data from Binance...")
while since < now:
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
    if not data:
        break
    ohlcv += data
    since = data[-1][0] + (5 * 60 * 1000)
    print(f"Fetched {len(ohlcv)} candles...", end="\r")

df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
df.set_index("timestamp", inplace=True)

# --------------------------
# CALCULATE EMAs & SIGNALS
# --------------------------
df["EMA12"] = df["close"].ewm(span=ema_fast, adjust=False).mean()
df["EMA20"] = df["close"].ewm(span=ema_slow, adjust=False).mean()
df["Signal"] = np.where(df["EMA12"] > df["EMA20"], 1, -1)

# Position = previous signal (execute next candle)
df["Position"] = df["Signal"].shift(1)

# --------------------------
# STRATEGY RETURNS
# --------------------------
df["Return"] = df["close"].pct_change()
df["Strategy_Return"] = df["Position"] * df["Return"]

# Apply cost when position changes
df["Trade_Change"] = df["Position"].diff().fillna(0) != 0
cost_per_trade = (fee_rate + slippage_rate) * 2  # round trip cost
df.loc[df["Trade_Change"], "Strategy_Return"] -= cost_per_trade

# --------------------------
# EQUITY CURVE
# --------------------------
df["Equity"] = (1 + df["Strategy_Return"]).cumprod()

# --------------------------
# METRICS
# --------------------------
total_return = df["Equity"].iloc[-1] - 1
cagr = (df["Equity"].iloc[-1]) ** (252*24*12 / len(df)) - 1
max_dd = (df["Equity"] / df["Equity"].cummax() - 1).min()
sharpe = (df["Strategy_Return"].mean() / df["Strategy_Return"].std()) * np.sqrt(252*24*12)

print("\n--- PERFORMANCE METRICS ---")
print(f"Total Return: {total_return*100:.2f}%")
print(f"CAGR: {cagr*100:.2f}%")
print(f"Max Drawdown: {max_dd*100:.2f}%")
print(f"Sharpe Ratio: {sharpe:.2f}")

# --------------------------
# PLOT EQUITY CURVE
# --------------------------
plt.figure(figsize=(12,6))
plt.plot(df.index, df["Equity"], label="Equity Curve")
plt.title(f"BTC/USDT EMA({ema_fast},{ema_slow}) Reversal Strategy - 5m")
plt.xlabel("Date")
plt.ylabel("Equity (relative)")
plt.legend()
plt.grid(True)
plt.show()

# --------------------------
# SAVE TRADES
# --------------------------
df.to_csv("btc_ema_strategy_results.csv")
print("Results saved to btc_ema_strategy_results.csv")
