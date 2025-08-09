##15 min timeframe
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from binance.client import Client
from datetime import datetime, timedelta

# ==== Binance API Keys (leave blank if just backtesting with public data) ====
API_KEY = ''
API_SECRET = ''
client = Client(API_KEY, API_SECRET)

# ==== Parameters ====
symbol = "BTCUSDT"
interval = Client.KLINE_INTERVAL_15MINUTE
lookback_months = 12
ema_fast = 12
ema_slow = 20
lot_size = 0.1  # Fixed BTC size

# ==== Fetch historical data ====
end_time = datetime.now()
start_time = end_time - timedelta(days=lookback_months * 30)
klines = client.get_historical_klines(symbol, interval, start_time.strftime("%d %b %Y %H:%M:%S"), end_time.strftime("%d %b %Y %H:%M:%S"))

# Convert to DataFrame
df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                   'quote_asset_volume', 'trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df['close'] = df['close'].astype(float)

# ==== Calculate EMAs ====
df['EMA_Fast'] = df['close'].ewm(span=ema_fast, adjust=False).mean()
df['EMA_Slow'] = df['close'].ewm(span=ema_slow, adjust=False).mean()

# ==== Backtest Logic ====
position = 0  # 0 = no position, 1 = long
entry_price = 0
balance = 0  # PnL in USDT
trade_log = []

for i in range(1, len(df)):
    # Buy signal
    if df['EMA_Fast'][i] > df['EMA_Slow'][i] and df['EMA_Fast'][i-1] <= df['EMA_Slow'][i-1]:
        if position == 0:
            position = 1
            entry_price = df['close'][i]
            trade_log.append((df['timestamp'][i], "BUY", entry_price))
    
    # Sell signal
    elif df['EMA_Fast'][i] < df['EMA_Slow'][i] and df['EMA_Fast'][i-1] >= df['EMA_Slow'][i-1]:
        if position == 1:
            position = 0
            exit_price = df['close'][i]
            profit = (exit_price - entry_price) * lot_size
            balance += profit
            trade_log.append((df['timestamp'][i], "SELL", exit_price, profit))

# If still in position at the end, close it
if position == 1:
    exit_price = df['close'].iloc[-1]
    profit = (exit_price - entry_price) * lot_size
    balance += profit
    trade_log.append((df['timestamp'].iloc[-1], "SELL_END", exit_price, profit))

# ==== Results ====
print(f"Final PnL: {balance:.2f} USDT")
print("\nTrade Log:")
for trade in trade_log:
    print(trade)

# ==== Plot ====
plt.figure(figsize=(14,7))
plt.plot(df['timestamp'], df['close'], label='Close Price', alpha=0.6)
plt.plot(df['timestamp'], df['EMA_Fast'], label=f'EMA {ema_fast}')
plt.plot(df['timestamp'], df['EMA_Slow'], label=f'EMA {ema_slow}')
plt.title(f"{symbol} EMA {ema_fast}/{ema_slow} Backtest - {lookback_months} Months - Lot Size: {lot_size} BTC")
plt.legend()
plt.show()
