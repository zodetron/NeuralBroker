## 1hr with 7% SL backtest
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Download 1 year of BTC-USD data
data = yf.download("BTC-USD", period="1y", interval="1d")

# Calculate EMAs manually
data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()

# Generate buy/sell signals
data['Signal'] = 0
data['Signal'][12:] = (data['EMA12'][12:] > data['EMA20'][12:]).astype(int)
data['Position'] = data['Signal'].diff()

# Plot results
plt.figure(figsize=(14,7))
plt.plot(data['Close'], label='BTC Price', alpha=0.5)
plt.plot(data['EMA12'], label='EMA 12', alpha=0.9)
plt.plot(data['EMA20'], label='EMA 20', alpha=0.9)

# Mark buy and sell signals
plt.plot(data[data['Position'] == 1].index,
         data['EMA12'][data['Position'] == 1],
         '^', markersize=12, color='g', label='Buy Signal')

plt.plot(data[data['Position'] == -1].index,
         data['EMA12'][data['Position'] == -1],
         'v', markersize=12, color='r', label='Sell Signal')

plt.title('BTC EMA Crossover (12 & 20) - 1 Year')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# Print last few rows to check signals
print(data.tail())
