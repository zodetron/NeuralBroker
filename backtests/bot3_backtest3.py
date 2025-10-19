import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ==========================
# CONFIG
# ==========================
EMA_SHORT = 12
EMA_LONG = 20
STOP_LOSS_PCT = 0.07  # 7%
VOLUME_SPIKE_RATIO = 1.5
START_BALANCE = 10000

# ==========================
# DOWNLOAD HISTORICAL DATA
# ==========================
data = yf.download("BTC-USD", period="2y", interval="1h")

# Flatten MultiIndex columns (fix for yfinance)
data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
data = data.reset_index()  # make 'Datetime' a column

# ==========================
# CALCULATE INDICATORS
# ==========================
data['EMA12'] = data['Close'].ewm(span=EMA_SHORT, adjust=False).mean()
data['EMA20'] = data['Close'].ewm(span=EMA_LONG, adjust=False).mean()
data['AvgVol'] = data['Volume'].rolling(10).mean()
data['VolRatio'] = data['Volume'] / data['AvgVol']

# ==========================
# GENERATE SIGNALS
# ==========================
data['Signal'] = 0
# Buy signal: EMA12 > EMA20 + volume spike
data.loc[(data['EMA12'] > data['EMA20']) & (data['VolRatio'] > VOLUME_SPIKE_RATIO), 'Signal'] = 1
# Sell signal: EMA12 < EMA20 + volume spike
data.loc[(data['EMA12'] < data['EMA20']) & (data['VolRatio'] > VOLUME_SPIKE_RATIO), 'Signal'] = -1

data['Position'] = data['Signal'].diff()  # 1 = buy, -1 = sell crossover

# ==========================
# BACKTEST
# ==========================
balance = START_BALANCE
position = None
entry_price = 1
trade_log = []

for i, row in data.iterrows():
    price = row['Close']
    signal = row['Signal']
    
    # Stop-loss check
    if position:
        if position == 'buy' and price <= entry_price * (1.2 - STOP_LOSS_PCT):
            pnl = price - entry_price
            balance += pnl
            trade_log.append({'Time': row['Datetime'], 'Side': 'Buy SL', 'Entry': entry_price, 'Exit': price, 'PnL': pnl})
            position = None
        elif position == 'sell' and price >= entry_price * (1 + STOP_LOSS_PCT):
            pnl = entry_price - price
            balance += pnl
            trade_log.append({'Time': row['Datetime'], 'Side': 'Sell SL', 'Entry': entry_price, 'Exit': price, 'PnL': pnl})
            position = None

    # Entry/exit based on crossover signals
    if not position:
        if signal == 1:
            position = 'buy'
            entry_price = price
        elif signal == -1:
            position = 'sell'
            entry_price = price
    else:
        # Reverse position if opposite signal appears
        if (position == 'buy' and signal == -1) or (position == 'sell' and signal == 1):
            pnl = (price - entry_price) if position == 'buy' else (entry_price - price)
            balance += pnl
            trade_log.append({'Time': row['Datetime'], 'Side': position, 'Entry': entry_price, 'Exit': price, 'PnL': pnl})
            position = None
            if signal == 1:
                position = 'buy'
                entry_price = price
            elif signal == -1:
                position = 'sell'
                entry_price = price

# ==========================
# RESULTS
# ==========================
trades_df = pd.DataFrame(trade_log)
print(f"Final Balance: ${balance:.2f}")
print(f"Total Trades: {len(trades_df)}")
if not trades_df.empty:
    print(trades_df.tail())

# ==========================
# PLOT
# ==========================
plt.figure(figsize=(14,7))
plt.plot(data['Datetime'], data['Close'], label='BTC Price', alpha=0.5)
plt.plot(data['Datetime'], data['EMA12'], label='EMA12', alpha=0.9)
plt.plot(data['Datetime'], data['EMA20'], label='EMA20', alpha=0.9)

# Mark buy/sell signals
plt.scatter(data[data['Position']==1]['Datetime'], data['EMA12'][data['Position']==1], marker='^', color='g', label='Buy Signal', s=100)
plt.scatter(data[data['Position']==-1]['Datetime'], data['EMA12'][data['Position']==-1], marker='v', color='r', label='Sell Signal', s=100)

plt.title('BTC EMA12/EMA20 Crossover + Volume Spike - 1 Year (1h)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()
