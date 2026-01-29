from binance.client import Client
import pandas as pd
from datetime import datetime, timedelta

client = Client()

symbol = "BTCUSDT"
interval = Client.KLINE_INTERVAL_5MINUTE

end_time = datetime.utcnow()
start_time = end_time - timedelta(days=182)

klines = []
while start_time < end_time:
    data = client.get_klines(
        symbol=symbol,
        interval=interval,
        startTime=int(start_time.timestamp() * 1000),
        limit=1000
    )
    if not data:
        break
    klines.extend(data)
    start_time = datetime.fromtimestamp(data[-1][0] / 1000) + timedelta(minutes=5)

df = pd.DataFrame(klines, columns=[
    "open_time","open","high","low","close","volume",
    "close_time","qav","num_trades","taker_base",
    "taker_quote","ignore"
])

df = df[["open_time","open","high","low","close"]]
df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")

df.to_csv("btcusd.csv", index=False)
print("Saved full 6-month BTC data")
