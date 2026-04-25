"""
Download BTC/USDT 5m candles from Binance — 3 years
Saves to: btcusdt_5m_3y.csv
Run once, reuse forever.

Usage:
    python download_data.py
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta, timezone

SYMBOL = "BTCUSDT"
DAYS   = 1095   # 3 years
OUTPUT = "btcusdt_5m_3y.csv"

def fetch_binance(symbol, days):
    print(f"📥 Fetching {symbol} 5m data ({days} days) from Binance...")
    print(f"   Estimated candles: {days * 24 * 60 // 5:,}")

    end_ms        = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms      = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    url           = "https://api.binance.com/api/v3/klines"
    all_candles   = []
    current_start = start_ms

    while current_start < end_ms:
        params = {
            "symbol":    symbol,
            "interval":  "5m",
            "startTime": current_start,
            "endTime":   end_ms,
            "limit":     1000,
        }
        try:
            resp    = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            candles = resp.json()
        except Exception as e:
            print(f"\n  ⚠ {e} — retrying in 2s...")
            time.sleep(2)
            continue

        if not candles:
            break

        all_candles.extend(candles)
        last_ts = candles[-1][0]
        pct     = (last_ts - start_ms) / (end_ms - start_ms) * 100
        print(f"  {len(all_candles):,} candles... {pct:.1f}%", end="\r")

        if len(candles) < 1000:
            break

        current_start = last_ts + 1
        time.sleep(0.05)

    print()

    cols = ["open_time", "Open", "High", "Low", "Close", "Volume",
            "close_time", "quote_vol", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"]
    df = pd.DataFrame(all_candles, columns=cols)

    # Keep only OHLCV + timestamp
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df[["open_time", "Open", "High", "Low", "Close", "Volume"]]
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = df[c].astype(float)

    df = df[~df["open_time"].duplicated(keep="first")]
    df.sort_values("open_time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.dropna(inplace=True)

    return df

df = fetch_binance(SYMBOL, DAYS)

print(f"✅ {len(df):,} candles | {df['open_time'].iloc[0].strftime('%Y-%m-%d')} → {df['open_time'].iloc[-1].strftime('%Y-%m-%d')}")

df.to_csv(OUTPUT, index=False)
print(f"💾 Saved → {OUTPUT}  ({df.memory_usage(deep=True).sum() / 1024**2:.1f} MB in memory)")
print(f"   File size: ~{len(df) * 6 * 10 // 1024} KB on disk")
print("✅ Done! Run any test_scalp*.py — they will load from this file automatically.\n")
