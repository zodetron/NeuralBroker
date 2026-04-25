"""
Shared data loader for all test_scalp4*.py files.
Loads from btcusdt_5m_3y.csv if it exists, otherwise downloads from Binance.

Usage in any script:
    from data_loader import load_data
    df = load_data()
"""

import os
import requests
import pandas as pd
import time
from datetime import datetime, timedelta, timezone

CSV_FILE = "btcusdt_5m_3y.csv"
SYMBOL   = "BTCUSDT"
DAYS     = 1095


def _fetch_binance(symbol, days):
    print(f"📥 Fetching {symbol} 5m data ({days} days) from Binance...")
    end_ms        = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms      = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    url           = "https://api.binance.com/api/v3/klines"
    all_candles   = []
    current_start = start_ms

    while current_start < end_ms:
        params = {"symbol": symbol, "interval": "5m",
                  "startTime": current_start, "endTime": end_ms, "limit": 1000}
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
        print(f"  fetched {len(all_candles):,} candles... {pct:.1f}%", end="\r")
        if len(candles) < 1000:
            break
        current_start = last_ts + 1
        time.sleep(0.05)

    print()
    cols = ["open_time", "Open", "High", "Low", "Close", "Volume",
            "close_time", "quote_vol", "trades", "tbb", "tbq", "ig"]
    df = pd.DataFrame(all_candles, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df[["open_time", "Open", "High", "Low", "Close", "Volume"]]
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = df[c].astype(float)
    df = df[~df["open_time"].duplicated(keep="first")]
    df.sort_values("open_time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.dropna(inplace=True)

    # Save for future use
    df.to_csv(CSV_FILE, index=False)
    print(f"💾 Saved → {CSV_FILE}")
    return df


def load_data(csv_file=CSV_FILE):
    """
    Load 5m BTC/USDT data.
    - If csv_file exists: load from disk instantly (~0.5s)
    - If not: download from Binance and save (~60s)
    Returns DataFrame with DatetimeIndex (UTC) and OHLCV columns.
    """
    if os.path.exists(csv_file):
        print(f"📂 Loading from {csv_file}...", end=" ")
        df = pd.read_csv(csv_file, parse_dates=["open_time"])
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        df = df[~df["open_time"].duplicated(keep="first")]
        df.sort_values("open_time", inplace=True)
        df.set_index("open_time", inplace=True)
        df.dropna(inplace=True)
        print(f"{len(df):,} candles | {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}")
        return df
    else:
        print(f"⚠  {csv_file} not found — downloading from Binance...")
        print(f"   Run 'python download_data.py' once to avoid this in future.\n")
        df = _fetch_binance(SYMBOL, DAYS)
        df.set_index("open_time", inplace=True)
        return df
