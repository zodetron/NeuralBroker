"""
data/pipeline.py — Fetch, clean, and cache BTC/USDT 15M and 1H OHLCV data.

Design:
  - Binance via ccxt, paginated from 2019-09-01 to present.
  - Cached as parquet (fast read, compressed).
  - Gap detection: reports any runs of missing 15M bars.
  - 1H bars fetched independently for higher-timeframe context.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_CFG


# ── FETCHER ───────────────────────────────────────────────────────────────────

def _fetch_ohlcv(symbol: str, timeframe: str, start: str) -> pd.DataFrame:
    """
    Paginate OHLCV from Binance via ccxt.
    Returns a UTC-indexed DataFrame with columns open/high/low/close/volume.
    """
    import ccxt

    exchange = ccxt.binance({"enableRateLimit": True})
    since_ms  = exchange.parse8601(f"{start}T00:00:00Z")

    all_rows = []
    page     = 0

    label = f"{symbol} {timeframe}"
    print(f"  Fetching {label} from Binance …", end="", flush=True)

    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=1000)
        if not batch:
            break
        all_rows.extend(batch)
        page += 1
        if page % 50 == 0:
            print(f" [{page}]", end="", flush=True)
        if len(batch) < 1000:
            break
        since_ms = batch[-1][0] + 1
        time.sleep(exchange.rateLimit / 1000)

    print(f" [{page}] done — {len(all_rows):,} rows")

    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    df   = pd.DataFrame(all_rows, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df.index.name = "ts"
    return df


# ── CLEANING ──────────────────────────────────────────────────────────────────

def _clean(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Standard cleaning:
      1. Drop rows with zero or NaN close.
      2. Remove duplicates.
      3. Reindex to expected frequency, forward-fill gaps ≤ 3 bars.
      4. Drop any remaining NaN closes.
    """
    df = df.copy()
    df = df[df["close"].notna() & (df["close"] > 0)]
    df = df[~df.index.duplicated(keep="first")]
    df.sort_index(inplace=True)

    freq_map = {"15m": "15min", "1h": "1h"}
    freq = freq_map.get(timeframe, timeframe)
    df = df.asfreq(freq).ffill(limit=3)
    df = df[df["close"].notna()]
    return df


# ── GAP DETECTION ─────────────────────────────────────────────────────────────

def _find_gaps(df: pd.DataFrame, timeframe: str, max_ffill: int = 3) -> pd.DataFrame:
    """
    Return a DataFrame of gap runs that exceed max_ffill consecutive missing bars.
    Gaps ≤ max_ffill are filled and not reported.
    """
    freq_map = {"15m": 15, "1h": 60}
    bar_minutes = freq_map.get(timeframe, 60)
    expected_delta = pd.Timedelta(minutes=bar_minutes)

    diffs  = df.index.to_series().diff().dropna()
    gaps   = diffs[diffs > expected_delta * (max_ffill + 1)]

    records = []
    for ts, gap in gaps.items():
        missing = int(gap / expected_delta) - 1
        records.append({
            "gap_start": ts - gap,
            "gap_end":   ts,
            "missing_bars": missing,
        })
    return pd.DataFrame(records)


# ── SUMMARY ───────────────────────────────────────────────────────────────────

def _print_summary(df: pd.DataFrame, label: str, timeframe: str) -> None:
    freq_map = {"15m": 15, "1h": 60}
    bar_min  = freq_map.get(timeframe, 60)
    n_days   = (df.index[-1] - df.index[0]).days

    expected_bars = n_days * 24 * 60 // bar_min
    completeness  = len(df) / expected_bars * 100 if expected_bars > 0 else 0.0

    print(f"\n  {'─'*62}")
    print(f"  {label}")
    print(f"  {'─'*62}")
    print(f"  Bars        : {len(df):>10,}")
    print(f"  Date range  : {df.index[0].date()}  →  {df.index[-1].date()}")
    print(f"  Calendar days: {n_days:,}  ({n_days/365.25:.1f} years)")
    print(f"  Completeness : {completeness:.1f}%  (vs perfect 24/7 grid)")
    print(f"  NaN counts  : {df.isna().sum().to_dict()}")

    gaps = _find_gaps(df, timeframe)
    if gaps.empty:
        print(f"  Gaps > 3 bars: none found ✓")
    else:
        total_missing = gaps["missing_bars"].sum()
        print(f"  Gaps > 3 bars: {len(gaps)} gap(s), "
              f"{total_missing:,} total missing bars")
        for _, g in gaps.head(5).iterrows():
            print(f"    {g['gap_start'].date()} → {g['gap_end'].date()}  "
                  f"({g['missing_bars']} bars missing)")
        if len(gaps) > 5:
            print(f"    … {len(gaps) - 5} more gaps")


# ── PUBLIC LOAD FUNCTIONS ─────────────────────────────────────────────────────

def load_15m(force_refresh: bool = False) -> pd.DataFrame:
    """
    Load BTC/USDT 15-minute OHLCV.
    Fetches from Binance on first run; loads from parquet cache on subsequent runs.
    """
    path = DATA_CFG["parquet_15m"]
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not force_refresh:
        print(f"  [15M] Loading from cache: {path.name}")
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        print(f"  [15M] Cache not found — fetching …")
        df = _fetch_ohlcv(
            DATA_CFG["symbol_15m"],
            DATA_CFG["tf_15m"],
            DATA_CFG["start"],
        )
        df = _clean(df, "15m")
        df.to_parquet(path, engine="pyarrow", compression="snappy")
        print(f"  [15M] Saved → {path.name}")

    _print_summary(df, "BTC/USDT  15M  (Binance)", "15m")
    return df


def load_1h(force_refresh: bool = False) -> pd.DataFrame:
    """
    Load BTC/USDT 1-hour OHLCV.
    Fetches from Binance on first run; loads from parquet cache on subsequent runs.
    """
    path = DATA_CFG["parquet_1h"]
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not force_refresh:
        print(f"  [1H] Loading from cache: {path.name}")
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        print(f"  [1H] Cache not found — fetching …")
        df = _fetch_ohlcv(
            DATA_CFG["symbol_1h"],
            DATA_CFG["tf_1h"],
            DATA_CFG["start"],
        )
        df = _clean(df, "1h")
        df.to_parquet(path, engine="pyarrow", compression="snappy")
        print(f"  [1H] Saved → {path.name}")

    _print_summary(df, "BTC/USDT  1H   (Binance)", "1h")
    return df
