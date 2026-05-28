"""
data/pipeline.py — Fetch, clean, cache, and load OHLCV data.

Design rules:
  - CSV cache: fetch once, load from disk on every subsequent run.
  - All timestamps normalised to UTC midnight (daily bars).
  - No forward-looking operations here — raw OHLCV only.
  - Missing rows: forward-fill up to 3 consecutive days (weekends/holidays),
    then drop remaining gaps to avoid phantom data.
"""

import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to path so config is importable from any working directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ASSETS, TIMEFRAME


# ── INTERNAL HELPERS ──────────────────────────────────────────────────────────

def _to_utc_date(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise any DatetimeTZDtype index to UTC, then strip time component."""
    idx = pd.to_datetime(df.index, utc=True)
    df.index = idx.normalize()            # keep date, zero the time
    df.index.name = "date"
    return df


def _clean(df: pd.DataFrame, asset: str) -> pd.DataFrame:
    """
    Standard cleaning applied to every dataset:
      1. Keep only OHLCV columns (lower-cased).
      2. Drop rows where close is NaN or zero.
      3. Forward-fill short gaps (≤3 bars) — covers weekends and holidays.
      4. Verify monotonic ascending index and no duplicate dates.
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # Keep only the five canonical columns
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            df[col] = np.nan

    df = df[["open", "high", "low", "close", "volume"]]

    # Drop rows with bad close prices
    df = df[df["close"].notna() & (df["close"] > 0)]

    # Forward-fill gaps ≤ 3 consecutive days (holidays, weekends)
    df = df.asfreq("D").ffill(limit=3)

    # Re-drop any remaining NaN closes after ffill
    df = df[df["close"].notna()]

    # Ensure ascending, unique index
    df = df[~df.index.duplicated(keep="first")]
    df.sort_index(inplace=True)

    return df


def _print_summary(df: pd.DataFrame, label: str) -> None:
    """Print dataset summary — shape, date range, first 5 rows, NaN check."""
    print(f"\n  {'─'*60}")
    print(f"  {label}")
    print(f"  {'─'*60}")
    print(f"  Shape      : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Date range : {df.index[0].date()}  →  {df.index[-1].date()}")
    print(f"  NaN counts : {df.isna().sum().to_dict()}")
    print(f"\n  First 5 rows:")
    print(df.head().to_string(
        float_format=lambda x: f"{x:,.4f}",
        justify="right"
    ))


# ── BTC FETCHER (Binance via ccxt) ────────────────────────────────────────────

def _fetch_btc_ccxt() -> pd.DataFrame:
    """
    Download BTC/USDT daily OHLCV from Binance via ccxt.
    Paginates automatically from ASSETS['BTC']['start'] to today.
    Returns a clean DataFrame indexed by UTC date.
    """
    import ccxt

    cfg      = ASSETS["BTC"]
    exchange = ccxt.binance({"enableRateLimit": True})
    symbol   = cfg["symbol"]
    since_ms = exchange.parse8601(cfg["start"] + "T00:00:00Z")

    all_ohlcv = []
    page      = 0

    print(f"  Downloading {symbol} daily from Binance …", end="", flush=True)
    while True:
        batch = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=since_ms, limit=1000)
        if not batch:
            break
        all_ohlcv.extend(batch)
        page += 1
        print(f" [{page}]", end="", flush=True)
        if len(batch) < 1000:
            break                              # reached present day
        since_ms = batch[-1][0] + 1            # start 1 ms after last candle
        time.sleep(exchange.rateLimit / 1000)  # respect rate limit

    print(f"\n  Fetched {len(all_ohlcv):,} candles total.")

    cols = ["date", "open", "high", "low", "close", "volume"]
    df   = pd.DataFrame(all_ohlcv, columns=cols)
    df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True)
    df.set_index("date", inplace=True)
    return df


# ── XAU FETCHER (yfinance) ────────────────────────────────────────────────────

def _fetch_xau_yfinance() -> pd.DataFrame:
    """
    Download XAU/USD daily OHLCV from yfinance using the GC=F gold futures
    continuous contract.  Returns a clean DataFrame indexed by UTC date.
    """
    import yfinance as yf

    cfg    = ASSETS["XAU"]
    ticker = yf.Ticker(cfg["ticker"])

    print(f"  Downloading {cfg['ticker']} daily from yfinance …", end="", flush=True)
    df = ticker.history(
        start=cfg["start"],
        auto_adjust=True,   # adjust for splits/dividends automatically
        actions=False,
    )
    print(f" {len(df):,} rows fetched.")

    if df.empty:
        raise RuntimeError(
            f"yfinance returned no data for {cfg['ticker']}. "
            "Check your internet connection or try ticker 'XAUUSD=X'."
        )

    df.index = pd.to_datetime(df.index, utc=True)
    df.columns = [c.lower() for c in df.columns]
    return df


# ── PUBLIC LOAD FUNCTIONS ─────────────────────────────────────────────────────

def load_btc(force_refresh: bool = False) -> pd.DataFrame:
    """
    Load BTC/USD daily OHLCV.

    On first run (or force_refresh=True) fetches from Binance and saves to CSV.
    On subsequent runs loads instantly from the cached CSV.

    Returns:
        pd.DataFrame — UTC-dated, columns: open high low close volume
    """
    csv_path = ASSETS["BTC"]["csv"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists() and not force_refresh:
        print(f"  [BTC] Loading from cache: {csv_path.name}")
        df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        print(f"  [BTC] Cache not found — fetching from Binance …")
        df = _fetch_btc_ccxt()
        df = _to_utc_date(df)
        df.to_csv(csv_path)
        print(f"  [BTC] Saved to {csv_path}")

    df = _clean(df, "BTC")

    # Slice to configured start date
    start = pd.Timestamp(ASSETS["BTC"]["start"], tz="UTC")
    df    = df[df.index >= start]

    _print_summary(df, "BTC/USD  (Binance daily)")
    return df


def load_xau(force_refresh: bool = False) -> pd.DataFrame:
    """
    Load XAU/USD daily OHLCV.

    On first run (or force_refresh=True) fetches from yfinance and saves to CSV.
    On subsequent runs loads instantly from the cached CSV.

    Returns:
        pd.DataFrame — UTC-dated, columns: open high low close volume
    """
    csv_path = ASSETS["XAU"]["csv"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists() and not force_refresh:
        print(f"  [XAU] Loading from cache: {csv_path.name}")
        df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        print(f"  [XAU] Cache not found — fetching from yfinance …")
        df = _fetch_xau_yfinance()
        df = _to_utc_date(df)
        df.to_csv(csv_path)
        print(f"  [XAU] Saved to {csv_path}")

    df = _clean(df, "XAU")

    # Slice to configured start date
    start = pd.Timestamp(ASSETS["XAU"]["start"], tz="UTC")
    df    = df[df.index >= start]

    _print_summary(df, "XAU/USD  (Gold futures GC=F via yfinance)")
    return df


def resample_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample a UTC-daily OHLCV DataFrame to weekly bars anchored on Friday.

    Aggregation:
      open   → first trading day's open in the week
      high   → week's highest high
      low    → week's lowest low
      close  → last trading day's close (Friday or preceding day if holiday)
      volume → sum of the week's volume

    Weeks with no valid close (entirely missing) are dropped.
    """
    weekly = df.resample("W-FRI").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    })
    weekly = weekly.dropna(subset=["close"])
    weekly = weekly[weekly["close"] > 0]
    weekly.index = pd.to_datetime(weekly.index, utc=True)
    weekly.index.name = "date"
    return weekly


def load_asset(key: str, force_refresh: bool = False) -> pd.DataFrame:
    """Convenience wrapper: load_asset('BTC') or load_asset('XAU')."""
    if key == "BTC":
        return load_btc(force_refresh)
    elif key == "XAU":
        return load_xau(force_refresh)
    else:
        raise ValueError(f"Unknown asset key '{key}'. Use 'BTC' or 'XAU'.")


# ── BTC 4H FETCHER ────────────────────────────────────────────────────────────

def _fetch_btc_4h_ccxt() -> pd.DataFrame:
    """
    Download BTC/USDT 4H OHLCV from Binance via ccxt.
    Paginates from 2018-01-01 to present.
    """
    import ccxt

    exchange = ccxt.binance({"enableRateLimit": True})
    symbol   = "BTC/USDT"
    since_ms = exchange.parse8601("2018-01-01T00:00:00Z")

    all_ohlcv = []
    page      = 0

    print("  Downloading BTC/USDT 4H from Binance …", end="", flush=True)
    while True:
        batch = exchange.fetch_ohlcv(symbol, "4h", since=since_ms, limit=1000)
        if not batch:
            break
        all_ohlcv.extend(batch)
        page += 1
        print(f" [{page}]", end="", flush=True)
        if len(batch) < 1000:
            break
        since_ms = batch[-1][0] + 1
        time.sleep(exchange.rateLimit / 1000)

    print(f"\n  Fetched {len(all_ohlcv):,} 4H candles total.")

    cols = ["date", "open", "high", "low", "close", "volume"]
    df   = pd.DataFrame(all_ohlcv, columns=cols)
    df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True)
    df.set_index("date", inplace=True)
    return df


def _clean_4h(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning for 4H crypto bars (trades 24/7, no weekend gaps).
    Keeps full UTC datetime index (does NOT strip time component).
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            df[col] = np.nan
    df = df[["open", "high", "low", "close", "volume"]]
    df = df[df["close"].notna() & (df["close"] > 0)]
    # Fill isolated gaps (≤2 consecutive missing 4H bars)
    df = df.asfreq("4h").ffill(limit=2)
    df = df[df["close"].notna()]
    df = df[~df.index.duplicated(keep="first")]
    df.sort_index(inplace=True)
    return df


def load_btc_4h(force_refresh: bool = False) -> pd.DataFrame:
    """
    Load BTC/USD 4-hour OHLCV from Binance.
    Fetches on first run and caches to data/btc_4h.csv.
    Returns a UTC-datetime-indexed DataFrame.
    """
    from config import DATA
    csv_path = DATA / "btc_4h.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists() and not force_refresh:
        print(f"  [BTC 4H] Loading from cache: {csv_path.name}")
        df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        print(f"  [BTC 4H] Cache not found — fetching from Binance …")
        df = _fetch_btc_4h_ccxt()
        df.to_csv(csv_path)
        print(f"  [BTC 4H] Saved to {csv_path}")

    df = _clean_4h(df)

    start = pd.Timestamp("2018-01-01", tz="UTC")
    df    = df[df.index >= start]

    print(f"\n  {'─'*60}")
    print(f"  BTC/USD  (Binance 4H)")
    print(f"  {'─'*60}")
    print(f"  Shape      : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Date range : {df.index[0]}  →  {df.index[-1]}")
    n_years = (df.index[-1] - df.index[0]).days / 365.25
    print(f"  Coverage   : {n_years:.1f} years  "
          f"({df.shape[0] / (n_years * 365.25 * 6):.1%} bar completeness vs 6×daily)")
    print(f"  NaN counts : {df.isna().sum().to_dict()}")

    return df
