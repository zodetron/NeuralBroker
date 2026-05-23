"""
TOP 5 STRATEGIES — XAUUSD 1m  |  1 Month  |  Spread: 0.025%
Tests the 5 winning strategies from test_hyper3 on 1-minute candles.
Capital: $200  |  Leverage: 1:1000  |  Min Lot: 0.01  |  Risk: 1%/trade
"""

import os, time, requests
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime, timedelta

# ── CONFIG ────────────────────────────────────────────────────────────────────
TWELVEDATA_API_KEY = "da2428fa53e14697989d2707c308a01d"
XAUUSD_1M_CSV      = "xauusd_1m_6mo.csv"

CAPITAL  = 100.0
RISK_PCT = 0.01
SPREAD   = 0.00025
LEVERAGE = 1000
MIN_LOT  = 0.01
LOT_STEP = 0.01

# ── TOP 5 STRATEGIES WITH EXACT PARAMS FROM test_hyper3 ──────────────────────
TOP5 = [
    {"name": "EMA_21_50_200", "sl": 0.002, "rr": 3.5},
    {"name": "ATR_Breakout",  "sl": 0.002, "rr": 3.5},
    {"name": "EMA_21_50_200", "sl": 0.002, "rr": 3.0},
    {"name": "EMA_5_13_50",   "sl": 0.002, "rr": 3.5},
    {"name": "RSI_MACD_EMA",  "sl": 0.002, "rr": 3.5},
]

# ── DOWNLOADER ────────────────────────────────────────────────────────────────
def download_1m(api_key, days=30, csv_file=XAUUSD_1M_CSV):
    print(f"Downloading XAUUSD 1m data ({days} days) from Twelve Data...")
    end_dt   = datetime.utcnow()
    start_dt = end_dt - timedelta(days=days)
    url      = "https://api.twelvedata.com/time_series"
    all_rows = []
    current_end = end_dt
    req_n = 0

    while current_end > start_dt:
        params = {
            "symbol":     "XAU/USD",
            "interval":   "1min",
            "outputsize": 5000,
            "apikey":     api_key,
            "end_date":   current_end.strftime("%Y-%m-%d %H:%M:%S"),
            "timezone":   "UTC",
            "format":     "JSON",
        }
        try:
            resp = requests.get(url, params=params, timeout=20)
            data = resp.json()
        except Exception as e:
            print(f"\n  ⚠ {e} — retrying in 10s...")
            time.sleep(10)
            continue

        if data.get("status") == "error" or "code" in data:
            print(f"\n  ✗ API error: {data.get('message', data)}")
            break

        values = data.get("values", [])
        if not values:
            break

        all_rows.extend(values)
        req_n += 1
        earliest = datetime.strptime(values[-1]["datetime"], "%Y-%m-%d %H:%M:%S")
        pct = (end_dt - earliest) / (end_dt - start_dt) * 100
        print(f"  [{req_n:2d} req] {len(all_rows):>7,} candles | {pct:.1f}% | to {earliest.strftime('%Y-%m-%d %H:%M')}",
              end="\r")
        if earliest <= start_dt:
            break
        current_end = earliest - timedelta(minutes=1)
        time.sleep(9)

    print()
    if not all_rows:
        return None

    df = pd.DataFrame(all_rows)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.rename(columns={"datetime": "open_time",
                             "open": "Open", "high": "High",
                             "low": "Low",   "close": "Close"})
    for c in ["Open", "High", "Low", "Close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0) if "volume" in df.columns else 0.0
    df = df[["open_time", "Open", "High", "Low", "Close", "Volume"]]
    df = df[~df["open_time"].duplicated(keep="first")]
    df.sort_values("open_time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.dropna(inplace=True)
    df.to_csv(csv_file, index=False)
    print(f"  Saved → {csv_file}  ({len(df):,} candles)")
    return df


def load_data():
    if os.path.exists(XAUUSD_1M_CSV):
        print(f"Loading {XAUUSD_1M_CSV}...", end=" ")
        df = pd.read_csv(XAUUSD_1M_CSV, parse_dates=["open_time"])
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        df = df[~df["open_time"].duplicated(keep="first")]
        df.sort_values("open_time", inplace=True)
        df.set_index("open_time", inplace=True)
        df.dropna(inplace=True)
        print(f"{len(df):,} candles | {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}")
        return df
    df = download_1m(TWELVEDATA_API_KEY, days=180)
    if df is None or df.empty:
        raise RuntimeError("Download failed.")
    df.set_index("open_time", inplace=True)
    return df


# ── LOAD ──────────────────────────────────────────────────────────────────────
df_raw = load_data()
# Use only last 6 months
cutoff = df_raw.index[-1] - pd.DateOffset(months=6)
df_raw = df_raw[df_raw.index >= cutoff].copy()
print(f"Using: {df_raw.index[0].strftime('%Y-%m-%d %H:%M')} → {df_raw.index[-1].strftime('%Y-%m-%d %H:%M')} | {len(df_raw):,} candles\n")

# ── INDICATORS ────────────────────────────────────────────────────────────────
def build_indicators(df):
    d = df.copy()
    c = d["Close"]

    for s in [5, 13, 21, 50, 100, 200]:
        d[f"ema{s}"] = c.ewm(span=s, adjust=False).mean()

    high_low = d["High"] - d["Low"]
    high_pc  = (d["High"] - c.shift()).abs()
    low_pc   = (d["Low"]  - c.shift()).abs()
    tr       = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
    d["atr14"] = tr.ewm(span=14, adjust=False).mean()

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    d["macd"]      = ema12 - ema26
    d["macd_hist"] = d["macd"] - d["macd"].ewm(span=9, adjust=False).mean()

    delta = c.diff()
    g = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    l = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    d["rsi14"] = 100 - 100 / (1 + g / l)

    return d.dropna()


print("Building indicators...")
df = build_indicators(df_raw)
print(f"Done. {len(df):,} candles.\n")

# ── SIGNAL GENERATORS ─────────────────────────────────────────────────────────
def sig_EMA_21_50_200(d):
    bull = (d["ema21"] > d["ema50"]) & (d["ema50"] > d["ema200"])
    bear = (d["ema21"] < d["ema50"]) & (d["ema50"] < d["ema200"])
    s = pd.Series(0, index=d.index)
    s[bull & ~bull.shift().fillna(False)] =  1
    s[bear & ~bear.shift().fillna(False)] = -1
    return s

def sig_ATR_Breakout(d):
    prev_c = d["Close"].shift()
    s = pd.Series(0, index=d.index)
    s[d["Close"] > prev_c + d["atr14"]] =  1
    s[d["Close"] < prev_c - d["atr14"]] = -1
    return s

def sig_EMA_5_13_50(d):
    bull = (d["ema5"] > d["ema13"]) & (d["ema13"] > d["ema50"])
    bear = (d["ema5"] < d["ema13"]) & (d["ema13"] < d["ema50"])
    s = pd.Series(0, index=d.index)
    s[bull & ~bull.shift().fillna(False)] =  1
    s[bear & ~bear.shift().fillna(False)] = -1
    return s

def sig_RSI_MACD_EMA(d):
    lc = (d["rsi14"] > 50) & (d["macd_hist"] > 0) & (d["Close"] > d["ema100"])
    sc = (d["rsi14"] < 50) & (d["macd_hist"] < 0) & (d["Close"] < d["ema100"])
    s = pd.Series(0, index=d.index)
    s[lc & ~lc.shift().fillna(False)] =  1
    s[sc & ~sc.shift().fillna(False)] = -1
    return s

SIG_FNS = {
    "EMA_21_50_200": sig_EMA_21_50_200,
    "ATR_Breakout":  sig_ATR_Breakout,
    "EMA_5_13_50":   sig_EMA_5_13_50,
    "RSI_MACD_EMA":  sig_RSI_MACD_EMA,
}

# ── LOT CALCULATOR ────────────────────────────────────────────────────────────
def calc_lots(capital, price, sl_pct):
    risk_usd       = capital * RISK_PCT
    sl_usd_per_lot = price * sl_pct
    lots           = max(MIN_LOT, round((risk_usd / sl_usd_per_lot) / LOT_STEP) * LOT_STEP)
    lots           = round(lots, 2)
    if lots * price / LEVERAGE > capital:
        lots = max(MIN_LOT, round((capital * LEVERAGE / price) / LOT_STEP) * LOT_STEP)
        lots = round(lots, 2)
    return lots

# ── BACKTEST ──────────────────────────────────────────────────────────────────
def backtest(df, signals, sl_pct, rr):
    cap        = CAPITAL
    pos        = None
    entry_fill = sl_p = tp_p = lots = 0.0
    tp_pct     = sl_pct * rr
    peak       = cap
    trades     = []

    for i in range(1, len(df)):
        if cap <= 0:
            break
        price = float(df["Close"].iloc[i])
        ts    = df.index[i]
        sig   = int(signals.iloc[i])

        if pos is not None:
            hit_tp = (pos ==  1 and price >= tp_p) or (pos == -1 and price <= tp_p)
            hit_sl = (pos ==  1 and price <= sl_p) or (pos == -1 and price >= sl_p)
            if hit_tp or hit_sl:
                exit_px = tp_p if hit_tp else sl_p
                pnl     = lots * abs(exit_px - entry_fill) * (1 if hit_tp else -1)
                cap     = max(0.0, cap + pnl)
                peak    = max(peak, cap)
                trades.append({
                    "exit_time": ts,
                    "direction": "LONG" if pos == 1 else "SHORT",
                    "result":    "TP" if hit_tp else "SL",
                    "lots":      lots,
                    "entry":     round(entry_fill, 3),
                    "exit":      round(exit_px, 3),
                    "pnl":       round(pnl, 4),
                    "capital":   round(cap, 4),
                    "peak":      round(peak, 4),
                })
                pos = None

        if pos is None and sig != 0 and cap > 0:
            lots       = calc_lots(cap, price, sl_pct)
            entry_fill = price * (1 + SPREAD) if sig == 1 else price * (1 - SPREAD)
            sl_p       = entry_fill * (1 - sl_pct) if sig == 1 else entry_fill * (1 + sl_pct)
            tp_p       = entry_fill * (1 + tp_pct) if sig == 1 else entry_fill * (1 - tp_pct)
            pos        = sig

    return pd.DataFrame(trades)

# ── RUN ALL 5 STRATEGIES ──────────────────────────────────────────────────────
print("═"*85)
print(f"  XAUUSD 1m  |  Last 6 Months  |  $100  |  0.025% Spread  |  0.01 Min Lot  |  1:1000")
print("═"*85)

all_summary = []

for cfg in TOP5:
    name   = cfg["name"]
    sl_pct = cfg["sl"]
    rr     = cfg["rr"]
    label  = f"{name}|SL{sl_pct*100:.1f}%|RR{rr}"

    sigs   = SIG_FNS[name](df)
    t      = backtest(df, sigs, sl_pct, rr)

    if t.empty:
        print(f"\n  {label}: no trades generated")
        continue

    wins    = (t["result"] == "TP").sum()
    losses  = (t["result"] == "SL").sum()
    wr      = wins / len(t) * 100
    final   = t["capital"].iloc[-1]
    total_r = (final - CAPITAL) / CAPITAL * 100
    avg_lot = t["lots"].mean()

    # Max drawdown
    caps  = pd.concat([pd.Series([CAPITAL]), t["capital"].reset_index(drop=True)])
    peak  = caps.cummax()
    dd    = (caps - peak) / peak * 100
    mdd   = dd.min()
    mdd_usd = (caps - peak).min()

    # Monthly breakdown
    t["month"] = pd.to_datetime(t["exit_time"]).dt.to_period("M")
    months   = sorted(t["month"].unique())
    prev_cap = CAPITAL
    mo_rows  = []
    for m in months:
        mt  = t[t["month"] == m]
        end = mt["capital"].iloc[-1]
        mo_rows.append({
            "month":   str(m),
            "trades":  len(mt),
            "win%":    (mt["result"] == "TP").sum() / len(mt) * 100,
            "ret%":    (end - prev_cap) / prev_cap * 100,
            "capital": end,
            "avg_lot": mt["lots"].mean(),
        })
        prev_cap = end

    mo_df = pd.DataFrame(mo_rows)
    rets  = mo_df["ret%"]

    all_summary.append({
        "label":    label,
        "trades":   len(t),
        "win%":     round(wr, 1),
        "total%":   round(total_r, 1),
        "mdd%":     round(mdd, 1),
        "mdd_$":    round(abs(mdd_usd), 2),
        "avg_lot":  round(avg_lot, 3),
        "final":    round(final, 2),
        "pos_mo":   (rets > 0).sum(),
        "n_mo":     len(rets),
        "avg_mo%":  round(rets.mean(), 2),
    })

    # ── PRINT RESULT ──────────────────────────────────────────────────────────
    print(f"\n  {'─'*75}")
    print(f"  {label}")
    print(f"  {'─'*75}")
    print(f"  Trades: {len(t):,}  |  Wins: {wins}  |  Losses: {losses}  |  Win Rate: {wr:.1f}%")
    print(f"  Final Capital: ${final:,.2f}  |  Total Return: {total_r:+.1f}%")
    print(f"  Max Drawdown: {mdd:.1f}%  (${abs(mdd_usd):.2f})  |  Avg Lot: {avg_lot:.3f}")
    print(f"\n  {'Month':<12} {'Trades':>7} {'Win%':>6} {'Return%':>10} {'Capital':>10} {'AvgLot':>8}  Bar")
    print(f"  {'-'*72}")
    for _, r in mo_df.iterrows():
        ret = r["ret%"]
        bar = ("█" * min(int(abs(ret)/3), 30)) if ret >= 0 else ("▓" * min(int(abs(ret)/3), 30))
        sgn = "+" if ret >= 0 else ""
        print(f"  {r['month']:<12} {int(r['trades']):>7} {r['win%']:>5.1f}% "
              f"{sgn}{ret:>9.2f}% ${r['capital']:>9.2f} {r['avg_lot']:>8.3f}  {bar}")
    print(f"  {'-'*72}")
    print(f"  Avg: {rets.mean():+.2f}%/mo  |  Best: {rets.max():+.2f}%  |  "
          f"Worst: {rets.min():+.2f}%  |  Positive: {(rets>0).sum()}/{len(rets)}")

# ── SUMMARY TABLE ─────────────────────────────────────────────────────────────
print(f"\n\n{'═'*95}")
print(f"  SUMMARY — All 5 Strategies on XAUUSD 1m  |  Last 6 Months  |  $100 start")
print(f"{'═'*95}")
print(f"  {'STRATEGY':<42} {'TRADES':>7} {'WIN%':>6} {'AVG/MO':>8} {'TOTAL%':>8} {'MDD%':>7} {'MDD$':>8} {'FINAL$':>9}")
print(f"{'═'*105}")

sumdf = pd.DataFrame(all_summary).sort_values("total%", ascending=False).reset_index(drop=True)
for i, r in sumdf.iterrows():
    mark = " ✓" if r["total%"] > 0 else " ✗"
    print(f"  {r['label']:<42} {int(r['trades']):>7} {r['win%']:>5.1f}% "
          f"{r['avg_mo%']:>+8.2f}% {r['total%']:>+8.1f}% {r['mdd%']:>7.1f}% "
          f"${r['mdd_$']:>7.2f} ${r['final']:>8.2f}{mark}")

print(f"{'═'*95}")
best = sumdf.iloc[0]
print(f"\n  WINNER  : {best['label']}")
print(f"  Return  : {best['total%']:+.1f}%  |  MDD: {best['mdd%']:.1f}% (${best['mdd_$']:.2f})")
print(f"  Final   : ${best['final']:,.2f}  from $100  over 6 months")
print(f"{'═'*95}")
