"""
BTC/USD 5m — MACD Zero-Cross: Original vs 4 Variants
Base: The original MACD Zero-Cross that hit +97% in 60 days
Goal: Beat it with smarter filters, same trade frequency or better

Variants:
  M0. Original       — MACD hist flip + EMA200 trend (baseline)
  M1. + Volume       — M0 + volume spike confirmation
  M2. + RSI Filter   — M0 + RSI not overbought/oversold at entry
  M3. + VWAP Side    — M0 + must be on correct side of VWAP
  M4. Full Stack     — All filters combined (M0+M1+M2+M3)

Data: 5m, last 58 days (yfinance max for 5m)
Capital: $100 | Leverage: 1:1000 | Risk: 1%/trade | RR: 2:1
"""

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CAPITAL   = 100.0
LEVERAGE  = 1000
RISK_PCT  = 0.01
RR        = 2.0
SL_PCT    = 0.0020   # 0.20% SL — same as original winner
TICKER    = "BTC-USD"
INTERVAL  = "5m"

# ─────────────────────────────────────────────
# FETCH DATA — full 58 days in 29-day chunks
# ─────────────────────────────────────────────
print("📥 Fetching BTC/USD 5m data (last 58 days)...")
end_date   = datetime.today()
start_date = end_date - timedelta(days=58)

chunks, chunk_start = [], start_date
while chunk_start < end_date:
    chunk_end = min(chunk_start + timedelta(days=29), end_date)
    try:
        chunk = yf.download(
            TICKER,
            start=chunk_start.strftime("%Y-%m-%d"),
            end=chunk_end.strftime("%Y-%m-%d"),
            interval=INTERVAL,
            auto_adjust=True,
            progress=False
        )
        if not chunk.empty:
            chunks.append(chunk)
    except Exception as e:
        print(f"  ⚠ chunk error: {e}")
    chunk_start = chunk_end

if not chunks:
    raise ValueError("No data fetched — check internet connection")

df = pd.concat(chunks)

# Fix MultiIndex columns (yfinance sometimes returns them)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
else:
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

df = df[~df.index.duplicated(keep="first")]
df.sort_index(inplace=True)
df.dropna(inplace=True)
print(f"✅ {len(df)} candles | {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}\n")

# ─────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────
def add_indicators(df):
    d = df.copy()

    # EMAs
    for span in [9, 21, 50, 200]:
        d[f"ema{span}"] = d["Close"].ewm(span=span, adjust=False).mean()

    # RSI (7-period — fast for scalping)
    delta      = d["Close"].diff()
    gain       = delta.clip(lower=0).rolling(7).mean()
    loss       = (-delta.clip(upper=0)).rolling(7).mean()
    d["rsi"]   = 100 - (100 / (1 + gain / loss))

    # MACD
    ema12          = d["Close"].ewm(span=12, adjust=False).mean()
    ema26          = d["Close"].ewm(span=26, adjust=False).mean()
    d["macd"]      = ema12 - ema26
    d["macd_sig"]  = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_sig"]

    # Volume
    d["vol_ma"]    = d["Volume"].rolling(20).mean()
    d["vol_spike"] = d["Volume"] > d["vol_ma"] * 1.5

    # VWAP — daily reset, no apply() bug
    d["date"]    = d.index.date
    tp           = d["Close"] * d["Volume"]
    d["cum_vol"] = d.groupby("date")["Volume"].cumsum()
    d["cum_tp"]  = tp.groupby(d["date"]).cumsum()
    d["vwap"]    = d["cum_tp"] / d["cum_vol"]

    # ATR
    hl  = d["High"] - d["Low"]
    hc  = (d["High"] - d["Close"].shift()).abs()
    lc  = (d["Low"]  - d["Close"].shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()

    return d

df = add_indicators(df)
df.dropna(inplace=True)
print(f"  Indicators ready. {len(df)} candles after warmup.\n")

# ─────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────
def backtest(df, signal_func, name, sl_pct=SL_PCT, rr=RR):
    tp_pct   = sl_pct * rr
    capital  = CAPITAL
    position = None
    trades   = []
    entry_price = sl_price = tp_price = risk_usd = 0.0
    entry_time  = None

    signals = signal_func(df)

    for i in range(1, len(df)):
        if capital <= 0:
            break

        price = float(df["Close"].iloc[i])
        ts    = df.index[i]
        sig   = int(signals.iloc[i])

        # Manage open position
        if position is not None:
            hit_tp = (position == "long"  and price >= tp_price) or \
                     (position == "short" and price <= tp_price)
            hit_sl = (position == "long"  and price <= sl_price) or \
                     (position == "short" and price >= sl_price)

            if hit_tp or hit_sl:
                pnl     = risk_usd * rr if hit_tp else -risk_usd
                capital = max(0.0, capital + pnl)
                trades.append({
                    "strategy":   name,
                    "entry_time": entry_time,
                    "exit_time":  ts,
                    "direction":  position,
                    "entry":      round(entry_price, 2),
                    "exit":       round(price, 2),
                    "result":     "TP ✅" if hit_tp else "SL ❌",
                    "pnl":        round(pnl, 4),
                    "capital":    round(capital, 4),
                })
                position = None
                continue

        # Open new position
        if position is None and sig != 0 and capital > 0:
            risk_usd    = round(capital * RISK_PCT, 6)
            entry_price = price
            entry_time  = ts
            if sig == 1:
                position = "long"
                sl_price = entry_price * (1 - sl_pct)
                tp_price = entry_price * (1 + tp_pct)
            else:
                position = "short"
                sl_price = entry_price * (1 + sl_pct)
                tp_price = entry_price * (1 - tp_pct)

    return pd.DataFrame(trades)

# ─────────────────────────────────────────────
# SIGNAL VARIANTS — all built on MACD Zero-Cross
# ─────────────────────────────────────────────

# M0: Original — MACD histogram flips + EMA200 trend filter
# This is the exact signal that hit +97% in 60 days
def sig_m0_original(df):
    s = pd.Series(0, index=df.index)
    hist_up   = (df["macd_hist"] > 0) & (df["macd_hist"].shift(1) <= 0) & (df["Close"] > df["ema200"])
    hist_down = (df["macd_hist"] < 0) & (df["macd_hist"].shift(1) >= 0) & (df["Close"] < df["ema200"])
    s[hist_up]   = 1
    s[hist_down] = -1
    return s

# M1: + Volume spike — only enter when volume confirms the move
# Filters out low-conviction MACD flips on thin volume
def sig_m1_volume(df):
    s = pd.Series(0, index=df.index)
    hist_up   = (df["macd_hist"] > 0) & (df["macd_hist"].shift(1) <= 0) & (df["Close"] > df["ema200"])
    hist_down = (df["macd_hist"] < 0) & (df["macd_hist"].shift(1) >= 0) & (df["Close"] < df["ema200"])
    s[hist_up   & df["vol_spike"]] = 1
    s[hist_down & df["vol_spike"]] = -1
    return s

# M2: + RSI filter — don't enter if RSI is already stretched
# Long only if RSI < 70 (room to run), Short only if RSI > 30
def sig_m2_rsi(df):
    s = pd.Series(0, index=df.index)
    hist_up   = (df["macd_hist"] > 0) & (df["macd_hist"].shift(1) <= 0) & (df["Close"] > df["ema200"])
    hist_down = (df["macd_hist"] < 0) & (df["macd_hist"].shift(1) >= 0) & (df["Close"] < df["ema200"])
    rsi_ok_long  = df["rsi"] < 70   # not overbought
    rsi_ok_short = df["rsi"] > 30   # not oversold
    s[hist_up   & rsi_ok_long]  = 1
    s[hist_down & rsi_ok_short] = -1
    return s

# M3: + VWAP side — only trade in the direction VWAP agrees with
# Long only above VWAP, Short only below VWAP
def sig_m3_vwap(df):
    s = pd.Series(0, index=df.index)
    hist_up   = (df["macd_hist"] > 0) & (df["macd_hist"].shift(1) <= 0) & (df["Close"] > df["ema200"])
    hist_down = (df["macd_hist"] < 0) & (df["macd_hist"].shift(1) >= 0) & (df["Close"] < df["ema200"])
    above_vwap = df["Close"] > df["vwap"]
    below_vwap = df["Close"] < df["vwap"]
    s[hist_up   & above_vwap] = 1
    s[hist_down & below_vwap] = -1
    return s

# M4: Full Stack — all 3 filters together
# Strictest entry: MACD flip + EMA200 + Volume + RSI room + VWAP side
def sig_m4_full(df):
    s = pd.Series(0, index=df.index)
    hist_up   = (df["macd_hist"] > 0) & (df["macd_hist"].shift(1) <= 0) & (df["Close"] > df["ema200"])
    hist_down = (df["macd_hist"] < 0) & (df["macd_hist"].shift(1) >= 0) & (df["Close"] < df["ema200"])
    s[hist_up   & df["vol_spike"] & (df["rsi"] < 70) & (df["Close"] > df["vwap"])] = 1
    s[hist_down & df["vol_spike"] & (df["rsi"] > 30) & (df["Close"] < df["vwap"])] = -1
    return s

# ─────────────────────────────────────────────
# RUN ALL VARIANTS
# ─────────────────────────────────────────────
variants = [
    ("M0. Original (baseline)",    sig_m0_original),
    ("M1. + Volume Spike",         sig_m1_volume),
    ("M2. + RSI Filter",           sig_m2_rsi),
    ("M3. + VWAP Side",            sig_m3_vwap),
    ("M4. Full Stack 🔥",          sig_m4_full),
]

all_trades = []
summary    = []

for name, sig_fn in variants:
    print(f"  Running {name}...")
    trades = backtest(df, sig_fn, name)
    all_trades.append(trades)

    if len(trades) == 0:
        summary.append({
            "Variant":     name,
            "Trades":      0,
            "Wins":        0,
            "Losses":      0,
            "Win%":        "0.0%",
            "PnL ($)":     0.0,
            "Capital ($)": CAPITAL,
            "Return%":     "0.0%",
            "MaxDD%":      "0.0%",
            "Avg Win":     0.0,
            "Avg Loss":    0.0,
        })
        continue

    tp_mask  = trades["result"].str.startswith("TP")
    sl_mask  = trades["result"].str.startswith("SL")
    wins     = tp_mask.sum()
    losses   = sl_mask.sum()
    pnl      = trades["pnl"].sum()
    final    = trades["capital"].iloc[-1]
    wr       = wins / len(trades) * 100
    ret      = (final - CAPITAL) / CAPITAL * 100
    avg_win  = trades.loc[tp_mask, "pnl"].mean() if wins   > 0 else 0.0
    avg_loss = trades.loc[sl_mask, "pnl"].mean() if losses > 0 else 0.0

    cap_s  = pd.concat([pd.Series([CAPITAL]), trades["capital"].reset_index(drop=True)])
    max_dd = ((cap_s - cap_s.cummax()) / cap_s.cummax() * 100).min()

    summary.append({
        "Variant":     name,
        "Trades":      len(trades),
        "Wins":        int(wins),
        "Losses":      int(losses),
        "Win%":        f"{wr:.1f}%",
        "PnL ($)":     round(pnl, 2),
        "Capital ($)": round(final, 2),
        "Return%":     f"{ret:.1f}%",
        "MaxDD%":      f"{max_dd:.1f}%",
        "Avg Win":     round(avg_win, 4),
        "Avg Loss":    round(avg_loss, 4),
    })

print()

# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────
print("=" * 115)
print(f"   BTC/USD {INTERVAL} | MACD Zero-Cross: Original vs Variants")
print(f"   Capital: ${CAPITAL}  |  Leverage: 1:{LEVERAGE}  |  Risk: {RISK_PCT*100}%/trade  |  RR: {RR}:1  |  SL: {SL_PCT*100}%")
print(f"   Period: {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}")
print("=" * 115)

summary_df = pd.DataFrame(summary)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 180)
print(summary_df.to_string(index=False))
print("=" * 115)

# Comparison vs baseline
base_pnl = summary_df.iloc[0]["PnL ($)"]
base_wr  = summary_df.iloc[0]["Win%"]
base_tr  = summary_df.iloc[0]["Trades"]
print(f"\n  📊 Improvement vs M0 baseline (PnL: ${base_pnl}, WR: {base_wr}, Trades: {base_tr}):")
for i in range(1, len(summary_df)):
    row   = summary_df.iloc[i]
    delta = round(row["PnL ($)"] - base_pnl, 2)
    sign  = "+" if delta >= 0 else ""
    print(f"     {row['Variant']:<35} PnL: ${row['PnL ($)']:>7}  ({sign}{delta})  |  WR: {row['Win%']}  |  Trades: {row['Trades']}")

# Last 15 trades for each variant
for i, (name, _) in enumerate(variants):
    trades = all_trades[i]
    if len(trades) == 0:
        print(f"\n  {name}: No signals.")
        continue
    print(f"\n{'─'*115}")
    print(f"  {name}  — Last 15 Trades")
    print(f"{'─'*115}")
    print(trades[["entry_time","exit_time","direction","entry","exit","result","pnl","capital"]].tail(15).to_string(index=False))

# Winner
print("\n" + "=" * 115)
best_idx = summary_df["PnL ($)"].idxmax()
best     = summary_df.iloc[best_idx]
print(f"  🏆  WINNER    : {best['Variant']}")
print(f"      Trades    : {best['Trades']}  |  Wins: {best['Wins']}  |  Losses: {best['Losses']}")
print(f"      Win Rate  : {best['Win%']}")
print(f"      PnL       : ${best['PnL ($)']}")
print(f"      Return    : {best['Return%']}")
print(f"      Max DD    : {best['MaxDD%']}")
print(f"      Avg Win   : ${best['Avg Win']}  |  Avg Loss: ${best['Avg Loss']}")
print("=" * 115)

# Save
valid = [t for t in all_trades if len(t) > 0]
if valid:
    out = pd.concat(valid, ignore_index=True)
    out.to_csv("macd_variants_trades.csv", index=False)
    print(f"\n📄 Full trade log → macd_variants_trades.csv")
print("✅ Done!\n")
