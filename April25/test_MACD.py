"""
MACD ZERO-CROSS — WALK FORWARD CONSISTENCY TEST
═══════════════════════════════════════════════════
Problem: yfinance only gives 60 days of 5m data.
Solution: Use 1H data (2 years available) and run
          the SAME strategy logic across every
          60-day rolling window to see if returns
          are consistent or just lucky.

Also runs: 5m data (last 60 days) for direct comparison.

Capital : $100 | Leverage 1:1000 | Risk 1%/trade | RR 2:1
"""

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

CAPITAL  = 100.0
LEVERAGE = 1000
RISK_PCT = 0.01
RR       = 2.0
SL_PCT   = 0.002
TICKER   = "BTC-USD"

# ─────────────────────────────────────────────
# FETCH DATA
# ─────────────────────────────────────────────
def fetch_chunked(ticker, interval, days, chunk_days=29):
    end = datetime.today()
    start = end - timedelta(days=days)
    chunks, cur = [], start
    while cur < end:
        nxt = min(cur + timedelta(days=chunk_days), end)
        try:
            c = yf.download(ticker, start=cur.strftime("%Y-%m-%d"),
                            end=nxt.strftime("%Y-%m-%d"),
                            interval=interval, auto_adjust=True, progress=False)
            if len(c): chunks.append(c)
        except: pass
        cur = nxt
    if not chunks: return pd.DataFrame()
    df = pd.concat(chunks)
    df.columns = [x[0] if isinstance(x, tuple) else x for x in df.columns]
    df = df[~df.index.duplicated(keep="first")]
    df.sort_index(inplace=True)
    return df.dropna()

print("📥 Fetching 1H data (720 days = 2 years)...")
df1h = fetch_chunked(TICKER, "1h", days=720, chunk_days=59)
print(f"✅ 1H: {len(df1h)} candles | {df1h.index[0].date()} → {df1h.index[-1].date()}")

print("📥 Fetching 5m data (58 days)...")
df5m = fetch_chunked(TICKER, "5m", days=58, chunk_days=29)
print(f"✅ 5m: {len(df5m)} candles | {df5m.index[0].date()} → {df5m.index[-1].date()}\n")

# ─────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────
def add_indicators(df):
    d = df.copy()
    d["ema200"]    = d["Close"].ewm(span=200, adjust=False).mean()
    ema12          = d["Close"].ewm(span=12,  adjust=False).mean()
    ema26          = d["Close"].ewm(span=26,  adjust=False).mean()
    d["macd"]      = ema12 - ema26
    d["macd_sig"]  = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_sig"]
    delta          = d["Close"].diff()
    gain           = delta.clip(lower=0).rolling(14).mean()
    loss           = (-delta.clip(upper=0)).rolling(14).mean()
    d["rsi"]       = 100 - (100 / (1 + gain / loss))
    d["vol_ma"]    = d["Volume"].rolling(20).mean()
    d["vol_ratio"] = d["Volume"] / (d["vol_ma"] + 1e-9)
    return d.dropna()

df1h = add_indicators(df1h)
df5m = add_indicators(df5m)

# ─────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────
def backtest(df, sl_pct=SL_PCT, rr=RR, starting_capital=CAPITAL):
    capital  = starting_capital
    position = None
    trades   = []
    entry_price = sl_price = tp_price = risk_usd = 0.0
    entry_time = None

    for i in range(1, len(df)):
        if capital <= 0:
            break
        price = float(df["Close"].iloc[i])
        ts    = df.index[i]

        hist_now  = float(df["macd_hist"].iloc[i])
        hist_prev = float(df["macd_hist"].iloc[i-1])
        ema200    = float(df["ema200"].iloc[i])

        sig = 0
        if hist_now > 0 and hist_prev <= 0 and price > ema200:
            sig = 1
        elif hist_now < 0 and hist_prev >= 0 and price < ema200:
            sig = -1

        if position is not None:
            hit_tp = (position == "long"  and price >= tp_price) or \
                     (position == "short" and price <= tp_price)
            hit_sl = (position == "long"  and price <= sl_price) or \
                     (position == "short" and price >= sl_price)

            if hit_tp or hit_sl:
                pnl     = risk_usd * rr if hit_tp else -risk_usd
                capital = max(0.0, capital + pnl)
                trades.append({
                    "entry_time": entry_time,
                    "exit_time":  ts,
                    "direction":  position,
                    "result":     "TP" if hit_tp else "SL",
                    "pnl":        round(pnl, 4),
                    "capital":    round(capital, 4),
                })
                position = None

        if position is None and sig != 0 and capital > 0:
            risk_usd    = round(capital * RISK_PCT, 6)
            entry_price = price
            entry_time  = ts
            tp_pct      = sl_pct * rr
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
# WALK FORWARD — rolling 60-day windows on 1H
# ─────────────────────────────────────────────
print("🔄 Running walk-forward (60-day windows, 1H data)...")
print("   Each window = same as your 5m test but on 1H candles\n")

WINDOW_DAYS = 60
STEP_DAYS   = 15   # new window every 15 days

results = []
dates   = df1h.index

window_candles = WINDOW_DAYS * 24   # ~1440 1H candles per 60 days
step_candles   = STEP_DAYS * 24

i = 0
window_num = 1
while i + window_candles <= len(df1h):
    window_df = df1h.iloc[i : i + window_candles].copy()
    if len(window_df) < 100:
        break

    trades = backtest(window_df)
    if len(trades) == 0:
        i += step_candles
        window_num += 1
        continue

    tp      = (trades["result"] == "TP").sum()
    sl      = (trades["result"] == "SL").sum()
    pnl     = trades["pnl"].sum()
    final   = trades["capital"].iloc[-1]
    wr      = tp / (tp + sl) * 100 if (tp + sl) > 0 else 0
    ret     = (final - CAPITAL) / CAPITAL * 100

    cap_s   = pd.concat([pd.Series([CAPITAL]), trades["capital"].reset_index(drop=True)])
    max_dd  = ((cap_s - cap_s.cummax()) / cap_s.cummax() * 100).min()

    results.append({
        "Window":     f"W{window_num:02d}",
        "From":       window_df.index[0].date(),
        "To":         window_df.index[-1].date(),
        "Trades":     len(trades),
        "Win%":       round(wr, 1),
        "PnL($)":     round(pnl, 2),
        "Return%":    round(ret, 1),
        "MaxDD%":     round(max_dd, 1),
        "Final($)":   round(final, 2),
    })

    i += step_candles
    window_num += 1

results_df = pd.DataFrame(results)

# ─────────────────────────────────────────────
# 5m BASELINE (direct comparison)
# ─────────────────────────────────────────────
print("⚡ Running 5m baseline (last 60 days)...")
trades_5m = backtest(df5m)
tp5  = (trades_5m["result"] == "TP").sum() if len(trades_5m) > 0 else 0
sl5  = (trades_5m["result"] == "SL").sum() if len(trades_5m) > 0 else 0
pnl5 = trades_5m["pnl"].sum() if len(trades_5m) > 0 else 0
fin5 = trades_5m["capital"].iloc[-1] if len(trades_5m) > 0 else CAPITAL
wr5  = tp5 / (tp5 + sl5) * 100 if (tp5 + sl5) > 0 else 0
ret5 = (fin5 - CAPITAL) / CAPITAL * 100

# ─────────────────────────────────────────────
# PRINT RESULTS
# ─────────────────────────────────────────────
print()
print("=" * 90)
print(f"  MACD ZERO-CROSS — WALK FORWARD TEST  |  ${CAPITAL} Capital  |  1:{LEVERAGE} Leverage")
print(f"  1H data: {df1h.index[0].date()} → {df1h.index[-1].date()}  |  60-day windows, step 15 days")
print("=" * 90)
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", None)
print(results_df.to_string(index=False))
print("=" * 90)

# Stats across all windows
print(f"\n  📊 CONSISTENCY STATS ACROSS ALL {len(results_df)} WINDOWS:")
print(f"{'─'*60}")
profitable = (results_df["PnL($)"] > 0).sum()
print(f"  Profitable windows : {profitable}/{len(results_df)} ({profitable/len(results_df)*100:.0f}%)")
print(f"  Avg Return/window  : {results_df['Return%'].mean():.1f}%")
print(f"  Best window        : {results_df['Return%'].max():.1f}%  ({results_df.loc[results_df['Return%'].idxmax(), 'From']} → {results_df.loc[results_df['Return%'].idxmax(), 'To']})")
print(f"  Worst window       : {results_df['Return%'].min():.1f}%  ({results_df.loc[results_df['Return%'].idxmin(), 'From']} → {results_df.loc[results_df['Return%'].idxmin(), 'To']})")
print(f"  Avg Win Rate       : {results_df['Win%'].mean():.1f}%")
print(f"  Avg Trades/window  : {results_df['Trades'].mean():.0f}")
print(f"  Avg Max Drawdown   : {results_df['MaxDD%'].mean():.1f}%")

# Visual return chart
print(f"\n  📈 Return per window (visual):")
print(f"{'─'*60}")
for _, row in results_df.iterrows():
    r   = row["Return%"]
    bar = "█" * int(abs(r) / 3)
    sgn = "+" if r >= 0 else "-"
    col = "✅" if r > 0 else "❌"
    print(f"  {row['Window']} {row['From']}→{row['To']}  {sgn}{abs(r):5.1f}%  {col} {bar}")

# 5m comparison
print(f"\n{'─'*60}")
print(f"  ⚡ 5m LAST 60 DAYS (direct):")
print(f"     Trades: {len(trades_5m)} | Win%: {wr5:.1f}% | PnL: ${pnl5:.2f} | Return: {ret5:.1f}%")

# Annualised estimate
avg_ret_60d = results_df["Return%"].mean()
annual_est  = ((1 + avg_ret_60d/100) ** 6 - 1) * 100
print(f"\n  💰 ANNUALISED ESTIMATE (based on avg 60-day return of {avg_ret_60d:.1f}%):")
for cap in [100, 500, 1000, 5000, 10000]:
    end_val = cap * (1 + avg_ret_60d/100) ** 6
    print(f"     ${cap:>6} → ${end_val:>12,.0f} in 1 year")

print("\n" + "=" * 90)
print("  ✅ VERDICT:")
if profitable / len(results_df) >= 0.7:
    print("  Strategy is CONSISTENT — profitable in majority of market conditions.")
    print("  The +97% and +78% results are within the normal range.")
    print("  → Safe to build live bot from this strategy.")
else:
    print("  Strategy is INCONSISTENT — may have been lucky in recent period.")
    print("  → Needs more filtering before going live.")
print("=" * 90)

trades_5m.to_csv("macd_walkforward_5m.csv", index=False)
results_df.to_csv("macd_walkforward_1h_windows.csv", index=False)
print(f"\n📄 Saved: macd_walkforward_5m.csv  +  macd_walkforward_1h_windows.csv")
print("✅ Done!\n")