"""
SUPERTREND PARAM SEARCH — BTC/USDT 5m  |  2 Years  |  Spread: 0.04%
Finds SL / RR / filter combos that are profitable at 0.04% spread.
Capital: $100  |  Leverage: 1:1000  |  Risk: 1%/trade
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from data_loader import load_data

# ── CONFIG ────────────────────────────────────────────────────────────────────
CAPITAL  = 100.0
RISK_PCT = 0.01
SPREAD   = 0.0004   # 0.04%

# Grids to search
SL_GRID  = [0.003, 0.005, 0.008, 0.010, 0.015]   # 0.3% → 1.5%
RR_GRID  = [2.0, 2.5, 3.0, 3.5, 4.0]

# ── LOAD DATA (last 2 years) ──────────────────────────────────────────────────
df_full = load_data()
cutoff  = df_full.index[-1] - pd.DateOffset(years=2)
df_raw  = df_full[df_full.index >= cutoff].copy()
print(f"Using: {df_raw.index[0].strftime('%Y-%m-%d')} → {df_raw.index[-1].strftime('%Y-%m-%d')} | {len(df_raw):,} candles\n")

# ── INDICATORS ────────────────────────────────────────────────────────────────
def build_indicators(df):
    d = df.copy()
    c = d["Close"]

    # SuperTrend (10, 3) — proper iterative
    hl2      = (d["High"] + d["Low"]) / 2
    high_low = d["High"] - d["Low"]
    high_pc  = (d["High"] - c.shift()).abs()
    low_pc   = (d["Low"]  - c.shift()).abs()
    tr       = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
    atr      = tr.ewm(span=10, adjust=False).mean()
    upper    = (hl2 + 3 * atr).values.copy()
    lower    = (hl2 - 3 * atr).values.copy()
    close_v  = c.values.copy()
    direction = np.zeros(len(d), dtype=int)

    for i in range(1, len(d)):
        upper[i] = min(upper[i], upper[i-1]) if close_v[i-1] <= upper[i-1] else upper[i]
        lower[i] = max(lower[i], lower[i-1]) if close_v[i-1] >= lower[i-1] else lower[i]
        prev_dir = direction[i-1]
        if prev_dir == -1:
            direction[i] = 1  if close_v[i] > upper[i] else -1
        else:
            direction[i] = -1 if close_v[i] < lower[i] else  1

    d["st_dir"] = direction

    # Filters
    d["ema50"]  = c.ewm(span=50,  adjust=False).mean()
    d["ema200"] = c.ewm(span=200, adjust=False).mean()

    delta = c.diff()
    g     = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    l     = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    d["rsi14"] = 100 - 100 / (1 + g / l)

    d["vol_ma"]    = d["Volume"].rolling(20).mean()
    d["vol_ratio"] = d["Volume"] / (d["vol_ma"] + 1e-9)

    dm_plus  = ((d["High"] - d["High"].shift()).clip(lower=0)).where(
                (d["High"] - d["High"].shift()) > (d["Low"].shift() - d["Low"]), 0)
    dm_minus = ((d["Low"].shift() - d["Low"]).clip(lower=0)).where(
                (d["Low"].shift() - d["Low"]) > (d["High"] - d["High"].shift()), 0)
    atr14s   = tr.ewm(span=14, adjust=False).mean()
    d["adx"] = (100 * (dm_plus.ewm(span=14, adjust=False).mean() -
                       dm_minus.ewm(span=14, adjust=False).mean()).abs() /
                (dm_plus.ewm(span=14, adjust=False).mean() +
                 dm_minus.ewm(span=14, adjust=False).mean() + 1e-9)
               ).ewm(span=14, adjust=False).mean()

    return d.dropna()

print("Building indicators...")
df = build_indicators(df_raw)
print(f"Done. {len(df):,} candles.\n")

# ── BASE SIGNALS ──────────────────────────────────────────────────────────────
flip_long  = (df["st_dir"] ==  1) & (df["st_dir"].shift() == -1)
flip_short = (df["st_dir"] == -1) & (df["st_dir"].shift() ==  1)
base_sig   = pd.Series(0, index=df.index)
base_sig[flip_long]  =  1
base_sig[flip_short] = -1

# ── FILTER VARIANTS ───────────────────────────────────────────────────────────
def apply_filter(sig, df, mode):
    s = sig.copy()
    if mode == "none":
        return s
    if mode == "ema200":
        s[(s ==  1) & (df["Close"] < df["ema200"])] = 0
        s[(s == -1) & (df["Close"] > df["ema200"])] = 0
    elif mode == "ema50":
        s[(s ==  1) & (df["Close"] < df["ema50"])]  = 0
        s[(s == -1) & (df["Close"] > df["ema50"])]  = 0
    elif mode == "rsi":
        s[(s ==  1) & (df["rsi14"] < 45)] = 0
        s[(s == -1) & (df["rsi14"] > 55)] = 0
    elif mode == "adx":
        s[df["adx"] < 20] = 0
    elif mode == "vol":
        s[df["vol_ratio"] < 1.2] = 0
    elif mode == "ema200+adx":
        s[(s ==  1) & (df["Close"] < df["ema200"])] = 0
        s[(s == -1) & (df["Close"] > df["ema200"])] = 0
        s[df["adx"] < 20] = 0
    elif mode == "ema200+rsi":
        s[(s ==  1) & (df["Close"] < df["ema200"])] = 0
        s[(s == -1) & (df["Close"] > df["ema200"])] = 0
        s[(s ==  1) & (df["rsi14"] < 45)] = 0
        s[(s == -1) & (df["rsi14"] > 55)] = 0
    elif mode == "ema50+adx":
        s[(s ==  1) & (df["Close"] < df["ema50"])]  = 0
        s[(s == -1) & (df["Close"] > df["ema50"])]  = 0
        s[df["adx"] < 20] = 0
    elif mode == "adx+vol":
        s[df["adx"] < 20] = 0
        s[df["vol_ratio"] < 1.2] = 0
    elif mode == "ema200+adx+rsi":
        s[(s ==  1) & (df["Close"] < df["ema200"])] = 0
        s[(s == -1) & (df["Close"] > df["ema200"])] = 0
        s[df["adx"] < 20] = 0
        s[(s ==  1) & (df["rsi14"] < 45)] = 0
        s[(s == -1) & (df["rsi14"] > 55)] = 0
    return s

FILTERS = ["none", "ema200", "ema50", "rsi", "adx", "vol",
           "ema200+adx", "ema200+rsi", "ema50+adx", "adx+vol", "ema200+adx+rsi"]

# ── BACKTEST ENGINE ───────────────────────────────────────────────────────────
def backtest(df, signals, sl_pct, rr):
    cap        = CAPITAL
    pos        = None
    entry_fill = sl_p = tp_p = risk = 0.0
    entry_time = None
    trades     = []
    tp_pct     = sl_pct * rr

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
                move    = abs(exit_px - entry_fill)
                sl_dist = abs(entry_fill - sl_p)
                pnl     = risk * (move / sl_dist) * (1 if hit_tp else -1)
                cap     = max(0.0, cap + pnl)
                trades.append({"exit_time": ts, "result": "TP" if hit_tp else "SL",
                               "pnl": pnl, "capital": cap})
                pos = None

        if pos is None and sig != 0 and cap > 0:
            risk = cap * RISK_PCT
            if sig == 1:
                entry_fill = price * (1 + SPREAD)
                sl_p = entry_fill * (1 - sl_pct)
                tp_p = entry_fill * (1 + tp_pct)
            else:
                entry_fill = price * (1 - SPREAD)
                sl_p = entry_fill * (1 + sl_pct)
                tp_p = entry_fill * (1 - tp_pct)
            pos = sig; entry_time = ts

    return pd.DataFrame(trades)

def score(trades_df):
    if trades_df.empty or len(trades_df) < 20:
        return None
    wins  = (trades_df["result"] == "TP").sum()
    wr    = wins / len(trades_df) * 100
    final = trades_df["capital"].iloc[-1]
    tot   = (final - CAPITAL) / CAPITAL * 100
    trades_df["month"] = pd.to_datetime(trades_df["exit_time"]).dt.to_period("M")
    months   = sorted(trades_df["month"].unique())
    prev_cap = CAPITAL
    mo_rets  = []
    for m in months:
        mt = trades_df[trades_df["month"] == m]
        end_cap = mt["capital"].iloc[-1]
        mo_rets.append((end_cap - prev_cap) / prev_cap * 100)
        prev_cap = end_cap
    mr = pd.Series(mo_rets)
    return {"trades": len(trades_df), "win_rate": round(wr, 1),
            "total%": round(tot, 1), "avg_mo%": round(mr.mean(), 2),
            "best_mo%": round(mr.max(), 2), "worst_mo%": round(mr.min(), 2),
            "final": round(final, 2)}

# ── RUN GRID ──────────────────────────────────────────────────────────────────
total = len(FILTERS) * len(SL_GRID) * len(RR_GRID)
print(f"Running {len(FILTERS)} filters × {len(SL_GRID)} SL × {len(RR_GRID)} RR = {total} combos...\n")

all_results = []
run_i = 0
for f in FILTERS:
    sig = apply_filter(base_sig, df, f)
    for sl in SL_GRID:
        for rr in RR_GRID:
            run_i += 1
            t = backtest(df, sig, sl, rr)
            s = score(t)
            if s:
                s["filter"] = f
                s["sl"]     = sl
                s["rr"]     = rr
                s["label"]  = f"ST|{f}|SL{sl*100:.1f}%|RR{rr}"
                all_results.append(s)
            if run_i % 20 == 0:
                print(f"  [{run_i}/{total}]...", end="\r")

print(f"  [{total}/{total}] done.   \n")

# ── RANK ──────────────────────────────────────────────────────────────────────
res = pd.DataFrame(all_results).sort_values("avg_mo%", ascending=False).reset_index(drop=True)
profitable = res[res["avg_mo%"] > 0]

print("═"*105)
print(f"  {'RANK':<5} {'COMBO':<45} {'TRADES':>7} {'WIN%':>6} {'AVG/MO%':>9} {'BEST/MO%':>10} {'WORST/MO%':>11} {'TOTAL%':>9}")
print("═"*105)
for i, row in res.head(20).iterrows():
    marker = " ✓" if row["avg_mo%"] > 0 else "  "
    print(f"  {i+1:<5} {row['label']:<45} {int(row['trades']):>7} {row['win_rate']:>5.1f}% "
          f"{row['avg_mo%']:>9.2f}% {row['best_mo%']:>9.2f}% {row['worst_mo%']:>10.2f}% "
          f"{row['total%']:>9.1f}%{marker}")
print("═"*105)
print(f"\n  Profitable combos (avg/month > 0): {len(profitable)} / {len(res)}\n")

# ── FULL MONTHLY BREAKDOWN FOR TOP 3 ─────────────────────────────────────────
print("═"*75)
print("  PER-MONTH BREAKDOWN — TOP 3 COMBOS  (0.04% spread)")
print("═"*75)

for rank in range(min(3, len(res))):
    row  = res.iloc[rank]
    sig  = apply_filter(base_sig, df, row["filter"])
    t    = backtest(df, sig, row["sl"], row["rr"])
    t["month"] = pd.to_datetime(t["exit_time"]).dt.to_period("M")
    months = sorted(t["month"].unique())
    prev_cap = CAPITAL
    mo_data  = []
    for m in months:
        mt = t[t["month"] == m]
        end_cap = mt["capital"].iloc[-1]
        mo_data.append((str(m), (end_cap - prev_cap) / prev_cap * 100))
        prev_cap = end_cap

    print(f"\n  #{rank+1}  {row['label']}")
    print(f"  Trades: {int(row['trades'])} | Win: {row['win_rate']}% | Total: {row['total%']}% | Final: ${row['final']:,.2f}")
    print(f"  {'Month':<10} {'Return%':>10}  Bar")
    print(f"  {'-'*55}")
    rets = []
    for month, ret in mo_data:
        bar_len = int(abs(ret) / 5)
        bar = ("█" * min(bar_len, 30)) if ret >= 0 else ("▓" * min(bar_len, 30))
        sign = "+" if ret >= 0 else ""
        print(f"  {month:<10} {sign}{ret:>9.2f}%  {bar}")
        rets.append(ret)
    s = pd.Series(rets)
    print(f"  {'-'*55}")
    print(f"  Avg: {s.mean():+.2f}%  |  Best: {s.max():+.2f}%  |  Worst: {s.min():+.2f}%")

print("\n" + "═"*75)
best = res.iloc[0]
print(f"  BEST COMBO : {best['label']}")
print(f"  Avg/Month  : {best['avg_mo%']:+.2f}%  |  Best Month: {best['best_mo%']:+.2f}%  |  Final: ${best['final']:,.2f}")
print("═"*75)
