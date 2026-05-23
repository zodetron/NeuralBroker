"""
SUPERTREND PARAM SEARCH — BTC/USDT 5m  |  2 Years  |  Spread: 0.04%
Capital: $400  |  Leverage: 1:1000  |  Min Lot: 0.01  |  Risk: 1%/trade
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from data_loader import load_data

# ── CONFIG ────────────────────────────────────────────────────────────────────
CAPITAL  = 400.0
RISK_PCT = 0.01
SPREAD   = 0.0004
LEVERAGE = 1000
MIN_LOT  = 0.01
LOT_STEP = 0.01

SL_GRID  = [0.003, 0.005, 0.008, 0.010, 0.015]
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
        direction[i] = (1 if close_v[i] > upper[i] else -1) if prev_dir == -1 else \
                       (-1 if close_v[i] < lower[i] else 1)

    d["st_dir"] = direction

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

# ── LOT CALCULATOR ────────────────────────────────────────────────────────────
def calc_lots(capital, price, sl_pct):
    risk_usd       = capital * RISK_PCT
    sl_usd_per_lot = price * sl_pct
    lots_ideal     = risk_usd / sl_usd_per_lot
    lots           = max(MIN_LOT, round(lots_ideal / LOT_STEP) * LOT_STEP)
    lots           = round(lots, 2)
    margin_req     = lots * price / LEVERAGE
    if margin_req > capital:
        lots = max(MIN_LOT, round((capital * LEVERAGE / price) / LOT_STEP) * LOT_STEP)
        lots = round(lots, 2)
    return lots

# ── BACKTEST ENGINE ───────────────────────────────────────────────────────────
def backtest(df, signals, sl_pct, rr):
    cap        = CAPITAL
    pos        = None
    entry_fill = sl_p = tp_p = lots = 0.0
    trades     = []
    tp_pct     = sl_pct * rr
    peak       = cap

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
                dd_pct  = (cap - peak) / peak * 100   # negative when in drawdown
                trades.append({
                    "exit_time": ts,
                    "result":    "TP" if hit_tp else "SL",
                    "lots":      lots,
                    "pnl":       round(pnl, 4),
                    "capital":   round(cap, 4),
                    "drawdown%": round(dd_pct, 2),
                })
                pos = None

        if pos is None and sig != 0 and cap > 0:
            lots       = calc_lots(cap, price, sl_pct)
            entry_fill = price * (1 + SPREAD) if sig == 1 else price * (1 - SPREAD)
            sl_p       = entry_fill * (1 - sl_pct) if sig == 1 else entry_fill * (1 + sl_pct)
            tp_p       = entry_fill * (1 + tp_pct) if sig == 1 else entry_fill * (1 - tp_pct)
            pos        = sig

    return pd.DataFrame(trades)

# ── MAX DRAWDOWN ──────────────────────────────────────────────────────────────
def max_drawdown(trades_df):
    if trades_df.empty:
        return 0.0, 0.0, 0, 0
    caps  = pd.concat([pd.Series([CAPITAL]), trades_df["capital"].reset_index(drop=True)])
    peak  = caps.cummax()
    dd    = (caps - peak) / peak * 100
    mdd   = dd.min()
    # dollar drawdown
    mdd_usd = (caps - peak).min()
    # longest drawdown duration (in trades)
    in_dd = dd < 0
    max_dur, cur_dur = 0, 0
    for v in in_dd:
        cur_dur = cur_dur + 1 if v else 0
        max_dur = max(max_dur, cur_dur)
    # recovery factor
    final_gain = trades_df["capital"].iloc[-1] - CAPITAL
    rf = final_gain / abs(mdd_usd) if mdd_usd != 0 else 0
    return round(mdd, 2), round(mdd_usd, 2), max_dur, round(rf, 2)

# ── SCORE ─────────────────────────────────────────────────────────────────────
def score(trades_df):
    if trades_df.empty or len(trades_df) < 20:
        return None
    wins  = (trades_df["result"] == "TP").sum()
    wr    = wins / len(trades_df) * 100
    final = trades_df["capital"].iloc[-1]
    tot   = (final - CAPITAL) / CAPITAL * 100
    mdd, mdd_usd, dd_dur, rf = max_drawdown(trades_df)
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
    return {
        "trades":    len(trades_df),
        "win_rate":  round(wr, 1),
        "total%":    round(tot, 1),
        "avg_mo%":   round(mr.mean(), 2),
        "best_mo%":  round(mr.max(), 2),
        "worst_mo%": round(mr.min(), 2),
        "mdd%":      mdd,
        "mdd_usd":   mdd_usd,
        "dd_dur":    dd_dur,
        "rec_factor":rf,
        "final":     round(final, 2),
    }

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

# ── RANK TABLE ────────────────────────────────────────────────────────────────
res = pd.DataFrame(all_results).sort_values("avg_mo%", ascending=False).reset_index(drop=True)
profitable = res[res["avg_mo%"] > 0]

print("═"*125)
print(f"  {'#':<4} {'COMBO':<42} {'TRADES':>7} {'WIN%':>6} {'AVG/MO%':>9} {'BEST/MO':>9} {'WORST/MO':>9} {'MDD%':>7} {'MDD_$':>8} {'TOTAL%':>8}")
print("═"*125)
for i, row in res.head(20).iterrows():
    mark = " ✓" if row["avg_mo%"] > 0 else "  "
    print(f"  {i+1:<4} {row['label']:<42} {int(row['trades']):>7} {row['win_rate']:>5.1f}% "
          f"{row['avg_mo%']:>9.2f}% {row['best_mo%']:>8.2f}% {row['worst_mo%']:>8.2f}% "
          f"{row['mdd%']:>7.1f}% ${row['mdd_usd']:>7.2f} {row['total%']:>8.1f}%{mark}")
print("═"*125)
print(f"\n  Profitable combos: {len(profitable)} / {len(res)}\n")

# ── TOP 3 FULL BREAKDOWN ──────────────────────────────────────────────────────
print("═"*80)
print(f"  PER-MONTH BREAKDOWN — TOP 3  |  $400 start | 0.04% spread | 1:1000")
print("═"*80)

for rank in range(min(3, len(res))):
    row = res.iloc[rank]
    sig = apply_filter(base_sig, df, row["filter"])
    t   = backtest(df, sig, row["sl"], row["rr"])
    mdd, mdd_usd, dd_dur, rf = max_drawdown(t)

    t["month"] = pd.to_datetime(t["exit_time"]).dt.to_period("M")
    months   = sorted(t["month"].unique())
    prev_cap = CAPITAL
    mo_rows  = []
    for m in months:
        mt      = t[t["month"] == m]
        end_cap = mt["capital"].iloc[-1]
        ret_pct = (end_cap - prev_cap) / prev_cap * 100
        mt_wins = (mt["result"] == "TP").sum()
        mo_rows.append({
            "month":    str(m),
            "trades":   len(mt),
            "win%":     mt_wins / len(mt) * 100,
            "ret%":     ret_pct,
            "end_cap":  end_cap,
            "avg_lot":  mt["lots"].mean(),
        })
        prev_cap = end_cap

    mo_df = pd.DataFrame(mo_rows)
    rets  = mo_df["ret%"]

    print(f"\n  #{rank+1}  {row['label']}")
    print(f"  Trades: {int(row['trades'])} | Win: {row['win_rate']}% | Final: ${row['final']:,.2f} | Total: {row['total%']:+.1f}%")
    print(f"  Max Drawdown: {mdd:.1f}%  (${abs(mdd_usd):.2f})  |  Longest DD streak: {dd_dur} trades  |  Recovery Factor: {rf:.2f}")
    print(f"\n  {'Month':<10} {'Trades':>7} {'Win%':>6} {'Return%':>10} {'Capital':>10} {'AvgLot':>8}  Bar")
    print(f"  {'-'*75}")
    for _, r in mo_df.iterrows():
        ret = r["ret%"]
        bar = ("█" * min(int(abs(ret)/4), 28)) if ret >= 0 else ("▓" * min(int(abs(ret)/4), 28))
        sign = "+" if ret >= 0 else ""
        print(f"  {r['month']:<10} {int(r['trades']):>7} {r['win%']:>5.1f}% {sign}{ret:>9.2f}% "
              f"${r['end_cap']:>9.2f} {r['avg_lot']:>8.3f}  {bar}")
    print(f"  {'-'*75}")
    print(f"  {'AVG':<10} {int(mo_df['trades'].mean()):>7}        {rets.mean():>+10.2f}%")
    print(f"  Best: {rets.max():+.2f}%  |  Worst: {rets.min():+.2f}%  |  Positive months: {(rets > 0).sum()}/{len(rets)}")

print("\n" + "═"*80)
best = res.iloc[0]
print(f"  WINNER  : {best['label']}")
print(f"  Avg/Mo  : {best['avg_mo%']:+.2f}% | Best: {best['best_mo%']:+.2f}% | MDD: {best['mdd%']:.1f}% (${abs(best['mdd_usd']):.2f})")
print(f"  Final   : ${best['final']:,.2f}  from $400  ({best['total%']:+.1f}% total)")
print("═"*80)
