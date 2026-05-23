"""
AGGRESSIVE HIGH-CONFLUENCE SEARCH — XAUUSD 5m  |  2 Years
8 strategies × 4 SL × 5 RR = 160 combos
Capital: $200  |  Leverage: 1:1000  |  Spread: 0.025%
Risk: 2% per trade (double previous tests — paper trading mode)
RR up to 10x — targets large moves inside trending sessions
"""

import os, sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

CAPITAL  = 200.0
RISK_PCT = 0.02          # 2% risk per trade — aggressive
SPREAD   = 0.00025
LEVERAGE = 1000
MIN_LOT  = 0.01
LOT_STEP = 0.01


# ── DATA ──────────────────────────────────────────────────────────────────────
def load_data():
    for path in ["../April25/xauusd_5m_2y.csv", "xauusd_5m_2y.csv",
                 os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "April25", "xauusd_5m_2y.csv")]:
        if os.path.exists(path):
            print(f"Loading {path}...", end=" ")
            df = pd.read_csv(path, parse_dates=["open_time"])
            df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
            df = df[~df["open_time"].duplicated(keep="first")]
            df.sort_values("open_time", inplace=True)
            df.set_index("open_time", inplace=True)
            df.dropna(inplace=True)
            print(f"{len(df):,} candles | {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}")
            return df
    print("ERROR: xauusd_5m_2y.csv not found.")
    sys.exit(1)


# ── INDICATORS ────────────────────────────────────────────────────────────────
def build_indicators(df):
    d = df.copy()
    c = d["Close"]

    for n in [9, 21, 50, 200]:
        d[f"ema{n}"] = c.ewm(span=n, adjust=False).mean()
    d["ema240"] = c.ewm(span=240, adjust=False).mean()

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    d["macd"]      = ema12 - ema26
    d["macd_sig"]  = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_sig"]

    delta = c.diff()
    g = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    l = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    d["rsi14"] = 100 - 100 / (1 + g / l)

    hl  = d["High"] - d["Low"]
    hpc = (d["High"] - c.shift()).abs()
    lpc = (d["Low"]  - c.shift()).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    d["atr14"]     = tr.ewm(span=14, adjust=False).mean()
    d["atr_ma30"]  = d["atr14"].rolling(30).mean()

    # Candle body size
    d["body"]      = (c - d["Open"]).abs()
    d["body_bull"] = c > d["Open"]
    d["body_bear"] = c < d["Open"]

    h = d.index.hour
    m = d.index.minute
    d["in_london"]  = pd.Series((h >= 7)  & (h < 10), index=d.index)
    d["in_ny"]      = pd.Series((h >= 13) & (h < 17), index=d.index)
    d["in_session"] = d["in_london"] | d["in_ny"]
    d["is_weekday"] = pd.Series(d.index.dayofweek <= 3, index=d.index)
    d["date_utc"]   = d.index.date

    return d.dropna()


def add_range_data(df):
    d = df.copy()
    dates = sorted(d["date_utc"].unique())
    a_hi, a_lo = {}, {}
    for date in dates:
        ds   = str(date)
        prev = str((pd.Timestamp(date) - pd.Timedelta(days=1)).date())
        t0 = pd.Timestamp(f"{prev} 22:00:00", tz="UTC")
        t1 = pd.Timestamp(f"{ds} 07:00:00", tz="UTC")
        m = (d.index >= t0) & (d.index < t1)
        if m.sum() >= 5:
            a_hi[date] = d.loc[m, "High"].max()
            a_lo[date] = d.loc[m, "Low"].min()
    d["asian_hi"] = d["date_utc"].map(a_hi)
    d["asian_lo"] = d["date_utc"].map(a_lo)
    return d


# ── SIGNAL GENERATORS ─────────────────────────────────────────────────────────
def _sig(d, up, dn):
    s = pd.Series(0, index=d.index)
    s[up.fillna(False)] = 1
    s[dn.fillna(False)] = -1
    return s

# 1. EMA Pullback — price dips below EMA9 then bounces back above in a full EMA stack trend
def sig_ema_pullback(d):
    bull_stack   = (d["ema9"] > d["ema21"]) & (d["ema21"] > d["ema50"]) & (d["Close"] > d["ema240"])
    bear_stack   = (d["ema9"] < d["ema21"]) & (d["ema21"] < d["ema50"]) & (d["Close"] < d["ema240"])
    bounce_up    = (d["Close"] > d["ema9"]) & (d["Close"].shift() < d["ema9"].shift())
    bounce_dn    = (d["Close"] < d["ema9"]) & (d["Close"].shift() > d["ema9"].shift())
    rsi_ok_up    = (d["rsi14"] > 45) & (d["rsi14"] < 68)
    rsi_ok_dn    = (d["rsi14"] < 55) & (d["rsi14"] > 32)
    return _sig(d,
        d["in_session"] & d["is_weekday"] & bull_stack & bounce_up & rsi_ok_up,
        d["in_session"] & d["is_weekday"] & bear_stack & bounce_dn & rsi_ok_dn)

# 2. Full Stack — all EMAs aligned + MACD + RSI all firing at once (state change entry)
def sig_full_stack(d):
    bull = (d["ema9"]>d["ema21"]) & (d["ema21"]>d["ema50"]) & (d["Close"]>d["ema240"]) \
         & (d["macd_hist"]>0) & (d["rsi14"]>52) & (d["rsi14"]<72)
    bear = (d["ema9"]<d["ema21"]) & (d["ema21"]<d["ema50"]) & (d["Close"]<d["ema240"]) \
         & (d["macd_hist"]<0) & (d["rsi14"]<48) & (d["rsi14"]>28)
    up = bull & ~bull.shift().fillna(False)
    dn = bear & ~bear.shift().fillna(False)
    return _sig(d, d["in_session"] & d["is_weekday"] & up, d["in_session"] & d["is_weekday"] & dn)

# 3. Engulfing candle in trend direction during session
def sig_engulfing_trend(d):
    prev_bull = d["Close"].shift() > d["Open"].shift()
    prev_bear = d["Close"].shift() < d["Open"].shift()
    bull_eng  = d["body_bull"] & prev_bear \
              & (d["Close"] > d["Open"].shift()) & (d["Open"] < d["Close"].shift()) \
              & (d["Close"] > d["ema240"])
    bear_eng  = d["body_bear"] & prev_bull \
              & (d["Close"] < d["Open"].shift()) & (d["Open"] > d["Close"].shift()) \
              & (d["Close"] < d["ema240"])
    # Require candle body > 0.5 ATR to avoid tiny engulfs
    big_enough = d["body"] > d["atr14"] * 0.5
    return _sig(d,
        d["in_session"] & d["is_weekday"] & bull_eng & big_enough,
        d["in_session"] & d["is_weekday"] & bear_eng & big_enough)

# 4. ATR surge + 12-bar range breakout + HTF trend — catches big momentum moves
def sig_atr_surge_break(d):
    atr_surge  = d["atr14"] > d["atr_ma30"] * 1.2
    range_hi   = d["High"].rolling(12).max().shift()
    range_lo   = d["Low"].rolling(12).min().shift()
    break_up   = (d["Close"] > range_hi) & (d["Close"].shift() <= range_hi.shift())
    break_dn   = (d["Close"] < range_lo) & (d["Close"].shift() >= range_lo.shift())
    htf_bull   = d["Close"] > d["ema240"]
    htf_bear   = d["Close"] < d["ema240"]
    return _sig(d,
        d["in_session"] & d["is_weekday"] & break_up & htf_bull & atr_surge,
        d["in_session"] & d["is_weekday"] & break_dn & htf_bear & atr_surge)

# 5. 8-bar swing break with HTF trend + MACD confirmation
def sig_swing_break_htf(d):
    swing_hi   = d["High"].rolling(8).max().shift()
    swing_lo   = d["Low"].rolling(8).min().shift()
    break_up   = (d["Close"] > swing_hi) & (d["Close"].shift() <= swing_hi.shift())
    break_dn   = (d["Close"] < swing_lo) & (d["Close"].shift() >= swing_lo.shift())
    htf_bull   = (d["Close"] > d["ema240"]) & (d["macd_hist"] > 0)
    htf_bear   = (d["Close"] < d["ema240"]) & (d["macd_hist"] < 0)
    return _sig(d,
        d["in_session"] & d["is_weekday"] & break_up & htf_bull,
        d["in_session"] & d["is_weekday"] & break_dn & htf_bear)

# 6. Session Multi + HTF — previous winner with added EMA240 filter
def sig_session_multi_htf(d):
    bull = (d["ema9"]>d["ema21"]) & (d["macd_hist"]>0) & (d["rsi14"]>50) \
         & (d["Close"]>d["ema50"]) & (d["Close"]>d["ema240"])
    bear = (d["ema9"]<d["ema21"]) & (d["macd_hist"]<0) & (d["rsi14"]<50) \
         & (d["Close"]<d["ema50"]) & (d["Close"]<d["ema240"])
    up = bull & ~bull.shift().fillna(False)
    dn = bear & ~bear.shift().fillna(False)
    return _sig(d, d["in_session"] & up, d["in_session"] & dn)

# 7. London breakout of Asian range with full EMA stack alignment
def sig_london_breakout_stack(d):
    w        = d["in_london"] & d["is_weekday"] & d["asian_hi"].notna()
    bull_stk = (d["ema9"] > d["ema21"]) & (d["Close"] > d["ema240"])
    bear_stk = (d["ema9"] < d["ema21"]) & (d["Close"] < d["ema240"])
    break_up = (d["Close"] > d["asian_hi"]) & (d["Close"].shift() <= d["asian_hi"].shift())
    break_dn = (d["Close"] < d["asian_lo"]) & (d["Close"].shift() >= d["asian_lo"].shift())
    return _sig(d, w & break_up & bull_stk, w & break_dn & bear_stk)

# 8. Strong momentum candle — body > 1.5 ATR in trend direction during session
def sig_strong_candle(d):
    strong_bull = d["body_bull"] & (d["body"] > d["atr14"] * 1.5) & (d["Close"] > d["ema240"]) \
                & (d["rsi14"] > 50) & (d["rsi14"] < 75)
    strong_bear = d["body_bear"] & (d["body"] > d["atr14"] * 1.5) & (d["Close"] < d["ema240"]) \
                & (d["rsi14"] < 50) & (d["rsi14"] > 25)
    return _sig(d,
        d["in_session"] & d["is_weekday"] & strong_bull,
        d["in_session"] & d["is_weekday"] & strong_bear)


STRATEGIES = {
    "EMA_Pullback":         sig_ema_pullback,
    "Full_Stack":           sig_full_stack,
    "Engulfing_Trend":      sig_engulfing_trend,
    "ATR_Surge_Break":      sig_atr_surge_break,
    "Swing_Break_HTF":      sig_swing_break_htf,
    "Session_Multi_HTF":    sig_session_multi_htf,
    "London_Break_Stack":   sig_london_breakout_stack,
    "Strong_Candle":        sig_strong_candle,
}


# ── LOT CALCULATOR ────────────────────────────────────────────────────────────
def calc_lots(capital, price, sl_pct):
    risk_usd = capital * RISK_PCT
    sl_usd   = price * sl_pct
    lots     = max(MIN_LOT, round(risk_usd / sl_usd / LOT_STEP) * LOT_STEP)
    if lots * price / LEVERAGE > capital:
        lots = max(MIN_LOT, round(capital * LEVERAGE / price / LOT_STEP) * LOT_STEP)
    return round(lots, 2)


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
            hit_tp = (pos==1 and price >= tp_p) or (pos==-1 and price <= tp_p)
            hit_sl = (pos==1 and price <= sl_p) or (pos==-1 and price >= sl_p)
            if hit_tp or hit_sl:
                exit_px = tp_p if hit_tp else sl_p
                pnl     = lots * abs(exit_px - entry_fill) * (1 if hit_tp else -1)
                cap     = max(0.0, cap + pnl)
                peak    = max(peak, cap)
                trades.append({
                    "exit_time": ts,
                    "result":    "TP" if hit_tp else "SL",
                    "lots":      lots,
                    "pnl":       round(pnl, 4),
                    "capital":   round(cap, 4),
                    "peak":      round(peak, 4),
                })
                pos = None

        if pos is None and sig != 0 and cap > 0:
            lots       = calc_lots(cap, price, sl_pct)
            entry_fill = price*(1+SPREAD) if sig==1 else price*(1-SPREAD)
            sl_p       = entry_fill*(1-sl_pct) if sig==1 else entry_fill*(1+sl_pct)
            tp_p       = entry_fill*(1+tp_pct) if sig==1 else entry_fill*(1-tp_pct)
            pos        = sig

    return pd.DataFrame(trades)


# ── STATS ─────────────────────────────────────────────────────────────────────
def calc_stats(t):
    if t.empty or len(t) < 5:
        return None
    wins    = (t["result"]=="TP").sum()
    final   = t["capital"].iloc[-1]
    tot     = (final - CAPITAL) / CAPITAL * 100
    caps    = pd.concat([pd.Series([CAPITAL]), t["capital"].reset_index(drop=True)])
    peak    = caps.cummax()
    mdd     = ((caps - peak) / peak * 100).min()
    mdd_usd = (caps - peak).min()
    t = t.copy()
    t["month"] = pd.to_datetime(t["exit_time"]).dt.to_period("M")
    prev_cap = CAPITAL
    mo_rets  = []
    for m in sorted(t["month"].unique()):
        mt = t[t["month"]==m]
        end = mt["capital"].iloc[-1]
        mo_rets.append((end - prev_cap) / prev_cap * 100)
        prev_cap = end
    mr = pd.Series(mo_rets)
    return {
        "trades":    len(t),
        "win%":      round(wins/len(t)*100, 1),
        "total%":    round(tot, 1),
        "avg_mo%":   round(mr.mean(), 2),
        "best_mo%":  round(mr.max(), 2),
        "worst_mo%": round(mr.min(), 2),
        "mdd%":      round(mdd, 1),
        "mdd_$":     round(abs(mdd_usd), 2),
        "final":     round(final, 2),
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────
SL_GRID = [0.003, 0.005, 0.008, 0.012]
RR_GRID = [2.0, 3.0, 5.0, 7.0, 10.0]

print("="*80)
print("  AGGRESSIVE HIGH-CONFLUENCE SEARCH — XAUUSD 5m  |  2 Years")
print(f"  {len(STRATEGIES)} strategies × {len(SL_GRID)} SL × {len(RR_GRID)} RR = {len(STRATEGIES)*len(SL_GRID)*len(RR_GRID)} combos")
print(f"  Risk per trade: {RISK_PCT*100:.0f}%  |  Max RR: {max(RR_GRID)}x  |  Sessions: London+NY  |  Mon-Thu")
print("="*80 + "\n")

df_raw = load_data()
cutoff  = df_raw.index[-1] - pd.DateOffset(years=2)
df_raw  = df_raw[df_raw.index >= cutoff].copy()
print(f"2-year window: {df_raw.index[0].strftime('%Y-%m-%d')} → {df_raw.index[-1].strftime('%Y-%m-%d')} | {len(df_raw):,} candles\n")

print("Building indicators...", end=" ", flush=True)
df = build_indicators(df_raw)
print("done.")
print("Computing Asian session ranges...", end=" ", flush=True)
df = add_range_data(df)
print("done.\n")

print("  Signal counts per strategy:")
for name, fn in STRATEGIES.items():
    s = fn(df)
    n_l, n_s = (s==1).sum(), (s==-1).sum()
    print(f"    {name:<25} long={n_l:5d}  short={n_s:5d}  total={n_l+n_s:5d}")
print()

total   = len(STRATEGIES) * len(SL_GRID) * len(RR_GRID)
results = []
run_i   = 0
print(f"Testing {total} combos...\n")

for name, fn in STRATEGIES.items():
    sigs = fn(df)
    for sl in SL_GRID:
        for rr in RR_GRID:
            run_i += 1
            t = backtest(df, sigs, sl, rr)
            s = calc_stats(t)
            if s:
                s["label"] = f"{name}|SL{sl*100:.1f}%|RR{rr}"
                s["name"]  = name
                s["sl"]    = sl
                s["rr"]    = rr
                results.append(s)
            res_str = "n/a" if not s else f"{s['avg_mo%']:+.2f}%/mo win={s['win%']}%"
            print(f"  [{run_i:3d}/{total}] {name:<25} SL={sl*100:.1f}% RR={rr:4.1f} → {res_str}", end="\r")

print(f"\n  [{total}/{total}] done.   \n")

if not results:
    print("No results — all strategies generated too few signals.")
    sys.exit(0)

res = pd.DataFrame(results).sort_values("avg_mo%", ascending=False).reset_index(drop=True)

# ── RANKINGS ──────────────────────────────────────────────────────────────────
print("═"*130)
print(f"  XAUUSD 5m  |  $200  |  1:1000  |  0.025% Spread  |  2% Risk/Trade  |  High-Confluence Strategies")
print("═"*130)
print(f"  {'#':<4} {'STRATEGY':<40} {'TRADES':>7} {'WIN%':>6} {'AVG/MO%':>9} {'BEST':>8} {'WORST':>9} {'MDD%':>7} {'MDD$':>7} {'FINAL$':>10}")
print("═"*130)
for i, row in res.head(20).iterrows():
    mark = " ✓" if row["avg_mo%"] > 0 else "  "
    print(f"  {i+1:<4} {row['label']:<40} {int(row['trades']):>7} {row['win%']:>5.1f}% "
          f"{row['avg_mo%']:>9.2f}% {row['best_mo%']:>7.2f}% {row['worst_mo%']:>8.2f}% "
          f"{row['mdd%']:>7.1f}% ${row['mdd_$']:>6.2f} ${row['final']:>9.2f}{mark}")
print("═"*130)
print(f"\n  Profitable: {(res['avg_mo%']>0).sum()} / {len(res)}")
print(f"  Survived (final > $200): {(res['final']>200).sum()} / {len(res)}\n")

# ── TOP 5 MONTHLY BREAKDOWN ───────────────────────────────────────────────────
print("═"*80)
print("  PER-MONTH BREAKDOWN — TOP 5 STRATEGIES")
print("═"*80)

for rank in range(min(5, len(res))):
    row  = res.iloc[rank]
    sigs = STRATEGIES[row["name"]](df)
    t    = backtest(df, sigs, row["sl"], row["rr"])

    caps    = pd.concat([pd.Series([CAPITAL]), t["capital"].reset_index(drop=True)])
    peak    = caps.cummax()
    mdd     = ((caps - peak)/peak*100).min()
    mdd_usd = (caps - peak).min()

    t = t.copy()
    t["month"] = pd.to_datetime(t["exit_time"]).dt.to_period("M")
    prev_cap = CAPITAL
    mo_rows  = []
    for m in sorted(t["month"].unique()):
        mt  = t[t["month"]==m]
        end = mt["capital"].iloc[-1]
        mo_rows.append({
            "month":   str(m),
            "trades":  len(mt),
            "win%":    (mt["result"]=="TP").sum()/len(mt)*100,
            "ret%":    (end - prev_cap)/prev_cap*100,
            "capital": end,
        })
        prev_cap = end

    mo   = pd.DataFrame(mo_rows)
    rets = mo["ret%"]

    print(f"\n  #{rank+1}  {row['label']}")
    print(f"  Trades: {int(row['trades'])} | Win: {row['win%']}% | Final: ${row['final']:,.2f} | Total: {row['total%']:+.1f}%")
    print(f"  Max Drawdown: {mdd:.1f}%  (${abs(mdd_usd):.2f})  |  Positive months: {(rets>0).sum()}/{len(rets)}")
    print(f"\n  {'Month':<10} {'Trades':>7} {'Win%':>6} {'Return%':>10} {'Capital':>11}  Bar")
    print(f"  {'-'*64}")
    for _, r in mo.iterrows():
        ret = r["ret%"]
        bar = ("█"*min(int(abs(ret)/3), 35)) if ret >= 0 else ("▓"*min(int(abs(ret)/3), 35))
        sgn = "+" if ret >= 0 else ""
        print(f"  {r['month']:<10} {int(r['trades']):>7} {r['win%']:>5.1f}% "
              f"{sgn}{ret:>9.2f}% ${r['capital']:>10.2f}  {bar}")
    print(f"  {'-'*64}")
    print(f"  Avg: {rets.mean():+.2f}%/mo | Best: {rets.max():+.2f}% | Worst: {rets.min():+.2f}%")

print("\n" + "═"*80)
w = res.iloc[0]
print(f"  WINNER : {w['label']}")
print(f"  Avg/Mo : {w['avg_mo%']:+.2f}%  |  Best: {w['best_mo%']:+.2f}%  |  MDD: {w['mdd%']:.1f}% (${w['mdd_$']:.2f})")
print(f"  Final  : ${w['final']:,.2f}  from $200  ({w['total%']:+.1f}% over 2 years)")
print(f"\n  NOTE: 2% risk/trade — aggressive mode. For live use dial back to 1%.")
print("═"*80)
