"""
SESSION-AWARE STRATEGY SEARCH — XAUUSD 5m  |  2 Years
10 strategies × 4 SL × 4 RR = 160 combos
Focus: London Breakout, NY Breakout, ORB, Session-filtered indicators, Round Numbers
Capital: $200  |  Leverage: 1:1000  |  Spread: 0.025%  |  Risk: 1%/trade
"""

import os, sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

CAPITAL  = 200.0
RISK_PCT = 0.01
SPREAD   = 0.00025
LEVERAGE = 1000
MIN_LOT  = 0.01
LOT_STEP = 0.01


# ── DATA ──────────────────────────────────────────────────────────────────────
def load_data():
    search = [
        "../April25/xauusd_5m_2y.csv",
        "xauusd_5m_2y.csv",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "April25", "xauusd_5m_2y.csv"),
    ]
    for path in search:
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
    print("ERROR: xauusd_5m_2y.csv not found. Run test_hyper3.py to download.")
    sys.exit(1)


# ── INDICATORS ────────────────────────────────────────────────────────────────
def build_indicators(df):
    d = df.copy()
    c = d["Close"]

    for n in [9, 21, 50]:
        d[f"ema{n}"] = c.ewm(span=n, adjust=False).mean()
    d["ema240"] = c.ewm(span=240, adjust=False).mean()   # ~20h slow trend

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
    d["atr14"] = tr.ewm(span=14, adjust=False).mean()

    hl2      = (d["High"] + d["Low"]) / 2
    atr10    = tr.ewm(span=10, adjust=False).mean()
    d["st_up"] = hl2 - 3*atr10   # SuperTrend support level
    d["st_dn"] = hl2 + 3*atr10   # SuperTrend resistance level

    h = d.index.hour
    m = d.index.minute
    d["in_london"]    = pd.Series((h >= 7)  & (h < 10), index=d.index)
    d["in_ny"]        = pd.Series((h >= 13) & (h < 17), index=d.index)
    d["in_session"]   = d["in_london"] | d["in_ny"]
    d["in_orb_trade"] = pd.Series(((h == 7) & (m >= 30)) | (h == 8) | (h == 9), index=d.index)
    d["in_ny_trade"]  = pd.Series((h >= 13) & (h < 16), index=d.index)
    d["is_weekday"]   = pd.Series(d.index.dayofweek <= 3, index=d.index)  # Mon-Thu
    d["date_utc"]     = d.index.date

    return d.dropna()


def add_range_data(df):
    """Compute Asian session range, ORB, and Pre-NY range — zero look-ahead."""
    d = df.copy()
    dates = sorted(d["date_utc"].unique())

    a_hi, a_lo = {}, {}
    o_hi, o_lo = {}, {}
    p_hi, p_lo = {}, {}

    for date in dates:
        ds   = str(date)
        prev = str((pd.Timestamp(date) - pd.Timedelta(days=1)).date())

        # Asian session: 22:00 prev day → 07:00 today
        t0, t1 = pd.Timestamp(f"{prev} 22:00:00", tz="UTC"), pd.Timestamp(f"{ds} 07:00:00", tz="UTC")
        m = (d.index >= t0) & (d.index < t1)
        if m.sum() >= 5:
            a_hi[date] = d.loc[m, "High"].max()
            a_lo[date] = d.loc[m, "Low"].min()

        # ORB: 07:00 → 07:30
        t0, t1 = pd.Timestamp(f"{ds} 07:00:00", tz="UTC"), pd.Timestamp(f"{ds} 07:30:00", tz="UTC")
        m = (d.index >= t0) & (d.index < t1)
        if m.sum() >= 3:
            o_hi[date] = d.loc[m, "High"].max()
            o_lo[date] = d.loc[m, "Low"].min()

        # Pre-NY: 11:00 → 13:00
        t0, t1 = pd.Timestamp(f"{ds} 11:00:00", tz="UTC"), pd.Timestamp(f"{ds} 13:00:00", tz="UTC")
        m = (d.index >= t0) & (d.index < t1)
        if m.sum() >= 5:
            p_hi[date] = d.loc[m, "High"].max()
            p_lo[date] = d.loc[m, "Low"].min()

    d["asian_hi"] = d["date_utc"].map(a_hi)
    d["asian_lo"] = d["date_utc"].map(a_lo)
    d["orb_hi"]   = d["date_utc"].map(o_hi)
    d["orb_lo"]   = d["date_utc"].map(o_lo)
    d["preny_hi"] = d["date_utc"].map(p_hi)
    d["preny_lo"] = d["date_utc"].map(p_lo)

    return d


# ── SIGNAL GENERATORS ─────────────────────────────────────────────────────────
def _sig(d, up, dn):
    s = pd.Series(0, index=d.index)
    s[up.fillna(False)] = 1
    s[dn.fillna(False)] = -1
    return s

# 1. London Breakout — first close above/below Asian session range
def sig_london_breakout(d):
    w  = d["in_london"] & d["is_weekday"] & d["asian_hi"].notna()
    up = (d["Close"] > d["asian_hi"]) & (d["Close"].shift() <= d["asian_hi"].shift())
    dn = (d["Close"] < d["asian_lo"]) & (d["Close"].shift() >= d["asian_lo"].shift())
    return _sig(d, w & up, w & dn)

# 2. London Breakout + HTF trend (EMA240) filter
def sig_london_breakout_htf(d):
    w  = d["in_london"] & d["is_weekday"] & d["asian_hi"].notna()
    up = (d["Close"] > d["asian_hi"]) & (d["Close"].shift() <= d["asian_hi"].shift()) & (d["Close"] > d["ema240"])
    dn = (d["Close"] < d["asian_lo"]) & (d["Close"].shift() >= d["asian_lo"].shift()) & (d["Close"] < d["ema240"])
    return _sig(d, w & up, w & dn)

# 3. NY Breakout — first close above/below pre-NY consolidation range
def sig_ny_breakout(d):
    w  = d["in_ny_trade"] & d["is_weekday"] & d["preny_hi"].notna()
    up = (d["Close"] > d["preny_hi"]) & (d["Close"].shift() <= d["preny_hi"].shift())
    dn = (d["Close"] < d["preny_lo"]) & (d["Close"].shift() >= d["preny_lo"].shift())
    return _sig(d, w & up, w & dn)

# 4. Opening Range Breakout — trade London ORB after 07:30
def sig_orb_london(d):
    w  = d["in_orb_trade"] & d["is_weekday"] & d["orb_hi"].notna()
    up = (d["Close"] > d["orb_hi"]) & (d["Close"].shift() <= d["orb_hi"].shift())
    dn = (d["Close"] < d["orb_lo"]) & (d["Close"].shift() >= d["orb_lo"].shift())
    return _sig(d, w & up, w & dn)

# 5. Session EMA 9/21 cross
def sig_session_ema(d):
    up = (d["ema9"] > d["ema21"]) & (d["ema9"].shift() <= d["ema21"].shift())
    dn = (d["ema9"] < d["ema21"]) & (d["ema9"].shift() >= d["ema21"].shift())
    return _sig(d, d["in_session"] & up, d["in_session"] & dn)

# 6. Session MACD histogram cross
def sig_session_macd(d):
    up = (d["macd_hist"] > 0) & (d["macd_hist"].shift() <= 0)
    dn = (d["macd_hist"] < 0) & (d["macd_hist"].shift() >= 0)
    return _sig(d, d["in_session"] & up, d["in_session"] & dn)

# 7. Session SuperTrend cross
def sig_session_supertrend(d):
    bull = d["Close"] > d["st_up"]
    bear = d["Close"] < d["st_dn"]
    up   = bull & ~bull.shift().fillna(False)
    dn   = bear & ~bear.shift().fillna(False)
    return _sig(d, d["in_session"] & up, d["in_session"] & dn)

# 8. Session MACD + HTF trend filter
def sig_session_macd_htf(d):
    up = (d["macd_hist"] > 0) & (d["macd_hist"].shift() <= 0) & (d["Close"] > d["ema240"])
    dn = (d["macd_hist"] < 0) & (d["macd_hist"].shift() >= 0) & (d["Close"] < d["ema240"])
    return _sig(d, d["in_session"] & up, d["in_session"] & dn)

# 9. Session multi-consensus (EMA + MACD + RSI all agree)
def sig_session_multi(d):
    bull = (d["ema9"] > d["ema21"]) & (d["macd_hist"] > 0) & (d["rsi14"] > 50) & (d["Close"] > d["ema50"])
    bear = (d["ema9"] < d["ema21"]) & (d["macd_hist"] < 0) & (d["rsi14"] < 50) & (d["Close"] < d["ema50"])
    up   = bull & ~bull.shift().fillna(False)
    dn   = bear & ~bear.shift().fillna(False)
    return _sig(d, d["in_session"] & up, d["in_session"] & dn)

# 10. Round number ($10 level) cross during sessions
def sig_round_number_session(d):
    rnd    = (d["Close"] / 10).round() * 10
    above  = d["Close"] > rnd          # close is above nearest $10 level
    was_bl = d["Close"].shift() < rnd  # previous close was below it
    was_ab = d["Close"].shift() > rnd  # previous close was above it
    near   = (d["Close"] - rnd).abs() < 2.5   # within $2.5 of level
    up     = near & above & was_bl             # bullish cross of round level
    dn     = near & ~above & was_ab            # bearish cross of round level
    return _sig(d, d["in_session"] & up, d["in_session"] & dn)


STRATEGIES = {
    "London_Breakout":      sig_london_breakout,
    "London_Breakout_HTF":  sig_london_breakout_htf,
    "NY_Breakout":          sig_ny_breakout,
    "ORB_London":           sig_orb_london,
    "Session_EMA_9_21":     sig_session_ema,
    "Session_MACD":         sig_session_macd,
    "Session_SuperTrend":   sig_session_supertrend,
    "Session_MACD_HTF":     sig_session_macd_htf,
    "Session_Multi":        sig_session_multi,
    "Round_Number_Session": sig_round_number_session,
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
RR_GRID = [1.5, 2.0, 3.0, 4.5]

print("="*80)
print("  SESSION-AWARE STRATEGY SEARCH — XAUUSD 5m  |  2 Years")
print(f"  {len(STRATEGIES)} strategies × {len(SL_GRID)} SL × {len(RR_GRID)} RR = {len(STRATEGIES)*len(SL_GRID)*len(RR_GRID)} combos")
print("  Sessions: London 07-10 UTC  |  NY 13-17 UTC  |  Mon-Thu only")
print("="*80 + "\n")

df_raw = load_data()
cutoff  = df_raw.index[-1] - pd.DateOffset(years=2)
df_raw  = df_raw[df_raw.index >= cutoff].copy()
print(f"2-year window: {df_raw.index[0].strftime('%Y-%m-%d')} → {df_raw.index[-1].strftime('%Y-%m-%d')} | {len(df_raw):,} candles\n")

print("Building indicators...", end=" ", flush=True)
df = build_indicators(df_raw)
print("done.")

print("Computing session ranges (Asian / ORB / Pre-NY)...", end=" ", flush=True)
df = add_range_data(df)
print("done.\n")

# Show signal counts per strategy
print("  Signal counts:")
for name, fn in STRATEGIES.items():
    s = fn(df)
    n_long = (s == 1).sum()
    n_short = (s == -1).sum()
    print(f"    {name:<30} long={n_long:4d}  short={n_short:4d}  total={n_long+n_short:4d}")
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
            result_str = "n/a" if not s else f"{s['avg_mo%']:+.2f}%/mo, win {s['win%']}%"
            print(f"  [{run_i:3d}/{total}] {name:<30} SL={sl*100:.1f}% RR={rr} → {result_str}", end="\r")

print(f"\n  [{total}/{total}] done.   \n")

if not results:
    print("No results — strategies generated too few signals for the backtest minimum.")
    sys.exit(0)

res = pd.DataFrame(results).sort_values("avg_mo%", ascending=False).reset_index(drop=True)

# ── RANKINGS ──────────────────────────────────────────────────────────────────
print("═"*125)
print(f"  XAUUSD 5m  |  $200  |  1:1000  |  0.025% Spread  |  Session-Aware Strategies")
print("═"*125)
print(f"  {'#':<4} {'STRATEGY':<45} {'TRADES':>7} {'WIN%':>6} {'AVG/MO%':>9} {'BEST':>8} {'WORST':>9} {'MDD%':>7} {'MDD$':>7} {'FINAL$':>9}")
print("═"*125)
for i, row in res.head(20).iterrows():
    mark = " ✓" if row["avg_mo%"] > 0 else "  "
    print(f"  {i+1:<4} {row['label']:<45} {int(row['trades']):>7} {row['win%']:>5.1f}% "
          f"{row['avg_mo%']:>9.2f}% {row['best_mo%']:>7.2f}% {row['worst_mo%']:>8.2f}% "
          f"{row['mdd%']:>7.1f}% ${row['mdd_$']:>6.2f} ${row['final']:>8.2f}{mark}")
print("═"*125)
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
    print(f"\n  {'Month':<10} {'Trades':>7} {'Win%':>6} {'Return%':>10} {'Capital':>10}  Bar")
    print(f"  {'-'*60}")
    for _, r in mo.iterrows():
        ret = r["ret%"]
        bar = ("█"*min(int(abs(ret)/2), 35)) if ret >= 0 else ("▓"*min(int(abs(ret)/2), 35))
        sgn = "+" if ret >= 0 else ""
        print(f"  {r['month']:<10} {int(r['trades']):>7} {r['win%']:>5.1f}% "
              f"{sgn}{ret:>9.2f}% ${r['capital']:>9.2f}  {bar}")
    print(f"  {'-'*60}")
    print(f"  Avg: {rets.mean():+.2f}%/mo | Best: {rets.max():+.2f}% | Worst: {rets.min():+.2f}%")

print("\n" + "═"*80)
w = res.iloc[0]
print(f"  WINNER : {w['label']}")
print(f"  Avg/Mo : {w['avg_mo%']:+.2f}%  |  Best: {w['best_mo%']:+.2f}%  |  MDD: {w['mdd%']:.1f}% (${w['mdd_$']:.2f})")
print(f"  Final  : ${w['final']:,.2f}  from $200  ({w['total%']:+.1f}% over 2 years)")
print("═"*80)
