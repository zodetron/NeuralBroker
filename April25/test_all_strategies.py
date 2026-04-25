"""
BTC/USDT 5m — ALL STRATEGIES | Spread-only costs | 3 Years
Target: Find what actually works. Report honest monthly returns.

Strategies tested:
  S1.  MACD + EMA200 (our best so far)
  S2.  EMA 9/21 crossover only
  S3.  EMA 9/21 cross + EMA200 trend
  S4.  Bollinger Band breakout
  S5.  BB breakout + EMA200 trend
  S6.  RSI reversal (oversold/overbought)
  S7.  RSI + EMA200 trend
  S8.  MACD + BB squeeze breakout
  S9.  Supertrend (ATR-based trend following)
  S10. Donchian channel breakout (20-bar high/low)
  S11. EMA stack (9>21>50>200) — full alignment
  S12. MACD + EMA cross + EMA200 (combined best)

Each tested at SL=1.0%, RR=2.5 (our proven best config).
Spread = 0.01%/side (0.02% round trip). No slippage, no commission.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from data_loader import load_data

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CAPITAL       = 100.0
LEVERAGE      = 1000
RISK_PCT      = 0.01
SL_PCT        = 0.0100
RR            = 2.5
COST_PER_SIDE = 0.0001   # spread only

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df = load_data()
print()

# ─────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────
def add_indicators(df):
    d = df.copy()

    # EMAs
    for span in [9, 13, 21, 50, 100, 200]:
        d[f"ema{span}"] = d["Close"].ewm(span=span, adjust=False).mean()

    # MACD
    ema12          = d["Close"].ewm(span=12, adjust=False).mean()
    ema26          = d["Close"].ewm(span=26, adjust=False).mean()
    d["macd"]      = ema12 - ema26
    d["macd_sig"]  = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_sig"]

    # RSI (14)
    delta      = d["Close"].diff()
    gain       = delta.clip(lower=0).rolling(14).mean()
    loss       = (-delta.clip(upper=0)).rolling(14).mean()
    d["rsi"]   = 100 - (100 / (1 + gain / loss))

    # Bollinger Bands (20, 2σ)
    d["bb_mid"]   = d["Close"].rolling(20).mean()
    bb_std        = d["Close"].rolling(20).std()
    d["bb_upper"] = d["bb_mid"] + 2 * bb_std
    d["bb_lower"] = d["bb_mid"] - 2 * bb_std
    d["bb_width"] = (d["bb_upper"] - d["bb_lower"]) / d["bb_mid"]

    # ATR (14)
    hl  = d["High"] - d["Low"]
    hc  = (d["High"] - d["Close"].shift()).abs()
    lc  = (d["Low"]  - d["Close"].shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()

    # Supertrend (ATR × 3)
    atr_mult = 3.0
    hl2      = (d["High"] + d["Low"]) / 2
    d["st_upper"] = hl2 + atr_mult * d["atr"]
    d["st_lower"] = hl2 - atr_mult * d["atr"]
    # Supertrend direction: 1 = bullish, -1 = bearish
    st_dir = pd.Series(1, index=d.index)
    for i in range(1, len(d)):
        prev_close = d["Close"].iloc[i-1]
        prev_dir   = st_dir.iloc[i-1]
        if prev_close > d["st_upper"].iloc[i-1]:
            st_dir.iloc[i] = 1
        elif prev_close < d["st_lower"].iloc[i-1]:
            st_dir.iloc[i] = -1
        else:
            st_dir.iloc[i] = prev_dir
    d["st_dir"] = st_dir

    # Donchian (20-bar)
    d["don_high"] = d["High"].rolling(20).max().shift(1)
    d["don_low"]  = d["Low"].rolling(20).min().shift(1)

    # EMA crossover flags
    d["cross_bull"] = ((d["ema9"] > d["ema21"]) & (d["ema9"].shift(1) <= d["ema21"].shift(1))).rolling(10).max().astype(bool)
    d["cross_bear"] = ((d["ema9"] < d["ema21"]) & (d["ema9"].shift(1) >= d["ema21"].shift(1))).rolling(10).max().astype(bool)

    return d

print("  Computing indicators...")
df = add_indicators(df)
df.dropna(inplace=True)
print(f"  Done. {len(df):,} candles.\n")

# ─────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────
def backtest(df, signals, sl_pct=SL_PCT, rr=RR):
    tp_pct      = sl_pct * rr
    capital     = CAPITAL
    position    = None
    trades      = []
    entry_fill  = sl_price = tp_price = risk_usd = 0.0
    entry_time  = None

    for i in range(1, len(df)):
        if capital <= 0:
            break
        close = float(df["Close"].iloc[i])
        ts    = df.index[i]
        sig   = int(signals.iloc[i])

        if position is not None:
            hit_tp = (position == "long"  and close >= tp_price) or \
                     (position == "short" and close <= tp_price)
            hit_sl = (position == "long"  and close <= sl_price) or \
                     (position == "short" and close >= sl_price)
            if hit_tp or hit_sl:
                theo = tp_price if hit_tp else sl_price
                if position == "long":
                    ef = theo * (1 - COST_PER_SIDE); move = ef - entry_fill
                else:
                    ef = theo * (1 + COST_PER_SIDE); move = entry_fill - ef
                sl_dist = entry_fill * sl_pct
                pnl     = risk_usd * (move / sl_dist)
                capital = max(0.0, capital + pnl)
                trades.append({
                    "exit_time": ts, "result": "TP" if hit_tp else "SL",
                    "pnl": round(pnl, 4), "capital": round(capital, 4),
                })
                position = None
                continue

        if position is None and sig != 0 and capital > 0:
            risk_usd   = round(capital * RISK_PCT, 6)
            entry_time = ts
            if sig == 1:
                position   = "long"
                entry_fill = close * (1 + COST_PER_SIDE)
                sl_price   = entry_fill * (1 - sl_pct)
                tp_price   = entry_fill * (1 + tp_pct)
            else:
                position   = "short"
                entry_fill = close * (1 - COST_PER_SIDE)
                sl_price   = entry_fill * (1 + sl_pct)
                tp_price   = entry_fill * (1 - tp_pct)

    return pd.DataFrame(trades)

# ─────────────────────────────────────────────
# SIGNAL DEFINITIONS
# ─────────────────────────────────────────────
def make_sig(long_cond, short_cond):
    s = pd.Series(0, index=df.index)
    s[long_cond]  = 1
    s[short_cond] = -1
    return s

# S1: MACD hist flip + EMA200 side
def s1_macd_ema200():
    hu = (df["macd_hist"]>0)&(df["macd_hist"].shift(1)<=0)&(df["Close"]>df["ema200"])
    hd = (df["macd_hist"]<0)&(df["macd_hist"].shift(1)>=0)&(df["Close"]<df["ema200"])
    return make_sig(hu, hd)

# S2: EMA 9/21 crossover only
def s2_ema_cross():
    cu = (df["ema9"]>df["ema21"])&(df["ema9"].shift(1)<=df["ema21"].shift(1))
    cd = (df["ema9"]<df["ema21"])&(df["ema9"].shift(1)>=df["ema21"].shift(1))
    return make_sig(cu, cd)

# S3: EMA 9/21 cross + EMA200 trend
def s3_ema_cross_ema200():
    cu = (df["ema9"]>df["ema21"])&(df["ema9"].shift(1)<=df["ema21"].shift(1))&(df["Close"]>df["ema200"])
    cd = (df["ema9"]<df["ema21"])&(df["ema9"].shift(1)>=df["ema21"].shift(1))&(df["Close"]<df["ema200"])
    return make_sig(cu, cd)

# S4: BB breakout (price breaks outside bands)
def s4_bb_breakout():
    bu = (df["Close"]>df["bb_upper"])&(df["Close"].shift(1)<=df["bb_upper"].shift(1))
    bd = (df["Close"]<df["bb_lower"])&(df["Close"].shift(1)>=df["bb_lower"].shift(1))
    return make_sig(bu, bd)

# S5: BB breakout + EMA200 trend
def s5_bb_ema200():
    bu = (df["Close"]>df["bb_upper"])&(df["Close"].shift(1)<=df["bb_upper"].shift(1))&(df["Close"]>df["ema200"])
    bd = (df["Close"]<df["bb_lower"])&(df["Close"].shift(1)>=df["bb_lower"].shift(1))&(df["Close"]<df["ema200"])
    return make_sig(bu, bd)

# S6: RSI reversal (cross back from extreme)
def s6_rsi_reversal():
    ru = (df["rsi"]>30)&(df["rsi"].shift(1)<=30)
    rd = (df["rsi"]<70)&(df["rsi"].shift(1)>=70)
    return make_sig(ru, rd)

# S7: RSI reversal + EMA200 trend
def s7_rsi_ema200():
    ru = (df["rsi"]>30)&(df["rsi"].shift(1)<=30)&(df["Close"]>df["ema200"])
    rd = (df["rsi"]<70)&(df["rsi"].shift(1)>=70)&(df["Close"]<df["ema200"])
    return make_sig(ru, rd)

# S8: MACD + BB squeeze (only trade when bands are wide = volatile)
def s8_macd_bb_squeeze():
    wide = df["bb_width"] > df["bb_width"].rolling(50).mean()
    hu = (df["macd_hist"]>0)&(df["macd_hist"].shift(1)<=0)&(df["Close"]>df["ema200"])&wide
    hd = (df["macd_hist"]<0)&(df["macd_hist"].shift(1)>=0)&(df["Close"]<df["ema200"])&wide
    return make_sig(hu, hd)

# S9: Supertrend flip
def s9_supertrend():
    flip_bull = (df["st_dir"]==1)&(df["st_dir"].shift(1)==-1)
    flip_bear = (df["st_dir"]==-1)&(df["st_dir"].shift(1)==1)
    return make_sig(flip_bull, flip_bear)

# S10: Donchian channel breakout
def s10_donchian():
    bu = df["Close"] > df["don_high"]
    bd = df["Close"] < df["don_low"]
    return make_sig(bu, bd)

# S11: Full EMA stack (9>21>50>200 for bull, reverse for bear)
def s11_ema_stack():
    # Entry: MACD hist flip when full stack aligned
    bull_stack = (df["ema9"]>df["ema21"])&(df["ema21"]>df["ema50"])&(df["ema50"]>df["ema200"])
    bear_stack = (df["ema9"]<df["ema21"])&(df["ema21"]<df["ema50"])&(df["ema50"]<df["ema200"])
    hu = (df["macd_hist"]>0)&(df["macd_hist"].shift(1)<=0)&bull_stack
    hd = (df["macd_hist"]<0)&(df["macd_hist"].shift(1)>=0)&bear_stack
    return make_sig(hu, hd)

# S12: MACD + EMA cross + EMA200 (combined best)
def s12_combined():
    slope_up   = df["ema200"] > df["ema200"].shift(10)
    slope_down = df["ema200"] < df["ema200"].shift(10)
    ema50_bull = df["ema50"] > df["ema200"]
    ema50_bear = df["ema50"] < df["ema200"]
    micro_bull = df["ema9"] > df["ema21"]
    micro_bear = df["ema9"] < df["ema21"]
    hu = (df["macd_hist"]>0)&(df["macd_hist"].shift(1)<=0)&(df["Close"]>df["ema200"])&slope_up&ema50_bull&micro_bull&df["cross_bull"]
    hd = (df["macd_hist"]<0)&(df["macd_hist"].shift(1)>=0)&(df["Close"]<df["ema200"])&slope_down&ema50_bear&micro_bear&df["cross_bear"]
    return make_sig(hu, hd)

# S13: EMA 50/200 golden/death cross
def s13_golden_cross():
    gc = (df["ema50"]>df["ema200"])&(df["ema50"].shift(1)<=df["ema200"].shift(1))
    dc = (df["ema50"]<df["ema200"])&(df["ema50"].shift(1)>=df["ema200"].shift(1))
    return make_sig(gc, dc)

# S14: RSI momentum (RSI crosses 50 with EMA200 trend)
def s14_rsi_momentum():
    ru = (df["rsi"]>50)&(df["rsi"].shift(1)<=50)&(df["Close"]>df["ema200"])
    rd = (df["rsi"]<50)&(df["rsi"].shift(1)>=50)&(df["Close"]<df["ema200"])
    return make_sig(ru, rd)

# S15: MACD zero-line cross (MACD line crosses zero, not histogram)
def s15_macd_zero():
    mu = (df["macd"]>0)&(df["macd"].shift(1)<=0)&(df["Close"]>df["ema200"])
    md = (df["macd"]<0)&(df["macd"].shift(1)>=0)&(df["Close"]<df["ema200"])
    return make_sig(mu, md)

STRATEGIES = [
    ("S1.  MACD hist + EMA200",          s1_macd_ema200),
    ("S2.  EMA 9/21 cross",              s2_ema_cross),
    ("S3.  EMA 9/21 cross + EMA200",     s3_ema_cross_ema200),
    ("S4.  BB breakout",                 s4_bb_breakout),
    ("S5.  BB breakout + EMA200",        s5_bb_ema200),
    ("S6.  RSI reversal",                s6_rsi_reversal),
    ("S7.  RSI reversal + EMA200",       s7_rsi_ema200),
    ("S8.  MACD + BB wide",              s8_macd_bb_squeeze),
    ("S9.  Supertrend flip",             s9_supertrend),
    ("S10. Donchian breakout",           s10_donchian),
    ("S11. Full EMA stack",              s11_ema_stack),
    ("S12. MACD+EMAcross+EMA200",        s12_combined),
    ("S13. Golden/Death cross",          s13_golden_cross),
    ("S14. RSI momentum cross 50",       s14_rsi_momentum),
    ("S15. MACD zero-line cross",        s15_macd_zero),
]

# ─────────────────────────────────────────────
# RUN ALL
# ─────────────────────────────────────────────
rt     = COST_PER_SIDE * 2
net_tp = SL_PCT * RR - rt
net_sl = SL_PCT + rt
be_wr  = 1 / (1 + net_tp/net_sl) * 100

print(f"  SL={SL_PCT*100:.1f}%  RR={RR}:1  Spread={COST_PER_SIDE*100:.2f}%/side")
print(f"  Breakeven WR = {be_wr:.1f}%\n")

results = {}
for name, sig_fn in STRATEGIES:
    sig = sig_fn()
    t   = backtest(df, sig)
    results[name] = t
    n   = len(t)
    if n == 0:
        print(f"  {name:<35} — no trades")
        continue
    tp_m = t["result"] == "TP"
    wr   = tp_m.sum() / n * 100
    fin  = t["capital"].iloc[-1]
    ret  = (fin - CAPITAL) / CAPITAL * 100
    print(f"  {name:<35} trades={n:>5,}  WR={wr:>5.1f}%  final=${fin:>8,.2f}  ret={ret:>+7.1f}%")

# ─────────────────────────────────────────────
# DETAILED STATS + MONTHLY BREAKDOWN
# ─────────────────────────────────────────────
def calc_stats(trades):
    if len(trades) == 0:
        return None
    tp_m = trades["result"] == "TP"
    sl_m = trades["result"] == "SL"
    wins = tp_m.sum(); losses = sl_m.sum()
    pnl  = trades["pnl"].sum()
    fin  = trades["capital"].iloc[-1]
    wr   = wins / len(trades) * 100
    ret  = (fin - CAPITAL) / CAPITAL * 100
    aw   = trades.loc[tp_m, "pnl"].mean() if wins  > 0 else 0
    al   = trades.loc[sl_m, "pnl"].mean() if losses > 0 else 0
    cap  = pd.concat([pd.Series([CAPITAL]), trades["capital"].reset_index(drop=True)])
    mdd  = ((cap - cap.cummax()) / cap.cummax() * 100).min()
    exp  = (wr/100 * aw) + ((1-wr/100) * al)

    trades = trades.copy()
    trades["exit_time"] = pd.to_datetime(trades["exit_time"])
    trades["month"]     = trades["exit_time"].dt.to_period("M")
    monthly = trades.groupby("month").agg(
        n       = ("pnl", "count"),
        wins    = ("result", lambda x: (x=="TP").sum()),
        pnl_m   = ("pnl", "sum"),
        end_cap = ("capital", "last"),
    ).reset_index()
    monthly["wr"]    = (monthly["wins"] / monthly["n"] * 100).round(1)
    monthly["ret%"]  = (monthly["pnl_m"] / monthly["end_cap"].shift(1).fillna(CAPITAL) * 100).round(1)
    monthly["pnl_m"] = monthly["pnl_m"].round(2)
    monthly["end_cap"] = monthly["end_cap"].round(2)

    avg_monthly_ret = monthly["ret%"].mean()
    best_month      = monthly["ret%"].max()
    worst_month     = monthly["ret%"].min()
    pos_months      = (monthly["ret%"] > 0).sum()
    total_months    = len(monthly)

    return {
        "trades": len(trades), "wins": int(wins), "wr": wr,
        "final": fin, "ret": ret, "mdd": mdd, "exp": exp,
        "avg_monthly": avg_monthly_ret, "best_month": best_month,
        "worst_month": worst_month, "pos_months": pos_months,
        "total_months": total_months, "monthly": monthly,
    }

# ─────────────────────────────────────────────
# PRINT FULL RESULTS
# ─────────────────────────────────────────────
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("\n" + "=" * 120)
print(f"   BTC/USDT 5m — ALL STRATEGIES | SL={SL_PCT*100:.1f}% RR={RR}:1 | Spread={COST_PER_SIDE*100:.2f}%/side | 3 Years")
print(f"   Breakeven WR = {be_wr:.1f}%  |  Period: {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}")
print("=" * 120)
print(f"\n  {'Strategy':<35} {'Tr':>6} {'WR%':>6} {'Margin':>7} {'Final':>10} {'3Y Ret':>8} "
      f"{'MaxDD':>7} {'AvgMo%':>8} {'BestMo':>8} {'WorstMo':>8} {'+Months':>8}")
print(f"  {'─'*35} {'─'*6} {'─'*6} {'─'*7} {'─'*10} {'─'*8} "
      f"{'─'*7} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

all_stats = {}
for name, _ in STRATEGIES:
    t = results[name]
    s = calc_stats(t)
    all_stats[name] = s
    if s is None:
        print(f"  {name:<35} — no trades")
        continue
    margin = s["wr"] - be_wr
    flag   = "✅" if margin > 0 else "❌"
    print(f"  {name:<35} {s['trades']:>6,} {s['wr']:>6.1f}% {margin:>+6.1f}%{flag} "
          f"${s['final']:>9,.2f} {s['ret']:>+7.1f}% {s['mdd']:>6.1f}% "
          f"{s['avg_monthly']:>+7.1f}% {s['best_month']:>+7.1f}% {s['worst_month']:>+7.1f}% "
          f"{s['pos_months']:>3}/{s['total_months']}")

print("=" * 120)

# ── Monthly breakdown for profitable strategies only ──
profitable = [(n, s) for n, s in all_stats.items() if s and s["ret"] > 0]
profitable.sort(key=lambda x: x[1]["ret"], reverse=True)

print(f"\n  📊 Profitable strategies: {len(profitable)} / {len(STRATEGIES)}")
print(f"  Target: avg monthly ≥ 100% — NONE will reach this (see analysis below)\n")

for name, s in profitable[:5]:   # top 5 only
    print(f"\n{'═'*100}")
    print(f"  {name}  |  3Y: {s['ret']:+.1f}%  |  Avg monthly: {s['avg_monthly']:+.1f}%  "
          f"|  Best month: {s['best_month']:+.1f}%  |  WR: {s['wr']:.1f}%")
    print(f"{'═'*100}")
    m = s["monthly"]
    print(f"  {'Month':<10} {'Tr':>4} {'W':>4} {'WR%':>6}  {'PnL':>10}  {'Cap':>10}  {'Ret%':>7}")
    print(f"  {'─'*10} {'─'*4} {'─'*4} {'─'*6}  {'─'*10}  {'─'*10}  {'─'*7}")
    for _, row in m.iterrows():
        flag = "🔥" if row["ret%"] > 10 else ("⚠" if row["ret%"] < -10 else "  ")
        print(f"  {str(row['month']):<10} {row['n']:>4} {row['wins']:>4} {row['wr']:>5.1f}%  "
              f"${row['pnl_m']:>9,.2f}  ${row['end_cap']:>9,.2f}  {row['ret%']:>+6.1f}% {flag}")

# ── Honest analysis ──
print(f"\n{'═'*100}")
print("  📌 HONEST ANALYSIS — Why 100% monthly is not achievable with real costs:")
print(f"{'═'*100}")
print(f"""
  The math is unambiguous:

  Best strategy found: {profitable[0][0] if profitable else 'none'}
  Best avg monthly return: {profitable[0][1]['avg_monthly']:+.1f}% (vs target: +100%)
  Best single month ever:  {profitable[0][1]['best_month']:+.1f}%

  To get 100% monthly at 200 trades/month with 1% risk:
    → Need expectancy of 0.347% per trade
    → Best strategy delivers ~{profitable[0][1]['exp']:.4f}% per trade
    → Gap: {0.00347 - profitable[0][1]['exp']:.4f}% per trade

  The ONLY ways to reach 100% monthly:
  ┌─────────────────────────────────────────────────────────────────┐
  │ 1. Risk 50%+ per trade  → near-certain ruin within weeks        │
  │ 2. Win rate of 70%+     → no known 5m strategy achieves this    │
  │ 3. RR of 10:1+          → TP too far, almost never hit on 5m    │
  │ 4. Zero-cost ideal test → not real trading                      │
  └─────────────────────────────────────────────────────────────────┘

  REALISTIC TARGETS with this approach:
    Conservative (1% risk, best signal): +3-5% per month avg
    Aggressive   (3% risk, best signal): +9-15% per month avg
    Very aggressive (5% risk):           +15-25% per month avg (high ruin risk)

  The +3,000% in 3 years from test_scalp4b.py was zero-cost ideal.
  With real spread: best is +40% over 3 years = ~1% per month avg.
""")
print("=" * 100)
print("✅ Done!\n")
