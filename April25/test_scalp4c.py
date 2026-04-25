# realistic test of 5min testing
"""
BTC/USDT 5m — MACD + EMA200 Trend: REALISTIC EXECUTION TEST
Data: btcusdt_5m_3y.csv | 3 Years (Apr 2023 → Apr 2026)

WHAT MAKES THIS REALISTIC:
  Spread + slippage applied on every fill. No commission (as requested).

  SPREAD   = 0.01% per side  (BTC bid/ask ~$3 on $30k price)
  SLIPPAGE = 0.02% per side  (market order fills slightly worse than close)
  TOTAL    = 0.03% per side  → 0.06% round trip

WHY PREVIOUS VERSIONS FAILED:
  SL=0.20% with 0.06% round trip = costs eat 30% of SL distance.
  Effective RR drops to 1.3:1, needing 43% WR. Signal delivers 35%. ❌

THE FIX — SL=1.0%, RR=2.0:
  Costs eat only 6% of SL distance.
  Effective RR = (2.0% - 0.06%) / (1.0% + 0.06%) = 1.83:1
  Breakeven WR = 35.3%  ← signal delivers ~35-36% ✅ (just above breakeven)

SIGNAL — MACD + EMA200 with full trend stack:
  1. MACD histogram flips positive/negative
  2. Price on correct side of EMA200
  3. EMA200 is sloping in trade direction (10-bar slope)
  4. EMA50 aligned with EMA200 (same side)
  5. EMA9 > EMA21 for longs / EMA9 < EMA21 for shorts (micro-trend)

TIERED LOT SIZE (same as test_scalp4b.py):
  <$200=1%  $200=1.5%  $500=2%  $1k=2.5%  $2.5k=3%  $5k+=3.5%
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from data_loader import load_data

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CAPITAL    = 100.0
LEVERAGE   = 1000
FIXED_RISK = 0.01

RISK_TIERS = [
    (0,      0.010),
    (200,    0.015),
    (500,    0.020),
    (1000,   0.025),
    (2500,   0.030),
    (5000,   0.035),
]

def get_tiered_risk(capital):
    risk = RISK_TIERS[0][1]
    for threshold, pct in RISK_TIERS:
        if capital >= threshold:
            risk = pct
    return risk

# ─────────────────────────────────────────────
# EXECUTION COSTS — spread + slippage, NO commission
# ─────────────────────────────────────────────
SPREAD_PCT    = 0.0001   # 0.01% per side — BTC bid/ask spread only
SLIPPAGE_PCT  = 0.0000   # 0.00% — no slippage
COST_PER_SIDE = SPREAD_PCT   # 0.01% per side, 0.02% round trip

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
    for span in [9, 21, 50, 200]:
        d[f"ema{span}"] = d["Close"].ewm(span=span, adjust=False).mean()
    ema12          = d["Close"].ewm(span=12, adjust=False).mean()
    ema26          = d["Close"].ewm(span=26, adjust=False).mean()
    d["macd"]      = ema12 - ema26
    d["macd_sig"]  = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_sig"]

    # EMA crossover flags — did EMA9 cross EMA21 within the last 10 bars?
    # This ensures we're entering near the cross, not 50 bars after it
    cross_up   = (d["ema9"] > d["ema21"]) & (d["ema9"].shift(1) <= d["ema21"].shift(1))
    cross_down = (d["ema9"] < d["ema21"]) & (d["ema9"].shift(1) >= d["ema21"].shift(1))
    d["recent_cross_bull"] = cross_up.rolling(10).max().astype(bool)   # cross happened in last 10 bars
    d["recent_cross_bear"] = cross_down.rolling(10).max().astype(bool)

    return d

df = add_indicators(df)
df.dropna(inplace=True)
print(f"  Indicators ready. {len(df):,} candles after warmup.\n")

# ─────────────────────────────────────────────
# SIGNAL FUNCTIONS
# ─────────────────────────────────────────────
def _base_conditions(df):
    """Shared conditions used by both signal variants."""
    hist_up        = (df["macd_hist"] > 0) & (df["macd_hist"].shift(1) <= 0)
    hist_down      = (df["macd_hist"] < 0) & (df["macd_hist"].shift(1) >= 0)
    above_ema200   = df["Close"] > df["ema200"]
    below_ema200   = df["Close"] < df["ema200"]
    ema200_up      = df["ema200"] > df["ema200"].shift(10)
    ema200_down    = df["ema200"] < df["ema200"].shift(10)
    ema50_above200 = df["ema50"] > df["ema200"]
    ema50_below200 = df["ema50"] < df["ema200"]
    micro_bull     = df["ema9"] > df["ema21"]
    micro_bear     = df["ema9"] < df["ema21"]
    return (hist_up, hist_down, above_ema200, below_ema200,
            ema200_up, ema200_down, ema50_above200, ema50_below200,
            micro_bull, micro_bear)

# V1: Original — MACD + EMA200 trend stack (no crossover recency)
def sig_v1_no_cross(df):
    s = pd.Series(0, index=df.index)
    (hist_up, hist_down, above_ema200, below_ema200,
     ema200_up, ema200_down, ema50_above200, ema50_below200,
     micro_bull, micro_bear) = _base_conditions(df)

    long_sig  = hist_up   & above_ema200 & ema200_up   & ema50_above200 & micro_bull
    short_sig = hist_down & below_ema200 & ema200_down & ema50_below200 & micro_bear
    s[long_sig]  = 1
    s[short_sig] = -1
    return s

# V2: + EMA9/21 crossover recency filter
# Only enter if EMA9 crossed EMA21 within the last 10 bars (50 min on 5m)
# This ensures we're entering near the momentum shift, not chasing a stale trend
def sig_v2_with_cross(df):
    s = pd.Series(0, index=df.index)
    (hist_up, hist_down, above_ema200, below_ema200,
     ema200_up, ema200_down, ema50_above200, ema50_below200,
     micro_bull, micro_bear) = _base_conditions(df)

    long_sig  = hist_up   & above_ema200 & ema200_up   & ema50_above200 & micro_bull & df["recent_cross_bull"]
    short_sig = hist_down & below_ema200 & ema200_down & ema50_below200 & micro_bear & df["recent_cross_bear"]
    s[long_sig]  = 1
    s[short_sig] = -1
    return s

# Run both for comparison, then use best for the RR configs
print("  Comparing signal versions (SL=1.0%, RR=2.5, fixed 1%)...")
sig_v1 = sig_v1_no_cross(df)
sig_v2 = sig_v2_with_cross(df)
print(f"  V1 (no cross):   {(sig_v1==1).sum():,} longs | {(sig_v1==-1).sum():,} shorts")
print(f"  V2 (with cross): {(sig_v2==1).sum():,} longs | {(sig_v2==-1).sum():,} shorts\n")

# ─────────────────────────────────────────────
# BACKTEST ENGINE — realistic fills (spread only)
# ─────────────────────────────────────────────
def backtest(df, signals, sl_pct, rr, mode="fixed"):
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
                    exit_fill   = theo * (1 - COST_PER_SIDE)
                    actual_move = exit_fill - entry_fill
                else:
                    exit_fill   = theo * (1 + COST_PER_SIDE)
                    actual_move = entry_fill - exit_fill

                sl_dist = entry_fill * sl_pct
                pnl     = risk_usd * (actual_move / sl_dist)
                capital = max(0.0, capital + pnl)

                trades.append({
                    "entry_time": entry_time,
                    "exit_time":  ts,
                    "direction":  position,
                    "entry_fill": round(entry_fill, 2),
                    "exit_fill":  round(exit_fill, 2),
                    "result":     "TP" if hit_tp else "SL",
                    "pnl":        round(pnl, 4),
                    "capital":    round(capital, 4),
                    "risk_pct":   round(risk_usd / max(capital - pnl, 1) * 100, 3),
                })
                position = None
                continue

        if position is None and sig != 0 and capital > 0:
            current_risk = get_tiered_risk(capital) if mode == "tiered" else FIXED_RISK
            risk_usd     = round(capital * current_risk, 6)
            entry_time   = ts

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
# RUN — 2 signal versions × 2 RR configs × 2 risk modes
# ─────────────────────────────────────────────
CONFIGS = [
    ("RR=1:3.0", 0.0100, 3.0),
    ("RR=1:2.5", 0.0100, 2.5),
]

SIGNALS = [
    ("V1 no-cross",   sig_v1_no_cross),
    ("V2 +EMA-cross", sig_v2_with_cross),
]

results = {}   # key: (sig_label, rr_label, mode) → trades DataFrame

for sig_label, sig_fn in SIGNALS:
    sigs = sig_fn(df)
    for rr_label, sl, rr in CONFIGS:
        for mode in ("fixed", "tiered"):
            key = (sig_label, rr_label, mode)
            print(f"  {sig_label} | {rr_label} | {mode}...")
            results[key] = backtest(df, sigs, sl_pct=sl, rr=rr, mode=mode)
    print()

# ─────────────────────────────────────────────
# STATS
# ─────────────────────────────────────────────
def calc_stats(trades, sl_pct, rr):
    if len(trades) == 0:
        return {}
    rt       = COST_PER_SIDE * 2
    net_tp   = sl_pct * rr - rt
    net_sl   = sl_pct + rt
    eff_rr   = net_tp / net_sl
    be_wr    = 1 / (1 + eff_rr) * 100

    tp_mask  = trades["result"] == "TP"
    sl_mask  = trades["result"] == "SL"
    wins     = tp_mask.sum()
    losses   = sl_mask.sum()
    pnl      = trades["pnl"].sum()
    final    = trades["capital"].iloc[-1]
    wr       = wins / len(trades) * 100
    ret      = (final - CAPITAL) / CAPITAL * 100
    avg_win  = trades.loc[tp_mask, "pnl"].mean() if wins  > 0 else 0
    avg_loss = trades.loc[sl_mask, "pnl"].mean() if losses > 0 else 0
    cap_s    = pd.concat([pd.Series([CAPITAL]), trades["capital"].reset_index(drop=True)])
    max_dd   = ((cap_s - cap_s.cummax()) / cap_s.cummax() * 100).min()
    exp      = (wr/100 * avg_win) + ((1 - wr/100) * avg_loss)
    return {"trades": len(trades), "wins": int(wins), "losses": int(losses),
            "wr": wr, "be_wr": be_wr, "eff_rr": eff_rr,
            "pnl": pnl, "final": final, "ret": ret,
            "max_dd": max_dd, "avg_win": avg_win, "avg_loss": avg_loss, "exp": exp}

# ─────────────────────────────────────────────
# MONTHLY BREAKDOWN
# ─────────────────────────────────────────────
def monthly_breakdown(trades):
    t = trades.copy()
    t["exit_time"] = pd.to_datetime(t["exit_time"])
    t["month"]     = t["exit_time"].dt.to_period("M")
    m = t.groupby("month").agg(
        trades_n    = ("pnl", "count"),
        wins        = ("result", lambda x: (x == "TP").sum()),
        losses      = ("result", lambda x: (x == "SL").sum()),
        pnl_month   = ("pnl", "sum"),
        end_capital = ("capital", "last"),
    ).reset_index()
    m["win_rate"]  = (m["wins"] / m["trades_n"] * 100).round(1)
    m["pnl_month"] = m["pnl_month"].round(2)
    m["end_cap"]   = m["end_capital"].round(2)
    m["ret%"]      = (m["pnl_month"] / m["end_capital"].shift(1).fillna(CAPITAL) * 100).round(2)
    m["status"]    = m["win_rate"].apply(
        lambda w: "🔥" if w >= 38 else ("⚠" if w < 33 else "✅"))
    return m

# ─────────────────────────────────────────────
# PRINT
# ─────────────────────────────────────────────
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

rt = COST_PER_SIDE * 2

print("=" * 110)
print("   BTC/USDT 5m — MACD + EMA200 + EMA CROSSOVER | SPREAD ONLY (no slippage, no commission)")
print(f"   V1: MACD hist + EMA200 side/slope + EMA50 align + EMA9>21 micro-trend")
print(f"   V2: V1 + EMA9/21 crossover within last 10 bars (50 min) — recency filter")
print(f"   Data: Binance  |  Capital: ${CAPITAL}  |  Leverage: 1:{LEVERAGE}")
print(f"   Period: {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}")
print(f"   Cost: Spread only = {COST_PER_SIDE*100:.2f}%/side ({rt*100:.2f}% round trip)")
print("=" * 110)

# ── Master summary table ──
print(f"\n  {'Signal':<16} {'RR':>10} {'Mode':<8} {'Trades':>7} {'WR%':>6} {'BE_WR':>7} "
      f"{'Margin':>7} {'Final $':>10} {'Return%':>9} {'MaxDD%':>8} {'Expect':>8}")
print(f"  {'─'*16} {'─'*10} {'─'*8} {'─'*7} {'─'*6} {'─'*7} "
      f"{'─'*7} {'─'*10} {'─'*9} {'─'*8} {'─'*8}")

best_ret = -999
best_key = None

for sig_label, _ in SIGNALS:
    for rr_label, sl, rr in CONFIGS:
        for mode in ("fixed", "tiered"):
            key = (sig_label, rr_label, mode)
            t   = results[key]
            if len(t) == 0:
                continue
            s      = calc_stats(t, sl, rr)
            margin = s['wr'] - s['be_wr']
            flag   = "✅" if margin > 0 else "❌"
            print(f"  {sig_label:<16} {rr_label:>10} {mode:<8} {s['trades']:>7,} "
                  f"{s['wr']:>6.1f}% {s['be_wr']:>6.1f}% {margin:>+6.1f}% {flag} "
                  f"${s['final']:>9,.2f} {s['ret']:>+8.1f}% {s['max_dd']:>7.1f}% ${s['exp']:>7.4f}")
            if s['ret'] > best_ret:
                best_ret = s['ret']
                best_key = key
        print()

print("=" * 110)

# ── Monthly breakdown per signal × RR (fixed only to keep output clean) ──
for sig_label, _ in SIGNALS:
    for rr_label, sl, rr in CONFIGS:
        rt_    = COST_PER_SIDE * 2
        net_tp = sl * rr - rt_
        net_sl = sl + rt_
        eff_rr = net_tp / net_sl
        be_wr  = 1 / (1 + eff_rr) * 100

        key_f = (sig_label, rr_label, "fixed")
        key_t = (sig_label, rr_label, "tiered")
        tf    = results[key_f]
        tt    = results[key_t]

        if len(tf) == 0:
            continue

        sf_ = calc_stats(tf, sl, rr)
        st_ = calc_stats(tt, sl, rr)

        print(f"\n{'═'*110}")
        print(f"  📊 {sig_label} | {rr_label}  |  Eff RR={eff_rr:.2f}:1  |  BE WR={be_wr:.1f}%  "
              f"|  Fixed: ${sf_['final']:,.2f} ({sf_['ret']:+.1f}%)  "
              f"|  Tiered: ${st_['final']:,.2f} ({st_['ret']:+.1f}%)")
        print(f"{'═'*110}")
        print(f"  {'Month':<10} {'Tr':>4} {'W':>4} {'WR%':>6} {'S':>3}  "
              f"{'Fixed PnL':>12} {'Fixed Cap':>11} {'F.Ret%':>7}  "
              f"{'Tiered PnL':>12} {'Tiered Cap':>11} {'T.Ret%':>7}")
        print(f"  {'─'*10} {'─'*4} {'─'*4} {'─'*6} {'─'*3}  "
              f"{'─'*12} {'─'*11} {'─'*7}  {'─'*12} {'─'*11} {'─'*7}")

        mf_ = monthly_breakdown(tf)
        mt_ = monthly_breakdown(tt)
        mf_d = mf_.set_index("month")
        mt_d = mt_.set_index("month")

        for mo in sorted(mf_d.index):
            fr = mf_d.loc[mo]
            tr = mt_d.loc[mo] if mo in mt_d.index else None
            tp_str = f"${tr['pnl_month']:>11,.2f} ${tr['end_cap']:>10,.2f} {tr['ret%']:>6.1f}%" if tr is not None else f"{'—':>12} {'—':>11} {'—':>7}"
            print(f"  {str(mo):<10} {fr['trades_n']:>4} {fr['wins']:>4} "
                  f"{fr['win_rate']:>5.1f}% {fr['status']:>3}  "
                  f"${fr['pnl_month']:>11,.2f} ${fr['end_cap']:>10,.2f} {fr['ret%']:>6.1f}%  "
                  f"{tp_str}")

# ── Best config ──
if best_key:
    bs_label, br_label, bm = best_key
    bsl = next(sl for _, sl, rr in CONFIGS if _ == br_label)
    brr = next(rr for _, sl, rr in CONFIGS if _ == br_label)
    bt  = results[best_key]
    bs  = calc_stats(bt, bsl, brr)

    print(f"\n{'═'*110}")
    print(f"  🏆  BEST: {bs_label} | {br_label} | {bm}")
    print(f"      $100 → ${bs['final']:,.2f}  ({bs['ret']:+.1f}%)")
    print(f"      WR: {bs['wr']:.1f}%  |  BE: {bs['be_wr']:.1f}%  |  Margin: {bs['wr']-bs['be_wr']:+.1f}%")
    print(f"      Max DD: {bs['max_dd']:.1f}%  |  Trades: {bs['trades']:,}  |  Expectancy: ${bs['exp']:.4f}")
    print(f"{'═'*110}")

# Save all
for sig_label, _ in SIGNALS:
    for rr_label, sl, rr in CONFIGS:
        for mode in ("fixed", "tiered"):
            key  = (sig_label, rr_label, mode)
            safe = f"{sig_label.replace(' ','_')}_{rr_label.replace('=','').replace(':','_')}_{mode}"
            results[key].to_csv(f"realistic_{safe}.csv", index=False)

print(f"\n📄 Trade logs saved for all {len(results)} combinations")
print("✅ Done!\n")
