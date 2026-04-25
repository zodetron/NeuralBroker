"""
BTC/USDT 5m — MACD Zero-Cross: TIERED LOT SCALING vs FIXED RISK
Data source: Binance public API | 3 Years (Apr 2023 → Apr 2026)

WHAT'S DIFFERENT FROM test_scalp4.py:
  test_scalp4.py  → fixed 1% risk every trade (standard compounding)
  test_scalp4b.py → tiered risk % that increases as account grows

TIERED LOT SIZE SCHEDULE:
  Capital < $200       → 1.0% risk  (starting tier, protect capital)
  $200  – $500         → 1.5% risk  (account proven, press a little)
  $500  – $1,000       → 2.0% risk  (solid base, increase size)
  $1,000 – $2,500      → 2.5% risk  (scaling up)
  $2,500 – $5,000      → 3.0% risk  (high confidence tier)
  $5,000+              → 3.5% risk  (max tier, full aggression)

WHY THIS WORKS:
  In the early choppy phase (2023) you're at 1% — minimum damage.
  Once the bull run kicks in and capital grows, you're at 2-3% —
  each winning trade compounds much harder on the larger base.
  The key insight: you only reach higher tiers AFTER proving the
  account, so you're never over-leveraged when you're losing.

Everything else identical: same signal, same SL/TP, same data.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from data_loader import load_data

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CAPITAL  = 100.0
LEVERAGE = 1000
RR       = 2.0
SL_PCT   = 0.0020

# Fixed risk (baseline from test_scalp4.py)
FIXED_RISK = 0.01

# Tiered risk schedule: (capital_threshold, risk_pct)
# Risk steps UP when capital crosses each threshold
RISK_TIERS = [
    (0,      0.010),   # < $200      → 1.0%
    (200,    0.015),   # $200-$500   → 1.5%
    (500,    0.020),   # $500-$1000  → 2.0%
    (1000,   0.025),   # $1000-$2500 → 2.5%
    (2500,   0.030),   # $2500-$5000 → 3.0%
    (5000,   0.035),   # $5000+      → 3.5%
]

def get_tiered_risk(capital):
    """Return the risk % for the current capital level."""
    risk = RISK_TIERS[0][1]
    for threshold, pct in RISK_TIERS:
        if capital >= threshold:
            risk = pct
    return risk

# ─────────────────────────────────────────────
# LOAD DATA — from CSV (instant) or Binance (first run)
# ─────────────────────────────────────────────
df = load_data()
print()

# ─────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────
def add_indicators(df):
    d = df.copy()
    d["ema200"]    = d["Close"].ewm(span=200, adjust=False).mean()
    ema12          = d["Close"].ewm(span=12, adjust=False).mean()
    ema26          = d["Close"].ewm(span=26, adjust=False).mean()
    d["macd"]      = ema12 - ema26
    d["macd_sig"]  = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_sig"]
    return d

df = add_indicators(df)
df.dropna(inplace=True)
print(f"  Indicators ready. {len(df):,} candles after warmup.\n")

# ─────────────────────────────────────────────
# SIGNAL — identical M0 original
# ─────────────────────────────────────────────
def sig_macd_original(df):
    s = pd.Series(0, index=df.index)
    hist_up   = (df["macd_hist"] > 0) & (df["macd_hist"].shift(1) <= 0) & (df["Close"] > df["ema200"])
    hist_down = (df["macd_hist"] < 0) & (df["macd_hist"].shift(1) >= 0) & (df["Close"] < df["ema200"])
    s[hist_up]   = 1
    s[hist_down] = -1
    return s

signals = sig_macd_original(df)

# ─────────────────────────────────────────────
# BACKTEST ENGINE — parameterised risk mode
# ─────────────────────────────────────────────
def backtest(df, signals, mode="fixed", sl_pct=SL_PCT, rr=RR):
    """
    mode = "fixed"  → always FIXED_RISK (1%)
    mode = "tiered" → risk % scales with capital via RISK_TIERS
    """
    tp_pct      = sl_pct * rr
    capital     = CAPITAL
    position    = None
    trades      = []
    entry_price = sl_price = tp_price = risk_usd = 0.0
    entry_time  = None

    for i in range(1, len(df)):
        if capital <= 0:
            break

        price = float(df["Close"].iloc[i])
        ts    = df.index[i]
        sig   = int(signals.iloc[i])

        # ── Manage open position ──
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
                    "entry":      round(entry_price, 2),
                    "exit":       round(price, 2),
                    "result":     "TP" if hit_tp else "SL",
                    "pnl":        round(pnl, 4),
                    "capital":    round(capital, 4),
                    "risk_pct":   round(risk_usd / max(capital - pnl, 1) * 100, 3),
                })
                position = None
                continue

        # ── Open new position ──
        if position is None and sig != 0 and capital > 0:
            if mode == "tiered":
                current_risk = get_tiered_risk(capital)
            else:
                current_risk = FIXED_RISK

            risk_usd    = round(capital * current_risk, 6)
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
# RUN BOTH
# ─────────────────────────────────────────────
print("  Running fixed 1% risk (baseline)...")
t_fixed  = backtest(df, signals, mode="fixed")
print(f"  {len(t_fixed):,} trades\n")

print("  Running tiered lot sizing...")
t_tiered = backtest(df, signals, mode="tiered")
print(f"  {len(t_tiered):,} trades\n")

# ─────────────────────────────────────────────
# STATS HELPER
# ─────────────────────────────────────────────
def calc_stats(trades):
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
            "wr": wr, "pnl": pnl, "final": final, "ret": ret,
            "max_dd": max_dd, "avg_win": avg_win, "avg_loss": avg_loss, "exp": exp}

sf = calc_stats(t_fixed)
st = calc_stats(t_tiered)

# ─────────────────────────────────────────────
# MONTHLY BREAKDOWN — both versions
# ─────────────────────────────────────────────
def monthly_breakdown(trades):
    trades = trades.copy()
    trades["exit_time"] = pd.to_datetime(trades["exit_time"])
    trades["month"]     = trades["exit_time"].dt.to_period("M")
    m = trades.groupby("month").agg(
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

mf = monthly_breakdown(t_fixed)
mt = monthly_breakdown(t_tiered)

# Merge for side-by-side monthly view
merged = mf[["month","trades_n","wins","win_rate","status"]].copy()
merged["fixed_pnl"]  = mf["pnl_month"]
merged["fixed_cap"]  = mf["end_cap"]
merged["fixed_ret%"] = mf["ret%"]
merged["tiered_pnl"] = mt["pnl_month"]
merged["tiered_cap"] = mt["end_cap"]
merged["tiered_ret%"]= mt["ret%"]
merged["delta_pnl"]  = (merged["tiered_pnl"] - merged["fixed_pnl"]).round(2)

# ─────────────────────────────────────────────
# YEARLY SUMMARY
# ─────────────────────────────────────────────
def yearly_summary(trades, label):
    trades = trades.copy()
    trades["exit_time"] = pd.to_datetime(trades["exit_time"])
    trades["year"]      = trades["exit_time"].dt.to_period("Y")
    y = trades.groupby("year").agg(
        trades_n    = ("pnl", "count"),
        wins        = ("result", lambda x: (x == "TP").sum()),
        pnl_year    = ("pnl", "sum"),
        end_capital = ("capital", "last"),
    ).reset_index()
    y["win_rate"] = (y["wins"] / y["trades_n"] * 100).round(1)
    y["pnl_year"] = y["pnl_year"].round(2)
    y["end_cap"]  = y["end_capital"].round(2)
    y["ret%"]     = ((y["end_capital"] / y["end_capital"].shift(1).fillna(CAPITAL) - 1) * 100).round(1)
    y["label"]    = label
    return y[["label","year","trades_n","wins","win_rate","pnl_year","end_cap","ret%"]]

yf = yearly_summary(t_fixed,  "Fixed 1%")
yt = yearly_summary(t_tiered, "Tiered")

# ─────────────────────────────────────────────
# COMPOUNDING MILESTONES
# ─────────────────────────────────────────────
def milestones(trades):
    targets = [200, 500, 1000, 2000, 5000, 10000, 25000, 50000]
    hit = {}
    for m in targets:
        crossed = trades[trades["capital"] >= m]
        if not crossed.empty:
            row = crossed.iloc[0]
            hit[m] = (pd.to_datetime(row["exit_time"]).strftime("%Y-%m-%d"), crossed.index[0])
    return hit

mf_hit = milestones(t_fixed)
mt_hit = milestones(t_tiered)

# ─────────────────────────────────────────────
# PRINT
# ─────────────────────────────────────────────
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("=" * 95)
print("   BTC/USDT 5m — MACD ZERO-CROSS | FIXED 1% vs TIERED LOT SIZING")
print(f"   Data: Binance  |  Capital: ${CAPITAL}  |  Leverage: 1:{LEVERAGE}")
print(f"   RR: {RR}:1  |  SL: {SL_PCT*100}%  |  Period: {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}")
print(f"   Tiers: <$200=1%  $200=1.5%  $500=2%  $1k=2.5%  $2.5k=3%  $5k+=3.5%")
print("=" * 95)

# Side-by-side summary
print(f"\n  {'Metric':<22} {'Fixed 1%':>18} {'Tiered':>18}  {'Delta':>14}")
print(f"  {'─'*22} {'─'*18} {'─'*18}  {'─'*14}")
rows = [
    ("Trades",        f"{sf['trades']:,}",          f"{st['trades']:,}",          "—"),
    ("Win Rate",      f"{sf['wr']:.1f}%",            f"{st['wr']:.1f}%",            f"{st['wr']-sf['wr']:+.1f}%"),
    ("Total PnL",     f"${sf['pnl']:,.2f}",          f"${st['pnl']:,.2f}",          f"${st['pnl']-sf['pnl']:+,.2f}"),
    ("Final Capital", f"${sf['final']:,.2f}",         f"${st['final']:,.2f}",         f"${st['final']-sf['final']:+,.2f}"),
    ("Return %",      f"{sf['ret']:,.1f}%",           f"{st['ret']:,.1f}%",           f"{st['ret']-sf['ret']:+,.1f}%"),
    ("Max Drawdown",  f"{sf['max_dd']:.1f}%",         f"{st['max_dd']:.1f}%",         f"{st['max_dd']-sf['max_dd']:+.1f}%"),
    ("Avg Win",       f"${sf['avg_win']:,.4f}",       f"${st['avg_win']:,.4f}",       ""),
    ("Avg Loss",      f"${sf['avg_loss']:,.4f}",      f"${st['avg_loss']:,.4f}",      ""),
    ("Expectancy",    f"${sf['exp']:,.4f}",           f"${st['exp']:,.4f}",           f"${st['exp']-sf['exp']:+,.4f}"),
]
for m, f, t, d in rows:
    print(f"  {m:<22} {f:>18} {t:>18}  {d:>14}")

print("=" * 95)

# Milestones
print(f"\n  {'💰 Compounding Milestones':<30} {'Fixed 1%':>25} {'Tiered':>25}")
print(f"  {'─'*30} {'─'*25} {'─'*25}")
all_targets = sorted(set(list(mf_hit.keys()) + list(mt_hit.keys())))
for m in all_targets:
    x    = round(m / CAPITAL, 0)
    f_dt = mf_hit[m][0] if m in mf_hit else "not reached"
    t_dt = mt_hit[m][0] if m in mt_hit else "not reached"
    print(f"  ${m:>6} ({x:>5.0f}x)  {f_dt:>25} {t_dt:>25}")

# Yearly
print(f"\n{'─'*95}")
print("  📆 Yearly Summary:")
print(f"{'─'*95}")
print(pd.concat([yf, yt]).sort_values(["year","label"]).to_string(index=False))

# Monthly side-by-side
print(f"\n{'─'*95}")
print("  📅 Monthly Breakdown — Fixed 1% vs Tiered:")
print(f"{'─'*95}")
print(f"  {'Month':<10} {'Tr':>4} {'W':>4} {'WR%':>6} {'S':>3}  "
      f"{'Fixed PnL':>12} {'Fixed Cap':>11} {'F.Ret%':>7}  "
      f"{'Tiered PnL':>12} {'Tiered Cap':>11} {'T.Ret%':>7}  {'Δ PnL':>10}")
print(f"  {'─'*10} {'─'*4} {'─'*4} {'─'*6} {'─'*3}  "
      f"{'─'*12} {'─'*11} {'─'*7}  "
      f"{'─'*12} {'─'*11} {'─'*7}  {'─'*10}")

for _, row in merged.iterrows():
    print(f"  {str(row['month']):<10} {row['trades_n']:>4} {row['wins']:>4} "
          f"{row['win_rate']:>5.1f}% {row['status']:>3}  "
          f"${row['fixed_pnl']:>11,.2f} ${row['fixed_cap']:>10,.2f} {row['fixed_ret%']:>6.1f}%  "
          f"${row['tiered_pnl']:>11,.2f} ${row['tiered_cap']:>10,.2f} {row['tiered_ret%']:>6.1f}%  "
          f"${row['delta_pnl']:>+9,.2f}")

# Capital growth every 500 trades — tiered only
print(f"\n{'─'*95}")
print("  📈 Capital Growth — Tiered (every 500 trades):")
print(f"{'─'*95}")
step = t_tiered.iloc[::500][["entry_time","exit_time","capital","risk_pct"]].copy()
step.index = range(0, len(step) * 500, 500)
step.index.name = "trade#"
print(step.to_string())

print("\n" + "=" * 95)
print(f"  🏆  TIERED LOT SIZING RESULT")
print(f"      $100  →  ${st['final']:,.2f}  ({st['ret']:,.1f}% return)")
print(f"      vs Fixed: $100  →  ${sf['final']:,.2f}  ({sf['ret']:,.1f}% return)")
print(f"      Tiered advantage: ${st['final']-sf['final']:+,.2f}  ({st['ret']-sf['ret']:+,.1f}%)")
print(f"      Win Rate: {st['wr']:.1f}%  |  Max DD: {st['max_dd']:.1f}%  |  {st['trades']:,} trades")
print("=" * 95)

# Save both
t_fixed.to_csv("macd_fixed_3y_trades.csv",  index=False)
t_tiered.to_csv("macd_tiered_3y_trades.csv", index=False)
print(f"\n📄 Fixed  → macd_fixed_3y_trades.csv")
print(f"📄 Tiered → macd_tiered_3y_trades.csv")
print("✅ Done!\n")
