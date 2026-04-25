"""
BTC/USDT 1m — MACD Zero-Cross HFT: 1-Year Test
Data source: Binance public API | ~525,600 candles

SAME STRATEGY as test_scalp4.py / test_scalp4b.py:
  Signal : MACD histogram flip + EMA200 trend filter
  Risk   : Tiered lot sizing (same schedule as test_scalp4b.py)
  SL/TP  : 0.20% SL | 2:1 RR

WHY 1m IS DIFFERENT FROM 5m:
  - ~5x more signals per day (~25,000+ trades/year vs ~8,400)
  - Each candle = 1 minute, so SL/TP hit much faster
  - MACD on 1m is noisier — more false flips, lower win rate expected
  - But compounding happens faster — more trades = more opportunities
  - EMA200 on 1m = 200 minutes (~3.3 hours) trend filter

TIERED LOT SIZE SCHEDULE (identical to test_scalp4b.py):
  Capital < $200       → 1.0% risk
  $200  – $500         → 1.5% risk
  $500  – $1,000       → 2.0% risk
  $1,000 – $2,500      → 2.5% risk
  $2,500 – $5,000      → 3.0% risk
  $5,000+              → 3.5% risk

Capital: $100 | Leverage: 1:1000 | RR: 2:1 | SL: 0.20%
"""

import requests
import pandas as pd
import numpy as np
import warnings
import time
from datetime import datetime, timedelta, timezone
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CAPITAL  = 100.0
LEVERAGE = 1000
RR       = 2.0
SL_PCT   = 0.0020     # 0.20% — same as 5m version
SYMBOL   = "BTCUSDT"
INTERVAL = "1m"
DAYS     = 365

# Tiered risk schedule
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
# FETCH DATA — Binance 1m, chunked
# 525,600 candles in 1 year — ~527 requests of 1000
# ─────────────────────────────────────────────
def fetch_binance(symbol, interval, days):
    print(f"📥 Fetching {symbol} {interval} data ({days} days) from Binance...")
    print(f"   Estimated candles: {days*24*60:,} — this will take ~2 minutes...")
    end_ms        = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms      = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    url           = "https://api.binance.com/api/v3/klines"
    all_candles   = []
    current_start = start_ms

    while current_start < end_ms:
        params = {"symbol": symbol, "interval": interval,
                  "startTime": current_start, "endTime": end_ms, "limit": 1000}
        try:
            resp    = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            candles = resp.json()
        except Exception as e:
            print(f"\n  ⚠ {e} — retrying in 2s...")
            time.sleep(2)
            continue
        if not candles:
            break
        all_candles.extend(candles)
        last_ts = candles[-1][0]
        pct     = (last_ts - start_ms) / (end_ms - start_ms) * 100
        print(f"  fetched {len(all_candles):,} candles... {pct:.1f}%", end="\r")
        if len(candles) < 1000:
            break
        current_start = last_ts + 1
        time.sleep(0.03)   # ~33 req/s — well within Binance limits

    print()
    cols = ["open_time","Open","High","Low","Close","Volume",
            "ct","qv","t","tbb","tbq","ig"]
    df = pd.DataFrame(all_candles, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("open_time", inplace=True)
    for c in ["Open","High","Low","Close","Volume"]:
        df[c] = df[c].astype(float)
    df = df[["Open","High","Low","Close","Volume"]]
    df = df[~df.index.duplicated(keep="first")]
    df.sort_index(inplace=True)
    df.dropna(inplace=True)
    return df

df = fetch_binance(SYMBOL, INTERVAL, DAYS)
print(f"✅ {len(df):,} candles | {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}\n")

# ─────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────
def add_indicators(df):
    d = df.copy()
    # EMA200 on 1m = 200-minute trend filter (~3.3 hours)
    d["ema200"]    = d["Close"].ewm(span=200, adjust=False).mean()
    # MACD standard settings
    ema12          = d["Close"].ewm(span=12, adjust=False).mean()
    ema26          = d["Close"].ewm(span=26, adjust=False).mean()
    d["macd"]      = ema12 - ema26
    d["macd_sig"]  = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_sig"]
    return d

print("  Computing indicators on 525k candles...")
df = add_indicators(df)
df.dropna(inplace=True)
print(f"  Done. {len(df):,} candles after warmup.\n")

# ─────────────────────────────────────────────
# SIGNAL — identical M0
# ─────────────────────────────────────────────
def sig_macd_original(df):
    s = pd.Series(0, index=df.index)
    hist_up   = (df["macd_hist"] > 0) & (df["macd_hist"].shift(1) <= 0) & (df["Close"] > df["ema200"])
    hist_down = (df["macd_hist"] < 0) & (df["macd_hist"].shift(1) >= 0) & (df["Close"] < df["ema200"])
    s[hist_up]   = 1
    s[hist_down] = -1
    return s

print("  Generating signals...")
signals = sig_macd_original(df)
print(f"  Signals: {(signals == 1).sum():,} longs | {(signals == -1).sum():,} shorts\n")

# ─────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────
def backtest(df, signals, mode="tiered", sl_pct=SL_PCT, rr=RR):
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
                    "risk_pct":   round(get_tiered_risk(capital - pnl) * 100, 2),
                })
                position = None
                continue

        # ── Open new position ──
        if position is None and sig != 0 and capital > 0:
            current_risk = get_tiered_risk(capital) if mode == "tiered" else 0.01
            risk_usd     = round(capital * current_risk, 6)
            entry_price  = price
            entry_time   = ts

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
# RUN BOTH MODES
# ─────────────────────────────────────────────
print("  Running fixed 1% risk...")
t_fixed  = backtest(df, signals, mode="fixed")
print(f"  {len(t_fixed):,} trades\n")

print("  Running tiered lot sizing...")
t_tiered = backtest(df, signals, mode="tiered")
print(f"  {len(t_tiered):,} trades\n")

# ─────────────────────────────────────────────
# STATS
# ─────────────────────────────────────────────
def calc_stats(trades):
    if len(trades) == 0:
        return {}
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

mf = monthly_breakdown(t_fixed)
mt = monthly_breakdown(t_tiered)

# Merge side-by-side
merged = mf[["month","trades_n","wins","win_rate","status"]].copy()
merged["fixed_pnl"]   = mf["pnl_month"]
merged["fixed_cap"]   = mf["end_cap"]
merged["fixed_ret%"]  = mf["ret%"]
merged["tiered_pnl"]  = mt["pnl_month"]
merged["tiered_cap"]  = mt["end_cap"]
merged["tiered_ret%"] = mt["ret%"]
merged["delta"]       = (merged["tiered_pnl"] - merged["fixed_pnl"]).round(2)

# ─────────────────────────────────────────────
# MILESTONES
# ─────────────────────────────────────────────
def milestones(trades):
    targets = [200, 500, 1000, 2000, 5000, 10000, 25000, 50000]
    hit = {}
    for m in targets:
        crossed = trades[trades["capital"] >= m]
        if not crossed.empty:
            hit[m] = pd.to_datetime(crossed.iloc[0]["exit_time"]).strftime("%Y-%m-%d")
    return hit

mf_hit = milestones(t_fixed)
mt_hit = milestones(t_tiered)

# ─────────────────────────────────────────────
# PRINT
# ─────────────────────────────────────────────
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("=" * 95)
print("   BTC/USDT 1m HFT — MACD ZERO-CROSS | FIXED 1% vs TIERED LOT SIZING")
print(f"   Data: Binance  |  Capital: ${CAPITAL}  |  Leverage: 1:{LEVERAGE}")
print(f"   RR: {RR}:1  |  SL: {SL_PCT*100}%  |  Period: {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}")
print(f"   Tiers: <$200=1%  $200=1.5%  $500=2%  $1k=2.5%  $2.5k=3%  $5k+=3.5%")
print("=" * 95)

# Summary comparison
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
print(f"\n  {'💰 Milestones':<28} {'Fixed 1%':>20} {'Tiered':>20}")
print(f"  {'─'*28} {'─'*20} {'─'*20}")
all_targets = sorted(set(list(mf_hit.keys()) + list(mt_hit.keys())))
for m in all_targets:
    x    = round(m / CAPITAL, 0)
    f_dt = mf_hit.get(m, "not reached")
    t_dt = mt_hit.get(m, "not reached")
    print(f"  ${m:>6} ({x:>5.0f}x)  {f_dt:>20} {t_dt:>20}")

# Monthly side-by-side
print(f"\n{'─'*95}")
print("  📅 Monthly Breakdown — Fixed 1% vs Tiered (1m timeframe):")
print(f"{'─'*95}")
print(f"  {'Month':<10} {'Tr':>5} {'W':>5} {'WR%':>6} {'S':>3}  "
      f"{'Fixed PnL':>12} {'Fixed Cap':>11} {'F.Ret%':>7}  "
      f"{'Tiered PnL':>12} {'Tiered Cap':>11} {'T.Ret%':>7}  {'Δ':>10}")
print(f"  {'─'*10} {'─'*5} {'─'*5} {'─'*6} {'─'*3}  "
      f"{'─'*12} {'─'*11} {'─'*7}  "
      f"{'─'*12} {'─'*11} {'─'*7}  {'─'*10}")

for _, row in merged.iterrows():
    print(f"  {str(row['month']):<10} {row['trades_n']:>5} {row['wins']:>5} "
          f"{row['win_rate']:>5.1f}% {row['status']:>3}  "
          f"${row['fixed_pnl']:>11,.2f} ${row['fixed_cap']:>10,.2f} {row['fixed_ret%']:>6.1f}%  "
          f"${row['tiered_pnl']:>11,.2f} ${row['tiered_cap']:>10,.2f} {row['tiered_ret%']:>6.1f}%  "
          f"${row['delta']:>+9,.2f}")

# Capital growth every 1000 trades — tiered
print(f"\n{'─'*95}")
print("  📈 Capital Growth — Tiered (every 1,000 trades):")
print(f"{'─'*95}")
step = t_tiered.iloc[::1000][["entry_time","exit_time","capital","risk_pct"]].copy()
step.index = range(0, len(step) * 1000, 1000)
step.index.name = "trade#"
print(step.to_string())

# 5m vs 1m comparison note
print(f"\n{'─'*95}")
print("  📊 1m vs 5m Comparison (same strategy, same period):")
print(f"{'─'*95}")
print(f"  Timeframe   Trades/yr   Win Rate   Fixed Return   Tiered Return   Max DD (tiered)")
print(f"  {'─'*14} {'─'*11} {'─'*10} {'─'*14} {'─'*15} {'─'*15}")
print(f"  5m          ~8,400      35.1%      +3,224%        +16,664%        -88.9%")
print(f"  1m          {sf['trades']:>7,}      {sf['wr']:.1f}%      {sf['ret']:>+,.1f}%        {st['ret']:>+,.1f}%        {st['max_dd']:.1f}%")

print("\n" + "=" * 95)
print(f"  🏆  1m TIERED RESULT")
print(f"      $100  →  ${st['final']:,.2f}  ({st['ret']:,.1f}% return)")
print(f"      vs Fixed 1%: $100  →  ${sf['final']:,.2f}  ({sf['ret']:,.1f}% return)")
print(f"      Win Rate: {st['wr']:.1f}%  |  Max DD: {st['max_dd']:.1f}%  |  {st['trades']:,} trades")
print("=" * 95)

# Save
t_fixed.to_csv("hft1_fixed_trades.csv",  index=False)
t_tiered.to_csv("hft1_tiered_trades.csv", index=False)
print(f"\n📄 Fixed  → hft1_fixed_trades.csv")
print(f"📄 Tiered → hft1_tiered_trades.csv")
print("✅ Done!\n")
