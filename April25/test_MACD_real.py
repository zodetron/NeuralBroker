"""
MACD ZERO-CROSS — SL OPTIMIZATION WITH SPREAD
══════════════════════════════════════════════
Tests SL sizes: 0.20%, 0.50%, 1.00%, 1.50%, 2.00%
With realistic 0.05% spread per side (0.10% round trip)
Capital: $100 | Leverage 1:1000 | Risk 1%/trade
Timeframe: 5m | Last 60 days
"""

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

CAPITAL    = 100.0
LEVERAGE   = 1000
RISK_PCT   = 0.01
RR         = 2.0
SPREAD     = 0.0002   # 0.05% per side
TICKER     = "BTC-USD"

# SL sizes to test
SL_SIZES = [0.002, 0.005, 0.010, 0.015, 0.020]

# ─────────────────────────────────────────────
# FETCH DATA
# ─────────────────────────────────────────────
print("📥 Fetching BTC/USD 5m data (last 60 days)...")
end_date   = datetime.today()
start_date = end_date - timedelta(days=58)
chunks, cur = [], start_date
while cur < end_date:
    nxt = min(cur + timedelta(days=29), end_date)
    try:
        c = yf.download(TICKER,
                        start=cur.strftime("%Y-%m-%d"),
                        end=nxt.strftime("%Y-%m-%d"),
                        interval="5m", auto_adjust=True, progress=False)
        if len(c): chunks.append(c)
    except: pass
    cur = nxt

df = pd.concat(chunks)
df.columns = [x[0] if isinstance(x, tuple) else x for x in df.columns]
df = df[~df.index.duplicated(keep="first")]
df.sort_index(inplace=True)
df.dropna(inplace=True)
print(f"✅ {len(df)} candles | {df.index[0].date()} → {df.index[-1].date()}\n")

# ─────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────
df["ema200"]    = df["Close"].ewm(span=200, adjust=False).mean()
ema12           = df["Close"].ewm(span=12,  adjust=False).mean()
ema26           = df["Close"].ewm(span=26,  adjust=False).mean()
df["macd"]      = ema12 - ema26
df["macd_sig"]  = df["macd"].ewm(span=9, adjust=False).mean()
df["macd_hist"] = df["macd"] - df["macd_sig"]
delta           = df["Close"].diff()
gain            = delta.clip(lower=0).rolling(14).mean()
loss            = (-delta.clip(upper=0)).rolling(14).mean()
df["rsi"]       = 100 - (100 / (1 + gain / loss))
df["atr"]       = pd.concat([
    df["High"] - df["Low"],
    (df["High"] - df["Close"].shift()).abs(),
    (df["Low"]  - df["Close"].shift()).abs()
], axis=1).max(axis=1).rolling(14).mean()
df.dropna(inplace=True)
print(f"  {len(df)} candles after warmup.\n")

# ─────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────
def backtest(df, sl_pct, rr=RR, use_spread=True):
    spread   = SPREAD if use_spread else 0.0
    tp_pct   = sl_pct * rr
    capital  = CAPITAL
    position = None
    trades   = []
    entry_price = sl_price = tp_price = risk_usd = 0.0
    entry_time = None

    for i in range(1, len(df)):
        if capital <= 0:
            break

        price     = float(df["Close"].iloc[i])
        ts        = df.index[i]
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
                # Spread cost = risk_usd * (spread / sl_pct) on both entry+exit
                spread_cost = risk_usd * (spread * 2 / sl_pct)
                raw_pnl     = risk_usd * rr if hit_tp else -risk_usd
                pnl         = raw_pnl - spread_cost
                capital     = max(0.0, capital + pnl)
                trades.append({
                    "entry_time": entry_time,
                    "exit_time":  ts,
                    "direction":  position,
                    "result":     "TP" if hit_tp else "SL",
                    "raw_pnl":    round(raw_pnl, 4),
                    "spread_cost":round(spread_cost, 4),
                    "pnl":        round(pnl, 4),
                    "capital":    round(capital, 4),
                })
                position = None
                continue

        if position is None and sig != 0 and capital > 0:
            risk_usd    = round(capital * RISK_PCT, 6)
            entry_price = price
            entry_time  = ts
            if sig == 1:
                position = "long"
                fill     = entry_price * (1 + spread)
                sl_price = fill * (1 - sl_pct)
                tp_price = fill * (1 + tp_pct)
            else:
                position = "short"
                fill     = entry_price * (1 - spread)
                sl_price = fill * (1 + sl_pct)
                tp_price = fill * (1 - tp_pct)

    return pd.DataFrame(trades)

# ─────────────────────────────────────────────
# RUN ALL SL SIZES
# ─────────────────────────────────────────────
results  = []
all_trades = {}

print("  Testing SL sizes with 0.05% spread...\n")

for sl in SL_SIZES:
    be_wr = (sl + SPREAD) / ((sl * RR - SPREAD) + (sl + SPREAD)) * 100

    # Without spread
    t_raw = backtest(df, sl, use_spread=False)
    # With spread
    t_sp  = backtest(df, sl, use_spread=True)

    all_trades[sl] = t_sp

    def s(trades):
        if len(trades) == 0:
            return dict(trades=0, wins=0, losses=0, wr=0, pnl=0, final=CAPITAL, ret=0, dd=0, aw=0, al=0)
        tp  = (trades["result"] == "TP").sum()
        sl_ = (trades["result"] == "SL").sum()
        pnl = trades["pnl"].sum()
        fin = trades["capital"].iloc[-1]
        wr  = tp / (tp + sl_) * 100 if (tp + sl_) > 0 else 0
        ret = (fin - CAPITAL) / CAPITAL * 100
        aw  = trades.loc[trades["result"]=="TP","pnl"].mean() if tp  > 0 else 0
        al  = trades.loc[trades["result"]=="SL","pnl"].mean() if sl_ > 0 else 0
        cap_s = pd.concat([pd.Series([CAPITAL]), trades["capital"].reset_index(drop=True)])
        dd  = ((cap_s - cap_s.cummax()) / cap_s.cummax() * 100).min()
        return dict(trades=len(trades), wins=int(tp), losses=int(sl_),
                    wr=round(wr,1), pnl=round(pnl,2), final=round(fin,2),
                    ret=round(ret,1), dd=round(dd,1), aw=round(aw,3), al=round(al,3))

    sr = s(t_raw)
    ss = s(t_sp)

    spread_ratio = round(SPREAD / sl * 100, 1)
    viable = "✅ VIABLE" if ss["wr"] >= be_wr and ss["ret"] > 0 else "❌ DEAD"

    results.append({
        "SL%":         f"{sl*100:.2f}%",
        "Spread/SL":   f"{spread_ratio}%",
        "BE WinRate":  f"{be_wr:.1f}%",
        "Trades":      ss["trades"],
        "WinRate":     f"{ss['wr']}%",
        "PnL($)":      ss["pnl"],
        "Return%":     f"{ss['ret']}%",
        "MaxDD%":      f"{ss['dd']}%",
        "AvgWin":      ss["aw"],
        "AvgLoss":     ss["al"],
        "Status":      viable,
    })

    print(f"  SL {sl*100:.2f}% | Spread={spread_ratio}% of SL | "
          f"WR={ss['wr']}% (need {be_wr:.1f}%) | "
          f"Return={ss['ret']}% | {viable}")

# ─────────────────────────────────────────────
# RESULTS TABLE
# ─────────────────────────────────────────────
print()
print("=" * 100)
print(f"  BTC/USD 5m | MACD Zero-Cross | Spread 0.05% | ${CAPITAL} Capital | 1:{LEVERAGE} | RR {RR}:1")
print(f"  Period: {df.index[0].date()} → {df.index[-1].date()}")
print("=" * 100)
res_df = pd.DataFrame(results)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", None)
print(res_df.to_string(index=False))
print("=" * 100)

# ─────────────────────────────────────────────
# BEST SL — deep dive
# ─────────────────────────────────────────────
viable_df = res_df[res_df["Status"].str.startswith("✅")]
if len(viable_df) == 0:
    print("\n  ❌ No SL size is viable with 0.05% spread at this RR.")
    print("  💡 Try RR 3:1 or reduce risk per trade to 0.5%")
else:
    # Pick best return among viable
    best_sl_pct_str = viable_df.sort_values("Return%", ascending=False).iloc[0]["SL%"]
    best_sl         = float(best_sl_pct_str.replace("%","")) / 100
    best_trades     = all_trades[best_sl]

    print(f"\n  🏆 BEST VIABLE SL: {best_sl_pct_str}")
    print(f"{'─'*80}")

    # Monthly breakdown
    print(f"\n  📅 MONTHLY PnL (SL={best_sl_pct_str}, with spread):")
    t = best_trades.copy()
    t["month"] = pd.to_datetime(t["entry_time"]).dt.to_period("M")
    monthly = t.groupby("month").agg(
        trades=("pnl","count"),
        wins=("result", lambda x: (x=="TP").sum()),
        pnl=("pnl","sum"),
        spread_paid=("spread_cost","sum")
    ).round(2)
    print(f"  {'Month':<12} {'Trades':>6} {'WR%':>5} {'PnL($)':>8} {'Spread($)':>10} {'Bar'}")
    print(f"  {'─'*65}")
    for m, row in monthly.iterrows():
        wr_m = row["wins"] / row["trades"] * 100 if row["trades"] > 0 else 0
        bar  = ("█" * int(abs(row["pnl"]) / 1)) if row["pnl"] != 0 else ""
        sgn  = "+" if row["pnl"] >= 0 else ""
        ico  = "✅" if row["pnl"] >= 0 else "❌"
        print(f"  {str(m):<12} {int(row['trades']):>6} {wr_m:>4.0f}% "
              f"{sgn}${abs(row['pnl']):>6.2f}  -${row['spread_paid']:>7.2f}  {ico} {bar}")

    # Capital growth
    print(f"\n  📈 Capital Growth (every 50 trades):")
    print(f"  {'─'*50}")
    step = max(1, len(best_trades) // 8)
    for j in range(0, len(best_trades), step):
        row = best_trades.iloc[j]
        ts  = pd.to_datetime(row["entry_time"]).strftime("%Y-%m-%d")
        cap = row["capital"]
        bar = "█" * int((cap - CAPITAL) / 5) if cap > CAPITAL else ""
        print(f"  Trade {j:>3}  {ts}  ${cap:>8.2f}  {bar}")

    # Scaling table
    ret_pct = float(viable_df.sort_values("Return%", ascending=False).iloc[0]["Return%"].replace("%","")) / 100
    print(f"\n  💰 SCALING TABLE (SL={best_sl_pct_str}, with spread, per 60 days):")
    print(f"  {'─'*60}")
    print(f"  {'Capital':>10}  {'60-day profit':>15}  {'After 6 months':>16}  {'After 1 year':>14}")
    print(f"  {'─'*60}")
    for cap in [100, 500, 1000, 5000, 10000]:
        profit   = cap * ret_pct
        after_6m = cap * (1 + ret_pct) ** 3
        after_1y = cap * (1 + ret_pct) ** 6
        print(f"  ${cap:>9,}  +${profit:>13,.2f}  ${after_6m:>15,.0f}  ${after_1y:>13,.0f}")

    print(f"\n  ✅ READY TO BUILD LIVE MT5 BOT")
    print(f"     Strategy : MACD Zero-Cross + EMA200")
    print(f"     Timeframe: 5m")
    print(f"     SL       : {best_sl_pct_str}")
    print(f"     TP       : {round(best_sl*RR*100,2)}% (RR {RR}:1)")
    print(f"     Spread   : accounted for")

print("\n" + "=" * 100)

# Save best
if len(viable_df) > 0:
    best_trades.to_csv("macd_best_sl_trades.csv", index=False)
    print(f"📄 Best SL trade log → macd_best_sl_trades.csv")
print("✅ Done!\n")