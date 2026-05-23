"""
ATR Breakout Strategy — XAUUSD 1m  |  XAU_1m_data.csv (2004–2025)
Spread: 0.025%  |  Capital: $100  |  Leverage: 1:1000  |  Min Lot: 0.01
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

CSV_FILE = "XAU_1m_data.csv"

CAPITAL  = 100.0
RISK_PCT = 0.01
SPREAD   = 0.00025
LEVERAGE = 1000
MIN_LOT  = 0.01
LOT_STEP = 0.01

SL_PCT = 0.002   # 0.2%
RR     = 3.5     # TP = SL * RR = 0.7%


# ── LOAD ──────────────────────────────────────────────────────────────────────
print(f"Loading {CSV_FILE}...", end=" ", flush=True)
df_raw = pd.read_csv(
    CSV_FILE,
    sep=";",
    parse_dates=["Date"],
    date_format="%Y.%m.%d %H:%M",
)
df_raw.rename(columns={"Date": "open_time"}, inplace=True)
df_raw = df_raw[["open_time", "Open", "High", "Low", "Close", "Volume"]]
for c in ["Open", "High", "Low", "Close"]:
    df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")
df_raw.dropna(inplace=True)
df_raw = df_raw[~df_raw["open_time"].duplicated(keep="first")]
df_raw.sort_values("open_time", inplace=True)
df_raw.set_index("open_time", inplace=True)
print(f"{len(df_raw):,} candles | {df_raw.index[0].strftime('%Y-%m-%d')} → {df_raw.index[-1].strftime('%Y-%m-%d')}")


# ── INDICATORS ────────────────────────────────────────────────────────────────
def build_indicators(df):
    d = df.copy()
    c = d["Close"]
    high_low = d["High"] - d["Low"]
    high_pc  = (d["High"] - c.shift()).abs()
    low_pc   = (d["Low"]  - c.shift()).abs()
    tr       = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
    d["atr14"] = tr.ewm(span=14, adjust=False).mean()
    return d.dropna()


print("Building ATR indicator...", end=" ", flush=True)
df = build_indicators(df_raw)
print(f"done. {len(df):,} candles.\n")


# ── SIGNAL ────────────────────────────────────────────────────────────────────
def sig_ATR_Breakout(d):
    prev_c = d["Close"].shift()
    s = pd.Series(0, index=d.index)
    s[d["Close"] > prev_c + d["atr14"]] =  1
    s[d["Close"] < prev_c - d["atr14"]] = -1
    return s


# ── LOT CALCULATOR ────────────────────────────────────────────────────────────
def calc_lots(capital, price, sl_pct):
    risk_usd       = capital * RISK_PCT
    sl_usd_per_lot = price * sl_pct
    lots           = max(MIN_LOT, round((risk_usd / sl_usd_per_lot) / LOT_STEP) * LOT_STEP)
    lots           = round(lots, 2)
    if lots * price / LEVERAGE > capital:
        lots = max(MIN_LOT, round((capital * LEVERAGE / price) / LOT_STEP) * LOT_STEP)
        lots = round(lots, 2)
    return lots


# ── BACKTEST ──────────────────────────────────────────────────────────────────
def backtest(df, signals, sl_pct, rr):
    cap        = CAPITAL
    pos        = None
    entry_fill = sl_p = tp_p = lots = 0.0
    tp_pct     = sl_pct * rr
    peak       = cap
    trades     = []

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
                trades.append({
                    "exit_time": ts,
                    "direction": "LONG" if pos == 1 else "SHORT",
                    "result":    "TP" if hit_tp else "SL",
                    "lots":      lots,
                    "entry":     round(entry_fill, 3),
                    "exit":      round(exit_px, 3),
                    "pnl":       round(pnl, 4),
                    "capital":   round(cap, 4),
                    "peak":      round(peak, 4),
                })
                pos = None

        if pos is None and sig != 0 and cap > 0:
            lots       = calc_lots(cap, price, sl_pct)
            entry_fill = price * (1 + SPREAD) if sig == 1 else price * (1 - SPREAD)
            sl_p       = entry_fill * (1 - sl_pct) if sig == 1 else entry_fill * (1 + sl_pct)
            tp_p       = entry_fill * (1 + tp_pct) if sig == 1 else entry_fill * (1 - tp_pct)
            pos        = sig

    return pd.DataFrame(trades)


# ── RUN ───────────────────────────────────────────────────────────────────────
print("Running ATR Breakout backtest on full historical data...")
signals = sig_ATR_Breakout(df)
t = backtest(df, signals, SL_PCT, RR)

if t.empty:
    print("No trades generated.")
    exit()

wins   = (t["result"] == "TP").sum()
losses = (t["result"] == "SL").sum()
wr     = wins / len(t) * 100
final  = t["capital"].iloc[-1]
total_r = (final - CAPITAL) / CAPITAL * 100
avg_lot = t["lots"].mean()

caps  = pd.concat([pd.Series([CAPITAL]), t["capital"].reset_index(drop=True)])
peak  = caps.cummax()
dd    = (caps - peak) / peak * 100
mdd   = dd.min()
mdd_usd = (caps - peak).min()

label = f"ATR_Breakout | SL {SL_PCT*100:.1f}% | RR {RR} | TP {SL_PCT*RR*100:.2f}%"

print(f"\n{'═'*85}")
print(f"  XAUUSD 1m  |  XAU_1m_data.csv (2004–2025)  |  $100  |  0.025% Spread  |  1:1000")
print(f"{'═'*85}")
print(f"  {label}")
print(f"  {'─'*75}")
print(f"  Trades: {len(t):,}  |  Wins: {wins:,}  |  Losses: {losses:,}  |  Win Rate: {wr:.1f}%")
print(f"  Final Capital: ${final:,.2f}  |  Total Return: {total_r:+.1f}%")
print(f"  Max Drawdown: {mdd:.1f}%  (${abs(mdd_usd):.2f})  |  Avg Lot: {avg_lot:.3f}")

# ── YEARLY BREAKDOWN ──────────────────────────────────────────────────────────
t["year"] = pd.to_datetime(t["exit_time"]).dt.year
years = sorted(t["year"].unique())
prev_cap = CAPITAL
yr_rows = []
for y in years:
    yt  = t[t["year"] == y]
    end = yt["capital"].iloc[-1]
    yr_rows.append({
        "year":    y,
        "trades":  len(yt),
        "win%":    (yt["result"] == "TP").sum() / len(yt) * 100,
        "ret%":    (end - prev_cap) / prev_cap * 100,
        "capital": end,
        "avg_lot": yt["lots"].mean(),
    })
    prev_cap = end

yr_df = pd.DataFrame(yr_rows)
rets  = yr_df["ret%"]

print(f"\n  {'Year':<8} {'Trades':>7} {'Win%':>6} {'Return%':>10} {'Capital':>12} {'AvgLot':>8}  Bar")
print(f"  {'-'*75}")
for _, r in yr_df.iterrows():
    ret = r["ret%"]
    bar = ("█" * min(int(abs(ret) / 5), 30)) if ret >= 0 else ("▓" * min(int(abs(ret) / 5), 30))
    sgn = "+" if ret >= 0 else ""
    print(f"  {int(r['year']):<8} {int(r['trades']):>7} {r['win%']:>5.1f}% "
          f"{sgn}{ret:>9.2f}% ${r['capital']:>10.2f} {r['avg_lot']:>8.3f}  {bar}")
print(f"  {'-'*75}")
print(f"  Avg: {rets.mean():+.2f}%/yr  |  Best: {rets.max():+.2f}%  |  "
      f"Worst: {rets.min():+.2f}%  |  Positive: {(rets>0).sum()}/{len(rets)} years")

print(f"\n{'═'*85}")
print(f"  Strategy  : {label}")
print(f"  Return    : {total_r:+.1f}%  |  MDD: {mdd:.1f}% (${abs(mdd_usd):.2f})")
print(f"  Final     : ${final:,.2f}  from $100  over {len(years)} years")
print(f"{'═'*85}")
