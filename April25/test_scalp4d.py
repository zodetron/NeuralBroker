"""
BTC/USDT 5m SCALPING BACKTEST — 3 Years
Base: test_scalp.py logic on btcusdt_5m_3y.csv
Spread: 0.01% on entry only (limit order exits, no spread on exit)

S1–S4: original 4 strategies (unchanged)
S2 variants — RSI+VWAP with 6 improvements:
  V0. S2 Baseline          — original RSI bounce near VWAP
  V1. + EMA200 trend       — only long above EMA200, short below
  V2. + Momentum distance  — price must be >0.3% from VWAP (has momentum)
  V3. + Candle confirm     — strong close candle (body > 60% of range)
  V4. + ATR filter         — only trade when ATR > 20-bar avg (volatile)
  V5. + Session filter     — London (7-12 UTC) + NY (13-20 UTC) only
  V6. + Tiered sizing      — <$500=1%, $500-$2k=1.5%, $2k+=2% (capped)
  V7. ALL combined         — all 6 improvements together
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from data_loader import load_data

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
STARTING_CAPITAL = 100.0
LEVERAGE         = 1000
RISK_PCT         = 0.01
RR               = 2.0
SL_BASE          = 0.0015   # S2 baseline SL
SPREAD_PCT       = 0.0001   # 0.01% entry-only spread

# Tiered sizing for V6/V7
RISK_TIERS = [(0, 0.010), (500, 0.015), (2000, 0.020)]
def get_risk(capital):
    r = RISK_TIERS[0][1]
    for thresh, pct in RISK_TIERS:
        if capital >= thresh:
            r = pct
    return r

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
    d["ema9"]   = d["Close"].ewm(span=9,   adjust=False).mean()
    d["ema21"]  = d["Close"].ewm(span=21,  adjust=False).mean()
    d["ema50"]  = d["Close"].ewm(span=50,  adjust=False).mean()
    d["ema200"] = d["Close"].ewm(span=200, adjust=False).mean()

    # RSI (7-period)
    delta      = d["Close"].diff()
    gain       = delta.clip(lower=0).rolling(7).mean()
    loss       = (-delta.clip(upper=0)).rolling(7).mean()
    d["rsi"]   = 100 - (100 / (1 + gain / loss))

    # MACD
    ema12          = d["Close"].ewm(span=12, adjust=False).mean()
    ema26          = d["Close"].ewm(span=26, adjust=False).mean()
    d["macd"]      = ema12 - ema26
    d["macd_sig"]  = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_sig"]

    # Bollinger Bands
    d["bb_mid"]   = d["Close"].rolling(20).mean()
    bb_std        = d["Close"].rolling(20).std()
    d["bb_upper"] = d["bb_mid"] + 2 * bb_std
    d["bb_lower"] = d["bb_mid"] - 2 * bb_std

    # VWAP — daily reset, no apply() bug
    d["date"]    = d.index.date
    tp           = d["Close"] * d["Volume"]
    d["cum_vol"] = d.groupby("date")["Volume"].cumsum()
    d["cum_tp"]  = tp.groupby(d["date"]).cumsum()
    d["vwap"]    = d["cum_tp"] / d["cum_vol"]

    # ATR (14-period)
    hl  = d["High"] - d["Low"]
    hc  = (d["High"] - d["Close"].shift()).abs()
    lc  = (d["Low"]  - d["Close"].shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    d["atr"]     = tr.rolling(14).mean()
    d["atr_avg"] = d["atr"].rolling(20).mean()   # 20-bar ATR average

    # Volume spike
    d["vol_avg"]   = d["Volume"].rolling(20).mean()
    d["vol_spike"] = d["Volume"] > d["vol_avg"] * 1.5

    # 20-bar high/low
    d["high20"] = d["High"].rolling(20).max().shift(1)
    d["low20"]  = d["Low"].rolling(20).min().shift(1)

    # Candle body strength (body / range)
    d["body"]      = (d["Close"] - d["Open"]).abs()
    d["candle_rng"]= (d["High"] - d["Low"]).replace(0, np.nan)
    d["body_pct"]  = d["body"] / d["candle_rng"]

    # Session (UTC hours)
    d["hour"] = d.index.hour
    # London: 7–11 UTC, NY: 13–19 UTC
    d["in_session"] = d["hour"].isin(list(range(7, 12)) + list(range(13, 20)))

    return d

df = add_indicators(df)
df.dropna(inplace=True)
print(f"  Indicators added. {len(df):,} candles after warmup.\n")

# ─────────────────────────────────────────────
# BACKTEST ENGINE — spread on entry only
# supports fixed risk OR tiered risk
# supports full TP OR partial TP (50% at 1R, rest at 2R)
# ─────────────────────────────────────────────
def backtest(df, signal_func, name, sl_pct=SL_BASE, rr=RR,
             tiered=False, partial_tp=False):
    tp_pct      = sl_pct * rr
    half_tp_pct = sl_pct * 1.0   # 1R for partial exit
    capital     = STARTING_CAPITAL
    position    = None
    trades      = []
    entry_fill  = sl_price = tp_price = half_tp = risk_usd = 0.0
    entry_time  = None
    half_closed = False

    signals = signal_func(df)

    for i in range(1, len(df)):
        if capital <= 0:
            break

        price = float(df["Close"].iloc[i])
        ts    = df.index[i]
        sig   = int(signals.iloc[i])

        # ── Manage open position ──
        if position is not None:
            # Partial TP: close 50% at 1R
            if partial_tp and not half_closed:
                hit_half = (position == "long"  and price >= half_tp) or \
                           (position == "short" and price <= half_tp)
                if hit_half:
                    half_pnl = (risk_usd * 0.5) * 1.0   # 50% of risk at 1R
                    capital  = max(0.0, capital + half_pnl)
                    risk_usd = risk_usd * 0.5            # remaining half
                    half_closed = True

            hit_tp = (position == "long"  and price >= tp_price) or \
                     (position == "short" and price <= tp_price)
            hit_sl = (position == "long"  and price <= sl_price) or \
                     (position == "short" and price >= sl_price)

            if hit_tp or hit_sl:
                exit_price = tp_price if hit_tp else sl_price
                if position == "long":
                    actual_move = exit_price - entry_fill
                else:
                    actual_move = entry_fill - exit_price
                sl_dist = entry_fill * sl_pct
                pnl     = risk_usd * (actual_move / sl_dist)
                if partial_tp and half_closed:
                    pnl += (risk_usd * 0.5) * 1.0  # already booked above
                capital = max(0.0, capital + pnl)
                trades.append({
                    "strategy":   name,
                    "entry_time": entry_time,
                    "exit_time":  ts,
                    "direction":  position,
                    "entry":      round(entry_fill, 2),
                    "exit":       round(exit_price, 2),
                    "result":     "TP ✅" if hit_tp else "SL ❌",
                    "pnl":        round(pnl, 4),
                    "capital":    round(capital, 4),
                })
                position    = None
                half_closed = False
                continue

        # ── Open new position ──
        if position is None and sig != 0 and capital > 0:
            risk_pct   = get_risk(capital) if tiered else RISK_PCT
            risk_usd   = round(capital * risk_pct, 6)
            entry_time = ts
            half_closed = False

            if sig == 1:
                position   = "long"
                entry_fill = price * (1 + SPREAD_PCT)
                sl_price   = entry_fill * (1 - sl_pct)
                tp_price   = entry_fill * (1 + tp_pct)
                half_tp    = entry_fill * (1 + half_tp_pct)
            else:
                position   = "short"
                entry_fill = price * (1 - SPREAD_PCT)
                sl_price   = entry_fill * (1 + sl_pct)
                tp_price   = entry_fill * (1 - tp_pct)
                half_tp    = entry_fill * (1 - half_tp_pct)

    return pd.DataFrame(trades)

# ─────────────────────────────────────────────
# ORIGINAL 4 SIGNAL FUNCTIONS (unchanged)
# ─────────────────────────────────────────────
def sig_ema_scalp(df):
    s = pd.Series(0, index=df.index)
    cross_up   = (df["ema9"] > df["ema21"]) & (df["ema9"].shift(1) <= df["ema21"].shift(1))
    cross_down = (df["ema9"] < df["ema21"]) & (df["ema9"].shift(1) >= df["ema21"].shift(1))
    s[cross_up   & df["vol_spike"] & (df["Close"] > df["ema50"])] = 1
    s[cross_down & df["vol_spike"] & (df["Close"] < df["ema50"])] = -1
    return s

def sig_rsi_vwap(df):
    s = pd.Series(0, index=df.index)
    near_vwap = (df["Close"] - df["vwap"]).abs() / df["vwap"] < 0.002
    rsi_up    = (df["rsi"] > 30) & (df["rsi"].shift(1) <= 30)
    rsi_down  = (df["rsi"] < 70) & (df["rsi"].shift(1) >= 70)
    s[rsi_up   & near_vwap & (df["Close"] >= df["vwap"] * 0.998)] = 1
    s[rsi_down & near_vwap & (df["Close"] <= df["vwap"] * 1.002)] = -1
    return s

def sig_breakout_scalp(df):
    s = pd.Series(0, index=df.index)
    break_up   = (df["Close"] > df["high20"]) & (df["macd_hist"] > 0) & df["vol_spike"]
    break_down = (df["Close"] < df["low20"])  & (df["macd_hist"] < 0) & df["vol_spike"]
    s[break_up]   = 1
    s[break_down] = -1
    return s

def sig_macd_scalp(df):
    s = pd.Series(0, index=df.index)
    hist_up   = (df["macd_hist"] > 0) & (df["macd_hist"].shift(1) <= 0) & (df["Close"] > df["ema200"])
    hist_down = (df["macd_hist"] < 0) & (df["macd_hist"].shift(1) >= 0) & (df["Close"] < df["ema200"])
    s[hist_up]   = 1
    s[hist_down] = -1
    return s

# ─────────────────────────────────────────────
# RSI+VWAP VARIANT SIGNAL FUNCTIONS
# ─────────────────────────────────────────────
def _rsi_vwap_base(df):
    """Core RSI+VWAP conditions — reused by all variants."""
    near_vwap = (df["Close"] - df["vwap"]).abs() / df["vwap"] < 0.002
    rsi_up    = (df["rsi"] > 30) & (df["rsi"].shift(1) <= 30)
    rsi_down  = (df["rsi"] < 70) & (df["rsi"].shift(1) >= 70)
    long_base  = rsi_up   & near_vwap & (df["Close"] >= df["vwap"] * 0.998)
    short_base = rsi_down & near_vwap & (df["Close"] <= df["vwap"] * 1.002)
    return long_base, short_base

# V1: + EMA200 trend filter
def sig_v1_ema200(df):
    s = pd.Series(0, index=df.index)
    lb, sb = _rsi_vwap_base(df)
    s[lb & (df["Close"] > df["ema200"])] = 1
    s[sb & (df["Close"] < df["ema200"])] = -1
    return s

# V2: + Momentum distance (price >0.3% from VWAP — has momentum)
def sig_v2_momentum(df):
    s = pd.Series(0, index=df.index)
    lb, sb = _rsi_vwap_base(df)
    # Price bounced from VWAP and is now moving away — min 0.3% distance
    away_up   = (df["Close"] - df["vwap"]) / df["vwap"] > 0.003
    away_down = (df["vwap"] - df["Close"]) / df["vwap"] > 0.003
    s[lb & away_up]   = 1
    s[sb & away_down] = -1
    return s

# V3: + Candle confirmation (strong close — body > 60% of range)
def sig_v3_candle(df):
    s = pd.Series(0, index=df.index)
    lb, sb = _rsi_vwap_base(df)
    strong_bull = (df["body_pct"] > 0.6) & (df["Close"] > df["Open"])
    strong_bear = (df["body_pct"] > 0.6) & (df["Close"] < df["Open"])
    s[lb & strong_bull] = 1
    s[sb & strong_bear] = -1
    return s

# V4: + ATR volatility filter (ATR > 20-bar average — volatile market)
def sig_v4_atr(df):
    s = pd.Series(0, index=df.index)
    lb, sb = _rsi_vwap_base(df)
    volatile = df["atr"] > df["atr_avg"]
    s[lb & volatile] = 1
    s[sb & volatile] = -1
    return s

# V5: + Session filter (London 7-11 UTC + NY 13-19 UTC)
def sig_v5_session(df):
    s = pd.Series(0, index=df.index)
    lb, sb = _rsi_vwap_base(df)
    s[lb & df["in_session"]] = 1
    s[sb & df["in_session"]] = -1
    return s

# V6: Tiered sizing — signal same as baseline, sizing handled in backtest()
def sig_v6_tiered(df):
    return sig_rsi_vwap(df)   # same signal, tiered=True in backtest call

# V7: ALL combined — EMA200 + momentum + candle + ATR + session
def sig_v7_all(df):
    s = pd.Series(0, index=df.index)
    lb, sb = _rsi_vwap_base(df)
    trend_bull = df["Close"] > df["ema200"]
    trend_bear = df["Close"] < df["ema200"]
    away_up    = (df["Close"] - df["vwap"]) / df["vwap"] > 0.003
    away_down  = (df["vwap"] - df["Close"]) / df["vwap"] > 0.003
    strong_bull= (df["body_pct"] > 0.6) & (df["Close"] > df["Open"])
    strong_bear= (df["body_pct"] > 0.6) & (df["Close"] < df["Open"])
    volatile   = df["atr"] > df["atr_avg"]
    sess       = df["in_session"]
    s[lb & trend_bull & away_up   & strong_bull & volatile & sess] = 1
    s[sb & trend_bear & away_down & strong_bear & volatile & sess] = -1
    return s

# ─────────────────────────────────────────────
# RUN ALL STRATEGIES
# ─────────────────────────────────────────────
# (name, signal_fn, sl_pct, rr, tiered, partial_tp)
strategies = [
    # Original 4
    ("S1. EMA 9/21 + Volume",        sig_ema_scalp,    0.0020, 2.0, False, False),
    ("S2. RSI+VWAP baseline",        sig_rsi_vwap,     0.0015, 2.0, False, False),
    ("S3. Breakout + Momentum",      sig_breakout_scalp,0.0020, 2.0, False, False),
    ("S4. MACD Zero-Cross",          sig_macd_scalp,   0.0020, 2.0, False, False),
    # RSI+VWAP variants
    ("V1. +EMA200 trend",            sig_v1_ema200,    0.0015, 2.0, False, False),
    ("V2. +Momentum distance",       sig_v2_momentum,  0.0015, 2.0, False, False),
    ("V3. +Candle confirm",          sig_v3_candle,    0.0015, 2.0, False, False),
    ("V4. +ATR volatility",          sig_v4_atr,       0.0015, 2.0, False, False),
    ("V5. +Session filter",          sig_v5_session,   0.0015, 2.0, False, False),
    ("V6. +Tiered sizing",           sig_v6_tiered,    0.0015, 2.0, True,  False),
    ("V7. ALL combined",             sig_v7_all,       0.0015, 2.0, True,  False),
]

all_trades = []
summary    = []

for name, sig_fn, sl, rr, tiered, partial in strategies:
    print(f"  Running {name}...")
    trades = backtest(df, sig_fn, name, sl_pct=sl, rr=rr,
                      tiered=tiered, partial_tp=partial)
    all_trades.append(trades)

    if len(trades) == 0:
        summary.append({
            "Strategy": name, "Trades": 0, "Wins": 0, "Losses": 0,
            "Win%": "0%", "PnL ($)": 0.0, "Capital ($)": STARTING_CAPITAL,
            "Return%": "0%", "MaxDD%": "0%", "Avg Win": 0.0, "Avg Loss": 0.0,
        })
        continue

    tp_mask  = trades["result"].str.startswith("TP")
    sl_mask  = trades["result"].str.startswith("SL")
    wins     = tp_mask.sum()
    losses   = sl_mask.sum()
    pnl      = trades["pnl"].sum()
    final    = trades["capital"].iloc[-1]
    wr       = wins / len(trades) * 100
    ret      = (final - STARTING_CAPITAL) / STARTING_CAPITAL * 100
    avg_win  = trades.loc[tp_mask, "pnl"].mean() if wins   > 0 else 0
    avg_loss = trades.loc[sl_mask, "pnl"].mean() if losses > 0 else 0
    cap_s    = pd.concat([pd.Series([STARTING_CAPITAL]), trades["capital"].reset_index(drop=True)])
    max_dd   = ((cap_s - cap_s.cummax()) / cap_s.cummax() * 100).min()

    summary.append({
        "Strategy":    name,
        "Trades":      len(trades),
        "Wins":        int(wins),
        "Losses":      int(losses),
        "Win%":        f"{wr:.1f}%",
        "PnL ($)":     round(pnl, 2),
        "Capital ($)": round(final, 2),
        "Return%":     f"{ret:.1f}%",
        "MaxDD%":      f"{max_dd:.1f}%",
        "Avg Win":     round(avg_win, 3),
        "Avg Loss":    round(avg_loss, 3),
    })

print()

# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────
print("=" * 110)
print(f"   BTC/USDT 5m  |  ${STARTING_CAPITAL} Capital  |  1:{LEVERAGE} Leverage  |  {RISK_PCT*100}% Risk/Trade")
print(f"   Data: btcusdt_5m_3y.csv  |  {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}")
print(f"   Spread: {SPREAD_PCT*100:.2f}% entry-only  |  S2 baseline SL={SL_BASE*100:.2f}%  |  RR={RR}:1")
print("=" * 110)

summary_df = pd.DataFrame(summary)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 180)
print(summary_df.to_string(index=False))
print("=" * 110)

# ── RSI+VWAP variants comparison ──
print(f"\n  📊 RSI+VWAP VARIANTS — head-to-head vs baseline (S2):")
print(f"  {'─'*90}")
base_row = summary_df[summary_df["Strategy"] == "S2. RSI+VWAP baseline"].iloc[0]
base_pnl = base_row["PnL ($)"]
base_wr  = base_row["Win%"]
print(f"  {'Variant':<28} {'Trades':>7} {'Win%':>6} {'PnL ($)':>12} {'Return%':>9} {'MaxDD%':>8} {'vs Baseline':>12}")
print(f"  {'─'*28} {'─'*7} {'─'*6} {'─'*12} {'─'*9} {'─'*8} {'─'*12}")
for _, row in summary_df.iterrows():
    if row["Strategy"] not in ["S1. EMA 9/21 + Volume","S3. Breakout + Momentum","S4. MACD Zero-Cross"]:
        delta = round(row["PnL ($)"] - base_pnl, 2)
        flag  = "✅" if delta >= 0 else "❌"
        print(f"  {row['Strategy']:<28} {row['Trades']:>7,} {row['Win%']:>6} "
              f"${row['PnL ($)']:>11,.2f} {row['Return%']:>9} {row['MaxDD%']:>8} "
              f"${delta:>+10,.2f} {flag}")

# ── Last 15 trades for each strategy ──
for i, (name, *_) in enumerate(strategies):
    trades = all_trades[i]
    if len(trades) == 0:
        print(f"\n  {name}: No signals.")
        continue
    print(f"\n{'─'*110}")
    print(f"  {name}  — Last 15 Trades")
    print(f"{'─'*110}")
    print(trades[["entry_time","exit_time","direction","entry","exit","result","pnl","capital"]].tail(15).to_string(index=False))

# ── Winner ──
print("\n" + "=" * 110)
best_idx = summary_df["PnL ($)"].idxmax()
best     = summary_df.iloc[best_idx]
print(f"  🏆  WINNER    : {best['Strategy']}")
print(f"      Trades    : {best['Trades']:,}")
print(f"      PnL       : ${best['PnL ($)']:,.2f}")
print(f"      Return    : {best['Return%']}")
print(f"      Win Rate  : {best['Win%']}")
print(f"      Max DD    : {best['MaxDD%']}")
print(f"      Avg Win   : ${best['Avg Win']:,.3f}  |  Avg Loss: ${best['Avg Loss']:,.3f}")
print("=" * 110)

# Save
valid = [t for t in all_trades if len(t) > 0]
if valid:
    pd.concat(valid, ignore_index=True).to_csv("scalp4d_trades.csv", index=False)
    print(f"\n📄 Full trade log → scalp4d_trades.csv")
print("✅ Done!\n")
