"""
BTC/USDT 5m SCALPING BACKTEST — Realistic Execution
Exact copy of test_scalp.py with two changes:
  1. Data loaded from btcusdt_5m_3y.csv (3 years, Binance) instead of yfinance
  2. Spread applied on every fill (0.01% per side, 0.02% round trip)
     No slippage, no commission — spread only.

4 Scalping Strategies:
  S1. EMA 9/21 + Volume Spike
  S2. RSI Bounce + VWAP
  S3. 20-Bar Breakout + Momentum
  S4. MACD Zero-Cross + EMA200

Capital: $100 | Leverage: 1:1000 | Risk: 1%/trade
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

# Spread only — no slippage, no commission
SPREAD_PCT    = 0.0001        # 0.01% per side
COST_PER_SIDE = SPREAD_PCT    # 0.01% per side → 0.02% round trip

# ─────────────────────────────────────────────
# LOAD DATA — from CSV (instant)
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

    # RSI (7-period for scalping)
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
    d["atr"] = tr.rolling(14).mean()

    # Volume spike (volume > 1.5x 20-bar avg)
    d["vol_avg"]   = d["Volume"].rolling(20).mean()
    d["vol_spike"] = d["Volume"] > d["vol_avg"] * 1.5

    # 20-bar high/low for breakout
    d["high20"] = d["High"].rolling(20).max().shift(1)
    d["low20"]  = d["Low"].rolling(20).min().shift(1)

    return d

df = add_indicators(df)
df.dropna(inplace=True)
print(f"  Indicators added. {len(df):,} candles after warmup.\n")

# ─────────────────────────────────────────────
# BACKTEST ENGINE — with spread on every fill
# ─────────────────────────────────────────────
def backtest(df, signal_func, name, sl_pct, rr=RR):
    """
    Spread applied on entry and exit:
      Long  entry: fill = close * (1 + COST_PER_SIDE)   — buy at ask
      Short entry: fill = close * (1 - COST_PER_SIDE)   — sell at bid
      Long  exit:  fill = theo  * (1 - COST_PER_SIDE)   — sell at bid
      Short exit:  fill = theo  * (1 + COST_PER_SIDE)   — buy at ask
    SL/TP levels set from actual fill price.
    PnL scaled by actual move vs SL distance.
    """
    tp_pct      = sl_pct * rr
    capital     = STARTING_CAPITAL
    position    = None
    trades      = []
    entry_fill  = sl_price = tp_price = risk_usd = 0.0
    entry_time  = None

    signals = signal_func(df)

    for i in range(1, len(df)):
        if capital <= 0:
            break

        close = float(df["Close"].iloc[i])
        ts    = df.index[i]
        sig   = int(signals.iloc[i])

        # ── Manage open position ──
        if position is not None:
            hit_tp = (position == "long"  and close >= tp_price) or \
                     (position == "short" and close <= tp_price)
            hit_sl = (position == "long"  and close <= sl_price) or \
                     (position == "short" and close >= sl_price)

            if hit_tp or hit_sl:
                # Exit fill is worse than theoretical SL/TP
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
                    "strategy":   name,
                    "entry_time": entry_time,
                    "exit_time":  ts,
                    "direction":  position,
                    "entry":      round(entry_fill, 2),
                    "exit":       round(exit_fill, 2),
                    "result":     "TP ✅" if hit_tp else "SL ❌",
                    "pnl":        round(pnl, 4),
                    "capital":    round(capital, 4),
                })
                position = None
                continue

        # ── Open new position ──
        if position is None and sig != 0 and capital > 0:
            risk_usd   = round(capital * RISK_PCT, 6)
            entry_time = ts

            if sig == 1:
                position   = "long"
                entry_fill = close * (1 + COST_PER_SIDE)   # buy at ask
                sl_price   = entry_fill * (1 - sl_pct)
                tp_price   = entry_fill * (1 + tp_pct)
            else:
                position   = "short"
                entry_fill = close * (1 - COST_PER_SIDE)   # sell at bid
                sl_price   = entry_fill * (1 + sl_pct)
                tp_price   = entry_fill * (1 - tp_pct)

    return pd.DataFrame(trades)

# ─────────────────────────────────────────────
# SCALPING SIGNAL FUNCTIONS — identical to test_scalp.py
# ─────────────────────────────────────────────

# S1: EMA 9/21 Crossover + Volume Spike
def sig_ema_scalp(df):
    s = pd.Series(0, index=df.index)
    cross_up   = (df["ema9"] > df["ema21"]) & (df["ema9"].shift(1) <= df["ema21"].shift(1))
    cross_down = (df["ema9"] < df["ema21"]) & (df["ema9"].shift(1) >= df["ema21"].shift(1))
    s[cross_up   & df["vol_spike"] & (df["Close"] > df["ema50"])] = 1
    s[cross_down & df["vol_spike"] & (df["Close"] < df["ema50"])] = -1
    return s

# S2: RSI Bounce off VWAP
def sig_rsi_vwap(df):
    s = pd.Series(0, index=df.index)
    rsi_up   = (df["rsi"] > 30) & (df["rsi"].shift(1) <= 30)
    rsi_down = (df["rsi"] < 70) & (df["rsi"].shift(1) >= 70)
    s[rsi_up   & (df["Close"] >= df["vwap"] * 0.998)] = 1
    s[rsi_down & (df["Close"] <= df["vwap"] * 1.002)] = -1
    return s

# S3: 20-Bar High/Low Breakout + Momentum
def sig_breakout_scalp(df):
    s = pd.Series(0, index=df.index)
    break_up   = (df["Close"] > df["high20"]) & (df["macd_hist"] > 0) & df["vol_spike"]
    break_down = (df["Close"] < df["low20"])  & (df["macd_hist"] < 0) & df["vol_spike"]
    s[break_up]   = 1
    s[break_down] = -1
    return s

# S4: MACD Zero-Cross + EMA200 Trend
def sig_macd_scalp(df):
    s = pd.Series(0, index=df.index)
    hist_up   = (df["macd_hist"] > 0) & (df["macd_hist"].shift(1) <= 0) & (df["Close"] > df["ema200"])
    hist_down = (df["macd_hist"] < 0) & (df["macd_hist"].shift(1) >= 0) & (df["Close"] < df["ema200"])
    s[hist_up]   = 1
    s[hist_down] = -1
    return s

# ─────────────────────────────────────────────
# RUN ALL STRATEGIES
# ─────────────────────────────────────────────
strategies = [
    ("S1. EMA 9/21 + Volume",   sig_ema_scalp,       0.0020),
    ("S2. RSI Bounce + VWAP",   sig_rsi_vwap,        0.0015),
    ("S3. Breakout + Momentum", sig_breakout_scalp,   0.0020),
    ("S4. MACD Zero-Cross",     sig_macd_scalp,       0.0020),
]

all_trades = []
summary    = []

for name, sig_fn, sl in strategies:
    print(f"  Running {name}...")
    trades = backtest(df, sig_fn, name, sl_pct=sl)
    all_trades.append(trades)

    if len(trades) == 0:
        summary.append({
            "Strategy":    name,
            "Trades":      0,
            "Wins":        0,
            "Losses":      0,
            "Win%":        "0%",
            "PnL ($)":     0.0,
            "Capital ($)": STARTING_CAPITAL,
            "Return%":     "0%",
            "MaxDD%":      "0%",
            "Avg Win":     0.0,
            "Avg Loss":    0.0,
            "Cost/trade":  f"{COST_PER_SIDE*2*100:.3f}%",
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

    cap_s  = pd.concat([pd.Series([STARTING_CAPITAL]), trades["capital"].reset_index(drop=True)])
    max_dd = ((cap_s - cap_s.cummax()) / cap_s.cummax() * 100).min()

    # Breakeven WR for this strategy's SL
    rt     = COST_PER_SIDE * 2
    net_tp = sl * RR - rt
    net_sl = sl + rt
    be_wr  = 1 / (1 + net_tp / net_sl) * 100

    summary.append({
        "Strategy":    name,
        "Trades":      len(trades),
        "Wins":        int(wins),
        "Losses":      int(losses),
        "Win%":        f"{wr:.1f}%",
        "BE_WR":       f"{be_wr:.1f}%",
        "Margin":      f"{wr - be_wr:+.1f}%",
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
print(f"   BTC/USDT 5m SCALPING — REALISTIC (Spread Only)")
print(f"   Data: btcusdt_5m_3y.csv  |  ${STARTING_CAPITAL} Capital  |  1:{LEVERAGE} Leverage  |  {RISK_PCT*100}% Risk/Trade")
print(f"   Spread: {COST_PER_SIDE*100:.2f}%/side ({COST_PER_SIDE*2*100:.2f}% round trip)  |  RR: {RR}:1")
print(f"   Period: {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}  |  3 Years")
print("=" * 110)

summary_df = pd.DataFrame(summary)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 180)
print(summary_df.to_string(index=False))
print("=" * 110)

# Trade logs — last 15 per strategy
for i, (name, _, _) in enumerate(strategies):
    trades = all_trades[i]
    if len(trades) == 0:
        print(f"\n  {name}: No signals generated.")
        continue
    print(f"\n{'─'*110}")
    print(f"  {name}  — Last 15 Trades")
    print(f"{'─'*110}")
    print(trades[["entry_time","exit_time","direction","entry","exit","result","pnl","capital"]].tail(15).to_string(index=False))

# Winner
print("\n" + "=" * 110)
best_idx = summary_df["PnL ($)"].idxmax()
best     = summary_df.iloc[best_idx]
print(f"  🏆  WINNER    : {best['Strategy']}")
print(f"      Trades    : {best['Trades']}")
print(f"      PnL       : ${best['PnL ($)']}")
print(f"      Return    : {best['Return%']}")
print(f"      Win Rate  : {best['Win%']}  (Breakeven: {best['BE_WR']}  Margin: {best['Margin']})")
print(f"      Max DD    : {best['MaxDD%']}")
print(f"      Avg Win   : ${best['Avg Win']}  |  Avg Loss: ${best['Avg Loss']}")
print(f"      Spread cost: {COST_PER_SIDE*100:.2f}%/side ({COST_PER_SIDE*2*100:.2f}% RT)")
print("=" * 110)

# Save
valid = [t for t in all_trades if len(t) > 0]
if valid:
    pd.concat(valid, ignore_index=True).to_csv("scalp4d_trades.csv", index=False)
    print(f"\n📄 Full trade log → scalp4d_trades.csv")
print("✅ Done!\n")
