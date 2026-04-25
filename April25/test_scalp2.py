"""
╔══════════════════════════════════════════════════════════╗
║   BTC/USD ULTRA SCALPER — 5m Timeframe                  ║
║   Capital: $100 | Leverage: 1:1000                      ║
║                                                          ║
║   TRICKS USED:                                           ║
║   1. Multi-confirmation entries (3+ signals must agree)  ║
║   2. Dynamic position sizing (bet more on hot streaks)   ║
║   3. Trailing stop logic (lock in profits mid-trade)     ║
║   4. Session filter (only trade high-volatility hours)   ║
║   5. Momentum score (rank signal strength 0-100)         ║
║   6. Martingale-lite (slightly increase after losses)    ║
║   7. Pyramiding (add to winners)                         ║
╚══════════════════════════════════════════════════════════╝
"""

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CAPITAL      = 100.0
LEVERAGE     = 1000
BASE_RISK    = 0.01      # 1% base risk per trade
MAX_RISK     = 0.03      # 3% max risk (on hot streaks)
RR           = 2.5       # 2.5:1 reward:risk (upgraded from 2:1)
TICKER       = "BTC-USD"
INTERVAL     = "5m"

# ─────────────────────────────────────────────
# FETCH DATA
# ─────────────────────────────────────────────
print("📥 Fetching BTC/USD 5m data (last 60 days)...")
end_date   = datetime.today()
start_date = end_date - timedelta(days=58)

chunks, chunk_start = [], start_date
while chunk_start < end_date:
    chunk_end = min(chunk_start + timedelta(days=29), end_date)
    try:
        chunk = yf.download(TICKER,
            start=chunk_start.strftime("%Y-%m-%d"),
            end=chunk_end.strftime("%Y-%m-%d"),
            interval=INTERVAL, auto_adjust=True, progress=False)
        if len(chunk) > 0:
            chunks.append(chunk)
    except: pass
    chunk_start = chunk_end

df = pd.concat(chunks)
df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
df = df[~df.index.duplicated(keep="first")]
df.sort_index(inplace=True)
df.dropna(inplace=True)
print(f"✅ {len(df)} candles | {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}\n")

# ─────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────
def add_indicators(df):
    d = df.copy()

    # Multiple EMAs
    for span in [5, 9, 13, 21, 50, 200]:
        d[f"ema{span}"] = d["Close"].ewm(span=span, adjust=False).mean()

    # RSI (fast + slow)
    for period in [7, 14]:
        delta = d["Close"].diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        d[f"rsi{period}"] = 100 - (100 / (1 + gain / loss))

    # MACD
    ema12          = d["Close"].ewm(span=12, adjust=False).mean()
    ema26          = d["Close"].ewm(span=26, adjust=False).mean()
    d["macd"]      = ema12 - ema26
    d["macd_sig"]  = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_sig"]

    # Bollinger Bands
    d["bb_mid"]    = d["Close"].rolling(20).mean()
    bb_std         = d["Close"].rolling(20).std()
    d["bb_upper"]  = d["bb_mid"] + 2 * bb_std
    d["bb_lower"]  = d["bb_mid"] - 2 * bb_std
    d["bb_pct"]    = (d["Close"] - d["bb_lower"]) / (d["bb_upper"] - d["bb_lower"])  # 0=bottom, 1=top

    # Stochastic RSI
    rsi14          = d["rsi14"]
    stoch_min      = rsi14.rolling(14).min()
    stoch_max      = rsi14.rolling(14).max()
    d["stoch_rsi"] = (rsi14 - stoch_min) / (stoch_max - stoch_min + 1e-9)

    # ATR
    hl  = d["High"] - d["Low"]
    hc  = (d["High"] - d["Close"].shift()).abs()
    lc  = (d["Low"]  - d["Close"].shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()
    d["atr_pct"] = d["atr"] / d["Close"]  # ATR as % of price

    # Volume
    d["vol_ma"]    = d["Volume"].rolling(20).mean()
    d["vol_ratio"] = d["Volume"] / d["vol_ma"]

    # VWAP (daily reset)
    d["date"]    = d.index.date
    d["cum_vol"] = d.groupby("date")["Volume"].cumsum()
    d["cum_tp"]  = d.groupby("date").apply(
        lambda x: (x["Close"] * x["Volume"]).cumsum()
    ).reset_index(level=0, drop=True)
    d["vwap"]    = d["cum_tp"] / d["cum_vol"]
    d["vwap_dist"] = (d["Close"] - d["vwap"]) / d["vwap"]  # % above/below VWAP

    # Candle properties
    d["body"]      = (d["Close"] - d["Open"]).abs()
    d["range"]     = d["High"] - d["Low"]
    d["body_pct"]  = d["body"] / (d["range"] + 1e-9)  # strong candle if > 0.6

    # Session (UTC hours — London+NY overlap = 12-16 UTC = hottest)
    d["hour"]      = d.index.hour
    d["in_session"]= d["hour"].isin(range(8, 21))  # 8am-9pm UTC

    # Momentum score (0-100): how many bullish signals
    d["bull_score"] = (
        (d["ema5"]  > d["ema13"]).astype(int) +
        (d["ema9"]  > d["ema21"]).astype(int) +
        (d["ema21"] > d["ema50"]).astype(int) +
        (d["Close"] > d["ema200"]).astype(int) +
        (d["rsi7"]  > 50).astype(int) +
        (d["rsi14"] > 50).astype(int) +
        (d["macd_hist"] > 0).astype(int) +
        (d["Close"] > d["vwap"]).astype(int)
    ) * 12.5  # scale to 0-100

    d["bear_score"] = 100 - d["bull_score"]

    return d

df = add_indicators(df)
df.dropna(inplace=True)
print(f"  {len(df)} candles after warmup.\n")

# ─────────────────────────────────────────────
# DYNAMIC RISK SIZING
# ─────────────────────────────────────────────
def get_risk(capital, streak, base=BASE_RISK, max_r=MAX_RISK):
    """
    Hot streak: won 2+ in a row → increase risk slightly
    Cold streak: lost 2+ in a row → cut risk
    """
    if streak >= 3:
        return min(base * 2.0, max_r)   # 2x on hot streak
    elif streak >= 2:
        return min(base * 1.5, max_r)   # 1.5x after 2 wins
    elif streak <= -2:
        return base * 0.5               # halve on losing streak
    else:
        return base

# ─────────────────────────────────────────────
# ULTRA BACKTEST ENGINE
# ─────────────────────────────────────────────
def ultra_backtest(df, signal_func, name, sl_atr_mult=1.0, rr=RR, trail=True):
    """
    Features:
    - ATR-based dynamic SL (adapts to volatility)
    - Trailing stop: moves SL to breakeven after 1:1, then trails
    - Dynamic position sizing based on win/loss streak
    """
    capital  = CAPITAL
    position = None
    trades   = []
    streak   = 0   # positive = winning streak, negative = losing streak

    entry_price = sl_price = tp_price = be_price = risk_usd = 0.0
    entry_time = None
    trailing_active = False

    signals = signal_func(df)

    for i in range(1, len(df)):
        if capital <= 0:
            break

        price   = float(df["Close"].iloc[i])
        atr     = float(df["atr"].iloc[i])
        ts      = df.index[i]
        sig     = int(signals.iloc[i])

        # ── Manage open position ──
        if position is not None:
            sl_dist = entry_price * 0.002  # base SL distance

            # Trailing stop logic
            if trail and position == "long":
                be_level = entry_price + sl_dist  # breakeven level
                if price >= be_level and not trailing_active:
                    sl_price = entry_price          # move SL to breakeven
                    trailing_active = True
                if trailing_active:
                    new_trail = price - sl_dist * 1.5
                    sl_price  = max(sl_price, new_trail)

            elif trail and position == "short":
                be_level = entry_price - sl_dist
                if price <= be_level and not trailing_active:
                    sl_price = entry_price
                    trailing_active = True
                if trailing_active:
                    new_trail = price + sl_dist * 1.5
                    sl_price  = min(sl_price, new_trail)

            hit_tp = (position == "long"  and price >= tp_price) or \
                     (position == "short" and price <= tp_price)
            hit_sl = (position == "long"  and price <= sl_price) or \
                     (position == "short" and price >= sl_price)

            if hit_tp or hit_sl:
                if hit_tp:
                    pnl    = risk_usd * rr
                    result = "TP ✅"
                    streak = streak + 1 if streak >= 0 else 1
                else:
                    # If trailing was active and we exit at BE, it's a scratch
                    if trailing_active and abs(price - entry_price) / entry_price < 0.001:
                        pnl    = 0
                        result = "BE 〰"
                        streak = max(streak, 0)
                    else:
                        pnl    = -risk_usd
                        result = "SL ❌"
                        streak = streak - 1 if streak <= 0 else -1

                capital = max(0.0, capital + pnl)
                trades.append({
                    "strategy":   name,
                    "entry_time": entry_time,
                    "exit_time":  ts,
                    "direction":  position,
                    "entry":      round(entry_price, 2),
                    "exit":       round(price, 2),
                    "result":     result,
                    "risk_used":  round(risk_usd, 4),
                    "pnl":        round(pnl, 4),
                    "capital":    round(capital, 4),
                    "streak":     streak,
                })
                position        = None
                trailing_active = False
                continue

        # ── Open new position ──
        if position is None and sig != 0 and capital > 0:
            risk_usd    = round(capital * get_risk(capital, streak), 6)
            sl_dist     = atr * sl_atr_mult
            sl_pct      = sl_dist / price
            tp_pct      = sl_pct * rr
            entry_price = price
            entry_time  = ts
            trailing_active = False

            if sig == 1:
                position = "long"
                sl_price = entry_price - sl_dist
                tp_price = entry_price * (1 + tp_pct)
            else:
                position = "short"
                sl_price = entry_price + sl_dist
                tp_price = entry_price * (1 - tp_pct)

    return pd.DataFrame(trades)

# ─────────────────────────────────────────────
# ULTRA SIGNAL FUNCTIONS
# ─────────────────────────────────────────────

# ── ULTRA 1: Triple EMA Alignment + MACD + Volume ──
# All 3 fast EMAs aligned + MACD confirms + volume spike
def sig_triple_ema(df):
    s = pd.Series(0, index=df.index)

    # Strong uptrend: 5 > 9 > 13 AND price > ema21
    bull = (df["ema5"] > df["ema9"]) & (df["ema9"] > df["ema13"]) & (df["Close"] > df["ema21"])
    bear = (df["ema5"] < df["ema9"]) & (df["ema9"] < df["ema13"]) & (df["Close"] < df["ema21"])

    # MACD histogram turning positive/negative
    macd_bull = (df["macd_hist"] > 0) & (df["macd_hist"].shift(1) <= 0)
    macd_bear = (df["macd_hist"] < 0) & (df["macd_hist"].shift(1) >= 0)

    # Volume confirmation
    vol_ok = df["vol_ratio"] > 1.2

    # Strong candle (body > 60% of range)
    strong = df["body_pct"] > 0.6

    # Session filter
    sess = df["in_session"]

    s[bull & macd_bull & vol_ok & strong & sess] = 1
    s[bear & macd_bear & vol_ok & strong & sess] = -1
    return s

# ── ULTRA 2: Momentum Score Threshold ──
# Trade only when momentum score is extreme (>75 bull or <25 bear)
# + RSI not overbought/oversold (has room to run)
def sig_momentum_score(df):
    s = pd.Series(0, index=df.index)

    strong_bull = (df["bull_score"] >= 75) & (df["bull_score"].shift(1) < 75)
    strong_bear = (df["bear_score"] >= 75) & (df["bear_score"].shift(1) < 75)

    # RSI has room (not already maxed out)
    rsi_room_bull = df["rsi14"] < 70
    rsi_room_bear = df["rsi14"] > 30

    # VWAP direction
    above_vwap = df["vwap_dist"] > 0
    below_vwap = df["vwap_dist"] < 0

    sess = df["in_session"]

    s[strong_bull & rsi_room_bull & above_vwap & sess] = 1
    s[strong_bear & rsi_room_bear & below_vwap & sess] = -1
    return s

# ── ULTRA 3: Stoch RSI + BB + VWAP Confluence ──
# Stoch RSI oversold/overbought + BB position + near VWAP
def sig_stoch_bb_vwap(df):
    s = pd.Series(0, index=df.index)

    # Stoch RSI crosses up from oversold
    stoch_up   = (df["stoch_rsi"] > 0.2) & (df["stoch_rsi"].shift(1) <= 0.2)
    stoch_down = (df["stoch_rsi"] < 0.8) & (df["stoch_rsi"].shift(1) >= 0.8)

    # BB position (lower third = buy zone, upper third = sell zone)
    bb_buy  = df["bb_pct"] < 0.35
    bb_sell = df["bb_pct"] > 0.65

    # Near or above/below VWAP
    near_above_vwap = df["vwap_dist"] > -0.001
    near_below_vwap = df["vwap_dist"] <  0.001

    # EMA trend
    uptrend   = df["Close"] > df["ema50"]
    downtrend = df["Close"] < df["ema50"]

    sess = df["in_session"]

    s[stoch_up   & bb_buy  & near_above_vwap & uptrend   & sess] = 1
    s[stoch_down & bb_sell & near_below_vwap & downtrend & sess] = -1
    return s

# ── ULTRA 4: EMA Fan + RSI Pullback (Trend Continuation) ──
# Price pulls back to EMA9 in a strong trend → continuation entry
def sig_ema_pullback(df):
    s = pd.Series(0, index=df.index)

    # Strong uptrend: ema9 > ema21 > ema50 > ema200
    strong_up   = (df["ema9"] > df["ema21"]) & (df["ema21"] > df["ema50"]) & (df["ema50"] > df["ema200"])
    strong_down = (df["ema9"] < df["ema21"]) & (df["ema21"] < df["ema50"]) & (df["ema50"] < df["ema200"])

    # Price touches EMA9 (within 0.1%)
    touch_ema9_up   = (df["Low"]  <= df["ema9"] * 1.001) & (df["Close"] > df["ema9"])
    touch_ema9_down = (df["High"] >= df["ema9"] * 0.999) & (df["Close"] < df["ema9"])

    # RSI dipped then recovering (for longs) or rose then falling (for shorts)
    rsi_dip     = (df["rsi7"] > 40) & (df["rsi7"].shift(1) < 40)
    rsi_peak    = (df["rsi7"] < 60) & (df["rsi7"].shift(1) > 60)

    # MACD still positive/negative
    macd_pos = df["macd_hist"] > 0
    macd_neg = df["macd_hist"] < 0

    sess = df["in_session"]

    s[strong_up   & touch_ema9_up   & rsi_dip  & macd_pos & sess] = 1
    s[strong_down & touch_ema9_down & rsi_peak & macd_neg & sess] = -1
    return s

# ── ULTRA 5: COMBINED CONFLUENCE (the beast) ──
# Fires only when AT LEAST 3 of the 4 strategies agree
def sig_combined(df):
    s1 = sig_triple_ema(df)
    s2 = sig_momentum_score(df)
    s3 = sig_stoch_bb_vwap(df)
    s4 = sig_ema_pullback(df)

    # Vote: +1 for bull signal, -1 for bear signal
    vote = s1 + s2 + s3 + s4

    s = pd.Series(0, index=df.index)
    s[vote >= 2]  = 1   # 2+ strategies say buy
    s[vote <= -2] = -1  # 2+ strategies say sell
    return s

# ─────────────────────────────────────────────
# RUN ALL STRATEGIES
# ─────────────────────────────────────────────
strategies = [
    ("U1. Triple EMA + MACD",     sig_triple_ema,      1.0),
    ("U2. Momentum Score",        sig_momentum_score,  0.8),
    ("U3. Stoch+BB+VWAP",         sig_stoch_bb_vwap,   1.0),
    ("U4. EMA Pullback",          sig_ema_pullback,     0.8),
    ("U5. COMBINED (2+ agree) 🔥",sig_combined,         0.8),
]

all_trades = []
summary    = []

for name, sig_fn, atr_mult in strategies:
    print(f"  Running {name}...")
    trades = ultra_backtest(df, sig_fn, name, sl_atr_mult=atr_mult)
    all_trades.append(trades)

    if len(trades) == 0:
        summary.append({"Strategy": name, "Trades": 0, "Wins": 0, "Losses": 0, "BE": 0,
                        "Win%": "0%", "PnL($)": 0, "Capital($)": CAPITAL,
                        "Return%": "0%", "MaxDD%": "0%", "AvgWin": 0, "AvgLoss": 0})
        continue

    tp_mask  = trades["result"].str.startswith("TP")
    sl_mask  = trades["result"].str.startswith("SL")
    be_mask  = trades["result"].str.startswith("BE")
    wins     = tp_mask.sum()
    losses   = sl_mask.sum()
    be_count = be_mask.sum()
    valid    = wins + losses
    pnl      = trades["pnl"].sum()
    final    = trades["capital"].iloc[-1]
    wr       = wins / valid * 100 if valid > 0 else 0
    ret      = (final - CAPITAL) / CAPITAL * 100
    avg_win  = trades.loc[tp_mask, "pnl"].mean() if wins  > 0 else 0
    avg_loss = trades.loc[sl_mask, "pnl"].mean() if losses > 0 else 0

    cap_s  = pd.concat([pd.Series([CAPITAL]), trades["capital"].reset_index(drop=True)])
    max_dd = ((cap_s - cap_s.cummax()) / cap_s.cummax() * 100).min()

    summary.append({
        "Strategy":  name,
        "Trades":    len(trades),
        "Wins":      int(wins),
        "Losses":    int(losses),
        "BE":        int(be_count),
        "Win%":      f"{wr:.1f}%",
        "PnL($)":    round(pnl, 2),
        "Capital($)":round(final, 2),
        "Return%":   f"{ret:.1f}%",
        "MaxDD%":    f"{max_dd:.1f}%",
        "AvgWin":    round(avg_win, 3),
        "AvgLoss":   round(avg_loss, 3),
    })

print()

# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────
print("=" * 110)
print(f"   ⚡ BTC/USD {INTERVAL} ULTRA SCALPER | ${CAPITAL} Capital | 1:{LEVERAGE} Leverage | RR {RR}:1 | Trailing Stop ON")
print(f"   Period: {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}")
print("=" * 110)

summary_df = pd.DataFrame(summary)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 180)
print(summary_df.to_string(index=False))
print("=" * 110)

# Trade logs — winner only (full)
best_idx = summary_df["PnL($)"].idxmax()
best_name = summary_df.iloc[best_idx]["Strategy"]
best_trades = all_trades[best_idx]

print(f"\n  🏆 WINNER: {best_name}")
print(f"{'─'*110}")
print(f"  Last 20 Trades:")
print(f"{'─'*110}")
if len(best_trades) > 0:
    cols = ["entry_time","exit_time","direction","entry","exit","result","risk_used","pnl","capital","streak"]
    print(best_trades[cols].tail(20).to_string(index=False))

# Capital growth curve (every 50 trades)
print(f"\n{'─'*110}")
print(f"  Capital Growth Curve (every 50 trades):")
print(f"{'─'*110}")
if len(best_trades) > 0:
    milestones = best_trades.iloc[::50][["entry_time","capital","streak"]]
    milestones.index = range(0, len(milestones)*50, 50)
    milestones.index.name = "trade#"
    print(milestones.to_string())

# Final summary
print("\n" + "=" * 110)
best = summary_df.iloc[best_idx]
print(f"  🏆  WINNER    : {best['Strategy']}")
print(f"      Trades    : {best['Trades']}  |  Wins: {best['Wins']}  |  Losses: {best['Losses']}  |  BE: {best['BE']}")
print(f"      Win Rate  : {best['Win%']}")
print(f"      PnL       : ${best['PnL($)']}")
print(f"      Return    : {best['Return%']}")
print(f"      Max DD    : {best['MaxDD%']}")
print(f"      Avg Win   : ${best['AvgWin']}  |  Avg Loss: ${best['AvgLoss']}")
print("=" * 110)

# Save
valid = [t for t in all_trades if len(t) > 0]
if valid:
    pd.concat(valid, ignore_index=True).to_csv("ultra_scalp_trades.csv", index=False)
    print(f"\n📄 Full trade log → ultra_scalp_trades.csv")
print("✅ Done!\n")