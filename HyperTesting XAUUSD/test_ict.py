"""
ICT (Inner Circle Trader) STRATEGY TEST — XAUUSD 5m  |  2 Years
8 strategies × 4 SL × 5 RR = 160 combos
Concepts: Judas Swing, Fair Value Gap, Order Block, OTE, Breaker Block
Capital: $200  |  Leverage: 1:1000  |  Risk: 2%/trade
Killzones: London 07-10 UTC  |  NY 13-16 UTC
"""

import os, sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

CAPITAL  = 200.0
RISK_PCT = 0.02
SPREAD   = 0.00025
LEVERAGE = 1000
MIN_LOT  = 0.01
LOT_STEP = 0.01


# ── DATA ──────────────────────────────────────────────────────────────────────
def load_data():
    for path in ["../April25/xauusd_5m_2y.csv", "xauusd_5m_2y.csv",
                 os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "April25", "xauusd_5m_2y.csv")]:
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
    print("ERROR: xauusd_5m_2y.csv not found.")
    sys.exit(1)


# ── INDICATORS ────────────────────────────────────────────────────────────────
def build_indicators(df):
    d = df.copy()
    c = d["Close"]
    for n in [9, 21, 50]:
        d[f"ema{n}"] = c.ewm(span=n, adjust=False).mean()
    d["ema240"] = c.ewm(span=240, adjust=False).mean()

    hl  = d["High"] - d["Low"]
    hpc = (d["High"] - c.shift()).abs()
    lpc = (d["Low"]  - c.shift()).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    d["atr14"] = tr.ewm(span=14, adjust=False).mean()

    delta = c.diff()
    g = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    l = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    d["rsi14"] = 100 - 100 / (1 + g / l)

    h = d.index.hour
    m = d.index.minute
    d["in_london"]    = pd.Series((h >= 7)  & (h < 10), index=d.index)
    d["in_ny"]        = pd.Series((h >= 13) & (h < 16), index=d.index)
    d["in_session"]   = d["in_london"] | d["in_ny"]
    d["is_weekday"]   = pd.Series(d.index.dayofweek <= 3, index=d.index)
    d["date_utc"]     = d.index.date
    d["body_bull"]    = (c > d["Open"]).astype(bool)
    d["body_bear"]    = (c < d["Open"]).astype(bool)
    d["body_size"]    = (c - d["Open"]).abs()
    return d.dropna()


# ── SESSION RANGES ────────────────────────────────────────────────────────────
def add_session_ranges(df):
    """Asian and London session ranges for Judas Swing and NY continuation."""
    d = df.copy()
    dates = sorted(d["date_utc"].unique())
    a_hi, a_lo = {}, {}
    l_hi, l_lo = {}, {}

    for date in dates:
        ds   = str(date)
        prev = str((pd.Timestamp(date) - pd.Timedelta(days=1)).date())

        t0 = pd.Timestamp(f"{prev} 22:00:00", tz="UTC")
        t1 = pd.Timestamp(f"{ds} 07:00:00", tz="UTC")
        m = (d.index >= t0) & (d.index < t1)
        if m.sum() >= 5:
            a_hi[date] = d.loc[m, "High"].max()
            a_lo[date] = d.loc[m, "Low"].min()

        t0 = pd.Timestamp(f"{ds} 07:00:00", tz="UTC")
        t1 = pd.Timestamp(f"{ds} 10:00:00", tz="UTC")
        m = (d.index >= t0) & (d.index < t1)
        if m.sum() >= 5:
            l_hi[date] = d.loc[m, "High"].max()
            l_lo[date] = d.loc[m, "Low"].min()

    d["asian_hi"] = d["date_utc"].map(a_hi)
    d["asian_lo"] = d["date_utc"].map(a_lo)
    d["london_hi"] = d["date_utc"].map(l_hi)
    d["london_lo"] = d["date_utc"].map(l_lo)
    return d


# ── FAIR VALUE GAP (FVG) ──────────────────────────────────────────────────────
def compute_fvg(df):
    """
    Stateful FVG tracker. Tracks the most recent unfilled FVG.
    Bullish FVG zone: [High[i-2], Low[i]]  — gap after upward impulse
    Bearish FVG zone: [High[i], Low[i-2]]  — gap after downward impulse
    Invalidated when price closes through the zone's outer boundary.
    """
    d = df.copy()
    n = len(d)
    h = d["High"].values
    l = d["Low"].values
    c = d["Close"].values

    bfvg_lo = np.full(n, np.nan)  # bottom of bullish FVG zone
    bfvg_hi = np.full(n, np.nan)  # top of bullish FVG zone
    sfvg_lo = np.full(n, np.nan)  # bottom of bearish FVG zone
    sfvg_hi = np.full(n, np.nan)  # top of bearish FVG zone

    active_bull = None  # (lo, hi)
    active_bear = None

    for i in range(2, n):
        # Detect new FVGs
        if l[i] > h[i-2]:          # bullish FVG: gap between candle i-2's high and candle i's low
            active_bull = (h[i-2], l[i])
        if h[i] < l[i-2]:          # bearish FVG: gap between candle i's high and candle i-2's low
            active_bear = (h[i], l[i-2])

        # Invalidate if price closed through the far side
        if active_bull is not None:
            if c[i] < active_bull[0]:   # closed below zone bottom → fully filled
                active_bull = None
            else:
                bfvg_lo[i], bfvg_hi[i] = active_bull

        if active_bear is not None:
            if c[i] > active_bear[1]:   # closed above zone top → fully filled
                active_bear = None
            else:
                sfvg_lo[i], sfvg_hi[i] = active_bear

    d["bfvg_lo"] = bfvg_lo
    d["bfvg_hi"] = bfvg_hi
    d["sfvg_lo"] = sfvg_lo
    d["sfvg_hi"] = sfvg_hi
    return d


# ── ORDER BLOCK (OB) ──────────────────────────────────────────────────────────
def compute_order_blocks(df):
    """
    Stateful OB tracker. Tracks the most recent active Order Block.
    Bullish OB: last bearish candle before a significant upward impulse.
    Bearish OB: last bullish candle before a significant downward impulse.
    Uses OB body (Open/Close range), not full wick.
    """
    d = df.copy()
    n = len(d)
    c = d["Close"].values
    o = d["Open"].values
    h = d["High"].values
    l = d["Low"].values
    atr = d["atr14"].values

    bob_lo = np.full(n, np.nan)  # bullish OB low (body)
    bob_hi = np.full(n, np.nan)  # bullish OB high (body)
    sob_lo = np.full(n, np.nan)  # bearish OB low (body)
    sob_hi = np.full(n, np.nan)  # bearish OB high (body)

    active_bull_ob = None
    active_bear_ob = None

    for i in range(5, n):
        # Detect bullish impulse: net gain over last 4 bars > 1.5×ATR
        if (c[i] - c[i-4]) > 1.5 * atr[i]:
            # Find the last bearish candle in the 4 bars before this move
            for k in range(i-1, max(i-6, 0)-1, -1):
                if c[k] < o[k]:
                    active_bull_ob = (min(o[k], c[k]), max(o[k], c[k]))
                    break

        # Detect bearish impulse: net drop over last 4 bars > 1.5×ATR
        if (c[i-4] - c[i]) > 1.5 * atr[i]:
            for k in range(i-1, max(i-6, 0)-1, -1):
                if c[k] > o[k]:
                    active_bear_ob = (min(o[k], c[k]), max(o[k], c[k]))
                    break

        # Invalidate if price closes through the OB
        if active_bull_ob is not None:
            if c[i] < active_bull_ob[0] - 0.5 * atr[i]:  # closed well below OB
                active_bull_ob = None
            else:
                bob_lo[i], bob_hi[i] = active_bull_ob

        if active_bear_ob is not None:
            if c[i] > active_bear_ob[1] + 0.5 * atr[i]:
                active_bear_ob = None
            else:
                sob_lo[i], sob_hi[i] = active_bear_ob

    d["bob_lo"] = bob_lo
    d["bob_hi"] = bob_hi
    d["sob_lo"] = sob_lo
    d["sob_hi"] = sob_hi
    return d


# ── OTE (FIBONACCI 61.8–78.6% RETRACEMENT) ───────────────────────────────────
def compute_ote(df, window=30):
    """
    Vectorized OTE zones using rolling window swing high/low.
    OTE bull zone: 61.8–78.6% pullback of recent upswing.
    OTE bear zone: 61.8–78.6% bounce of recent downswing.
    """
    d = df.copy()
    swing_hi = d["High"].rolling(window).max()
    swing_lo = d["Low"].rolling(window).min()
    rng = swing_hi - swing_lo

    # Bullish OTE: price pulled back 61.8–78.6% from recent high
    d["ote_bull_lo"] = swing_hi - 0.786 * rng  # deeper retracement (lower price)
    d["ote_bull_hi"] = swing_hi - 0.618 * rng  # shallower retracement (higher price)

    # Bearish OTE: price bounced 61.8–78.6% from recent low
    d["ote_bear_lo"] = swing_lo + 0.618 * rng
    d["ote_bear_hi"] = swing_lo + 0.786 * rng
    return d


# ── HELPERS ───────────────────────────────────────────────────────────────────
def _sig(d, up, dn):
    s = pd.Series(0, index=d.index)
    s[up.fillna(False)] = 1
    s[dn.fillna(False)] = -1
    return s

def _first_per_day(signal, date_col):
    """Return a boolean Series keeping only the first True per day."""
    result = pd.Series(False, index=signal.index)
    dates  = date_col.values
    sigs   = signal.values
    seen   = set()
    for i in range(len(sigs)):
        if sigs[i] and dates[i] not in seen:
            result.iloc[i] = True
            seen.add(dates[i])
    return result


# ── ICT SIGNAL GENERATORS ────────────────────────────────────────────────────

# 1. Judas Swing — Asian range sweep reversal at London open
def sig_ict_judas_swing(d):
    """
    London 07-10 UTC: price sweeps Asian session high/low (wick outside,
    close back inside) → enter reversal.
    """
    w        = d["in_london"] & d["is_weekday"] & d["asian_hi"].notna()
    sweep_up = (d["High"] > d["asian_hi"]) & (d["Close"] < d["asian_hi"])   # fake break up → SHORT
    sweep_dn = (d["Low"]  < d["asian_lo"]) & (d["Close"] > d["asian_lo"])   # fake break dn → LONG
    first_up = _first_per_day(sweep_up & w, d["date_utc"])
    first_dn = _first_per_day(sweep_dn & w, d["date_utc"])
    return _sig(d, first_dn, first_up)

# 2. Judas Swing + OB alignment — only enter if also at an active Order Block
def sig_ict_judas_ob(d):
    """
    Judas Swing confirmed by active Order Block at the entry level.
    More selective — OB provides the precise entry zone.
    """
    w        = d["in_london"] & d["is_weekday"] & d["asian_hi"].notna()
    sweep_up = (d["High"] > d["asian_hi"]) & (d["Close"] < d["asian_hi"])
    sweep_dn = (d["Low"]  < d["asian_lo"]) & (d["Close"] > d["asian_lo"])
    # OB alignment: close is inside (or near) the OB zone
    near_bob = d["sob_lo"].notna() & (d["Close"] <= d["sob_hi"])             # close near bearish OB → short
    near_bull_ob = d["bob_lo"].notna() & (d["Close"] >= d["bob_lo"])         # close near bullish OB → long
    first_up = _first_per_day(sweep_up & w & near_bob, d["date_utc"])
    first_dn = _first_per_day(sweep_dn & w & near_bull_ob, d["date_utc"])
    return _sig(d, first_dn, first_up)

# 3. FVG retracement entry during London killzone
def sig_ict_fvg_london(d):
    """
    Price retraces into an active Fair Value Gap during London session.
    HTF trend filter: trade in direction of EMA240.
    """
    htf_bull = d["Close"] > d["ema240"]
    htf_bear = d["Close"] < d["ema240"]

    # Bullish FVG entry: price dips into zone (Low touches zone top), close inside/above bottom
    bull_entry = (d["Low"] <= d["bfvg_hi"]) & (d["Close"] >= d["bfvg_lo"]) & \
                 d["bfvg_lo"].notna() & d["body_bull"]
    bear_entry = (d["High"] >= d["sfvg_lo"]) & (d["Close"] <= d["sfvg_hi"]) & \
                 d["sfvg_lo"].notna() & d["body_bear"]

    # First entry per day per FVG
    prev_bull = (d["Low"].shift() <= d["bfvg_hi"].shift()) & d["bfvg_lo"].shift().notna()
    prev_bear = (d["High"].shift() >= d["sfvg_lo"].shift()) & d["sfvg_lo"].shift().notna()

    return _sig(d,
        d["in_london"] & d["is_weekday"] & htf_bull & bull_entry & ~prev_bull,
        d["in_london"] & d["is_weekday"] & htf_bear & bear_entry & ~prev_bear)

# 4. FVG retracement during NY killzone
def sig_ict_fvg_ny(d):
    """Same as FVG London but during New York session (13-16 UTC)."""
    htf_bull = d["Close"] > d["ema240"]
    htf_bear = d["Close"] < d["ema240"]
    bull_entry = (d["Low"] <= d["bfvg_hi"]) & (d["Close"] >= d["bfvg_lo"]) & \
                 d["bfvg_lo"].notna() & d["body_bull"]
    bear_entry = (d["High"] >= d["sfvg_lo"]) & (d["Close"] <= d["sfvg_hi"]) & \
                 d["sfvg_lo"].notna() & d["body_bear"]
    prev_bull = (d["Low"].shift() <= d["bfvg_hi"].shift()) & d["bfvg_lo"].shift().notna()
    prev_bear = (d["High"].shift() >= d["sfvg_lo"].shift()) & d["sfvg_lo"].shift().notna()
    return _sig(d,
        d["in_ny"] & d["is_weekday"] & htf_bull & bull_entry & ~prev_bull,
        d["in_ny"] & d["is_weekday"] & htf_bear & bear_entry & ~prev_bear)

# 5. Order Block retracement during London killzone
def sig_ict_ob_london(d):
    """
    Price retraces into an active Order Block during London session.
    Enter on the first candle that touches the OB body.
    """
    htf_bull = d["Close"] > d["ema240"]
    htf_bear = d["Close"] < d["ema240"]

    bull_ob_touch = (d["Low"] <= d["bob_hi"]) & (d["Close"] >= d["bob_lo"]) & \
                    d["bob_lo"].notna() & d["body_bull"]
    bear_ob_touch = (d["High"] >= d["sob_lo"]) & (d["Close"] <= d["sob_hi"]) & \
                    d["sob_lo"].notna() & d["body_bear"]

    prev_bull = (d["Low"].shift() <= d["bob_hi"].shift()) & d["bob_lo"].shift().notna()
    prev_bear = (d["High"].shift() >= d["sob_lo"].shift()) & d["sob_lo"].shift().notna()

    return _sig(d,
        d["in_london"] & d["is_weekday"] & htf_bull & bull_ob_touch & ~prev_bull,
        d["in_london"] & d["is_weekday"] & htf_bear & bear_ob_touch & ~prev_bear)

# 6. OTE (Fibonacci 61.8–78.6%) retracement during London session
def sig_ict_ote_london(d):
    """
    Price enters the Fibonacci OTE zone (61.8–78.6% retracement) during London.
    Enter when price is in zone and shows a reversal candle.
    """
    htf_bull = d["Close"] > d["ema240"]
    htf_bear = d["Close"] < d["ema240"]

    in_bull_ote = (d["Close"] >= d["ote_bull_lo"]) & (d["Close"] <= d["ote_bull_hi"]) & \
                  d["body_bull"] & (d["rsi14"] > 40) & (d["rsi14"] < 65)
    in_bear_ote = (d["Close"] >= d["ote_bear_lo"]) & (d["Close"] <= d["ote_bear_hi"]) & \
                  d["body_bear"] & (d["rsi14"] > 35) & (d["rsi14"] < 60)

    prev_bull_ote = (d["Close"].shift() >= d["ote_bull_lo"].shift()) & \
                    (d["Close"].shift() <= d["ote_bull_hi"].shift())
    prev_bear_ote = (d["Close"].shift() >= d["ote_bear_lo"].shift()) & \
                    (d["Close"].shift() <= d["ote_bear_hi"].shift())

    return _sig(d,
        d["in_london"] & d["is_weekday"] & htf_bull & in_bull_ote & ~prev_bull_ote,
        d["in_london"] & d["is_weekday"] & htf_bear & in_bear_ote & ~prev_bear_ote)

# 7. ICT Full Model — Judas Swing + FVG active in reversal direction + HTF
def sig_ict_full_model(d):
    """
    Complete ICT Power of Three checklist:
    Asian range → London Judas Sweep → FVG in reversal direction → HTF trend aligned.
    Highest confluence, fewest signals.
    """
    w        = d["in_london"] & d["is_weekday"] & d["asian_hi"].notna()
    htf_bull = d["Close"] > d["ema240"]
    htf_bear = d["Close"] < d["ema240"]

    sweep_up = (d["High"] > d["asian_hi"]) & (d["Close"] < d["asian_hi"])
    sweep_dn = (d["Low"]  < d["asian_lo"]) & (d["Close"] > d["asian_lo"])

    # After a bearish sweep (up then back), look for bearish FVG active → short
    # After a bullish sweep (down then back), look for bullish FVG active → long
    fvg_bear_active = d["sfvg_lo"].notna()
    fvg_bull_active = d["bfvg_lo"].notna()

    first_up = _first_per_day(sweep_up & w & htf_bear & fvg_bear_active, d["date_utc"])
    first_dn = _first_per_day(sweep_dn & w & htf_bull & fvg_bull_active, d["date_utc"])
    return _sig(d, first_dn, first_up)

# 8. NY Continuation — NY session sweep of London range → continue in direction
def sig_ict_ny_sweep(d):
    """
    At NY open (13-16 UTC): price sweeps above/below the London session range,
    then closes back inside → expect continuation in sweep direction (not reversal).
    In ICT NY session often 'distributes' in the same direction as London trend.
    """
    w        = d["in_ny"] & d["is_weekday"] & d["london_hi"].notna()
    htf_bull = d["Close"] > d["ema240"]
    htf_bear = d["Close"] < d["ema240"]

    # NY breaks above London high and holds above → continuation long (not reversal)
    ny_break_up = (d["Close"] > d["london_hi"]) & (d["Close"].shift() <= d["london_hi"].shift())
    # NY breaks below London low and holds → continuation short
    ny_break_dn = (d["Close"] < d["london_lo"]) & (d["Close"].shift() >= d["london_lo"].shift())

    first_up = _first_per_day(ny_break_up & w & htf_bull, d["date_utc"])
    first_dn = _first_per_day(ny_break_dn & w & htf_bear, d["date_utc"])
    return _sig(d, first_up, first_dn)


STRATEGIES = {
    "ICT_Judas_Swing":   sig_ict_judas_swing,
    "ICT_Judas_OB":      sig_ict_judas_ob,
    "ICT_FVG_London":    sig_ict_fvg_london,
    "ICT_FVG_NY":        sig_ict_fvg_ny,
    "ICT_OB_London":     sig_ict_ob_london,
    "ICT_OTE_London":    sig_ict_ote_london,
    "ICT_Full_Model":    sig_ict_full_model,
    "ICT_NY_Sweep":      sig_ict_ny_sweep,
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
RR_GRID = [2.0, 3.0, 5.0, 7.0, 10.0]

print("="*80)
print("  ICT STRATEGY TEST — XAUUSD 5m  |  2 Years")
print(f"  {len(STRATEGIES)} strategies × {len(SL_GRID)} SL × {len(RR_GRID)} RR = {len(STRATEGIES)*len(SL_GRID)*len(RR_GRID)} combos")
print("  Concepts: Judas Swing | FVG | Order Block | OTE | NY Continuation")
print("  Killzones: London 07-10 UTC  |  NY 13-16 UTC  |  Mon-Thu only")
print("="*80 + "\n")

df_raw = load_data()
cutoff  = df_raw.index[-1] - pd.DateOffset(years=2)
df_raw  = df_raw[df_raw.index >= cutoff].copy()
print(f"Window: {df_raw.index[0].strftime('%Y-%m-%d')} → {df_raw.index[-1].strftime('%Y-%m-%d')} | {len(df_raw):,} candles\n")

print("Building indicators...", end=" ", flush=True)
df = build_indicators(df_raw)
print("done.")

print("Computing session ranges...", end=" ", flush=True)
df = add_session_ranges(df)
print("done.")

print("Computing Fair Value Gaps...", end=" ", flush=True)
df = compute_fvg(df)
print("done.")

print("Computing Order Blocks...", end=" ", flush=True)
df = compute_order_blocks(df)
print("done.")

print("Computing OTE Fibonacci zones...", end=" ", flush=True)
df = compute_ote(df)
print("done.\n")

print("  Signal counts per strategy:")
for name, fn in STRATEGIES.items():
    s = fn(df)
    nl, ns = (s==1).sum(), (s==-1).sum()
    print(f"    {name:<22} long={nl:5d}  short={ns:5d}  total={nl+ns:5d}")
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
            res_str = "n/a" if not s else f"{s['avg_mo%']:+.2f}%/mo win={s['win%']}%"
            print(f"  [{run_i:3d}/{total}] {name:<22} SL={sl*100:.1f}% RR={rr:4.1f} → {res_str}", end="\r")

print(f"\n  [{total}/{total}] done.   \n")

if not results:
    print("No results — check signal generators.")
    sys.exit(0)

res = pd.DataFrame(results).sort_values("avg_mo%", ascending=False).reset_index(drop=True)

# ── RANKINGS ──────────────────────────────────────────────────────────────────
print("═"*125)
print("  ICT STRATEGY RESULTS — XAUUSD 5m  |  $200  |  1:1000  |  0.025% Spread  |  2% Risk/Trade")
print("═"*125)
print(f"  {'#':<4} {'STRATEGY':<40} {'TRADES':>7} {'WIN%':>6} {'AVG/MO%':>9} {'BEST':>8} {'WORST':>9} {'MDD%':>7} {'MDD$':>7} {'FINAL$':>10}")
print("═"*125)
for i, row in res.head(20).iterrows():
    mark = " ✓" if row["avg_mo%"] > 0 else "  "
    print(f"  {i+1:<4} {row['label']:<40} {int(row['trades']):>7} {row['win%']:>5.1f}% "
          f"{row['avg_mo%']:>9.2f}% {row['best_mo%']:>7.2f}% {row['worst_mo%']:>8.2f}% "
          f"{row['mdd%']:>7.1f}% ${row['mdd_$']:>6.2f} ${row['final']:>9.2f}{mark}")
print("═"*125)
print(f"\n  Profitable: {(res['avg_mo%']>0).sum()} / {len(res)}")
print(f"  Survived (final > $200): {(res['final']>200).sum()} / {len(res)}\n")

# ── TOP 5 MONTHLY BREAKDOWN ───────────────────────────────────────────────────
print("═"*80)
print("  PER-MONTH BREAKDOWN — TOP 5 ICT STRATEGIES")
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
    print(f"\n  {'Month':<10} {'Trades':>7} {'Win%':>6} {'Return%':>10} {'Capital':>11}  Bar")
    print(f"  {'-'*64}")
    for _, r in mo.iterrows():
        ret = r["ret%"]
        bar = ("█"*min(int(abs(ret)/3), 35)) if ret >= 0 else ("▓"*min(int(abs(ret)/3), 35))
        sgn = "+" if ret >= 0 else ""
        print(f"  {r['month']:<10} {int(r['trades']):>7} {r['win%']:>5.1f}% "
              f"{sgn}{ret:>9.2f}% ${r['capital']:>10.2f}  {bar}")
    print(f"  {'-'*64}")
    print(f"  Avg: {rets.mean():+.2f}%/mo | Best: {rets.max():+.2f}% | Worst: {rets.min():+.2f}%")

print("\n" + "═"*80)
w = res.iloc[0]
print(f"  WINNER  : {w['label']}")
print(f"  Avg/Mo  : {w['avg_mo%']:+.2f}%  |  Best: {w['best_mo%']:+.2f}%  |  MDD: {w['mdd%']:.1f}% (${w['mdd_$']:.2f})")
print(f"  Final   : ${w['final']:,.2f}  from $200  ({w['total%']:+.1f}% over 2 years)")
print(f"  NOTE: 2% risk/trade — reduce to 1% for live trading.")
print("═"*80)
