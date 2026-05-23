"""
test_5m_longrun.py
Strategy 1 (Sweep + iFVG Scalp) and Strategy 2 (Four-EMA Band) on 5-minute bars.

Source : XAU_1m_data.csv  (MT4 format, 2004–2025, ~6.7 M rows)
         → resampled to 5m  (~1.35 M bars, ~21 years)

Timezone note: MT4 data timestamps are typically broker-local time (UTC+2/+3 EET).
               Session filter is adjusted accordingly:
                 NY Open Killzone UTC 14:30–16:00  →  EET 16:30–18:00
               If your data is already UTC, change SESSION_OFFSET = 0 below.

Parameter adjustments vs 1m version:
  • Cooldown (S2) : 100 bars × 1m = 100 min → 20 bars × 5m = 100 min (same wall-clock)
  • iFVG lookback : 5 bars × 5m = 25 min (tighter than 15m, wider than 1m)
  • EMA periods   : same numbers (become multi-hour filters on 5m — good trend anchors)
"""

import pandas as pd
import numpy as np
from itertools import product
import os

DATA_FILE = "/Users/hardik/Projects/NeuralBroker/HyperTesting XAUUSD/XAU_1m_data.csv"

CAPITAL     = 200.0
CONTRACT_SZ = 100
MIN_LOT     = 0.01
RISK_PCT    = 0.02
SPREAD      = 0.00025   # 0.025% — applied to both entry and exit

# Broker offset: MT4 data is typically UTC+2 (EET winter) or UTC+3 (EET summer).
# We use UTC+2 as conservative estimate.
# NY Open Killzone in UTC: 14:30–16:00  →  broker local: 16:30–18:00
# Set SESSION_OFFSET = 0 if your data timestamps are already UTC.
SESSION_OFFSET = 2   # hours to ADD to UTC to reach broker local time

# ── Grids ─────────────────────────────────────────────────────────────────────
S1_SL_GRID = [0.002, 0.003, 0.005, 0.008]
S1_RR_GRID = [2.0, 3.0, 5.0, 7.0, 10.0]

S2_SL_GRID  = [0.0015, 0.002, 0.003]
S2_TP_GRID  = [0.015, 0.025, 0.04]
S2_BE_PCT   = 0.003
S2_COOLDOWN = 20    # 20 × 5m = 100 min (same as 100 × 1m)

# total combos: 4×5 + 3×3 = 29


# ── Data ──────────────────────────────────────────────────────────────────────

def load_and_resample():
    """
    Load MT4 semicolon-CSV and resample 1m → 5m.
    MT4 date format: '2004.06.11 07:18'
    """
    print("  Reading CSV (6.7 M rows — may take ~20 s) ...")
    df = pd.read_csv(
        DATA_FILE,
        sep=";",
        parse_dates=["Date"],
        date_format="%Y.%m.%d %H:%M",
    )
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.rename(columns={
        "Date": "ts", "Open": "o", "High": "h",
        "Low": "l", "Close": "c", "Volume": "v"
    })
    df = df.set_index("ts")

    print("  Resampling 1m → 5m ...")
    r = df.resample("5min").agg({
        "o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"
    }).dropna(subset=["o"])

    r = r.reset_index()

    # Session columns based on broker-local time (add SESSION_OFFSET to UTC)
    local_ts  = r["ts"] + pd.Timedelta(hours=SESSION_OFFSET)
    r["hour"]   = local_ts.dt.hour
    r["minute"] = local_ts.dt.minute
    r["dow"]    = local_ts.dt.dayofweek   # 0=Mon … 6=Sun
    return r


# ── Indicators ────────────────────────────────────────────────────────────────

def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def build_indicators(df):
    df["ema80"]  = _ema(df["c"], 80)
    df["ema85"]  = _ema(df["c"], 85)
    df["ema340"] = _ema(df["c"], 340)
    df["ema500"] = _ema(df["c"], 500)
    df["sma325"] = df["c"].rolling(325).mean()
    return df


# ── Strategy 1 helpers ────────────────────────────────────────────────────────

def compute_pivot_sweep(df, window=20):
    prior_hi = df["h"].rolling(window).max().shift(1)
    prior_lo = df["l"].rolling(window).min().shift(1)
    df["bull_sweep"] = (df["l"] < prior_lo) & (df["c"] > prior_lo)
    df["bear_sweep"] = (df["h"] > prior_hi) & (df["c"] < prior_hi)
    return df


def compute_ifvg(df):
    h = df["h"].values
    l = df["l"].values
    c = df["c"].values
    n = len(df)

    bull_ifvg = np.zeros(n, dtype=bool)
    bear_ifvg = np.zeros(n, dtype=bool)

    bear_fvg_zones = []
    bull_fvg_zones = []

    for i in range(2, n):
        # Check inversions BEFORE forming new zones
        new_bear = []
        for thr, bound, exp in bear_fvg_zones:
            if i > exp:
                continue
            if c[i] > thr:
                bull_ifvg[i] = True
            else:
                new_bear.append((thr, bound, exp))
        bear_fvg_zones = new_bear

        new_bull = []
        for bound, thr, exp in bull_fvg_zones:
            if i > exp:
                continue
            if c[i] < thr:
                bear_ifvg[i] = True
            else:
                new_bull.append((bound, thr, exp))
        bull_fvg_zones = new_bull

        if h[i] < l[i - 2]:
            bear_fvg_zones.append((h[i], l[i - 2], i + 10))
        if l[i] > h[i - 2]:
            bull_fvg_zones.append((h[i - 2], l[i], i + 10))

    df["bull_ifvg"] = bull_ifvg
    df["bear_ifvg"] = bear_ifvg
    return df


def strategy1_signals(df, sweep_window=20, ifvg_lookback=5):
    """
    NY Open Killzone in broker-local time (UTC+2): 16:30–18:00, Mon–Thu
    """
    df = compute_pivot_sweep(df, sweep_window)
    df = compute_ifvg(df)

    h_arr  = df["hour"].values
    mi_arr = df["minute"].values
    dw_arr = df["dow"].values
    b_ifvg = df["bull_ifvg"].values
    s_ifvg = df["bear_ifvg"].values
    b_sw   = df["bull_sweep"].values
    s_sw   = df["bear_sweep"].values
    n      = len(df)

    long_sig  = np.zeros(n, dtype=bool)
    short_sig = np.zeros(n, dtype=bool)

    last_bull_sw = -9999
    last_bear_sw = -9999

    # Killzone in broker local time (UTC + SESSION_OFFSET)
    kz_start_h, kz_start_m = 14 + SESSION_OFFSET, 30   # 16:30 local
    kz_end_h,   kz_end_m   = 16 + SESSION_OFFSET, 0    # 18:00 local

    for i in range(n):
        if b_sw[i]:
            last_bull_sw = i
        if s_sw[i]:
            last_bear_sw = i

        ho, mi, dw = int(h_arr[i]), int(mi_arr[i]), int(dw_arr[i])
        in_nyo = (
            dw <= 3 and (
                (ho == kz_start_h and mi >= kz_start_m) or
                (ho == kz_start_h + 1) or
                (ho == kz_end_h and mi == kz_end_m)
            )
        )
        if not in_nyo:
            continue

        if b_ifvg[i] and (last_bull_sw >= 0) and (i - last_bull_sw) <= ifvg_lookback:
            long_sig[i] = True
        if s_ifvg[i] and (last_bear_sw >= 0) and (i - last_bear_sw) <= ifvg_lookback:
            short_sig[i] = True

    df["s1_long"]  = long_sig
    df["s1_short"] = short_sig
    return df


# ── Strategy 2 helpers ────────────────────────────────────────────────────────

def strategy2_signals(df):
    c    = df["c"].values
    e80  = df["ema80"].values
    e85  = df["ema85"].values
    e340 = df["ema340"].values
    e500 = df["ema500"].values
    s325 = df["sma325"].values
    n    = len(df)

    long_sig  = np.zeros(n, dtype=bool)
    short_sig = np.zeros(n, dtype=bool)

    for i in range(1, n):
        if np.isnan(s325[i]):
            continue
        if (c[i] > e85[i] and c[i - 1] <= e85[i - 1] and
                e340[i] > e500[i] and c[i] > s325[i] and c[i] > e340[i]):
            long_sig[i] = True
        if (c[i] < e80[i] and c[i - 1] >= e80[i - 1] and
                e340[i] < e500[i] and c[i] < s325[i] and c[i] < e340[i]):
            short_sig[i] = True

    df["s2_long"]  = long_sig
    df["s2_short"] = short_sig
    return df


# ── Backtest engines ──────────────────────────────────────────────────────────

def backtest(df, long_col, short_col, sl_pct, rr,
             risk_pct=RISK_PCT, spread=SPREAD):
    capital  = CAPITAL
    equity   = [capital]
    trades   = []
    in_trade = False

    ls = df[long_col].values
    ss = df[short_col].values
    c  = df["c"].values
    h  = df["h"].values
    l  = df["l"].values
    ts = df["ts"].values
    n  = len(df)

    i = 0
    while i < n:
        if not in_trade:
            sig = "long" if ls[i] else ("short" if ss[i] else None)
            if sig:
                entry    = c[i] * (1 + spread) if sig == "long" else c[i] * (1 - spread)
                sl_dist  = entry * sl_pct
                sl_price = entry - sl_dist if sig == "long" else entry + sl_dist
                tp_price = entry + sl_dist * rr if sig == "long" else entry - sl_dist * rr
                direction, entry_i = sig, i
                in_trade = True
            i += 1
            continue

        hit_sl = hit_tp = False
        if direction == "long":
            hit_sl, exit_price = (True, sl_price) if l[i] <= sl_price else (False, None)
            if not hit_sl:
                hit_tp, exit_price = (True, tp_price) if h[i] >= tp_price else (False, None)
        else:
            hit_sl, exit_price = (True, sl_price) if h[i] >= sl_price else (False, None)
            if not hit_sl:
                hit_tp, exit_price = (True, tp_price) if l[i] <= tp_price else (False, None)

        if hit_sl or hit_tp:
            actual_exit = exit_price * (1 - spread) if direction == "long" else exit_price * (1 + spread)
            pnl_pct = (actual_exit - entry) / entry if direction == "long" else (entry - actual_exit) / entry
            pnl     = capital * risk_pct * (pnl_pct / sl_pct)
            capital += pnl
            equity.append(capital)
            trades.append({
                "entry_ts": ts[entry_i], "exit_ts": ts[i],
                "dir": direction, "pnl": pnl, "win": hit_tp, "capital": capital,
            })
            in_trade = False
        i += 1

    return trades, equity


def backtest_4ema(df, sl_pct, tp_pct,
                  be_pct=S2_BE_PCT, cooldown=S2_COOLDOWN,
                  risk_pct=RISK_PCT, spread=SPREAD):
    capital   = CAPITAL
    equity    = [capital]
    trades    = []
    in_trade  = False
    last_exit = -cooldown

    ls = df["s2_long"].values
    ss = df["s2_short"].values
    c  = df["c"].values
    h  = df["h"].values
    l  = df["l"].values
    ts = df["ts"].values
    n  = len(df)

    entry = sl_price = tp_price = be_price = 0.0
    direction = "long"
    entry_i = 0
    be_triggered = False

    i = 0
    while i < n:
        if not in_trade:
            if (i - last_exit) >= cooldown:
                sig = "long" if ls[i] else ("short" if ss[i] else None)
                if sig:
                    entry     = c[i] * (1 + spread) if sig == "long" else c[i] * (1 - spread)
                    sl_dist   = entry * sl_pct
                    sl_price  = entry - sl_dist   if sig == "long" else entry + sl_dist
                    tp_price  = entry + entry * tp_pct if sig == "long" else entry - entry * tp_pct
                    be_price  = entry + entry * be_pct if sig == "long" else entry - entry * be_pct
                    be_triggered = False
                    direction, entry_i = sig, i
                    in_trade  = True
            i += 1
            continue

        if not be_triggered:
            if direction == "long"  and h[i] >= be_price:
                sl_price = entry; be_triggered = True
            elif direction == "short" and l[i] <= be_price:
                sl_price = entry; be_triggered = True

        hit_sl = hit_tp = False
        if direction == "long":
            hit_sl, exit_price = (True, sl_price) if l[i] <= sl_price else (False, None)
            if not hit_sl:
                hit_tp, exit_price = (True, tp_price) if h[i] >= tp_price else (False, None)
        else:
            hit_sl, exit_price = (True, sl_price) if h[i] >= sl_price else (False, None)
            if not hit_sl:
                hit_tp, exit_price = (True, tp_price) if l[i] <= tp_price else (False, None)

        if hit_sl or hit_tp:
            actual_exit = exit_price * (1 - spread) if direction == "long" else exit_price * (1 + spread)
            pnl_pct = (actual_exit - entry) / entry if direction == "long" else (entry - actual_exit) / entry
            pnl     = capital * risk_pct * (pnl_pct / sl_pct)
            capital += pnl
            equity.append(capital)
            trades.append({
                "entry_ts": ts[entry_i], "exit_ts": ts[i],
                "dir": direction, "pnl": pnl,
                "win": hit_tp, "capital": capital, "be": be_triggered,
            })
            in_trade  = False
            last_exit = i
        i += 1

    return trades, equity


# ── Statistics ────────────────────────────────────────────────────────────────

def calc_stats(trades, equity):
    if len(trades) < 10:
        return None
    wins    = sum(1 for t in trades if t["win"])
    win_pct = 100 * wins / len(trades)

    tdf = pd.DataFrame(trades)
    tdf["ym"]  = pd.to_datetime(tdf["exit_ts"]).dt.to_period("M")
    monthly    = tdf.groupby("ym")["pnl"].sum()

    eq   = np.array(equity)
    peak = np.maximum.accumulate(eq)
    mdd  = ((eq - peak) / peak * 100).min()

    return {
        "trades":    len(trades),
        "win%":      round(win_pct, 1),
        "avg_mo%":   round(monthly.mean() / CAPITAL * 100, 2),
        "best_mo%":  round(monthly.max()  / CAPITAL * 100, 2),
        "worst_mo%": round(monthly.min()  / CAPITAL * 100, 2),
        "mo_pos":    int((monthly > 0).sum()),
        "mo_total":  len(monthly),
        "final_cap": round(equity[-1], 2),
        "mdd%":      round(mdd, 1),
    }


# ── Runners ───────────────────────────────────────────────────────────────────

def run_strategy1(df):
    print("\n" + "=" * 76)
    print("STRATEGY 1 — Sweep + iFVG Scalp  (NY Open Killzone)  [5m, ~21 years]")
    print(f"  Session: broker-local {14+SESSION_OFFSET}:30–{16+SESSION_OFFSET}:00  "
          f"(UTC 14:30–16:00 + {SESSION_OFFSET}h offset), Mon–Thu")
    print("=" * 76)

    df = strategy1_signals(df)
    n_long  = int(df["s1_long"].sum())
    n_short = int(df["s1_short"].sum())
    print(f"Raw signals: {n_long + n_short:,}  (long={n_long:,}, short={n_short:,})")

    results = []
    for sl, rr in product(S1_SL_GRID, S1_RR_GRID):
        trades, equity = backtest(df, "s1_long", "s1_short", sl, rr)
        s = calc_stats(trades, equity)
        if s:
            results.append((f"iFVG|SL{sl*100:.1f}%|RR{rr}", sl, rr, s))

    if not results:
        print("\n  *** No combo produced ≥10 trades. ***")
        return

    results.sort(key=lambda x: x[3]["avg_mo%"], reverse=True)

    hdr = f"{'Label':<28} {'#Tr':>6} {'Win%':>6} {'AvgMo%':>8} {'BestMo%':>9} {'MDD%':>7} {'Final$':>9} {'MoPos':>9}"
    print(f"\n{hdr}")
    print("-" * len(hdr))
    for label, sl, rr, s in results[:15]:
        print(f"{label:<28} {s['trades']:>6} {s['win%']:>6.1f} "
              f"{s['avg_mo%']:>+8.2f}% {s['best_mo%']:>+9.2f}% {s['mdd%']:>7.1f}% "
              f"${s['final_cap']:>8.0f} {s['mo_pos']}/{s['mo_total']}")

    best = results[0]
    b = best[3]
    print(f"\nBEST → {best[0]}  |  avg {b['avg_mo%']:+.2f}%/mo  |  {b['win%']}% win  |  "
          f"MDD {b['mdd%']}%  |  ${b['final_cap']:.0f} final  |  {b['mo_pos']}/{b['mo_total']} months +ve")


def run_strategy2(df):
    print("\n" + "=" * 76)
    print("STRATEGY 2 — Four-EMA Band Scalp  (EMA80/85 · EMA340/500 · SMA325)  [5m]")
    print(f"Breakeven +{S2_BE_PCT*100:.1f}%  |  Cooldown {S2_COOLDOWN} bars = {S2_COOLDOWN*5} min")
    print("=" * 76)

    df = strategy2_signals(df)
    n_long  = int(df["s2_long"].sum())
    n_short = int(df["s2_short"].sum())
    print(f"Raw signals: {n_long + n_short:,}  (long={n_long:,}, short={n_short:,})")

    results = []
    for sl, tp in product(S2_SL_GRID, S2_TP_GRID):
        trades, equity = backtest_4ema(df, sl_pct=sl, tp_pct=tp)
        s = calc_stats(trades, equity)
        if s:
            be_hits = sum(1 for t in trades if t.get("be"))
            s["be_pct"] = round(100 * be_hits / len(trades), 1)
            results.append((f"4EMA|SL{sl*100:.2f}%|TP{tp*100:.1f}%", sl, tp, s))

    if not results:
        print("\n  *** No combo produced ≥10 trades. ***")
        return

    results.sort(key=lambda x: x[3]["avg_mo%"], reverse=True)

    hdr = f"{'Label':<30} {'#Tr':>6} {'Win%':>6} {'BE%':>5} {'AvgMo%':>8} {'MDD%':>7} {'Final$':>9} {'MoPos':>9}"
    print(f"\n{hdr}")
    print("-" * len(hdr))
    for label, sl, tp, s in results:
        print(f"{label:<30} {s['trades']:>6} {s['win%']:>6.1f} "
              f"{s['be_pct']:>5.1f} {s['avg_mo%']:>+8.2f}% {s['mdd%']:>7.1f}% "
              f"${s['final_cap']:>8.0f} {s['mo_pos']}/{s['mo_total']}")

    best = results[0]
    b = best[3]
    print(f"\nBEST → {best[0]}  |  avg {b['avg_mo%']:+.2f}%/mo  |  {b['win%']}% win  |  "
          f"BE {b['be_pct']}%  |  MDD {b['mdd%']}%  |  ${b['final_cap']:.0f} final")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    total_combos = len(S1_SL_GRID) * len(S1_RR_GRID) + len(S2_SL_GRID) * len(S2_TP_GRID)
    print(f"test_5m_longrun.py  —  XAUUSD 5m  |  RISK {RISK_PCT*100:.0f}%  |  "
          f"Spread {SPREAD*100:.3f}% both sides  |  {total_combos} combos")
    print(f"  Source: {DATA_FILE}")

    df = load_and_resample()
    print(f"  Loaded {len(df):,} × 5m bars  "
          f"{df['ts'].iloc[0].strftime('%Y-%m-%d')}  →  {df['ts'].iloc[-1].strftime('%Y-%m-%d')}")

    print("  Building indicators ...")
    df = build_indicators(df)

    run_strategy1(df)
    run_strategy2(df)

    print("\n" + "=" * 76)
    print("Done.")
