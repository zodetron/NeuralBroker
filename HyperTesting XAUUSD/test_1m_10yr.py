"""
test_1m_10yr.py
Strategy 1 (Sweep + iFVG Scalp) and Strategy 2 (Four-EMA Band) on 1-minute bars.

Source : XAU_1m_data.csv  (MT4, 2004–2025)  — filtered to last 10 years (2016–2025)
         ~3.5 M 1m bars

Session offset: MT4 broker time assumed UTC+2.
  NY Open Killzone UTC 14:30–16:00  →  broker local 16:30–18:00
"""

import pandas as pd
import numpy as np
from itertools import product
import os

DATA_FILE      = "/Users/hardik/Projects/NeuralBroker/HyperTesting XAUUSD/XAU_1m_data.csv"
YEAR_START     = 2016      # change to narrow/widen the window
SESSION_OFFSET = 2         # UTC+2 broker local; set 0 if data is already UTC

CAPITAL     = 200.0
CONTRACT_SZ = 100
MIN_LOT     = 0.01
RISK_PCT    = 0.02
SPREAD      = 0.00025      # 0.025% both entry and exit

S1_SL_GRID = [0.002, 0.003, 0.005, 0.008]
S1_RR_GRID = [2.0, 3.0, 5.0, 7.0, 10.0]

S2_SL_GRID  = [0.0015, 0.002, 0.003]
S2_TP_GRID  = [0.015, 0.025, 0.04]
S2_BE_PCT   = 0.003
S2_COOLDOWN = 100          # bars (= 100 min on 1m, same as original)

# ── Data ──────────────────────────────────────────────────────────────────────

def load_data():
    print(f"  Reading {DATA_FILE} ...")
    df = pd.read_csv(
        DATA_FILE,
        sep=";",
        parse_dates=["Date"],
        date_format="%Y.%m.%d %H:%M",
    )
    df = df.sort_values("Date").reset_index(drop=True)
    df.rename(columns={
        "Date": "ts", "Open": "o", "High": "h",
        "Low": "l", "Close": "c", "Volume": "v"
    }, inplace=True)

    df = df[df["ts"].dt.year >= YEAR_START].reset_index(drop=True)
    print(f"  Filtered to {YEAR_START}+: {len(df):,} bars  "
          f"{df['ts'].iloc[0].strftime('%Y-%m-%d')}  →  {df['ts'].iloc[-1].strftime('%Y-%m-%d')}")

    # Session columns in broker-local time
    local = df["ts"] + pd.Timedelta(hours=SESSION_OFFSET)
    df["hour"]   = local.dt.hour
    df["minute"] = local.dt.minute
    df["dow"]    = local.dt.dayofweek
    df["year"]   = df["ts"].dt.year
    return df


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


# ── Strategy 1 ────────────────────────────────────────────────────────────────

def compute_pivot_sweep(df, window=20):
    prior_hi = df["h"].rolling(window).max().shift(1)
    prior_lo = df["l"].rolling(window).min().shift(1)
    df["bull_sweep"] = (df["l"] < prior_lo) & (df["c"] > prior_lo)
    df["bear_sweep"] = (df["h"] > prior_hi) & (df["c"] < prior_hi)
    return df


def compute_ifvg(df):
    """O(n) iFVG detector — zones expire after 10 bars to keep the list short."""
    h = df["h"].values
    l = df["l"].values
    c = df["c"].values
    n = len(df)

    bull_ifvg = np.zeros(n, dtype=bool)
    bear_ifvg = np.zeros(n, dtype=bool)
    bear_zones = []   # (inversion_thr, expiry)
    bull_zones = []   # (inversion_thr, expiry)

    for i in range(2, n):
        # check inversions first
        new_bear = [(t, e) for t, e in bear_zones if i <= e and (c[i] <= t or (bull_ifvg.__setitem__(i, True) or False))]
        # cleaner version:
        nb, nb2 = [], []
        for thr, exp in bear_zones:
            if i > exp:
                continue
            if c[i] > thr:
                bull_ifvg[i] = True
            else:
                nb.append((thr, exp))
        bear_zones = nb

        for thr, exp in bull_zones:
            if i > exp:
                continue
            if c[i] < thr:
                bear_ifvg[i] = True
            else:
                nb2.append((thr, exp))
        bull_zones = nb2

        if h[i] < l[i - 2]:
            bear_zones.append((h[i], i + 10))       # inverted when c > h[i]
        if l[i] > h[i - 2]:
            bull_zones.append((l[i], i + 10))        # inverted when c < l[i]

    df["bull_ifvg"] = bull_ifvg
    df["bear_ifvg"] = bear_ifvg
    return df


def strategy1_signals(df, sweep_window=20, ifvg_lookback=5):
    """
    Vectorized session filter + sweep-recency check.
    Killzone (broker-local, UTC+2): 16:30–18:00 Mon–Thu
    """
    df = compute_pivot_sweep(df, sweep_window)
    df = compute_ifvg(df)

    ho = df["hour"].values
    mi = df["minute"].values
    dw = df["dow"].values
    kh = 14 + SESSION_OFFSET    # 16
    eh = 16 + SESSION_OFFSET    # 18

    # Vectorised killzone mask
    in_nyo = (
        (dw <= 3) &
        (
            ((ho == kh) & (mi >= 30)) |
            (ho == kh + 1) |
            ((ho == eh) & (mi == 0))
        )
    )

    b_sw = df["bull_sweep"].values
    s_sw = df["bear_sweep"].values
    b_iv = df["bull_ifvg"].values
    s_iv = df["bear_ifvg"].values
    n    = len(df)

    # Rolling "recent sweep": True if any sweep in [i-lookback, i]
    # Build using cumsum trick: recent = any True in last K bars
    def recent_within(arr, k):
        out = np.zeros(n, dtype=bool)
        for lag in range(k + 1):
            if lag == 0:
                out |= arr
            else:
                out[lag:] |= arr[:-lag]
        return out

    recent_bull = recent_within(b_sw, ifvg_lookback)
    recent_bear = recent_within(s_sw, ifvg_lookback)

    df["s1_long"]  = in_nyo & b_iv & recent_bull
    df["s1_short"] = in_nyo & s_iv & recent_bear
    return df


# ── Strategy 2 ────────────────────────────────────────────────────────────────

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
            ax = exit_price * (1 - spread) if direction == "long" else exit_price * (1 + spread)
            pnl_pct = (ax - entry) / entry if direction == "long" else (entry - ax) / entry
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
            ax = exit_price * (1 - spread) if direction == "long" else exit_price * (1 + spread)
            pnl_pct = (ax - entry) / entry if direction == "long" else (entry - ax) / entry
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
    tdf["ym"]   = pd.to_datetime(tdf["exit_ts"]).dt.to_period("M")
    tdf["year"] = pd.to_datetime(tdf["exit_ts"]).dt.year
    monthly     = tdf.groupby("ym")["pnl"].sum()
    annual      = tdf.groupby("year")["pnl"].sum()

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
        "annual":    annual,
    }


def print_annual(annual, label):
    print(f"\n  Year-by-year  [{label}]")
    print(f"  {'Year':>6}  {'P&L $':>9}  {'Return':>8}")
    print(f"  {'----':>6}  {'---------':>9}  {'------':>8}")
    for yr, pnl in annual.items():
        ret = pnl / CAPITAL * 100
        marker = " ◄" if ret < 0 else ""
        print(f"  {yr:>6}  ${pnl:>8.0f}  {ret:>+7.1f}%{marker}")


# ── Runners ───────────────────────────────────────────────────────────────────

def run_strategy1(df):
    print("\n" + "=" * 78)
    print("STRATEGY 1 — Sweep + iFVG Scalp  [1m, 10yr]")
    print(f"  NY Open Killzone: broker-local {14+SESSION_OFFSET}:30–{16+SESSION_OFFSET}:00 "
          f"(UTC 14:30–16:00 + {SESSION_OFFSET}h), Mon–Thu")
    print("=" * 78)

    print("  Computing signals ...")
    df = strategy1_signals(df)
    n_long  = int(df["s1_long"].sum())
    n_short = int(df["s1_short"].sum())
    print(f"  Raw signals: {n_long + n_short:,}  (long={n_long:,}, short={n_short:,})")

    results = []
    print("  Running combos ...")
    for sl, rr in product(S1_SL_GRID, S1_RR_GRID):
        trades, equity = backtest(df, "s1_long", "s1_short", sl, rr)
        s = calc_stats(trades, equity)
        if s:
            results.append((f"iFVG|SL{sl*100:.1f}%|RR{rr}", sl, rr, s))

    if not results:
        print("  *** No combo produced ≥10 trades. ***")
        return

    results.sort(key=lambda x: x[3]["avg_mo%"], reverse=True)

    hdr = f"  {'Label':<26} {'#Tr':>6} {'Win%':>6} {'AvgMo%':>8} {'BestMo':>8} {'WorstMo':>9} {'MDD%':>7} {'Final$':>8} {'Mo+':>7}"
    print(f"\n{hdr}")
    print("  " + "-" * (len(hdr) - 2))
    for label, sl, rr, s in results:
        print(f"  {label:<26} {s['trades']:>6} {s['win%']:>6.1f} "
              f"{s['avg_mo%']:>+8.2f}% {s['best_mo%']:>+8.2f}% {s['worst_mo%']:>+9.2f}% "
              f"{s['mdd%']:>7.1f}% ${s['final_cap']:>7.0f} {s['mo_pos']}/{s['mo_total']}")

    best = results[0]
    b = best[3]
    print(f"\n  BEST → {best[0]}  |  avg {b['avg_mo%']:+.2f}%/mo  |  {b['win%']}% win  |  "
          f"MDD {b['mdd%']}%  |  ${b['final_cap']:.0f} final  |  {b['mo_pos']}/{b['mo_total']} months +ve")
    print_annual(b["annual"], best[0])

    # Also print top 3 by MDD (safest profitable combos)
    profitable = [(lbl, sl, rr, s) for lbl, sl, rr, s in results if s["avg_mo%"] > 0]
    if profitable:
        safest = sorted(profitable, key=lambda x: x[3]["mdd%"], reverse=True)
        if safest[0][0] != best[0]:
            sb = safest[0]
            print(f"\n  SAFEST (lowest MDD with +avg): {sb[0]}  |  avg {sb[3]['avg_mo%']:+.2f}%/mo  |  "
                  f"MDD {sb[3]['mdd%']}%  |  {sb[3]['win%']}% win")
            print_annual(sb[3]["annual"], sb[0])


def run_strategy2(df):
    print("\n" + "=" * 78)
    print("STRATEGY 2 — Four-EMA Band  [1m, 10yr]")
    print(f"  BE +{S2_BE_PCT*100:.1f}%  |  Cooldown {S2_COOLDOWN} bars")
    print("=" * 78)

    print("  Computing signals ...")
    df = strategy2_signals(df)
    n_long  = int(df["s2_long"].sum())
    n_short = int(df["s2_short"].sum())
    print(f"  Raw signals: {n_long + n_short:,}  (long={n_long:,}, short={n_short:,})")

    results = []
    print("  Running combos ...")
    for sl, tp in product(S2_SL_GRID, S2_TP_GRID):
        trades, equity = backtest_4ema(df, sl_pct=sl, tp_pct=tp)
        s = calc_stats(trades, equity)
        if s:
            be_hits = sum(1 for t in trades if t.get("be"))
            s["be_pct"] = round(100 * be_hits / len(trades), 1)
            results.append((f"4EMA|SL{sl*100:.2f}%|TP{tp*100:.1f}%", sl, tp, s))

    if not results:
        print("  *** No combo produced ≥10 trades. ***")
        return

    results.sort(key=lambda x: x[3]["avg_mo%"], reverse=True)

    hdr = f"  {'Label':<28} {'#Tr':>6} {'Win%':>6} {'BE%':>5} {'AvgMo%':>8} {'MDD%':>7} {'Final$':>8} {'Mo+':>7}"
    print(f"\n{hdr}")
    print("  " + "-" * (len(hdr) - 2))
    for label, sl, tp, s in results:
        print(f"  {label:<28} {s['trades']:>6} {s['win%']:>6.1f} "
              f"{s['be_pct']:>5.1f} {s['avg_mo%']:>+8.2f}% {s['mdd%']:>7.1f}% "
              f"${s['final_cap']:>7.0f} {s['mo_pos']}/{s['mo_total']}")

    best = results[0]
    b = best[3]
    print(f"\n  BEST → {best[0]}  |  avg {b['avg_mo%']:+.2f}%/mo  |  {b['win%']}% win  |  "
          f"BE {b['be_pct']}%  |  MDD {b['mdd%']}%  |  ${b['final_cap']:.0f} final")
    if b["avg_mo%"] > 0:
        print_annual(b["annual"], best[0])


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    total = len(S1_SL_GRID) * len(S1_RR_GRID) + len(S2_SL_GRID) * len(S2_TP_GRID)
    print(f"test_1m_10yr.py  —  XAUUSD 1m  |  {YEAR_START}–2025  |  "
          f"RISK {RISK_PCT*100:.0f}%  |  Spread {SPREAD*100:.3f}% both sides  |  {total} combos")

    df = load_data()
    print("  Building indicators ...")
    df = build_indicators(df)

    run_strategy1(df)
    run_strategy2(df)

    print("\n" + "=" * 78)
    print("Done.")
