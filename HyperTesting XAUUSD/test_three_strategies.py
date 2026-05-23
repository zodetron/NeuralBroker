"""
test_three_strategies.py
XAUUSD 1-minute backtesting for three requested strategies.

  Strategy 1 : Sweep + iFVG Scalp  (NY Open Killzone 14:31–16:00 UTC)
  Strategy 2 : Four-EMA Band Scalping with Breakeven  (EMA80/85/340/500 + SMA325)
  Strategy 3 : Pure Arbitrage — NOT IMPLEMENTABLE (see strategy3_note() below)

WHY STRATEGY 3 CANNOT BE BACKTESTED
-------------------------------------
Pure price arbitrage requires simultaneous real-time quotes from ≥2 separate
venues (e.g. two brokers, XAU spot vs futures). This dataset is single-venue
OHLCV at 1-minute resolution with Volume = 0.0 throughout.

  • No second price stream to compare against.
  • Arb windows close in <100 ms; a 1-min bar loses all intra-bar path info.
  • No bid/ask spread per venue — arb profit is precisely the inter-venue spread diff.
  • Volume = 0 → cannot model order-book depth or venue imbalances.

To test it you would need: tick-level dual-feed data from two brokers (or XAU
futures + spot), a sub-second latency model, and per-venue commission/spread data.

NOTE ON ABSORPTION BUBBLE (Strategy 1)
----------------------------------------
The original strategy 1 calls for an "absorption bubble" detector based on
volume divergence (large size absorbed at a level without price movement).
Volume = 0.0 for every candle in this dataset — it is structurally impossible
to implement. The volume-based component is replaced with the iFVG inversion
signal which captures the same idea (price rejected at the swept level).
"""

import pandas as pd
import numpy as np
from itertools import product
import os

DATA_DIR    = "/Users/hardik/Projects/NeuralBroker/April25"
DATA_FILE   = os.path.join(DATA_DIR, "xauusd_1m_6mo.csv")

CAPITAL     = 200.0
CONTRACT_SZ = 100          # oz per standard lot for XAUUSD
MIN_LOT     = 0.01
RISK_PCT    = 0.02          # 2% per trade
SPREAD      = 0.00025       # 0.025% on entry

# ── Strategy 1 grid (Sweep + iFVG Scalp) ─────────────────────────────────────
S1_SL_GRID  = [0.002, 0.003, 0.005, 0.008]   # 0.2 / 0.3 / 0.5 / 0.8 %
S1_RR_GRID  = [2.0, 3.0, 5.0, 7.0, 10.0]    # TP = SL × RR

# ── Strategy 2 grid (Four-EMA Band with breakeven) ───────────────────────────
S2_SL_GRID  = [0.0015, 0.002, 0.003]         # 0.15 / 0.2 / 0.3 %
S2_TP_GRID  = [0.015, 0.025, 0.04]           # 1.5 / 2.5 / 4.0 %
S2_BE_PCT   = 0.003                           # move SL→entry when unrealised ≥ +0.3%
S2_COOLDOWN = 100                             # bars between trades

# total combos: S1 = 4×5 = 20 │ S2 = 3×3 = 9 │ Grand total = 29  (< 200)


# ── Data ──────────────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_FILE, parse_dates=["open_time"])
    df = df.sort_values("open_time").reset_index(drop=True)
    df.rename(columns={
        "open_time": "ts", "Open": "o", "High": "h",
        "Low": "l", "Close": "c", "Volume": "v"
    }, inplace=True)
    df["hour"] = df["ts"].dt.hour
    df["minute"] = df["ts"].dt.minute
    df["dow"]  = df["ts"].dt.dayofweek   # 0=Mon … 6=Sun
    return df


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
    """
    Bullish sweep : candle wicks *below* the prior-window pivot low but closes above it
                    → stop-hunt of long-side liquidity, expect reversal up
    Bearish sweep : candle wicks *above* the prior-window pivot high but closes below it
                    → stop-hunt of short-side liquidity, expect reversal down
    """
    prior_hi = df["h"].rolling(window).max().shift(1)
    prior_lo = df["l"].rolling(window).min().shift(1)
    df["bull_sweep"] = (df["l"] < prior_lo) & (df["c"] > prior_lo)
    df["bear_sweep"] = (df["h"] > prior_hi) & (df["c"] < prior_hi)
    return df


def compute_ifvg(df):
    """
    An Inverted FVG (iFVG) is a Fair Value Gap that price has already
    re-entered and closed through — signalling the gap is now acting as
    support / resistance rather than being a void.

    Bearish FVG : h[i] < l[i-2]  → gap = (h[i], l[i-2])
    Bullish iFVG: close later crosses *above* h[i]  (gap flips to support)

    Bullish FVG : l[i] > h[i-2]  → gap = (h[i-2], l[i])
    Bearish iFVG: close later crosses *below* l[i]  (gap flips to resistance)

    Each zone expires after 10 bars to keep signals fresh.
    """
    h = df["h"].values
    l = df["l"].values
    c = df["c"].values
    n = len(df)

    bull_ifvg = np.zeros(n, dtype=bool)
    bear_ifvg = np.zeros(n, dtype=bool)

    # (inversion_threshold, raw_bound, expiry_bar)
    bear_fvg_zones = []   # becomes bullish iFVG when c > threshold
    bull_fvg_zones = []   # becomes bearish iFVG when c < threshold

    for i in range(2, n):
        # 1. Check inversions of existing zones (BEFORE forming new ones)
        new_bear = []
        for thr, bound, exp in bear_fvg_zones:
            if i > exp:
                continue
            if c[i] > thr:
                bull_ifvg[i] = True   # consumed
            else:
                new_bear.append((thr, bound, exp))
        bear_fvg_zones = new_bear

        new_bull = []
        for bound, thr, exp in bull_fvg_zones:
            if i > exp:
                continue
            if c[i] < thr:
                bear_ifvg[i] = True   # consumed
            else:
                new_bull.append((bound, thr, exp))
        bull_fvg_zones = new_bull

        # 2. Form new FVG zones for future inversion
        if h[i] < l[i - 2]:                           # bearish FVG
            bear_fvg_zones.append((h[i], l[i - 2], i + 10))
        if l[i] > h[i - 2]:                           # bullish FVG
            bull_fvg_zones.append((h[i - 2], l[i], i + 10))

    df["bull_ifvg"] = bull_ifvg
    df["bear_ifvg"] = bear_ifvg
    return df


def strategy1_signals(df, sweep_window=20, ifvg_lookback=5):
    """
    Entry conditions (both must be met on the same bar):

      Long  : bullish iFVG at bar i  AND  a bullish sweep occurred within
              the last `ifvg_lookback` bars
      Short : bearish iFVG at bar i  AND  a bearish sweep occurred within
              the last `ifvg_lookback` bars

    Session filter: NY Open Killzone 14:31–16:00 UTC, Monday–Thursday only.
    """
    df = compute_pivot_sweep(df, sweep_window)
    df = compute_ifvg(df)

    h_arr   = df["hour"].values
    mi_arr  = df["minute"].values
    dw_arr  = df["dow"].values
    b_ifvg  = df["bull_ifvg"].values
    s_ifvg  = df["bear_ifvg"].values
    b_sw    = df["bull_sweep"].values
    s_sw    = df["bear_sweep"].values
    n       = len(df)

    long_sig  = np.zeros(n, dtype=bool)
    short_sig = np.zeros(n, dtype=bool)

    last_bull_sw = -9999
    last_bear_sw = -9999

    for i in range(n):
        # Update sweep trackers regardless of session
        if b_sw[i]:
            last_bull_sw = i
        if s_sw[i]:
            last_bear_sw = i

        # NY Open Killzone: 14:31–16:00 UTC, Mon–Thu
        ho, mi, dw = int(h_arr[i]), int(mi_arr[i]), int(dw_arr[i])
        in_nyo = (
            dw <= 3 and (
                (ho == 14 and mi >= 31) or
                (ho == 15) or
                (ho == 16 and mi == 0)
            )
        )
        if not in_nyo:
            continue

        recent_bull_sw = (last_bull_sw >= 0) and ((i - last_bull_sw) <= ifvg_lookback)
        recent_bear_sw = (last_bear_sw >= 0) and ((i - last_bear_sw) <= ifvg_lookback)

        if b_ifvg[i] and recent_bull_sw:
            long_sig[i] = True
        if s_ifvg[i] and recent_bear_sw:
            short_sig[i] = True

    df["s1_long"]  = long_sig
    df["s1_short"] = short_sig
    return df


# ── Strategy 2 helpers ────────────────────────────────────────────────────────

def strategy2_signals(df):
    """
    Long  : close crosses *above* EMA85 (top of 80/85 band) from below,
            EMA340 > EMA500 (uptrend), close > SMA325, close > EMA340
    Short : close crosses *below* EMA80 (bottom of band) from above,
            EMA340 < EMA500 (downtrend), close < SMA325, close < EMA340

    The EMA80/85 band acts as a dynamic equilibrium — a cross out of the band
    in the direction of the trend is the entry trigger.
    """
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

        # Long: cross up through EMA85
        if (c[i] > e85[i] and c[i - 1] <= e85[i - 1] and
                e340[i] > e500[i] and c[i] > s325[i] and c[i] > e340[i]):
            long_sig[i] = True

        # Short: cross down through EMA80
        if (c[i] < e80[i] and c[i - 1] >= e80[i - 1] and
                e340[i] < e500[i] and c[i] < s325[i] and c[i] < e340[i]):
            short_sig[i] = True

    df["s2_long"]  = long_sig
    df["s2_short"] = short_sig
    return df


# ── Backtest engines ──────────────────────────────────────────────────────────

def backtest(df, long_col, short_col, sl_pct, rr,
             risk_pct=RISK_PCT, spread=SPREAD):
    """Standard percentage-SL, fixed-RR engine (Strategy 1)."""
    capital  = CAPITAL
    equity   = [capital]
    trades   = []
    in_trade = False

    ls  = df[long_col].values
    ss  = df[short_col].values
    c   = df["c"].values
    h   = df["h"].values
    l   = df["l"].values
    ts  = df["ts"].values
    n   = len(df)

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
            # Apply spread on exit: long sells at bid (−spread), short covers at ask (+spread)
            actual_exit = exit_price * (1 - spread) if direction == "long" else exit_price * (1 + spread)
            pnl_pct  = (actual_exit - entry) / entry if direction == "long" else (entry - actual_exit) / entry
            pnl      = capital * risk_pct * (pnl_pct / sl_pct)
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
    """
    Modified engine for Strategy 2:
      • TP defined as a fixed % of entry (not RR of SL)
      • Breakeven: when unrealised gain ≥ be_pct, SL moves to entry (zero-loss floor)
      • Cooldown: minimum `cooldown` bars between trade exits and next entry
    """
    capital   = CAPITAL
    equity    = [capital]
    trades    = []
    in_trade  = False
    last_exit = -cooldown

    ls  = df["s2_long"].values
    ss  = df["s2_short"].values
    c   = df["c"].values
    h   = df["h"].values
    l   = df["l"].values
    ts  = df["ts"].values
    n   = len(df)

    # These are assigned inside the trade-open block; satisfy linter with dummies
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

        # ── Breakeven trigger ──────────────────────────────────────────────
        if not be_triggered:
            if direction == "long"  and h[i] >= be_price:
                sl_price     = entry
                be_triggered = True
            elif direction == "short" and l[i] <= be_price:
                sl_price     = entry
                be_triggered = True

        # ── SL / TP check ─────────────────────────────────────────────────
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
            pnl_pct  = (exit_price - entry) / entry if direction == "long" else (entry - exit_price) / entry
            # size based on original SL; if BE triggered and SL at entry, pnl_pct=0 → pnl=0
            pnl      = capital * risk_pct * (pnl_pct / sl_pct)
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
    if len(trades) < 5:
        return None
    wins    = sum(1 for t in trades if t["win"])
    win_pct = 100 * wins / len(trades)

    tdf = pd.DataFrame(trades)
    tdf["ym"]   = pd.to_datetime(tdf["exit_ts"]).dt.to_period("M")
    monthly     = tdf.groupby("ym")["pnl"].sum()

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
    print("\n" + "=" * 72)
    print("STRATEGY 1 — Sweep + iFVG Scalp   (NY Open Killzone 14:31–16:00 UTC)")
    print("NOTE: Absorption bubble SKIPPED — Volume=0.0 throughout dataset")
    print("=" * 72)

    df = strategy1_signals(df)
    n_long  = int(df["s1_long"].sum())
    n_short = int(df["s1_short"].sum())
    print(f"Raw signals before backtest: {n_long + n_short}  "
          f"(long={n_long}, short={n_short})")

    results = []
    for sl, rr in product(S1_SL_GRID, S1_RR_GRID):
        trades, equity = backtest(df, "s1_long", "s1_short", sl, rr)
        s = calc_stats(trades, equity)
        if s:
            results.append((f"iFVG_Scalp|SL{sl*100:.1f}%|RR{rr}", sl, rr, s))

    if not results:
        print("\n  *** No combo produced ≥5 trades — signals too sparse for this dataset. ***")
        print("  Consider widening the iFVG lookback or the session window.")
        return

    results.sort(key=lambda x: x[3]["avg_mo%"], reverse=True)

    hdr = f"{'Label':<35} {'#Tr':>5} {'Win%':>6} {'AvgMo%':>8} {'MDD%':>7} {'Final$':>8} {'MoPos':>7}"
    print(f"\n{hdr}")
    print("-" * len(hdr))
    for label, sl, rr, s in results[:12]:
        print(f"{label:<35} {s['trades']:>5} {s['win%']:>6.1f} "
              f"{s['avg_mo%']:>+8.2f}% {s['mdd%']:>7.1f}% "
              f"${s['final_cap']:>7.0f} {s['mo_pos']}/{s['mo_total']}")

    best = results[0]
    b = best[3]
    print(f"\nBEST → {best[0]} | avg {b['avg_mo%']:+.2f}%/mo | {b['win%']}% win | "
          f"MDD {b['mdd%']}% | ${b['final_cap']:.0f} final")


def run_strategy2(df):
    print("\n" + "=" * 72)
    print("STRATEGY 2 — Four-EMA Band Scalp  (EMA80/85 band · EMA340/500 · SMA325)")
    print(f"Breakeven: SL→entry at +{S2_BE_PCT*100:.1f}%  |  Cooldown: {S2_COOLDOWN} bars")
    print("=" * 72)

    df = strategy2_signals(df)
    n_long  = int(df["s2_long"].sum())
    n_short = int(df["s2_short"].sum())
    print(f"Raw signals before backtest: {n_long + n_short}  "
          f"(long={n_long}, short={n_short})")

    results = []
    for sl, tp in product(S2_SL_GRID, S2_TP_GRID):
        trades, equity = backtest_4ema(df, sl_pct=sl, tp_pct=tp)
        s = calc_stats(trades, equity)
        if s:
            be_hits = sum(1 for t in trades if t.get("be"))
            s["be_pct"] = round(100 * be_hits / len(trades), 1)
            results.append((f"4EMA|SL{sl*100:.2f}%|TP{tp*100:.1f}%", sl, tp, s))

    if not results:
        print("\n  *** No combo produced ≥5 trades. ***")
        return

    results.sort(key=lambda x: x[3]["avg_mo%"], reverse=True)

    hdr = f"{'Label':<34} {'#Tr':>5} {'Win%':>6} {'BE%':>5} {'AvgMo%':>8} {'MDD%':>7} {'Final$':>8} {'MoPos':>7}"
    print(f"\n{hdr}")
    print("-" * len(hdr))
    for label, sl, tp, s in results:
        print(f"{label:<34} {s['trades']:>5} {s['win%']:>6.1f} "
              f"{s['be_pct']:>5.1f} {s['avg_mo%']:>+8.2f}% {s['mdd%']:>7.1f}% "
              f"${s['final_cap']:>7.0f} {s['mo_pos']}/{s['mo_total']}")

    best = results[0]
    b = best[3]
    print(f"\nBEST → {best[0]} | avg {b['avg_mo%']:+.2f}%/mo | {b['win%']}% win | "
          f"BE used {b['be_pct']}% of trades | MDD {b['mdd%']}% | ${b['final_cap']:.0f} final")


def strategy3_note():
    print("\n" + "=" * 72)
    print("STRATEGY 3 — Pure Arbitrage   *** NOT IMPLEMENTABLE — SKIPPED ***")
    print("=" * 72)
    print("""
  Reason: requires simultaneous real-time quotes from ≥2 venues.
  This dataset: single-venue OHLCV, 1-min bars, Volume=0.0 throughout.

  Blockers
  --------
  1.  No second price stream to compute a spread between venues.
  2.  Arb windows close in < 1 second; 1-min bars have no intra-bar resolution.
  3.  No bid/ask data → cannot model the inter-venue spread differential that
      is the actual source of arb profit.
  4.  Volume = 0 → no order-flow signal for venue imbalance detection.

  To actually test this you would need:
  • Dual tick feeds (e.g., two MT4 brokers or XAU spot + CME futures)
  • Sub-second execution latency model
  • Per-venue spread + commission model
""")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    total_combos = len(S1_SL_GRID) * len(S1_RR_GRID) + len(S2_SL_GRID) * len(S2_TP_GRID)
    print(f"test_three_strategies.py  —  XAUUSD 1m  |  RISK {RISK_PCT*100:.0f}%  |  "
          f"Spread {SPREAD*100:.3f}%  |  {total_combos} total combos")
    print(f"  S1 (iFVG Scalp)   : {len(S1_SL_GRID)} SL × {len(S1_RR_GRID)} RR = "
          f"{len(S1_SL_GRID)*len(S1_RR_GRID)} combos")
    print(f"  S2 (4-EMA Band)   : {len(S2_SL_GRID)} SL × {len(S2_TP_GRID)} TP = "
          f"{len(S2_SL_GRID)*len(S2_TP_GRID)} combos")
    print(f"  S3 (Pure Arb)     : 0 combos — skipped (see note)")

    print(f"\nLoading {DATA_FILE} ...")
    df = load_data()
    print(f"Loaded {len(df):,} candles  {df['ts'].iloc[0]}  →  {df['ts'].iloc[-1]}")

    print("Building indicators ...")
    df = build_indicators(df)

    run_strategy1(df)
    run_strategy2(df)
    strategy3_note()

    print("\n" + "=" * 72)
    print("Done.")
