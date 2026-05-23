"""
$100 → $1,000,000 COMPOUND OPTIMIZER — XAUUSD 5m | 2 Years
Finds the strategy + risk% that maximises P($1M) with full geometric compounding.

DATA    : ../April25/xauusd_5m_2y.csv  (164 k candles, UTC)
FEATURES:
  • 7 strategies — 5 proven winners + 2 new filtered combos
  • SL sweep 0.002–0.005 × RR sweep 3.0–4.0 × risk% sweep 0.5–5%
  • London + NY session filter (UTC 08:00–22:00)
  • Drawdown circuit-breaker (half risk at -15%, quarter risk at -30%)
  • Milestone log: $200 / $1K / $5K / $10K / $100K / $1M
  • Monte Carlo: 500 bootstrap paths per top-5 combos
  • Kelly Criterion: optimal risk% per strategy
  • Compound growth curves + MC percentile bands (matplotlib)
"""

import os, sys, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless; swap to TkAgg / Qt5Agg for live window
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
START_CAPITAL = 100.0
LEVERAGE      = 1000
MIN_LOT       = 0.01
LOT_STEP      = 0.01
SPREAD        = 0.00025        # 0.025% (2.5 pips on gold ≈ $0.60)

SL_GRID       = [0.002, 0.003, 0.005]   # 0.2 / 0.3 / 0.5 %
RR_GRID       = [3.0, 3.5, 4.0]
RISK_GRID     = [0.005, 0.010, 0.015, 0.020, 0.030, 0.050]  # 0.5 % … 5 %

SESSION_UTC_START = 8    # London open
SESSION_UTC_END   = 22   # NY close

MILESTONES    = [200, 500, 1_000, 5_000, 10_000, 100_000, 250_000, 500_000, 1_000_000]
MC_PATHS      = 500
TOP_N_MONTEC  = 5

# Drawdown circuit-breaker thresholds (fraction from peak)
DD_HALF       = -0.15   # reduce to 50 % risk
DD_QUARTER    = -0.30   # reduce to 25 % risk

# ── DATA LOADING ──────────────────────────────────────────────────────────────
CSV = os.path.join(os.path.dirname(__file__), "..", "April25", "xauusd_5m_2y.csv")

def load_data():
    if not os.path.exists(CSV):
        sys.exit(f"ERROR: data not found at {CSV}\nRun test_hyper3.py first to download it.")
    df = pd.read_csv(CSV, parse_dates=["open_time"])
    df.set_index("open_time", inplace=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df.sort_index(inplace=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    # use last 2 years
    cutoff = df.index[-1] - pd.DateOffset(years=2)
    return df[df.index >= cutoff].copy()

# ── INDICATOR BUILDER ─────────────────────────────────────────────────────────
def build_indicators(df):
    d = df.copy()
    c = d["Close"]

    for s in [5, 8, 9, 13, 20, 21, 50, 100, 200]:
        d[f"ema{s}"] = c.ewm(span=s, adjust=False).mean()

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    d["macd"]      = ema12 - ema26
    d["macd_sig"]  = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_sig"]

    delta     = c.diff()
    g         = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    l         = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    d["rsi14"] = 100 - 100 / (1 + g / l)

    hl = d["High"] - d["Low"]
    hp = (d["High"] - c.shift()).abs()
    lp = (d["Low"]  - c.shift()).abs()
    tr = pd.concat([hl, hp, lp], axis=1).max(axis=1)
    d["atr14"] = tr.ewm(span=14, adjust=False).mean()
    d["atr10"] = tr.ewm(span=10, adjust=False).mean()

    bb_mid      = c.rolling(20).mean()
    bb_std      = c.rolling(20).std()
    d["bb_up"]  = bb_mid + 2 * bb_std
    d["bb_lo"]  = bb_mid - 2 * bb_std

    kc_mid      = c.ewm(span=20, adjust=False).mean()
    d["kc_up"]  = kc_mid + 1.5 * d["atr14"]
    d["kc_lo"]  = kc_mid - 1.5 * d["atr14"]

    dm_p  = ((d["High"] - d["High"].shift()).clip(lower=0)).where(
             (d["High"] - d["High"].shift()) > (d["Low"].shift() - d["Low"]), 0)
    dm_m  = ((d["Low"].shift() - d["Low"]).clip(lower=0)).where(
             (d["Low"].shift() - d["Low"]) > (d["High"] - d["High"].shift()), 0)
    di_p  = 100 * dm_p.ewm(span=14, adjust=False).mean() / (d["atr14"] + 1e-9)
    di_m  = 100 * dm_m.ewm(span=14, adjust=False).mean() / (d["atr14"] + 1e-9)
    dx    = 100 * (di_p - di_m).abs() / (di_p + di_m + 1e-9)
    d["adx"] = dx.ewm(span=14, adjust=False).mean()

    # SuperTrend (factor=3, period=10)
    hl2     = (d["High"] + d["Low"]) / 2
    upper   = (hl2 + 3 * d["atr10"]).values.copy()
    lower   = (hl2 - 3 * d["atr10"]).values.copy()
    cv      = c.values.copy()
    st_dir  = np.zeros(len(d), dtype=int)
    for i in range(1, len(d)):
        upper[i] = min(upper[i], upper[i-1]) if cv[i-1] <= upper[i-1] else upper[i]
        lower[i] = max(lower[i], lower[i-1]) if cv[i-1] >= lower[i-1] else lower[i]
        pd_ = st_dir[i-1]
        st_dir[i] = (1 if cv[i] > upper[i] else -1) if pd_ == -1 else \
                    (-1 if cv[i] < lower[i] else 1)
    d["st_dir"] = st_dir

    # Squeeze momentum proxy
    mid12        = (d["High"].rolling(12).max() + d["Low"].rolling(12).min()) / 2
    d["sqz_mom"] = c - (mid12 + c.rolling(12).mean()) / 2
    d["squeeze"] = (d["bb_up"] < d["kc_up"]) & (d["bb_lo"] > d["kc_lo"])

    # Session mask (UTC hours 08–22 = London + NY)
    d["session"] = (d.index.hour >= SESSION_UTC_START) & (d.index.hour < SESSION_UTC_END)

    return d.dropna()


# ── SIGNAL GENERATORS ─────────────────────────────────────────────────────────
def _sig(d, up, dn):
    s = pd.Series(0, index=d.index)
    s[up] = 1
    s[dn] = -1
    return s

def sig_ema_21_50_200(d):
    bull = (d["ema21"] > d["ema50"]) & (d["ema50"] > d["ema200"])
    bear = (d["ema21"] < d["ema50"]) & (d["ema50"] < d["ema200"])
    return _sig(d, bull & ~bull.shift().fillna(False),
                   bear & ~bear.shift().fillna(False))

def sig_atr_breakout(d):
    pc = d["Close"].shift()
    return _sig(d,
        d["Close"] > pc + d["atr14"],
        d["Close"] < pc - d["atr14"])

def sig_ema_5_13_50(d):
    bull = (d["ema5"] > d["ema13"]) & (d["ema13"] > d["ema50"])
    bear = (d["ema5"] < d["ema13"]) & (d["ema13"] < d["ema50"])
    return _sig(d, bull & ~bull.shift().fillna(False),
                   bear & ~bear.shift().fillna(False))

def sig_rsi_macd_ema(d):
    lc = (d["rsi14"] > 50) & (d["macd_hist"] > 0) & (d["Close"] > d["ema100"])
    sc = (d["rsi14"] < 50) & (d["macd_hist"] < 0) & (d["Close"] < d["ema100"])
    return _sig(d, lc & ~lc.shift().fillna(False),
                   sc & ~sc.shift().fillna(False))

def sig_supertrend(d):
    bull = d["st_dir"] == 1
    bear = d["st_dir"] == -1
    return _sig(d, bull & ~bull.shift().fillna(False),
                   bear & ~bear.shift().fillna(False))

# ── NEW COMBINATION STRATEGIES ────────────────────────────────────────────────
def sig_ema_atr_combo(d):
    """EMA_21_50_200 trend filter + ATR breakout entry = fewer signals, higher WR."""
    trend_bull = (d["ema21"] > d["ema50"]) & (d["ema50"] > d["ema200"])
    trend_bear = (d["ema21"] < d["ema50"]) & (d["ema50"] < d["ema200"])
    pc         = d["Close"].shift()
    atr_bull   = d["Close"] > pc + d["atr14"]
    atr_bear   = d["Close"] < pc - d["atr14"]
    up = trend_bull & atr_bull
    dn = trend_bear & atr_bear
    # edge: on transition only (avoid re-entries while already in same-direction trend)
    return _sig(d, up & ~up.shift().fillna(False),
                   dn & ~dn.shift().fillna(False))

def sig_triple_filter(d):
    """EMA200 direction + MACD histogram + RSI momentum + ADX>20 strength."""
    lc  = (d["Close"] > d["ema200"]) & (d["macd_hist"] > 0) & \
          (d["rsi14"] > 52) & (d["adx"] > 20)
    sc  = (d["Close"] < d["ema200"]) & (d["macd_hist"] < 0) & \
          (d["rsi14"] < 48) & (d["adx"] > 20)
    return _sig(d, lc & ~lc.shift().fillna(False),
                   sc & ~sc.shift().fillna(False))

STRATEGIES = {
    "EMA_21_50_200":  sig_ema_21_50_200,
    "ATR_Breakout":   sig_atr_breakout,
    "EMA_5_13_50":    sig_ema_5_13_50,
    "RSI_MACD_EMA":   sig_rsi_macd_ema,
    "SuperTrend":     sig_supertrend,
    "EMA_ATR_Combo":  sig_ema_atr_combo,   # NEW
    "Triple_Filter":  sig_triple_filter,   # NEW
}


# ── LOT CALCULATOR ────────────────────────────────────────────────────────────
def calc_lots(capital, price, sl_pct, risk_pct):
    risk_usd       = capital * risk_pct
    sl_usd_per_lot = price * sl_pct
    lots           = max(MIN_LOT, round(risk_usd / sl_usd_per_lot / LOT_STEP) * LOT_STEP)
    lots           = round(lots, 2)
    if lots * price / LEVERAGE > capital:
        lots = max(MIN_LOT, round(capital * LEVERAGE / price / LOT_STEP) * LOT_STEP)
        lots = round(lots, 2)
    return lots


# ── BACKTEST ENGINE ───────────────────────────────────────────────────────────
def backtest(df, signals, sl_pct, rr, risk_pct, session_filter=True):
    cap        = START_CAPITAL
    pos        = None
    entry_px   = sl_p = tp_p = lots = 0.0
    tp_pct     = sl_pct * rr
    peak       = cap
    trades     = []
    milestones_hit = {}

    for i in range(1, len(df)):
        if cap <= 0:
            break
        price   = float(df["Close"].iloc[i])
        ts      = df.index[i]
        sig     = int(signals.iloc[i])
        in_sess = bool(df["session"].iloc[i])

        # ── milestone check ──────────────────────────────────────────────────
        for m in MILESTONES:
            if m not in milestones_hit and cap >= m:
                milestones_hit[m] = {"trade": len(trades), "time": ts, "capital": cap}

        # ── open position management ─────────────────────────────────────────
        if pos is not None:
            hit_tp = (pos == 1 and price >= tp_p) or (pos == -1 and price <= tp_p)
            hit_sl = (pos == 1 and price <= sl_p) or (pos == -1 and price >= sl_p)
            if hit_tp or hit_sl:
                exit_px = tp_p if hit_tp else sl_p
                pnl     = lots * abs(exit_px - entry_px) * (1 if hit_tp else -1)
                cap     = max(0.0, cap + pnl)
                peak    = max(peak, cap)
                trades.append({
                    "ts":      ts,
                    "result":  "TP" if hit_tp else "SL",
                    "pnl":     pnl,
                    "pnl_pct": pnl / (cap - pnl + 1e-9) * 100,
                    "capital": cap,
                    "peak":    peak,
                    "lots":    lots,
                })
                pos = None

        # ── new entry ────────────────────────────────────────────────────────
        if pos is None and sig != 0 and cap > 0:
            if session_filter and not in_sess:
                continue  # skip trades outside London/NY

            # drawdown circuit-breaker
            dd_frac     = (cap - peak) / peak
            eff_risk    = risk_pct
            if dd_frac <= DD_QUARTER:
                eff_risk = risk_pct * 0.25
            elif dd_frac <= DD_HALF:
                eff_risk = risk_pct * 0.50

            lots     = calc_lots(cap, price, sl_pct, eff_risk)
            entry_px = price * (1 + SPREAD) if sig == 1 else price * (1 - SPREAD)
            sl_p     = entry_px * (1 - sl_pct) if sig == 1 else entry_px * (1 + sl_pct)
            tp_p     = entry_px * (1 + tp_pct) if sig == 1 else entry_px * (1 - tp_pct)
            pos      = sig

    return pd.DataFrame(trades), milestones_hit


# ── STATISTICS ────────────────────────────────────────────────────────────────
def calc_stats(trades_df, milestones_hit, risk_pct, sl_pct, rr):
    t = trades_df
    if t.empty or len(t) < 10:
        return None

    wins       = (t["result"] == "TP").sum()
    win_rate   = wins / len(t)
    final_cap  = t["capital"].iloc[-1]
    total_ret  = (final_cap - START_CAPITAL) / START_CAPITAL * 100

    caps   = pd.concat([pd.Series([START_CAPITAL]), t["capital"].reset_index(drop=True)])
    peak   = caps.cummax()
    dd_pct = (caps - peak) / peak * 100
    mdd    = dd_pct.min()

    # Kelly fraction: f* = WR - (1-WR)/RR
    kelly = win_rate - (1 - win_rate) / rr

    # Approximate trades-to-$1M under this compound rate
    avg_pct_per_trade = (t["pnl"] / (t["capital"] - t["pnl"].clip(lower=0) + 1e-9)).mean() * 100
    trades_to_1M      = None
    if avg_pct_per_trade > 0:
        import math
        trades_to_1M = int(math.log(1_000_000 / START_CAPITAL) / math.log(1 + avg_pct_per_trade / 100)) + 1

    return {
        "trades":         len(t),
        "win_rate":       round(win_rate * 100, 1),
        "total_ret%":     round(total_ret, 1),
        "final_cap":      round(final_cap, 2),
        "mdd%":           round(mdd, 1),
        "kelly":          round(kelly * 100, 1),
        "trades_to_1M":   trades_to_1M,
        "avg_pct":        round(avg_pct_per_trade, 4),
        "milestones":     milestones_hit,
        "cap_series":     t["capital"].values,
    }


# ── MONTE CARLO ───────────────────────────────────────────────────────────────
def monte_carlo(trades_df, risk_pct, sl_pct, rr, n_paths=MC_PATHS, n_trades=None):
    t = trades_df
    if t.empty:
        return None

    wins     = (t["result"] == "TP").sum()
    win_rate = wins / len(t)
    tp_ret   = risk_pct * rr   # fractional gain on capital per TP
    sl_ret   = risk_pct        # fractional loss on capital per SL

    if n_trades is None:
        n_trades = len(t)

    rng     = np.random.default_rng(42)
    results = np.zeros((n_paths, n_trades + 1))
    results[:, 0] = START_CAPITAL

    for p in range(n_paths):
        cap = START_CAPITAL
        for k in range(1, n_trades + 1):
            if rng.random() < win_rate:
                cap *= (1 + tp_ret)
            else:
                cap *= (1 - sl_ret)
            results[p, k] = max(0, cap)

    return results


# ── MAIN RUN ──────────────────────────────────────────────────────────────────
def main():
    print("Loading data…")
    df_raw = load_data()
    print(f"  {df_raw.index[0].strftime('%Y-%m-%d')} → {df_raw.index[-1].strftime('%Y-%m-%d')} "
          f"| {len(df_raw):,} candles")

    print("Building indicators…")
    df = build_indicators(df_raw)
    print(f"  {len(df):,} candles after dropna\n")

    total_combos = len(STRATEGIES) * len(SL_GRID) * len(RR_GRID) * len(RISK_GRID)
    print(f"Running {len(STRATEGIES)} strategies × {len(SL_GRID)} SL × "
          f"{len(RR_GRID)} RR × {len(RISK_GRID)} risk% = {total_combos} combos…\n")

    all_results = []
    done = 0

    for sname, sfn in STRATEGIES.items():
        sigs = sfn(df)
        for sl in SL_GRID:
            for rr in RR_GRID:
                for rp in RISK_GRID:
                    t_df, milestones = backtest(df, sigs, sl, rr, rp)
                    s = calc_stats(t_df, milestones, rp, sl, rr)
                    if s:
                        s["strategy"] = sname
                        s["sl"]       = sl
                        s["rr"]       = rr
                        s["risk_pct"] = rp
                        s["label"]    = f"{sname}|SL{sl*100:.1f}%|RR{rr}|R{rp*100:.1f}%"
                        s["trades_df"]= t_df
                        all_results.append(s)
                    done += 1
                    if done % 50 == 0:
                        print(f"  [{done:4d}/{total_combos}]…", end="\r")

    print(f"  [{total_combos}/{total_combos}] done.    \n")

    if not all_results:
        print("No valid results.")
        return

    # ── SORT: primary = reached $1M? secondary = speed; tertiary = final cap ──
    def sort_key(r):
        hit_1m  = 1 if 1_000_000 in r["milestones"] else 0
        t_1m    = r["milestones"].get(1_000_000, {}).get("trade", 999_999)
        final   = r["final_cap"]
        mdd_pen = abs(r["mdd%"])
        return (-hit_1m, t_1m, -final, mdd_pen)

    all_results.sort(key=sort_key)

    # ── RESULTS TABLE ─────────────────────────────────────────────────────────
    hdr = (f"{'#':<4} {'STRATEGY':<35} {'SL':>5} {'RR':>4} {'R%':>5} "
           f"{'TRADES':>7} {'WIN%':>6} {'MDD%':>7} {'KELLY':>7} "
           f"{'FINAL$':>12} {'→$1M?':>7} {'TRADE#':>8}")
    print("═" * len(hdr))
    print(f"  $100 COMPOUND BACKTEST — XAUUSD 5m 2yr | Spread 0.025% | "
          f"London+NY session | DD circuit-breaker")
    print("═" * len(hdr))
    print("  " + hdr)
    print("  " + "─" * len(hdr))

    for rank, r in enumerate(all_results[:25], 1):
        hit = r["milestones"].get(1_000_000)
        hit_str  = "YES" if hit else "no"
        trade_n  = str(hit["trade"]) if hit else "—"
        print(
            f"  {rank:<4} {r['label']:<35} "
            f"{r['sl']*100:>4.1f}% {r['rr']:>4.1f} {r['risk_pct']*100:>4.1f}% "
            f"{r['trades']:>7,} {r['win_rate']:>5.1f}% {r['mdd%']:>7.1f}% "
            f"{r['kelly']:>6.1f}% ${r['final_cap']:>11,.2f} "
            f"{hit_str:>7} {trade_n:>8}"
        )
    print("═" * len(hdr))

    # ── MILESTONE BREAKDOWN — TOP 3 ───────────────────────────────────────────
    print("\n\n  MILESTONE BREAKDOWN — TOP 3 STRATEGIES")
    print("═" * 70)
    for r in all_results[:3]:
        print(f"\n  {r['label']}")
        print(f"  Win: {r['win_rate']}%  |  MDD: {r['mdd%']}%  |  "
              f"Kelly: {r['kelly']}%  |  Trades: {r['trades']:,}")
        print(f"  {'Target':>10}  {'Trade #':>9}  {'Date':>22}  {'Capital':>12}")
        print(f"  {'─'*60}")
        for m in MILESTONES:
            if m in r["milestones"]:
                h = r["milestones"][m]
                print(f"  ${m:>9,}  {h['trade']:>9,}  "
                      f"{str(h['time'])[:19]:>22}  ${h['capital']:>11,.2f}")
            else:
                print(f"  ${m:>9,}  {'—':>9}")
        print(f"\n  Final capital: ${r['final_cap']:>12,.2f}")

    # ── KELLY TABLE ───────────────────────────────────────────────────────────
    print("\n\n  KELLY CRITERION — OPTIMAL RISK% PER STRATEGY")
    print("═" * 55)
    print(f"  {'Strategy':<25} {'Win%':>6}  {'RR':>4}  {'Kelly%':>8}  {'½Kelly':>7}")
    print("  " + "─" * 53)
    seen_strats = set()
    for r in all_results:
        key = (r["strategy"], r["sl"], r["rr"])
        if key not in seen_strats:
            seen_strats.add(key)
            half_k = r["kelly"] / 2
            print(f"  {r['strategy']:<25} {r['win_rate']:>5.1f}%  "
                  f"{r['rr']:>4.1f}  {r['kelly']:>7.1f}%  {half_k:>6.1f}%")
        if len(seen_strats) >= 7:
            break
    print("═" * 55)

    # ── MONTE CARLO — TOP 5 ───────────────────────────────────────────────────
    print(f"\n\n  MONTE CARLO ({MC_PATHS} paths) — TOP {TOP_N_MONTEC} COMBOS")
    print("═" * 70)
    mc_data = {}
    for r in all_results[:TOP_N_MONTEC]:
        paths = monte_carlo(r["trades_df"], r["risk_pct"], r["sl"], r["rr"])
        if paths is None:
            continue
        final_vals = paths[:, -1]
        p_1m       = (final_vals >= 1_000_000).mean() * 100
        p_100k     = (final_vals >= 100_000).mean() * 100
        med_final  = np.median(final_vals)
        p5         = np.percentile(final_vals, 5)
        p95        = np.percentile(final_vals, 95)
        mc_data[r["label"]] = paths
        print(f"\n  {r['label']}")
        print(f"  P(≥$100K): {p_100k:5.1f}%  |  P(≥$1M): {p_1m:5.1f}%")
        print(f"  Median: ${med_final:>12,.0f}  |  5th pct: ${p5:>12,.0f}  |  "
              f"95th pct: ${p95:>12,.0f}")
    print("═" * 70)

    # ── PLOTS ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots…")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("$100 → $1,000,000 Compound Analysis — XAUUSD 5m 2yr",
                 fontsize=14, fontweight="bold")
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # Plot 1: Growth curves — top 5
    ax = axes[0, 0]
    ax.set_title("Compound Growth Curves (Top 5)", fontsize=11)
    for i, r in enumerate(all_results[:5]):
        caps = np.concatenate([[START_CAPITAL], r["cap_series"]])
        ax.semilogy(caps, color=colors[i], linewidth=1.2,
                    label=f"#{i+1} {r['strategy']} R{r['risk_pct']*100:.0f}%")
    ax.axhline(1_000_000, color="gold", linestyle="--", linewidth=1.5, label="$1M target")
    ax.axhline(100_000,   color="green", linestyle=":",  linewidth=1.0, label="$100K")
    ax.set_xlabel("Trade #"); ax.set_ylabel("Capital ($, log scale)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Plot 2: Risk% sensitivity — top strategy
    ax = axes[0, 1]
    ax.set_title(f"Risk% Sensitivity — {all_results[0]['strategy']}", fontsize=11)
    best_strat = all_results[0]["strategy"]
    best_sl    = all_results[0]["sl"]
    best_rr    = all_results[0]["rr"]
    risk_finals = {rp: None for rp in RISK_GRID}
    for r in all_results:
        if r["strategy"] == best_strat and r["sl"] == best_sl and r["rr"] == best_rr:
            risk_finals[r["risk_pct"]] = r["final_cap"]
    rps = [rp * 100 for rp in RISK_GRID if risk_finals[rp] is not None]
    fcs = [risk_finals[rp] for rp in RISK_GRID if risk_finals[rp] is not None]
    bars = ax.bar(rps, fcs, color="#1f77b4", alpha=0.8, width=0.35)
    ax.axhline(1_000_000, color="gold", linestyle="--", linewidth=1.5, label="$1M")
    ax.set_xlabel("Risk % per trade"); ax.set_ylabel("Final Capital ($)")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    # Plot 3: Monte Carlo fan — best strategy
    ax = axes[1, 0]
    ax.set_title(f"Monte Carlo Fan — {all_results[0]['label']}", fontsize=11)
    if mc_data:
        key  = list(mc_data.keys())[0]
        pths = mc_data[key]
        pcts = [5, 25, 50, 75, 95]
        q    = np.percentile(pths, pcts, axis=0)
        xs   = np.arange(pths.shape[1])
        ax.fill_between(xs, q[0], q[4], alpha=0.15, color=colors[0], label="5–95%")
        ax.fill_between(xs, q[1], q[3], alpha=0.30, color=colors[0], label="25–75%")
        ax.semilogy(xs, q[2], color=colors[0], linewidth=2, label="Median")
        ax.axhline(1_000_000, color="gold",  linestyle="--", linewidth=1.5, label="$1M")
        ax.axhline(100_000,   color="green", linestyle=":",  linewidth=1.0, label="$100K")
        ax.set_xlabel("Trade #"); ax.set_ylabel("Capital ($, log scale)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Plot 4: Win rate vs Final capital scatter
    ax = axes[1, 1]
    ax.set_title("Win Rate vs Final Capital (all combos)", fontsize=11)
    wrs    = [r["win_rate"] for r in all_results]
    finals = [r["final_cap"] for r in all_results]
    mdds   = [abs(r["mdd%"]) for r in all_results]
    sc = ax.scatter(wrs, finals, c=mdds, cmap="RdYlGn_r", alpha=0.6,
                    s=20, vmin=0, vmax=60)
    plt.colorbar(sc, ax=ax, label="Max Drawdown %")
    ax.axhline(1_000_000, color="gold", linestyle="--", linewidth=1.5, label="$1M")
    ax.set_xlabel("Win Rate (%)"); ax.set_ylabel("Final Capital ($)")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = os.path.join(os.path.dirname(__file__), "compound_to_1M.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_png}")

    # ── WINNER SUMMARY ────────────────────────────────────────────────────────
    w = all_results[0]
    print("\n" + "═" * 70)
    print("  WINNER CONFIGURATION")
    print("═" * 70)
    print(f"  Strategy : {w['strategy']}")
    print(f"  SL       : {w['sl']*100:.1f}%   RR: {w['rr']}   Risk/trade: {w['risk_pct']*100:.1f}%")
    print(f"  Win rate : {w['win_rate']}%   Kelly: {w['kelly']}%   (½Kelly = {w['kelly']/2:.1f}%)")
    print(f"  Trades   : {w['trades']:,}   MDD: {w['mdd%']}%")
    print(f"  Final    : ${w['final_cap']:>15,.2f}  (from $100)")
    hit1m = w["milestones"].get(1_000_000)
    if hit1m:
        print(f"  $1M hit  : trade #{hit1m['trade']:,} on {str(hit1m['time'])[:19]}")
    else:
        print(f"  $1M hit  : not reached — increase risk% or select higher-Kelly strategy")
    print("═" * 70)
    print()


if __name__ == "__main__":
    main()
