"""
ATR Breakout — Parameter Sweep  |  XAU_1m_data.csv (2004–2025)
Sweeps ATR multiplier, EMA trend filter, SL%, RR to find profitable configs.
Capital: $100 | Leverage: 1:1000 | Min Lot: 0.01 | Spread: 0.025%
"""

import pandas as pd
import numpy as np
import warnings, os, tempfile, shutil, time
from itertools import product
from multiprocessing import Pool, cpu_count
warnings.filterwarnings("ignore")

CSV_FILE = "XAU_1m_data.csv"

CAPITAL  = 100.0
RISK_PCT = 0.01
SPREAD   = 0.00025
LEVERAGE = 1000
MIN_LOT  = 0.01
LOT_STEP = 0.01

ATR_MULTS   = [1.5, 2.0, 2.5, 3.0]
EMA_FILTERS = [0, 50, 200]       # 0 = no filter
SL_PCTS     = [0.003, 0.005, 0.008]
RRS         = [2.0, 2.5, 3.0]

# ── MODULE-LEVEL GLOBALS (set by worker_init in spawned child processes) ──────
_g_close      = None
_g_prev_close = None
_g_atr        = None
_g_ema50      = None
_g_ema200     = None
_g_n          = 0


def _worker_init(data_dir):
    global _g_close, _g_prev_close, _g_atr, _g_ema50, _g_ema200, _g_n
    _g_close      = np.load(os.path.join(data_dir, 'close.npy'),      mmap_mode='r')
    _g_prev_close = np.load(os.path.join(data_dir, 'prev_close.npy'), mmap_mode='r')
    _g_atr        = np.load(os.path.join(data_dir, 'atr.npy'),        mmap_mode='r')
    _g_ema50      = np.load(os.path.join(data_dir, 'ema50.npy'),       mmap_mode='r')
    _g_ema200     = np.load(os.path.join(data_dir, 'ema200.npy'),      mmap_mode='r')
    _g_n          = len(_g_close)


def _run_combo(args):
    atr_mult, ema_filt, sl_pct, rr = args
    close      = _g_close
    prev_close = _g_prev_close
    atr        = _g_atr
    n          = _g_n

    # Build signal (vectorized)
    sig = np.zeros(n, dtype=np.int8)
    lc = close > prev_close + atr_mult * atr
    sc = close < prev_close - atr_mult * atr
    if ema_filt == 50:
        lc &= close > _g_ema50
        sc &= close < _g_ema50
    elif ema_filt == 200:
        lc &= close > _g_ema200
        sc &= close < _g_ema200
    sig[lc] =  1
    sig[sc] = -1

    # Sequential backtest
    tp_pct = sl_pct * rr
    cap = CAPITAL
    pos = 0
    entry_fill = sl_p = tp_p = lots = 0.0
    peak = cap
    wins = losses = 0
    min_ratio = 1.0

    for i in range(1, n):
        if cap <= 0:
            break
        price = float(close[i])
        s     = int(sig[i])

        if pos:
            if pos == 1:
                hit_tp = price >= tp_p
                hit_sl = price <= sl_p
            else:
                hit_tp = price <= tp_p
                hit_sl = price >= sl_p
            if hit_tp or hit_sl:
                exit_px = tp_p if hit_tp else sl_p
                pnl = lots * abs(exit_px - entry_fill) * (1 if hit_tp else -1)
                cap += pnl
                if cap < 0:
                    cap = 0.0
                if cap > peak:
                    peak = cap
                r = cap / peak if peak > 0 else 1.0
                if r < min_ratio:
                    min_ratio = r
                wins   += hit_tp
                losses += (not hit_tp)
                pos = 0

        if not pos and s and cap > 0:
            risk   = cap * RISK_PCT
            sl_amt = price * sl_pct
            lots   = max(MIN_LOT, round((risk / sl_amt) / LOT_STEP) * LOT_STEP)
            lots   = round(lots, 2)
            if lots * price / LEVERAGE > cap:
                lots = max(MIN_LOT, round((cap * LEVERAGE / price) / LOT_STEP) * LOT_STEP)
                lots = round(lots, 2)
            if s == 1:
                entry_fill = price * (1 + SPREAD)
                sl_p = entry_fill * (1 - sl_pct)
                tp_p = entry_fill * (1 + tp_pct)
            else:
                entry_fill = price * (1 - SPREAD)
                sl_p = entry_fill * (1 + sl_pct)
                tp_p = entry_fill * (1 - tp_pct)
            pos = s

    total      = wins + losses
    filt_label = f"EMA{ema_filt}" if ema_filt else "NoFilt"
    return {
        "label":    f"ATR×{atr_mult:.1f}|{filt_label}|SL{sl_pct*100:.1f}%|RR{rr}",
        "atr_mult": atr_mult,
        "ema_filt": ema_filt,
        "sl_pct":   sl_pct,
        "rr":       rr,
        "trades":   total,
        "wr":       wins / total * 100 if total else 0,
        "final":    round(cap, 2),
        "ret%":     round((cap - CAPITAL) / CAPITAL * 100, 2),
        "mdd%":     round((min_ratio - 1) * 100, 2),
    }


if __name__ == "__main__":

    # ── LOAD ──────────────────────────────────────────────────────────────────
    print(f"Loading {CSV_FILE}...", end=" ", flush=True)
    df_raw = pd.read_csv(
        CSV_FILE, sep=";", parse_dates=["Date"], date_format="%Y.%m.%d %H:%M",
    )
    df_raw.rename(columns={"Date": "open_time"}, inplace=True)
    df_raw = df_raw[["open_time", "Open", "High", "Low", "Close", "Volume"]]
    for col in ["Open", "High", "Low", "Close"]:
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
    df_raw.dropna(inplace=True)
    df_raw = df_raw[~df_raw["open_time"].duplicated(keep="first")]
    df_raw.sort_values("open_time", inplace=True)
    df_raw.set_index("open_time", inplace=True)
    print(f"{len(df_raw):,} candles | "
          f"{df_raw.index[0].strftime('%Y-%m-%d')} → {df_raw.index[-1].strftime('%Y-%m-%d')}")

    # ── INDICATORS ────────────────────────────────────────────────────────────
    print("Building indicators...", end=" ", flush=True)
    c = df_raw["Close"]
    h = df_raw["High"]
    l = df_raw["Low"]

    high_low = h - l
    high_pc  = (h - c.shift()).abs()
    low_pc   = (l - c.shift()).abs()
    tr = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)

    atr_arr   = tr.ewm(span=14,  adjust=False).mean().values
    ema50_arr = c.ewm(span=50,   adjust=False).mean().values
    ema200_arr= c.ewm(span=200,  adjust=False).mean().values
    close_arr = c.values
    prev_arr  = np.roll(close_arr, 1)
    prev_arr[0] = close_arr[0]
    timestamps  = df_raw.index
    N = len(close_arr)
    print(f"done. {N:,} candles.")

    # ── SAVE ARRAYS FOR WORKER PROCESSES ──────────────────────────────────────
    tmpdir = tempfile.mkdtemp(prefix="atr_sweep_")
    np.save(os.path.join(tmpdir, "close.npy"),      close_arr)
    np.save(os.path.join(tmpdir, "prev_close.npy"), prev_arr)
    np.save(os.path.join(tmpdir, "atr.npy"),        atr_arr)
    np.save(os.path.join(tmpdir, "ema50.npy"),       ema50_arr)
    np.save(os.path.join(tmpdir, "ema200.npy"),      ema200_arr)

    combos = list(product(ATR_MULTS, EMA_FILTERS, SL_PCTS, RRS))
    n_cpu  = min(cpu_count(), len(combos))

    print(f"\nGrid search: {len(combos)} combos | {n_cpu} CPUs | 2004–2025 data")
    print("(may take a few minutes)\n")
    t0 = time.time()

    try:
        with Pool(n_cpu, initializer=_worker_init, initargs=(tmpdir,)) as pool:
            results = pool.map(_run_combo, combos)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s.\n")

    # ── RESULTS TABLE ─────────────────────────────────────────────────────────
    rdf = pd.DataFrame(results).sort_values("ret%", ascending=False).reset_index(drop=True)
    profitable = rdf[rdf["ret%"] > 0]

    W = 112
    print("═" * W)
    print(f"  GRID SEARCH — ATR Breakout XAUUSD 1m | 2004–2025 | $100 Start | "
          f"Profitable: {len(profitable)}/{len(rdf)}")
    print("═" * W)
    print(f"  {'#':<4} {'CONFIGURATION':<50} {'TRADES':>7} {'WIN%':>6} "
          f"{'RET%':>10} {'MDD%':>8} {'FINAL$':>10}")
    print("═" * W)

    for i, r in rdf.head(20).iterrows():
        mark = "✓" if r["ret%"] > 0 else "✗"
        print(f"  {i+1:<4} {r['label']:<50} {int(r['trades']):>7} {r['wr']:>5.1f}% "
              f"{r['ret%']:>+10.1f}% {r['mdd%']:>7.1f}% ${r['final']:>9.2f} {mark}")

    print("═" * W)

    if profitable.empty:
        print("\n  No profitable config found on 2004–2025 data.")
        print("  Suggestions: add session filter (London/NY only), multi-TF confirmation,")
        print("  or volume filter. The ATR breakout alone may not be edge on 1m XAUUSD long-term.")
        exit()

    # ── DEEP DIVE ON BEST CONFIG ───────────────────────────────────────────────
    best = rdf.iloc[0]
    atr_m  = best["atr_mult"]
    ef     = int(best["ema_filt"])
    sl_pct = best["sl_pct"]
    rr     = best["rr"]
    tp_pct = sl_pct * rr

    filt_label = f"EMA{ef}" if ef else "NoFilter"
    label = (f"ATR×{atr_m:.1f} | {filt_label} | "
             f"SL {sl_pct*100:.1f}% | RR {rr} | TP {tp_pct*100:.2f}%")

    print(f"\n  BEST: {label}  →  {best['ret%']:+.1f}%")
    print("  Running full trade-log backtest for yearly breakdown...\n")

    # Rebuild best signal
    sig = np.zeros(N, dtype=np.int8)
    lc = close_arr > prev_arr + atr_m * atr_arr
    sc = close_arr < prev_arr - atr_m * atr_arr
    if ef == 50:
        lc &= close_arr > ema50_arr
        sc &= close_arr < ema50_arr
    elif ef == 200:
        lc &= close_arr > ema200_arr
        sc &= close_arr < ema200_arr
    sig[lc] =  1
    sig[sc] = -1

    # Full backtest with trade log
    cap = CAPITAL
    pos = 0
    entry_fill = sl_p = tp_p = lots = 0.0
    peak = cap
    trades = []

    for i in range(1, N):
        if cap <= 0:
            break
        price = float(close_arr[i])
        s     = int(sig[i])
        ts    = timestamps[i]

        if pos:
            if pos == 1:
                hit_tp = price >= tp_p
                hit_sl = price <= sl_p
            else:
                hit_tp = price <= tp_p
                hit_sl = price >= sl_p
            if hit_tp or hit_sl:
                exit_px = tp_p if hit_tp else sl_p
                pnl = lots * abs(exit_px - entry_fill) * (1 if hit_tp else -1)
                cap = max(0.0, cap + pnl)
                peak = max(peak, cap)
                trades.append({
                    "exit_time": ts,
                    "result":    "TP" if hit_tp else "SL",
                    "pnl":       round(pnl, 4),
                    "capital":   round(cap, 4),
                    "peak":      round(peak, 4),
                })
                pos = 0

        if not pos and s and cap > 0:
            risk   = cap * RISK_PCT
            sl_amt = price * sl_pct
            lots   = max(MIN_LOT, round((risk / sl_amt) / LOT_STEP) * LOT_STEP)
            lots   = round(lots, 2)
            if lots * price / LEVERAGE > cap:
                lots = max(MIN_LOT, round((cap * LEVERAGE / price) / LOT_STEP) * LOT_STEP)
                lots = round(lots, 2)
            if s == 1:
                entry_fill = price * (1 + SPREAD)
                sl_p = entry_fill * (1 - sl_pct)
                tp_p = entry_fill * (1 + tp_pct)
            else:
                entry_fill = price * (1 - SPREAD)
                sl_p = entry_fill * (1 + sl_pct)
                tp_p = entry_fill * (1 - tp_pct)
            pos = s

    t = pd.DataFrame(trades)
    wins    = (t["result"] == "TP").sum()
    losses  = (t["result"] == "SL").sum()
    wr      = wins / len(t) * 100
    final   = t["capital"].iloc[-1]
    total_r = (final - CAPITAL) / CAPITAL * 100
    caps    = pd.concat([pd.Series([CAPITAL]), t["capital"].reset_index(drop=True)])
    peak_s  = caps.cummax()
    mdd     = ((caps - peak_s) / peak_s * 100).min()
    mdd_usd = (caps - peak_s).min()

    print("═" * 85)
    print(f"  BEST CONFIG: {label}")
    print("═" * 85)
    print(f"  Trades: {len(t):,}  |  Wins: {wins:,}  |  Losses: {losses:,}  |  Win Rate: {wr:.1f}%")
    print(f"  Final Capital: ${final:,.2f}  |  Total Return: {total_r:+.1f}%")
    print(f"  Max Drawdown: {mdd:.1f}%  (${abs(mdd_usd):.2f})")

    # Yearly breakdown
    t["year"] = pd.to_datetime(t["exit_time"]).dt.year
    years = sorted(t["year"].unique())
    prev_cap = CAPITAL
    yr_rows = []
    for y in years:
        yt  = t[t["year"] == y]
        end = yt["capital"].iloc[-1]
        yr_rows.append({
            "year":   y,
            "trades": len(yt),
            "win%":   (yt["result"] == "TP").sum() / len(yt) * 100,
            "ret%":   (end - prev_cap) / prev_cap * 100,
            "capital": end,
        })
        prev_cap = end

    yr_df = pd.DataFrame(yr_rows)
    rets  = yr_df["ret%"]

    print(f"\n  {'Year':<8} {'Trades':>7} {'Win%':>6} {'Return%':>10} {'Capital':>12}  Bar")
    print(f"  {'-'*68}")
    for _, r in yr_df.iterrows():
        ret = r["ret%"]
        bar = ("█" * min(int(abs(ret) / 5), 30)) if ret >= 0 else ("▓" * min(int(abs(ret) / 5), 30))
        sgn = "+" if ret >= 0 else ""
        print(f"  {int(r['year']):<8} {int(r['trades']):>7} {r['win%']:>5.1f}% "
              f"{sgn}{ret:>9.2f}% ${r['capital']:>10.2f}  {bar}")
    print(f"  {'-'*68}")
    print(f"  Avg: {rets.mean():+.2f}%/yr  |  Best: {rets.max():+.2f}%  |  "
          f"Worst: {rets.min():+.2f}%  |  Positive: {(rets>0).sum()}/{len(rets)} years")

    print(f"\n{'═'*85}")
    print(f"  Config   : {label}")
    print(f"  Return   : {total_r:+.1f}%  |  MDD: {mdd:.1f}%  (${abs(mdd_usd):.2f})")
    print(f"  Final    : ${final:,.2f}  from $100  over {len(years)} years")
    print(f"{'═'*85}")
