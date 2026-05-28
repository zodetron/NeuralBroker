"""
strategy/signals_d.py — Strategy D: Manipulation Fade (ICT Fib 2.5/4.0)

Concept:
  A big-body candle near a 20-bar extreme is the setup ("manipulation candle").
  Fib levels are LOCKED at setup detection time using the candle's full range:

    origin = candle high  (bull fade)  / candle low   (bear fade)
    sweep  = candle low   (bull fade)  / candle high  (bear fade)
    leg    = origin − sweep   (always positive)

    fib_2_5 = origin − 2.5 × leg   (bull fade: below sweep)
    fib_4_0 = origin − 4.0 × leg   (bull fade: invalidation)
    TP      = origin               (full reversal to setup-candle origin)

  Signal fires when price in a subsequent bar touches fib_2_5 AND that
  bar shows a same-bar displacement confirmation (pin-bar body in the
  reversal direction, body ≥ 0.5× avg body).

Zero-lookahead: setup detected at bar t_setup (big candle at key level).
Fib levels fixed immediately. Signal fires at bar t ≥ t_setup + 1 when
fib_2_5 is touched with displacement. Entry at open[t+1].
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import STRAT_D

_AVG_BODY_WINDOW = 20


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=window - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=window - 1, adjust=False).mean()
    return 100 - 100 / (1 + gain / (loss + 1e-9))


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate Manipulation Fade signals.

    Parameters
    ----------
    df : classified 15M DataFrame with columns:
         open, high, low, close, volume, atr_14, atr_avg,
         market_state, trend_1h, vwap (optional — used for confluence).
    """
    big_body_mult    = STRAT_D["big_body_mult"]       # 1.2
    key_level_pct    = STRAT_D["key_level_pct"]       # 0.015 = 1.5%
    atr_spike_thresh = STRAT_D["atr_spike_thresh"]    # 2.5
    atr_news_thresh  = STRAT_D["atr_news_thresh"]     # 3.0
    fib_entry_mult   = STRAT_D["fib_entry"]           # 2.5
    fib_stop_mult    = STRAT_D["fib_stop"]            # 4.0
    disp_body_mult   = STRAT_D["disp_body_mult"]      # 0.5
    stop_buf_atr     = STRAT_D["stop_buffer_atr"]     # 0.1
    min_rr           = STRAT_D["min_rr"]              # 1.5
    stale_bars       = STRAT_D["staleness_bars"]      # 20
    conf_roll50_pct  = STRAT_D["conf_roll50_pct"]     # 0.003
    conf_vwap_dev    = STRAT_D["conf_vwap_dev"]       # 0.004
    conf_rsi_bull    = STRAT_D["conf_rsi_bull"]       # 35
    conf_rsi_bear    = STRAT_D["conf_rsi_bear"]       # 65

    d = df.copy()

    body       = (d["close"] - d["open"]).abs()
    avg_body   = body.rolling(_AVG_BODY_WINDOW).mean()
    atr_ratio  = d["atr_14"] / (d["atr_avg"] + 1e-9)

    roll20_high = d["high"].shift(1).rolling(20).max()
    roll20_low  = d["low"].shift(1).rolling(20).min()
    roll50_high = d["high"].rolling(50).max()
    roll50_low  = d["low"].rolling(50).min()

    rsi14    = _rsi(d["close"], 14)
    has_vwap = "vwap" in d.columns
    vwap_s   = d["vwap"] if has_vwap else pd.Series(np.nan, index=d.index)

    opens      = d["open"].values
    highs      = d["high"].values
    lows       = d["low"].values
    closes     = d["close"].values
    atrs       = d["atr_14"].values
    avg_bodies = avg_body.values
    atr_ratios = atr_ratio.values
    roll20h    = roll20_high.values
    roll20l    = roll20_low.values
    roll50h    = roll50_high.values
    roll50l    = roll50_low.values
    rsi_arr    = rsi14.values
    vwap_arr   = vwap_s.values
    trend_arr  = d["trend_1h"].values if "trend_1h" in d.columns else np.zeros(len(d))

    n = len(d)

    # ONE active setup per direction. New setup candle replaces old one.
    bull_setup = None   # expect LONG fade
    bear_setup = None   # expect SHORT fade

    rows = []

    for i in range(_AVG_BODY_WINDOW + 2, n - 1):
        avg_b = avg_bodies[i]
        atr_r = atr_ratios[i]

        # ── ATR spike: invalidate setups ──────────────────────────────────────
        if atr_r >= atr_spike_thresh:
            bull_setup = None
            bear_setup = None
            continue

        # ── Expire stale setups ───────────────────────────────────────────────
        if bull_setup is not None and (i - bull_setup["bar"]) > stale_bars:
            bull_setup = None
        if bear_setup is not None and (i - bear_setup["bar"]) > stale_bars:
            bear_setup = None

        # ── Recent news spike guard ───────────────────────────────────────────
        if any(atr_ratios[max(0, i - 3):i + 1] >= atr_news_thresh):
            continue

        # ── SIGNAL CHECK ──────────────────────────────────────────────────────
        kb     = abs(closes[i] - opens[i])
        kb_avg = avg_bodies[i] if avg_bodies[i] > 0 else 1.0

        if bull_setup is not None:
            fib_2_5 = bull_setup["fib_2_5"]
            fib_4_0 = bull_setup["fib_4_0"]
            if lows[i] <= fib_2_5 and lows[i] > fib_4_0:
                if closes[i] > opens[i] and kb >= disp_body_mult * kb_avg:
                    atr_cur  = atrs[i]
                    sl_price = fib_4_0 - stop_buf_atr * atr_cur
                    tp_price = bull_setup["origin"]
                    dist_tp  = tp_price - fib_2_5
                    dist_sl  = fib_2_5 - sl_price
                    rr       = dist_tp / max(dist_sl, 1e-9)

                    if rr >= min_rr:
                        conf_score = 0
                        r50l = roll50l[i]
                        if not np.isnan(r50l):
                            if abs(lows[i] - r50l) / (r50l + 1e-9) <= conf_roll50_pct:
                                conf_score += 1
                        vwap_v = vwap_arr[i]
                        if not np.isnan(vwap_v):
                            if abs(closes[i] - vwap_v) / (vwap_v + 1e-9) <= conf_vwap_dev:
                                conf_score += 1
                        if rsi_arr[i] <= conf_rsi_bull:
                            conf_score += 1

                        rows.append({
                            "ts":              d.index[i],
                            "direction":       1,
                            "origin":          bull_setup["origin"],
                            "sweep":           bull_setup["sweep"],
                            "fib_2_5":         fib_2_5,
                            "fib_4_0":         fib_4_0,
                            "tp_price":        tp_price,
                            "sl_price":        sl_price,
                            "rr":              rr,
                            "close":           closes[i],
                            "atr_14":          atrs[i],
                            "rsi14":           rsi_arr[i],
                            "vwap_dev_pct":    abs(closes[i] - vwap_arr[i]) / (vwap_arr[i] + 1e-9) * 100 if not np.isnan(vwap_arr[i]) else np.nan,
                            "conf_score":      conf_score,
                            "high_confluence": int(conf_score >= 2),
                            "setup_age_bars":  i - bull_setup["bar"],
                            "hour":            d.index[i].hour,
                            "trend_1h":        int(trend_arr[i]),
                        })
                        bull_setup = None   # consumed

        if bear_setup is not None:
            fib_2_5 = bear_setup["fib_2_5"]
            fib_4_0 = bear_setup["fib_4_0"]
            if highs[i] >= fib_2_5 and highs[i] < fib_4_0:
                if closes[i] < opens[i] and kb >= disp_body_mult * kb_avg:
                    atr_cur  = atrs[i]
                    sl_price = fib_4_0 + stop_buf_atr * atr_cur
                    tp_price = bear_setup["origin"]
                    dist_tp  = fib_2_5 - tp_price
                    dist_sl  = sl_price - fib_2_5
                    rr       = dist_tp / max(dist_sl, 1e-9)

                    if rr >= min_rr:
                        conf_score = 0
                        r50h = roll50h[i]
                        if not np.isnan(r50h):
                            if abs(highs[i] - r50h) / (r50h + 1e-9) <= conf_roll50_pct:
                                conf_score += 1
                        vwap_v = vwap_arr[i]
                        if not np.isnan(vwap_v):
                            if abs(closes[i] - vwap_v) / (vwap_v + 1e-9) <= conf_vwap_dev:
                                conf_score += 1
                        if rsi_arr[i] >= conf_rsi_bear:
                            conf_score += 1

                        rows.append({
                            "ts":              d.index[i],
                            "direction":       -1,
                            "origin":          bear_setup["origin"],
                            "sweep":           bear_setup["sweep"],
                            "fib_2_5":         fib_2_5,
                            "fib_4_0":         fib_4_0,
                            "tp_price":        tp_price,
                            "sl_price":        sl_price,
                            "rr":              rr,
                            "close":           closes[i],
                            "atr_14":          atrs[i],
                            "rsi14":           rsi_arr[i],
                            "vwap_dev_pct":    abs(closes[i] - vwap_arr[i]) / (vwap_arr[i] + 1e-9) * 100 if not np.isnan(vwap_arr[i]) else np.nan,
                            "conf_score":      conf_score,
                            "high_confluence": int(conf_score >= 2),
                            "setup_age_bars":  i - bear_setup["bar"],
                            "hour":            d.index[i].hour,
                            "trend_1h":        int(trend_arr[i]),
                        })
                        bear_setup = None   # consumed

        # ── SETUP CANDLE DETECTION (new setup replaces old one) ───────────────
        cur_body = abs(closes[i] - opens[i])
        is_big   = (avg_b > 0) and (cur_body >= big_body_mult * avg_b)
        if not is_big:
            continue

        is_bull_candle = closes[i] > opens[i]
        is_bear_candle = closes[i] < opens[i]
        r20h = roll20h[i]
        r20l = roll20l[i]

        # Bull fade: big BEARISH candle near 20-bar low
        if is_bear_candle and not np.isnan(r20l):
            dist_pct = abs(lows[i] - r20l) / (r20l + 1e-9)
            if dist_pct <= key_level_pct:
                origin = highs[i]
                sweep  = lows[i]
                leg    = origin - sweep
                if leg > 0:
                    bull_setup = {
                        "origin":  origin,
                        "sweep":   sweep,
                        "fib_2_5": origin - fib_entry_mult * leg,
                        "fib_4_0": origin - fib_stop_mult  * leg,
                        "bar":     i,
                    }

        # Bear fade: big BULLISH candle near 20-bar high
        if is_bull_candle and not np.isnan(r20h):
            dist_pct = abs(highs[i] - r20h) / (r20h + 1e-9)
            if dist_pct <= key_level_pct:
                origin = lows[i]
                sweep  = highs[i]
                leg    = sweep - origin
                if leg > 0:
                    bear_setup = {
                        "origin":  origin,
                        "sweep":   sweep,
                        "fib_2_5": origin + fib_entry_mult * leg,
                        "fib_4_0": origin + fib_stop_mult  * leg,
                        "bar":     i,
                    }

    if not rows:
        return pd.DataFrame()

    sig_df = pd.DataFrame(rows).set_index("ts")
    sig_df.index.name = "ts"
    sig_df.insert(0, "strategy",  "D")
    sig_df.insert(0, "signal_id", [f"D{num:04d}" for num in range(1, len(sig_df) + 1)])
    return sig_df


def signal_stats(sig: pd.DataFrame, df: pd.DataFrame) -> None:
    if sig.empty:
        print("  Strategy D: no signals generated.")
        return

    n      = len(sig)
    n_long = (sig["direction"] == 1).sum()
    n_days = (df.index[-1] - df.index[0]).days
    avg_pd = n / n_days

    print(f"\n  {'─'*56}")
    print(f"  Strategy D — Manipulation Fade (ICT Fib 2.5/4.0)")
    print(f"  {'─'*56}")
    print(f"  Total signals      : {n:>7,}")
    print(f"  Long / Short       : {n_long:>7,} / {n - n_long:,}")
    print(f"  Avg per day        : {avg_pd:>7.2f}")
    print(f"  Avg RR             : {sig['rr'].mean():>7.2f}")
    print(f"  High confluence    : {sig['high_confluence'].sum():>7,}  "
          f"({sig['high_confluence'].mean()*100:.1f}%)")
    print(f"  Avg setup age bars : {sig['setup_age_bars'].mean():>7.1f}")
    print(f"  Avg RSI14          : {sig['rsi14'].mean():>7.2f}")

    hour_counts = sig["hour"].value_counts().sort_values(ascending=False)
    print(f"\n  Top 5 hours (UTC):")
    for hr, cnt in hour_counts.head(5).items():
        bar_chart = "█" * int(cnt / hour_counts.iloc[0] * 20)
        print(f"    {hr:02d}:00  {cnt:>5,}  {bar_chart}")
