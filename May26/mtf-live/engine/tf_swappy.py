"""
engine/tf_swappy.py — Swappy ICT Fib 2.5/4.0 manipulation-fade detector.

Logic (15M bars, same as backtest Strategy D):
  1. WATCH:   Detect big-body setup candle near a key level.
              Body > big_body_mult × avg body AND close in top/bottom 30% of range.
              Wait for displacement bar (body > disp_body_mult × avg).
  2. TRACK:   Once displacement confirmed, compute Fibonacci 2.5 and 4.0 extensions
              of the manipulation leg. Wait for price to touch fib_2_5 ± touch_pct.
  3. SIGNAL:  On touch, emit trade signal.  Auto-expire after max_bars.

State is held in SwappyState and updated each 15M bar via update().
Call signal() to retrieve a pending signal (returns None if none).

Output signal dict:
  direction    : +1 (long at 2.5 ext below) / -1 (short at 2.5 ext above)
  entry        : Fib 2.5 extension price
  stop         : Fib 4.0 extension price
  tp           : 50% of distance back to setup origin
  rr           : reward/risk ratio
  confluence   : True if high-confluence conditions met
  setup_ts     : timestamp of the setup candle
  signal_ts    : timestamp when touch triggered
  atr          : ATR at signal time
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import ta
from engine.builder import build_candles
from config import (
    SWAPPY_MIN_RR, SWAPPY_MAX_BARS, SWAPPY_TP_PCT, SWAPPY_TOUCH_PCT,
    SWAPPY_SLIP_GUARD_PCT, SWAPPY_BIG_BODY_MULT, SWAPPY_SUPPORT_PCT,
    SWAPPY_ATR_SPIKE, SWAPPY_DISP_BODY_MULT, SWAPPY_DISP_TOP_PCT,
    SWAPPY_VOL_MULT, SWAPPY_RSI_LONG_MAX, SWAPPY_RSI_SHORT_MIN,
    SWAPPY_CONF_RSI_LONG, SWAPPY_CONF_RSI_SHORT, SWAPPY_CONF_ZONE_PCT,
)


@dataclass
class _Setup:
    """One pending Swappy setup waiting for a Fib 2.5 touch."""
    direction:  int          # +1 bull, -1 bear
    origin:     float        # high (bear) or low (bull) of the setup candle
    extreme:    float        # low (bear) or high (bull) of manipulation leg
    fib_2_5:    float
    fib_4_0:    float
    tp:         float
    setup_bar:  int          # bar index when setup was detected
    setup_ts:   pd.Timestamp
    nearest_4h: float        # nearest 4H key level for confluence check


class SwappyState:
    """
    Persistent state machine updated every 15M bar.
    Keep one instance alive for the lifetime of the live system.
    """

    def __init__(self) -> None:
        self._setup: Optional[_Setup] = None
        self._bar_count: int = 0
        self._pending_signal: Optional[dict] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(self, df15: pd.DataFrame, nearest_4h_level: float = float("nan")) -> None:
        """
        Feed the latest 15M candles DataFrame. Must include at least 30 bars.
        nearest_4h_level: closest 4H support/resistance for confluence scoring.
        """
        if len(df15) < 30:
            return

        self._bar_count += 1
        self._pending_signal = None

        close = df15["close"]
        high  = df15["high"]
        low   = df15["low"]
        open_ = df15["open"]
        vol   = df15["volume"]

        atr_s   = ta.volatility.average_true_range(high, low, close, window=14)
        rsi_s   = ta.momentum.RSIIndicator(close, window=14).rsi()
        avg_atr = float(atr_s.rolling(20).mean().iloc[-1])
        atr_now = float(atr_s.iloc[-1])
        avg_body = float(abs(close - open_).rolling(20).mean().iloc[-1]) or 1.0
        avg_vol  = float(vol.rolling(20).mean().iloc[-1])

        last     = df15.iloc[-1]
        last_c   = float(last["close"])
        last_ts  = df15["timestamp"].iloc[-1]

        # ── Expire stale setup ─────────────────────────────────────────────────
        if self._setup is not None:
            bars_since = self._bar_count - self._setup.setup_bar
            if bars_since > SWAPPY_MAX_BARS:
                self._setup = None

        # ── Check for touch on active setup ───────────────────────────────────
        if self._setup is not None:
            self._check_touch(last, last_ts, atr_now, rsi_s, nearest_4h_level)
            if self._pending_signal is not None:
                self._setup = None   # consumed
                return

        # ── Scan for new setup (only if no active setup) ──────────────────────
        if self._setup is None:
            self._scan_setup(df15, atr_s, avg_atr, avg_body, avg_vol, rsi_s, nearest_4h_level)

    def signal(self) -> Optional[dict]:
        """Return the pending signal (or None). Consumed after one read."""
        sig = self._pending_signal
        self._pending_signal = None
        return sig

    # ── Internal methods ───────────────────────────────────────────────────────

    def _scan_setup(
        self,
        df: pd.DataFrame,
        atr_s: pd.Series,
        avg_atr: float,
        avg_body: float,
        avg_vol: float,
        rsi_s: pd.Series,
        nearest_4h: float,
    ) -> None:
        """Check the last completed bar (index -2) for setup conditions."""
        if len(df) < 4:
            return

        close = df["close"]
        high  = df["high"]
        low   = df["low"]
        open_ = df["open"]
        vol   = df["volume"]

        # Setup candle is index -2; last closed bar is -1 (displacement candidate)
        idx = len(df) - 2

        s_close = float(close.iloc[idx])
        s_open  = float(open_.iloc[idx])
        s_high  = float(high.iloc[idx])
        s_low   = float(low.iloc[idx])
        s_body  = abs(s_close - s_open)
        s_range = s_high - s_low or 1e-8
        s_atr   = float(atr_s.iloc[idx])
        s_vol   = float(vol.iloc[idx])
        s_ts    = df["timestamp"].iloc[idx]

        # ── Setup candle filters ───────────────────────────────────────────────
        if s_body < avg_body * SWAPPY_BIG_BODY_MULT:
            return
        if s_atr > avg_atr * SWAPPY_ATR_SPIKE:
            return   # news spike — skip

        # Direction: bull setup = big DOWN candle (manipulation wick down)
        is_bull_setup = s_close < s_open   # bearish manipulation leg
        is_bear_setup = s_close > s_open   # bullish manipulation leg

        if not (is_bull_setup or is_bear_setup):
            return

        direction = 1 if is_bull_setup else -1

        # Close must be in bottom 30% of range (bull) or top 30% (bear)
        pct_from_bottom = (s_close - s_low) / s_range
        if direction == 1 and pct_from_bottom > SWAPPY_DISP_TOP_PCT:
            return
        if direction == -1 and (1.0 - pct_from_bottom) > SWAPPY_DISP_TOP_PCT:
            return

        # ── Displacement bar (index -1) ────────────────────────────────────────
        d_close = float(close.iloc[-1])
        d_open  = float(open_.iloc[-1])
        d_body  = abs(d_close - d_open)
        d_vol   = float(vol.iloc[-1])

        if d_body < avg_body * SWAPPY_DISP_BODY_MULT:
            return
        if d_vol < avg_vol * SWAPPY_VOL_MULT:
            return

        # Displacement must move away from setup in the correct direction
        if direction == 1 and d_close <= d_open:
            return   # bull fade: displacement should be bullish
        if direction == -1 and d_close >= d_open:
            return   # bear fade: displacement should be bearish

        # ── Compute Fib levels ─────────────────────────────────────────────────
        # Manipulation leg: setup candle low (bull) or high (bear)
        if direction == 1:
            origin  = s_high   # pre-manipulation anchor (top of setup candle)
            extreme = s_low    # tip of manipulation wick
        else:
            origin  = s_low
            extreme = s_high

        leg = abs(origin - extreme)
        if leg < 1e-6:
            return

        if direction == 1:
            fib_2_5 = extreme - leg * (2.5 - 1.0)   # extension below extreme
            fib_4_0 = extreme - leg * (4.0 - 1.0)
            tp      = extreme + leg * SWAPPY_TP_PCT
        else:
            fib_2_5 = extreme + leg * (2.5 - 1.0)
            fib_4_0 = extreme + leg * (4.0 - 1.0)
            tp      = extreme - leg * SWAPPY_TP_PCT

        sl_dist = abs(fib_4_0 - fib_2_5)
        tp_dist = abs(tp - fib_2_5)
        rr = tp_dist / sl_dist if sl_dist > 1e-6 else 0.0

        if rr < SWAPPY_MIN_RR:
            return

        self._setup = _Setup(
            direction  = direction,
            origin     = origin,
            extreme    = extreme,
            fib_2_5    = fib_2_5,
            fib_4_0    = fib_4_0,
            tp         = tp,
            setup_bar  = self._bar_count,
            setup_ts   = s_ts,
            nearest_4h = nearest_4h,
        )

    def _check_touch(
        self,
        last_bar,
        last_ts: pd.Timestamp,
        atr_now: float,
        rsi_s: pd.Series,
        nearest_4h: float,
    ) -> None:
        setup = self._setup
        assert setup is not None

        last_low  = float(last_bar["low"])
        last_high = float(last_bar["high"])
        last_c    = float(last_bar["close"])
        rsi_val   = float(rsi_s.iloc[-1])

        fib = setup.fib_2_5
        tol = fib * SWAPPY_TOUCH_PCT

        # Touch condition: price tag within tolerance
        touched = False
        if setup.direction == 1:
            touched = last_low <= fib + tol
        else:
            touched = last_high >= fib - tol

        if not touched:
            return

        # Slip guard: reject if entry is too far from fib_2_5
        if setup.direction == 1:
            slip = (last_c - fib) / fib
        else:
            slip = (fib - last_c) / fib
        if slip > SWAPPY_SLIP_GUARD_PCT:
            return

        # RSI filter
        if setup.direction == 1 and rsi_val > SWAPPY_RSI_LONG_MAX:
            return
        if setup.direction == -1 and rsi_val < SWAPPY_RSI_SHORT_MIN:
            return

        # Confluence scoring
        conf = False
        if setup.direction == 1 and rsi_val < SWAPPY_CONF_RSI_LONG:
            conf = True
        if setup.direction == -1 and rsi_val > SWAPPY_CONF_RSI_SHORT:
            conf = True
        if not pd.isna(nearest_4h) and abs(fib - nearest_4h) / fib < SWAPPY_CONF_ZONE_PCT:
            conf = True

        sl_dist = abs(setup.fib_4_0 - fib)
        tp_dist = abs(setup.tp - fib)
        rr      = tp_dist / sl_dist if sl_dist > 1e-6 else 0.0

        self._pending_signal = {
            "direction":  setup.direction,
            "entry":      fib,
            "stop":       setup.fib_4_0,
            "tp":         setup.tp,
            "rr":         round(rr, 3),
            "confluence": conf,
            "setup_ts":   setup.setup_ts,
            "signal_ts":  last_ts,
            "atr":        atr_now,
        }


# ── Convenience wrapper ────────────────────────────────────────────────────────

# Module-level singleton so the feed module can call update_swappy(df, lvl)
_state = SwappyState()


def update_swappy(df15: pd.DataFrame, nearest_4h_level: float = float("nan")) -> Optional[dict]:
    """Update state and immediately return any pending signal."""
    _state.update(df15, nearest_4h_level)
    return _state.signal()


if __name__ == "__main__":
    df15 = build_candles(15, n_bars=100)
    sig  = update_swappy(df15)
    if sig:
        print("Signal:", sig)
    else:
        print("No Swappy signal at this time.")
