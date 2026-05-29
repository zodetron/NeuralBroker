"""
signals/generator.py — MTF signal assembly.

Reads live analysis snapshots from all TF modules and assembles trade signals.
Each signal dict is passed through the ML filter before being emitted.

Signal types:
  MTF_LONG  / MTF_SHORT  — multi-timeframe confluence (5M trigger + 15M momentum
                            + 1H structure + 4H zone + 1D bias all aligned)
  SWAPPY_LONG / SWAPPY_SHORT — ICT Fib 2.5/4.0 manipulation fade

Signal dict schema (both types share this structure):
  type         : "MTF_LONG" | "MTF_SHORT" | "SWAPPY_LONG" | "SWAPPY_SHORT"
  direction    : +1 | -1
  entry        : float
  stop         : float
  tp           : float
  rr           : float
  atr          : float
  confluence   : bool
  timestamp    : pd.Timestamp
  raw_score    : float  (ML probability 0-1, filled in by ml_filter)
  approved     : bool   (True after ML filter passes)
  tf_stack     : dict   (snapshot of all TF states at signal time)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from typing import Optional

import pandas as pd

from config import (
    ATR_TP_MULT_1, ATR_SL_MULT,
    MIN_RR_RATIO, MIN_DAILY_RANGE_PCT,
    SIGNAL_COOLDOWN_MINS,
    SWAPPY_MIN_RR,
)
from engine.builder import build_candles

log = logging.getLogger(__name__)


class SignalGenerator:
    """
    Assembled on every closed 1M bar tick.
    Call evaluate() with current TF snapshots; it returns a signal or None.
    """

    def __init__(self) -> None:
        self._last_signal_ts: Optional[pd.Timestamp] = None

    # ── Entry point called by feed on each closed 1M bar ──────────────────────

    def evaluate(
        self,
        snap_1d:  dict,
        snap_4h:  dict,
        snap_1h:  dict,
        snap_15m: dict,
        snap_5m:  dict,
        swappy_signal: Optional[dict],
    ) -> Optional[dict]:
        """
        Returns a signal dict if conditions are met, else None.
        The signal is NOT yet ML-filtered — caller must pass through ml_filter.
        """
        # ── Dead-market gate ──────────────────────────────────────────────────
        if snap_1d.get("daily_range_pct", 0) < MIN_DAILY_RANGE_PCT / 100:
            return None

        # ── Cooldown ──────────────────────────────────────────────────────────
        now_ts = snap_5m.get("timestamp")
        if now_ts is not None and self._last_signal_ts is not None:
            elapsed = (now_ts - self._last_signal_ts).total_seconds() / 60
            if elapsed < SIGNAL_COOLDOWN_MINS:
                return None

        # ── Swappy ICT signal (higher priority) ──────────────────────────────
        if swappy_signal is not None:
            sig = self._wrap_swappy(swappy_signal, snap_15m, snap_1h, snap_4h, snap_1d)
            if sig is not None:
                self._last_signal_ts = now_ts
                return sig

        # ── MTF confluence signal ─────────────────────────────────────────────
        sig = self._check_mtf(snap_1d, snap_4h, snap_1h, snap_15m, snap_5m)
        if sig is not None:
            self._last_signal_ts = now_ts
            return sig

        return None

    # ── Swappy wrapper ────────────────────────────────────────────────────────

    def _wrap_swappy(
        self,
        swappy: dict,
        snap_15m: dict,
        snap_1h: dict,
        snap_4h: dict,
        snap_1d: dict,
    ) -> Optional[dict]:
        direction = swappy["direction"]
        rr        = swappy["rr"]

        if rr < SWAPPY_MIN_RR:
            return None

        # 1H structure must not be directly opposed
        structure = snap_1h.get("structure", "RANGING")
        if direction == 1 and structure == "TRENDING_DOWN":
            return None
        if direction == -1 and structure == "TRENDING_UP":
            return None

        sig_type = "SWAPPY_LONG" if direction == 1 else "SWAPPY_SHORT"

        return {
            "type":       sig_type,
            "direction":  direction,
            "entry":      swappy["entry"],
            "stop":       swappy["stop"],
            "tp":         swappy["tp"],
            "rr":         rr,
            "atr":        swappy["atr"],
            "confluence": swappy["confluence"],
            "timestamp":  swappy["signal_ts"],
            "raw_score":  0.0,
            "approved":   False,
            "tf_stack": {
                "1d":  snap_1d,
                "4h":  snap_4h,
                "1h":  snap_1h,
                "15m": snap_15m,
            },
        }

    # ── MTF confluence ────────────────────────────────────────────────────────

    def _check_mtf(
        self,
        snap_1d: dict,
        snap_4h: dict,
        snap_1h: dict,
        snap_15m: dict,
        snap_5m: dict,
    ) -> Optional[dict]:
        trigger   = snap_5m.get("trigger", "NONE")
        direction = snap_5m.get("direction", 0)

        if trigger == "NONE" or direction == 0:
            return None

        # ── Alignment check ───────────────────────────────────────────────────
        bias_1d  = snap_1d.get("bias", 0)
        zone_4h  = snap_4h.get("zone", "MIDZONE")
        struct_1h = snap_1h.get("structure", "RANGING")
        momentum  = snap_15m.get("momentum", "NEUTRAL")

        aligned = self._is_aligned(direction, bias_1d, zone_4h, struct_1h, momentum)
        if not aligned:
            return None

        # ── Compute entry / stop / TP ─────────────────────────────────────────
        atr    = snap_15m.get("atr") or snap_5m.get("atr", 0.0)
        entry  = snap_5m["last_close"]
        stop   = entry - direction * atr * ATR_SL_MULT
        tp     = entry + direction * atr * ATR_TP_MULT_1

        sl_dist = abs(entry - stop)
        tp_dist = abs(tp - entry)
        rr      = tp_dist / sl_dist if sl_dist > 1e-6 else 0.0

        if rr < MIN_RR_RATIO:
            return None

        confluence = self._confluence_score(direction, snap_1d, snap_4h, snap_1h, snap_15m)

        sig_type = "MTF_LONG" if direction == 1 else "MTF_SHORT"

        return {
            "type":       sig_type,
            "direction":  direction,
            "entry":      entry,
            "stop":       stop,
            "tp":         tp,
            "rr":         round(rr, 3),
            "atr":        atr,
            "confluence": confluence,
            "timestamp":  snap_5m.get("timestamp"),
            "raw_score":  0.0,
            "approved":   False,
            "tf_stack": {
                "1d":  snap_1d,
                "4h":  snap_4h,
                "1h":  snap_1h,
                "15m": snap_15m,
                "5m":  snap_5m,
            },
        }

    def _is_aligned(
        self,
        direction: int,
        bias_1d: int,
        zone_4h: str,
        struct_1h: str,
        momentum: str,
    ) -> bool:
        """At least 3 of 4 higher TFs must agree with the trigger direction."""
        score = 0

        # 1D bias
        if direction == 1 and bias_1d >= 0:
            score += 1
        elif direction == -1 and bias_1d <= 0:
            score += 1

        # 4H zone
        if direction == 1 and zone_4h in ("DEMAND", "MIDZONE"):
            score += 1
        elif direction == -1 and zone_4h in ("SUPPLY", "MIDZONE"):
            score += 1

        # 1H structure
        bull_struct = {"TRENDING_UP", "BREAKOUT_UP", "RANGING"}
        bear_struct = {"TRENDING_DOWN", "BREAKOUT_DOWN", "RANGING"}
        if direction == 1 and struct_1h in bull_struct:
            score += 1
        elif direction == -1 and struct_1h in bear_struct:
            score += 1

        # 15M momentum
        bull_mom = {"STRONG_BULL", "PULLBACK_BULL", "NEUTRAL"}
        bear_mom = {"STRONG_BEAR", "PULLBACK_BEAR", "NEUTRAL"}
        if direction == 1 and momentum in bull_mom:
            score += 1
        elif direction == -1 and momentum in bear_mom:
            score += 1

        return score >= 3

    def _confluence_score(self, direction, snap_1d, snap_4h, snap_1h, snap_15m) -> bool:
        """True if 3 or more strong-alignment conditions are met."""
        count = 0

        bias_1d = snap_1d.get("bias", 0)
        if direction == 1 and bias_1d == 1:
            count += 1
        elif direction == -1 and bias_1d == -1:
            count += 1

        zone_4h = snap_4h.get("zone", "MIDZONE")
        if direction == 1 and zone_4h == "DEMAND":
            count += 1
        elif direction == -1 and zone_4h == "SUPPLY":
            count += 1

        struct_1h = snap_1h.get("structure", "RANGING")
        strong_bull = {"TRENDING_UP", "BREAKOUT_UP"}
        strong_bear = {"TRENDING_DOWN", "BREAKOUT_DOWN"}
        if direction == 1 and struct_1h in strong_bull:
            count += 1
        elif direction == -1 and struct_1h in strong_bear:
            count += 1

        momentum = snap_15m.get("momentum", "NEUTRAL")
        if direction == 1 and momentum == "STRONG_BULL":
            count += 1
        elif direction == -1 and momentum == "STRONG_BEAR":
            count += 1

        return count >= 3
