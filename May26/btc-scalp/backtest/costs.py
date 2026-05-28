"""
backtest/costs.py — Dynamic trade cost model for BTC 15M scalping.

Cost components per trade (one side at entry, one side at exit):
  commission : flat 0.05% per side (maker limit order fee)
  spread     : dynamic — depends on time-of-day + ATR regime
  slippage   : flat 0.02% per side (adverse fill)

Round-trip totals (commission + spread + slippage × 2 sides):
  Normal hours     :  0.13%  (0.10 + 0.01 + 0.02)
  Low-liq (02-06Z) :  0.15%  (0.10 + 0.03 + 0.02)
  High-vol (3×ATR) :  0.17%  (0.10 + 0.05 + 0.02)
"""

from __future__ import annotations
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BT


# ── SPREAD FUNCTION ───────────────────────────────────────────────────────────

def get_spread(
    timestamp:   pd.Timestamp,
    current_atr: float,
    avg_atr:     float,
) -> float:
    """
    Return the bid-ask half-spread cost fraction for a single trade entry.

    Decision order (most urgent condition wins):
      1. ATR spike > 3× normal  → high-vol spread (0.05%)
      2. Low-liquidity window   → low-liq spread  (0.03%)  [02:00–06:00 UTC]
      3. Everything else        → normal spread    (0.01%)

    Parameters
    ----------
    timestamp   : bar open/close time (UTC-aware)
    current_atr : ATR at the trade bar
    avg_atr     : rolling average ATR (same lookback as ATR period)

    Returns
    -------
    float — spread fraction (e.g. 0.0001 = 0.01%)
    """
    atr_ratio = current_atr / (avg_atr + 1e-12)
    hour      = timestamp.hour

    if atr_ratio > BT["spread_atr_mult_thresh"]:
        return BT["spread_high_vol"]
    if BT["spread_low_liq_start"] <= hour <= BT["spread_low_liq_end"]:
        return BT["spread_low_liq"]
    return BT["spread_normal"]


def total_cost_one_side(
    timestamp:   pd.Timestamp,
    current_atr: float,
    avg_atr:     float,
) -> float:
    """
    Commission + spread + slippage for ONE side of a trade.
    Multiply by 2 for round-trip.
    """
    return BT["commission"] + get_spread(timestamp, current_atr, avg_atr) + BT["slippage"]


def round_trip_cost(
    entry_ts:    pd.Timestamp,
    exit_ts:     pd.Timestamp,
    entry_atr:   float,
    exit_atr:    float,
    avg_atr:     float,
) -> float:
    """Full round-trip cost fraction (entry + exit sides combined)."""
    return (
        total_cost_one_side(entry_ts, entry_atr, avg_atr)
        + total_cost_one_side(exit_ts,  exit_atr,  avg_atr)
    )


# ── COST LEDGER ───────────────────────────────────────────────────────────────

@dataclass
class CostLedger:
    """
    Accumulates per-component cost totals across all trades.
    Pass one instance to the backtest engine; call add() per trade.
    """
    commission_total: float = 0.0
    spread_total:     float = 0.0
    slippage_total:   float = 0.0

    # breakdown by cost regime
    _normal_count:   int = field(default=0, repr=False)
    _low_liq_count:  int = field(default=0, repr=False)
    _high_vol_count: int = field(default=0, repr=False)

    def add(
        self,
        trade_value:   float,      # notional value of the trade (price × units)
        entry_ts:      pd.Timestamp,
        exit_ts:       pd.Timestamp,
        entry_atr:     float,
        exit_atr:      float,
        avg_atr:       float,
    ) -> float:
        """
        Record costs for one completed trade. Returns total cost $ amount.
        trade_value = entry_price × position_units (notional, both sides same).
        """
        entry_spread = get_spread(entry_ts, entry_atr, avg_atr)
        exit_spread  = get_spread(exit_ts,  exit_atr,  avg_atr)

        comm_cost  = BT["commission"] * 2 * trade_value
        spread_cost= (entry_spread + exit_spread) * trade_value
        slip_cost  = BT["slippage"]  * 2 * trade_value

        self.commission_total += comm_cost
        self.spread_total     += spread_cost
        self.slippage_total   += slip_cost

        # classify by entry regime
        entry_atr_ratio = entry_atr / (avg_atr + 1e-12)
        entry_hour = entry_ts.hour
        if entry_atr_ratio > BT["spread_atr_mult_thresh"]:
            self._high_vol_count += 1
        elif BT["spread_low_liq_start"] <= entry_hour <= BT["spread_low_liq_end"]:
            self._low_liq_count += 1
        else:
            self._normal_count += 1

        return comm_cost + spread_cost + slip_cost

    @property
    def total(self) -> float:
        return self.commission_total + self.spread_total + self.slippage_total

    def print_summary(self, net_pnl: Optional[float] = None) -> None:
        """Print the cost breakdown table."""
        gross_pnl = (net_pnl + self.total) if net_pnl is not None else None

        print(f"\n  {'─'*52}")
        print(f"  COST BREAKDOWN")
        print(f"  {'─'*52}")
        print(f"  {'Component':<22} {'Amount':>12} {'% of gross':>12}")
        print(f"  {'─'*52}")

        components = [
            ("Commission (0.05%/side)", self.commission_total),
            ("Spread (dynamic)",         self.spread_total),
            ("Slippage (0.02%/side)",    self.slippage_total),
        ]
        for name, amt in components:
            pct = (amt / gross_pnl * 100) if gross_pnl and gross_pnl > 0 else float("nan")
            pct_str = f"{pct:.1f}%" if not np.isnan(pct) else "—"
            print(f"  {name:<22} ${amt:>10,.2f} {pct_str:>12}")

        print(f"  {'─'*52}")
        pct_t = (self.total / gross_pnl * 100) if gross_pnl and gross_pnl > 0 else float("nan")
        pct_t_str = f"{pct_t:.1f}%" if not np.isnan(pct_t) else "—"
        print(f"  {'TOTAL COSTS':<22} ${self.total:>10,.2f} {pct_t_str:>12}")

        if net_pnl is not None:
            print(f"  {'Gross PnL':<22} ${gross_pnl:>10,.2f}")
            print(f"  {'Net PnL':<22} ${net_pnl:>10,.2f}")

        total_trades = self._normal_count + self._low_liq_count + self._high_vol_count
        if total_trades > 0:
            print(f"\n  Spread regime at entry ({total_trades} trades):")
            for label, cnt in [
                ("Normal (0.01%)",       self._normal_count),
                ("Low-liq 02-06Z (0.03%)", self._low_liq_count),
                ("High-vol >3×ATR (0.05%)", self._high_vol_count),
            ]:
                print(f"    {label:<28} {cnt:>5}  ({cnt/total_trades*100:.1f}%)")
        print(f"  {'─'*52}")
