"""
run_phase3.py — Build and verify Market State Detector + cost model.

Usage:
    venv/bin/python3 run_phase3.py
"""

import sys
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import warnings
import time
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

from data.pipeline import load_15m, load_1h
from strategy.market_state import classify_market_state, print_state_summary, plot_state_distribution
from backtest.costs import get_spread, CostLedger
from config import BT, RESULTS, DATA_CFG


def verify_cost_model() -> None:
    """Smoke-test the cost model and print representative round-trip costs."""
    print(f"\n  {'─'*60}")
    print(f"  COST MODEL VERIFICATION")
    print(f"  {'─'*60}")

    # Build representative timestamps and ATR conditions
    scenarios = [
        ("Normal  (10:00Z, no spike)",
         pd.Timestamp("2024-01-15 10:00", tz="UTC"), 100.0, 100.0),
        ("Low-liq (03:00Z, no spike)",
         pd.Timestamp("2024-01-15 03:00", tz="UTC"), 100.0, 100.0),
        ("High-vol (10:00Z, 3.5× ATR)",
         pd.Timestamp("2024-01-15 10:00", tz="UTC"), 350.0, 100.0),
        ("High-vol + low-liq (04:00Z, 4× ATR)",
         pd.Timestamp("2024-01-15 04:00", tz="UTC"), 400.0, 100.0),
    ]

    print(f"  {'Scenario':<38} {'Spread':>8} {'One-side':>10} {'Round-trip':>12}")
    print(f"  {'─'*60}")
    for label, ts, atr, avg in scenarios:
        spread    = get_spread(ts, atr, avg)
        one_side  = BT["commission"] + spread + BT["slippage"]
        rt        = one_side * 2
        print(f"  {label:<38} {spread*100:.3f}%  {one_side*100:.4f}%  {rt*100:.4f}%")

    print(f"\n  Reference costs (round-trip %):  commission={BT['commission']*200:.3f}%"
          f"  slippage={BT['slippage']*200:.3f}%")


def main():
    start_time = time.time()

    print("\n" + "═"*62)
    print("  BTC SCALP — PHASE 3: MARKET STATE DETECTOR")
    print("═"*62)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n  Loading cached data …")
    df_15m = load_15m()
    df_1h  = load_1h()

    # ── Classify market states ────────────────────────────────────────────────
    print("\n  Classifying 15M bars …", end="", flush=True)
    t0 = time.time()
    df = classify_market_state(df_15m, df_1h)
    elapsed = time.time() - t0
    print(f" done in {elapsed:.1f}s  ({len(df):,} bars)")

    # Verify no lookahead — warmup rows have NaN state (expected)
    nan_states = df["market_state"].isna().sum()
    print(f"  Warmup NaN states: {nan_states}  (expected ~40 rows for ADX warmup)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print_state_summary(df)

    # ── Charts ────────────────────────────────────────────────────────────────
    plot_state_distribution(df, RESULTS / "market_state_distribution.png")

    # ── Cost model verification ───────────────────────────────────────────────
    verify_cost_model()

    # ── Apply cost model to a sample of 100 mock trades ───────────────────────
    print(f"\n  {'─'*60}")
    print(f"  COST LEDGER DEMO  (100 simulated trades)")
    print(f"  {'─'*60}")

    ledger     = CostLedger()
    rng        = np.random.default_rng(42)
    sample_idx = rng.choice(len(df) - 20, size=100, replace=False)

    for idx in sample_idx:
        row       = df.iloc[idx]
        exit_row  = df.iloc[idx + 8]   # 8-bar trade
        price     = row["close"]
        units     = BT["initial_capital"] / price
        notional  = price * units

        ledger.add(
            trade_value = notional,
            entry_ts    = row.name,
            exit_ts     = exit_row.name,
            entry_atr   = row["atr_14"],
            exit_atr    = exit_row["atr_14"],
            avg_atr     = row["atr_avg"] if not np.isnan(row["atr_avg"]) else row["atr_14"],
        )

    mock_net = rng.normal(50, 200, 100).sum()   # random net PnL for illustration
    mock_gross = mock_net + ledger.total
    ledger.print_summary(net_pnl=mock_net)

    # ── Save classified data ───────────────────────────────────────────────────
    out = DATA_CFG["parquet_15m"].parent / "btc_15m_states.parquet"
    df.to_parquet(out, engine="pyarrow", compression="snappy")
    print(f"\n  Classified data saved → {out.name}  "
          f"({out.stat().st_size / 1e6:.1f} MB)")

    total_time = time.time() - start_time
    print(f"\n{'═'*62}")
    print(f"  PHASE 3 COMPLETE  ({total_time:.1f}s)")
    print(f"{'═'*62}")
    print(f"  Outputs in {RESULTS}/")
    print(f"  Ready for Phase 4 — Strategy Signals.\n")


if __name__ == "__main__":
    main()
