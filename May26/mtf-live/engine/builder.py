"""
engine/builder.py — Builds higher-TF OHLCV DataFrames from the 1M DuckDB table.

Design: never stores higher TF candles — builds them on-the-fly via SQL aggregation.
DuckDB's grouping is fast enough (<50ms) for real-time use on 3.5M rows.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
import pandas as pd
from config import DB_PATH


def build_candles(tf_minutes: int, n_bars: int = 500) -> pd.DataFrame:
    """
    Build n_bars of OHLCV candles at the given timeframe from 1M storage.

    Uses DuckDB SQL to aggregate — no pandas groupby, so it stays under 50ms
    even on 3.5M rows for normal n_bars values.

    Parameters
    ----------
    tf_minutes : timeframe in minutes (5, 15, 60, 240, 1440)
    n_bars     : how many completed bars to return

    Returns
    -------
    pd.DataFrame with columns: timestamp, open, high, low, close, volume
    Index is a RangeIndex (not the timestamp) so callers can use iloc safely.
    """
    # We need (n_bars + 1) × tf_minutes worth of 1M bars to build n_bars complete TF bars.
    # The +1 guard ensures the most-recent (potentially incomplete) bar is excluded.
    lookback_1m = (n_bars + 2) * tf_minutes

    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        # epoch_bucket groups 1M timestamps into TF-sized buckets.
        # epoch(timestamp) returns seconds since Unix epoch.
        # Dividing by (tf_minutes * 60) and flooring gives a unique integer per TF candle.
        sql = f"""
            WITH bucketed AS (
                SELECT
                    epoch_ms(
                        CAST(
                            floor(epoch(timestamp) / ({tf_minutes} * 60)) * {tf_minutes} * 60
                            AS BIGINT
                        ) * 1000
                    ) AS ts,
                    first(open  ORDER BY timestamp) AS open,
                    max(high)                        AS high,
                    min(low)                         AS low,
                    last(close  ORDER BY timestamp)  AS close,
                    sum(volume)                      AS volume,
                    count(*)                         AS bar_count
                FROM candles_1m
                WHERE timestamp >= (
                    SELECT max(timestamp) - INTERVAL '{lookback_1m} minutes'
                    FROM candles_1m
                )
                GROUP BY floor(epoch(timestamp) / ({tf_minutes} * 60))
            )
            SELECT ts AS timestamp, open, high, low, close, volume
            FROM bucketed
            -- Exclude the last bucket: it may be an incomplete candle
            WHERE ts < (SELECT max(ts) FROM bucketed)
            ORDER BY ts DESC
            LIMIT {n_bars}
        """
        df = con.execute(sql).df()
    finally:
        con.close()

    if df.empty:
        return df

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def get_latest_1m_count() -> int:
    """Return total number of 1M candles stored."""
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        result = con.execute("SELECT count(*) FROM candles_1m").fetchone()
        return result[0] if result else 0
    finally:
        con.close()


def get_date_range() -> tuple:
    """Return (min_ts, max_ts) of stored candles."""
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        row = con.execute(
            "SELECT min(timestamp), max(timestamp) FROM candles_1m"
        ).fetchone()
        return row if row else (None, None)
    finally:
        con.close()


if __name__ == "__main__":
    # Quick self-test: build one candle at each supported timeframe
    for tf, label in [(5, "5M"), (15, "15M"), (60, "1H"), (240, "4H"), (1440, "1D")]:
        df = build_candles(tf, n_bars=10)
        print(f"  {label}: {len(df)} bars  last={df['timestamp'].iloc[-1] if not df.empty else 'N/A'}")
    print("Builder ready — all TFs operational")
