# backtest_upstox_nifty_options.py
import os
import math
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dateutil import rrule, parser
from datetime import datetime, timedelta, time

# ------------------ CONFIG ------------------
API_BASE = os.getenv("API_BASE", "https://api-hft.upstox.com")  # change if needed
ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN", None)
UNDERLYING_NAME = "NIFTY"  # used to identify index instruments
INVEST_PER_TRADE = float(os.getenv("INVEST_PER_TRADE", 20000))  # INR per trade
SL_PCT = 0.05   # 5% of invested amount (converted to price move)
TP_PCT = 0.02   # 2% take-profit on entry price
VOLUME_RATIO_THRESHOLD = 1.5
EMA_FAST = 12
EMA_SLOW = 20
EMA_HTF = 50    # HTF EMA on 15-min bars
TIMEFRAME_MIN = 5
HTF_MIN = 15

# backtest date range (1 month default)
END_DATE = datetime.utcnow().date()
START_DATE = END_DATE - timedelta(days=30)

# trading hours (NSE approx) - modifies per exchange - we assume 09:15 - 15:30 local
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)

# ------------------ HELPERS ------------------
HEADERS = {"Authorization": f"Bearer {ACCESS_TOKEN}"} if ACCESS_TOKEN else {}

def safe_get(url, params=None):
    r = requests.get(url, headers=HEADERS, params=params, timeout=30)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code} GET {url} -> {r.text}")
    ctype = r.headers.get("content-type","")
    if "application/json" in ctype:
        return r.json()
    # some endpoints may return plain text (but we expect json)
    return r.text

def calculate_ema(series, period):
    # returns pandas Series (same length)
    return series.ewm(span=period, adjust=False).mean()

def floor_int(x):
    return int(math.floor(x))

# ------------------ UPSTOX FETCHERS (ADJUST IF YOUR API DIFFERS) ------------------

def fetch_instruments():
    """
    Fetch instruments list. Upstox often provides a daily instruments JSON.
    This function assumes a GET endpoint /v2/instruments (or similar). If your Upstox
    account provides a static file URL, replace this function accordingly.
    """
    url = f"{API_BASE}/v2/instruments"
    try:
        return safe_get(url)  # expecting list of dicts
    except Exception as e:
        # If /v2/instruments not available, try a fallback public doc or raise
        raise RuntimeError("Could not fetch instruments from Upstox API. Update fetch_instruments() to your instruments file. " + str(e))

def fetch_intraday_candles(instrument_key, interval_minutes, from_iso, to_iso):
    """
    v3 historical candlestick endpoint pattern used earlier. Adjust if your API differs.
    Returns list of candles with fields: {close, open, high, low, volume, time}
    """
    # Example path used earlier: /v3/historical-candle/intraday/{instrument_key}/minutes/{interval}?from={from}&to={to}
    url = f"{API_BASE}/v3/historical-candle/intraday/{instrument_key}/minutes/{interval_minutes}"
    params = {"from": from_iso, "to": to_iso}
    data = safe_get(url, params=params)
    # Possible returned shapes: {"candles":[...]} or {"data": [...]} or raw list
    candles = data.get("candles") if isinstance(data, dict) and "candles" in data else data.get("data") if isinstance(data, dict) and "data" in data else data
    # Normalize to DataFrame
    df = pd.DataFrame(candles)
    # Normalize field names
    if df.empty:
        return df
    # find best column names
    for col in ["close", "c", "close_price", "closePrice"]:
        if col in df.columns:
            df["close"] = df[col]; break
    for col in ["volume", "v", "vol"]:
        if col in df.columns:
            df["volume"] = df[col]; break
    # time field unify
    for col in ["time", "datetime", "t", "timestamp"]:
        if col in df.columns:
            df["time"] = pd.to_datetime(df[col]); break
    return df[["time","close","volume"]].dropna()

# ------------------ INSTRUMENT CHOOSER ------------------

def parse_expiry(exp):
    # attempt to parse expiry string to date; fallback to large timestamp
    try:
        return parser.parse(exp).date()
    except:
        return None

def choose_atm_option_for_day(instruments, target_date, underlying_spot, option_type_hint=None):
    """
    Choose nearest-expiry ATM option for the provided underlying spot price on target_date.
    - instruments: list/dict from fetch_instruments
    - target_date: date object
    - underlying_spot: numeric value
    - option_type_hint: 'CE' to prefer calls, 'PE' to prefer puts, or None
    Returns chosen instrument dict or raises if none.
    """
    # Normalize instruments into DataFrame
    df = pd.DataFrame(instruments)
    if df.empty:
        raise RuntimeError("Instruments list empty")
    # filtering underlying symbol/name
    # Common fields: 'underlying', 'underlying_symbol', 'symbol', 'name'
    mask_under = df.apply(lambda r: any([
        str(r.get(k,"")).lower().find(UNDERLYING_NAME.lower()) != -1
        for k in ["underlying","underlying_symbol","symbol","name"]
    ]), axis=1)
    df = df[mask_under]
    if df.empty:
        raise RuntimeError(f"No instruments matching underlying {UNDERLYING_NAME}")

    # Filter options only
    # Common field: instrument_type or option_type or segment
    opt_mask = df.apply(lambda r: any([
        "opt" in str(r.get("instrument_type","")).lower(),
        str(r.get("segment","")).lower().find("fo") != -1 and ("option" in str(r.get("instrument_type","")).lower() or "opt" in str(r.get("instrument_type","")).lower()),
        str(r.get("option_type","")).upper() in ["CE","PE"]
    ]), axis=1)
    df_opt = df[opt_mask].copy()
    if df_opt.empty:
        raise RuntimeError("No option instruments found for underlying in instruments file")

    # compute expiry and strike columns
    df_opt["expiry_dt"] = df_opt["expiry"].apply(lambda x: parse_expiry(x))
    df_opt["strike_val"] = df_opt.apply(lambda r: float(r.get("strike") or r.get("strike_price") or 0), axis=1)
    df_opt["lot_size"] = df_opt.apply(lambda r: int(r.get("lot_size") or r.get("lotSize") or r.get("lots") or 1), axis=1)
    df_opt["opt_type"] = df_opt.apply(lambda r: (str(r.get("option_type") or r.get("opt_type") or "")).upper(), axis=1)

    # Keep only expiries >= target_date
    df_opt = df_opt[df_opt["expiry_dt"].notnull() & (df_opt["expiry_dt"] >= target_date)]
    if df_opt.empty:
        raise RuntimeError("No option expiries >= target date found")

    # sort by expiry (nearest), then strike distance to ATM, then smaller lot size
    df_opt["strike_dist"] = abs(df_opt["strike_val"] - underlying_spot)
    df_opt.sort_values(by=["expiry_dt","strike_dist","lot_size"], inplace=True)

    # If option_type_hint is provided, filter that
    if option_type_hint:
        cand = df_opt[df_opt["opt_type"] == option_type_hint]
        if not cand.empty:
            df_opt = cand

    chosen = df_opt.iloc[0].to_dict()
    return chosen

# ------------------ BACKTEST ENGINE ------------------

def compute_lots_from_invested(invested_amt, price, lot_size):
    # number of contracts (lots) we can buy with invested_amt
    if price <= 0 or lot_size <= 0:
        return 0
    per_lot_notional = price * lot_size
    return int(invested_amt // per_lot_notional)  # floor

def compute_stop_price_for_5pct(invested_amt, lots, lot_size, entry_price, side):
    # loss_total = invested_amt * 0.05
    # loss_per_unit = loss_total / (lots * lot_size)
    if lots <= 0:
        return None
    loss_total = invested_amt * SL_PCT
    loss_per_unit = loss_total / (lots * lot_size)
    if side == "BUY":
        return entry_price - loss_per_unit
    else:
        return entry_price + loss_per_unit

def backtest_one_day(instruments, date):
    """
    Simulate the bot running every 5-min for a single trading date.
    Returns list of trades executed that day (each trade dict includes pnl and timestamps).
    """
    # Determine market day timeframe
    day_start = datetime.combine(date, MARKET_OPEN)
    day_end = datetime.combine(date, MARKET_CLOSE)

    # 1) find index instrument to fetch spot series (to pick ATM)
    df_inst = pd.DataFrame(instruments)
    idx_mask = df_inst.apply(lambda r: any([
        str(r.get("segment","")).upper().find("INDEX") != -1,
        (str(r.get("symbol","")).upper().find("NIFTY") != -1),
        (str(r.get("name","")).upper().find("NIFTY") != -1)
    ]), axis=1)
    idx_cands = df_inst[idx_mask]
    if idx_cands.empty:
        # fallback: try any instrument with 'NIFTY' in name/symbol
        idx_cands = df_inst[df_inst.apply(lambda r: ("nifty" in str(r.get("symbol","")).lower()) or ("nifty" in str(r.get("name","")).lower()), axis=1)]
        if idx_cands.empty:
            raise RuntimeError("Could not find index instrument for NIFTY to determine spot price")

    index_inst = idx_cands.iloc[0].to_dict()
    index_key = index_inst.get("instrument_key") or index_inst.get("instrument_token") or index_inst.get("exchange_token")

    # fetch 5-min spot candles for the day
    from_iso = (day_start - timedelta(minutes=30)).isoformat()  # small padding
    to_iso = (day_end + timedelta(minutes=30)).isoformat()
    spot_candles = fetch_intraday_candles(index_key, TIMEFRAME_MIN, from_iso, to_iso)
    if spot_candles.empty:
        raise RuntimeError(f"No spot candles for index on {date}")

    # choose ATM option at open (use first available 5-min close at/after market open)
    spot_at_open_row = spot_candles[spot_candles["time"] >= pd.to_datetime(day_start)].head(1)
    if spot_at_open_row.empty:
        spot_price = float(spot_candles.iloc[0]["close"])
    else:
        spot_price = float(spot_at_open_row.iloc[0]["close"])

    # choose call and put instruments (nearest expiry ATM) - strategy will pick CE for LONG, PE for SHORT
    chosen_ce = choose_atm_option_for_day(instruments, date, spot_price, option_type_hint="CE")
    chosen_pe = choose_atm_option_for_day(instruments, date, spot_price, option_type_hint="PE")

    # pick the option we will trade based on EMA signal — we need time series for both instruments to compute EMAs
    # fetch intraday candles for chosen CE and PE (5-min and 15-min). If 15-min not available, we will resample from 5-min.
    def fetch_opt_df(inst):
        key = inst.get("instrument_key") or inst.get("instrument_token") or inst.get("exchange_token")
        df5 = fetch_intraday_candles(key, TIMEFRAME_MIN, from_iso, to_iso)
        # if 15-min endpoint exists, fetch it, else resample from 5-min
        try:
            df15 = fetch_intraday_candles(key, HTF_MIN, from_iso, to_iso)
        except Exception:
            # resample 5-min to 15-min
            tmp = df5.set_index("time").copy()
            df15 = tmp["close"].resample("15T").agg("last").to_frame()
            df15["volume"] = tmp["volume"].resample("15T").sum()
            df15 = df15.reset_index().rename(columns={"close":"close","volume":"volume"})
        return df5, df15

    ce_5, ce_15 = fetch_opt_df(chosen_ce)
    pe_5, pe_15 = fetch_opt_df(chosen_pe)

    # Create a unified timeline of 5-min bars between market open and close
    timeline = pd.date_range(start=day_start, end=day_end, freq=f"{TIMEFRAME_MIN}T")
    trades = []

    # We'll maintain entry state (single position at a time per day)
    position = None

    # iterate over timeline
    for ts in timeline:
        # get historical bars up to current ts for the chosen instrument price series
        # but we must compute EMA on the instrument that we will trade (option chosen after signal)
        # We'll compute EMAs for both CE and PE and then decide which instrument to use based on signal.
        # Align bars: find rows <= ts
        def get_row_slice(df, t):
            df_t = df[df["time"] <= pd.to_datetime(t)].copy()
            return df_t

        ce_hist_5 = get_row_slice(ce_5, ts)
        pe_hist_5 = get_row_slice(pe_5, ts)
        ce_hist_15 = get_row_slice(ce_15, ts)
        pe_hist_15 = get_row_slice(pe_15, ts)

        # need at least max(EMA_SLOW, EMA_HTF) bars
        if len(ce_hist_5) < EMA_SLOW or len(ce_hist_15) < EMA_HTF or len(pe_hist_5) < EMA_SLOW or len(pe_hist_15) < EMA_HTF:
            continue

        # compute EMAs for CE
        ce_close_5 = ce_hist_5["close"].astype(float)
        ce_ema12 = calculate_ema(ce_close_5, EMA_FAST).iloc[-1]
        ce_ema20 = calculate_ema(ce_close_5, EMA_SLOW).iloc[-1]
        ce_close_15 = ce_hist_15["close"].astype(float)
        ce_ema50_htf = calculate_ema(ce_close_15, EMA_HTF).iloc[-1]

        # compute EMAs for PE
        pe_close_5 = pe_hist_5["close"].astype(float)
        pe_ema12 = calculate_ema(pe_close_5, EMA_FAST).iloc[-1]
        pe_ema20 = calculate_ema(pe_close_5, EMA_SLOW).iloc[-1]
        pe_close_15 = pe_hist_15["close"].astype(float)
        pe_ema50_htf = calculate_ema(pe_close_15, EMA_HTF).iloc[-1]

        # compute volumes & ratio on latest 5-min for both
        def vol_ratio(hist5):
            vols = hist5["volume"].astype(float)
            if len(vols) < 11: return 1.0
            avg_prev10 = vols.iloc[-11:-1].mean()
            latest_vol = vols.iloc[-1]
            return latest_vol / avg_prev10 if avg_prev10 > 0 else 1.0

        ce_vol_ratio = vol_ratio(ce_hist_5)
        pe_vol_ratio = vol_ratio(pe_hist_5)

        # determine crossover signals (we detect change-of-relation by looking at last two EMA comparisons)
        def ema_relation(close_series, efast_period=EMA_FAST, eslow_period=EMA_SLOW):
            ema_fast = calculate_ema(close_series, efast_period)
            ema_slow = calculate_ema(close_series, eslow_period)
            if len(ema_fast) < 2 or len(ema_slow) < 2: return None
            prev_rel = "above" if ema_fast.iloc[-2] > ema_slow.iloc[-2] else ("below" if ema_fast.iloc[-2] < ema_slow.iloc[-2] else "equal")
            curr_rel = "above" if ema_fast.iloc[-1] > ema_slow.iloc[-1] else ("below" if ema_fast.iloc[-1] < ema_slow.iloc[-1] else "equal")
            return prev_rel, curr_rel

        ce_prev_rel, ce_curr_rel = ema_relation(ce_close_5)
        pe_prev_rel, pe_curr_rel = ema_relation(pe_close_5)

        # decide signals:
        # - CE LONG when EMA crosses above AND CE HTF confirms AND volume spike
        # - PE SHORT when EMA crosses below AND PE HTF confirms AND volume spike
        signal = None
        trade_inst = None
        trade_price = None
        trade_lots = 0

        # Last bar close price for entry estimation
        ce_price = float(ce_hist_5.iloc[-1]["close"])
        pe_price = float(pe_hist_5.iloc[-1]["close"])

        # Determine CE long
        if ce_prev_rel == "below" and ce_curr_rel == "above" and ce_ema12 > ce_ema50_htf and ce_vol_ratio > VOLUME_RATIO_THRESHOLD:
            signal = "LONG_CE"
            trade_inst = chosen_ce
            trade_price = ce_price

        # Determine PE short (we buy PE to short underlying — actually buying PE is LONG PUT; but your original bot used 'sell' for short futures.
        # For simplicity here: signal SHORT -> we BUY PUT (long put) as the directional instrument)
        if pe_prev_rel == "above" and pe_curr_rel == "below" and pe_ema12 < pe_ema50_htf and pe_vol_ratio > VOLUME_RATIO_THRESHOLD:
            signal = "LONG_PE"
            trade_inst = chosen_pe
            trade_price = pe_price

        # place simulated order if signal and no existing position
        if signal and position is None:
            lot_size = int(trade_inst.get("lot_size") or trade_inst.get("lotSize") or trade_inst.get("lots") or 1)
            lots = compute_lots_from_invested(INVEST_PER_TRADE, trade_price, lot_size)
            if lots <= 0:
                # cannot buy even 1 lot -> skip
                continue

            invested_actual = lots * lot_size * trade_price
            stop_price = compute_stop_price_for_5pct(invested_actual, lots, lot_size, trade_price, "BUY")
            tp_price = trade_price * (1 + TP_PCT) if signal == "LONG_CE" else trade_price * (1 + TP_PCT)  # same for options
            position = {
                "entry_time": ts,
                "instrument": trade_inst,
                "side": "BUY",
                "lots": lots,
                "lot_size": lot_size,
                "entry_price": trade_price,
                "stop_price": stop_price,
                "tp_price": tp_price,
                "invested": invested_actual,
                "status": "OPEN"
            }
            # assume order filled at next bar open realistically; we already used the last bar price as proxy
            position["filled_price"] = trade_price
            # store
            print(f"{date} {ts} -> ENTER {signal} {trade_inst.get('symbol') or trade_inst.get('instrument_key')} price {trade_price:.2f} lots {lots} invested {invested_actual:.2f}")
            continue

        # if position exists, check SL/TP on new price (use current bar close)
        if position is not None:
            # pick price series of the instrument in position
            inst = position["instrument"]
            inst_key = inst.get("instrument_key") or inst.get("instrument_token") or inst.get("exchange_token")
            # get relevant candle df (ce or pe)
            if position["instrument"].get("option_type","").upper().startswith("C") or position["instrument"].get("option_type","").upper().startswith("CE") or position["instrument"].get("opt_type","").upper().startswith("CE") or position["instrument"] == chosen_ce:
                df_now = ce_hist_5
            else:
                df_now = pe_hist_5
            if df_now.empty:
                continue
            curr_price = float(df_now.iloc[-1]["close"])
            # check SL hit
            if position["side"] == "BUY":
                if curr_price <= position["stop_price"]:
                    pnl = (curr_price - position["entry_price"]) * position["lots"] * position["lot_size"]
                    trades.append({
                        "entry_time": position["entry_time"],
                        "exit_time": ts,
                        "instrument": position["instrument"],
                        "side": position["side"],
                        "entry_price": position["entry_price"],
                        "exit_price": curr_price,
                        "lots": position["lots"],
                        "lot_size": position["lot_size"],
                        "pnl": pnl
                    })
                    print(f"{date} {ts} -> EXIT STOP LOSS price {curr_price:.2f} pnl {pnl:.2f}")
                    position = None
                    continue
                if curr_price >= position["tp_price"]:
                    pnl = (curr_price - position["entry_price"]) * position["lots"] * position["lot_size"]
                    trades.append({
                        "entry_time": position["entry_time"],
                        "exit_time": ts,
                        "instrument": position["instrument"],
                        "side": position["side"],
                        "entry_price": position["entry_price"],
                        "exit_price": curr_price,
                        "lots": position["lots"],
                        "lot_size": position["lot_size"],
                        "pnl": pnl
                    })
                    print(f"{date} {ts} -> EXIT TAKE PROFIT price {curr_price:.2f} pnl {pnl:.2f}")
                    position = None
                    continue
                # trailing: if price gained 1% from entry, move stop to entry (breakeven)
                if curr_price >= position["entry_price"] * 1.01:
                    new_stop = max(position["stop_price"], position["entry_price"])
                    if new_stop != position["stop_price"]:
                        position["stop_price"] = new_stop
                        # print trailing update
                        print(f"{date} {ts} -> Trailing stop updated to {new_stop:.2f}")

        # continue loop to next bar
    # End of day: if position still open, close at last available price (EOD close)
    if position:
        # find last price at or before market close
        inst = position["instrument"]
        df_for = ce_5 if inst == chosen_ce else pe_5
        last_row = df_for[df_for["time"] <= pd.to_datetime(day_end)].tail(1)
        if last_row.empty:
            exit_price = position["entry_price"]
        else:
            exit_price = float(last_row.iloc[0]["close"])
        pnl = (exit_price - position["entry_price"]) * position["lots"] * position["lot_size"]
        trades.append({
            "entry_time": position["entry_time"],
            "exit_time": datetime.combine(date, MARKET_CLOSE),
            "instrument": position["instrument"],
            "side": position["side"],
            "entry_price": position["entry_price"],
            "exit_price": exit_price,
            "lots": position["lots"],
            "lot_size": position["lot_size"],
            "pnl": pnl
        })
        print(f"{date} EOD -> EXIT at close price {exit_price:.2f} pnl {pnl:.2f}")
        position = None

    return trades

# ------------------ RUN BACKTEST ------------------

def run_backtest(start_date, end_date):
    # fetch instruments once
    instruments = fetch_instruments()
    all_trades = []
    # Loop trading days (Mon-Fri)
    for dt in rrule.rrule(rrule.DAILY, dtstart=start_date, until=end_date):
        if dt.weekday() >= 5:
            continue
        day = dt.date()
        try:
            trades = backtest_one_day(instruments, day)
            all_trades.extend(trades)
        except Exception as e:
            print(f"Skipping {day} due to error: {e}")

    # build DataFrame of trades
    if not all_trades:
        print("No trades executed in the backtest period.")
        return None
    
    def run_backtest2(start_date-30, end_date-30):
        for dt in rrule.rrule(rrule.WEEKLY, dtstart=start_date-30, until=end_date-30):
        if dt.weekday() >= 6:
            continue
        day = dt.date()
        try:
            trades = backtest_one_day(instruments, day)
            all_trades.extend(trades)
        except Exception as e:
            print(f"Skipping {day} due to error: {e}")

    def Run_backtest3(start_date-30, end_date-30):
        for dt in rrule:rrule(rrule.MONTHLY, distance=start_date,unitl=end_date)
        if dt.weekday() >= 5:
            continue
        day = dt.date()
        try:
            trades = backtest_one_day(instruments, day)
            all_trades.extend(trades)
        except Exception as e:
            print(f"Skipping {day} due to error: {e}")


    df = pd.DataFrame(all_trades)
    df["pnl"] = df["pnl"].astype(float)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"] = pd.to_datetime(df["exit_time"])

    # compute metrics
    total_pnl = df["pnl"].sum()
    win_rate = (df["pnl"] > 0).mean()
    avg_win = df[df["pnl"] > 0]["pnl"].mean() if (df["pnl"] > 0).any() else 0
    avg_loss = df[df["pnl"] <= 0]["pnl"].mean() if (df["pnl"] <= 0).any() else 0
    trades_count = len(df)

    print("Backtest summary:")
    print(f"Period: {start_date} to {end_date}")
    print(f"Trades: {trades_count}, Total PnL: {total_pnl:.2f}, Win rate: {win_rate:.2%}")
    print(f"Avg Win: {avg_win:.2f}, Avg Loss: {avg_loss:.2f}")

    # equity curve
    df = df.sort_values(by="exit_time").reset_index(drop=True)
    df["cum_pnl"] = df["pnl"].cumsum()

    # Plot equity curve
    plt.figure(figsize=(10,5))
    plt.plot(df["exit_time"], df["cum_pnl"], marker="o")
    plt.title("Equity Curve - Cumulative PnL")
    plt.xlabel("Time")
    plt.ylabel("Cumulative PnL (INR)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return df

if __name__ == "__main__":
    # Run backtest for START_DATE to END_DATE
    print(f"Running backtest from {START_DATE} to {END_DATE}")
    trades_df = run_backtest(START_DATE, END_DATE)
    if trades_df is not None:
        trades_df.to_csv("backtest_trades.csv", index=False)
        print("Trades saved to backtest_trades.csv")
