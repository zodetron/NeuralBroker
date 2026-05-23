"""
HYPER STRATEGY SEARCH — XAUUSD 5m  |  2 Years  |  Spread: 0.025%
Tests 37 strategies, ranks by monthly return with max drawdown.
Capital: $200  |  Leverage: 1:1000  |  Min Lot: 0.01  |  Risk: 1%/trade

DATA SOURCE: Twelve Data API (free — get key at https://twelvedata.com)
  1. Register free at twelvedata.com (no credit card)
  2. Copy your API key below
"""

import os, time, requests
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime, timedelta

# ── FREE API KEY (optional but recommended for full 2-year history) ───────────
# Get a free key at https://twelvedata.com  (no credit card, 2-min signup)
# Without it: falls back to yfinance which only gives ~70 days of 5m data.
TWELVEDATA_API_KEY = "da2428fa53e14697989d2707c308a01d"
XAUUSD_CSV        = "xauusd_5m_2y.csv"

# ── CONFIG ────────────────────────────────────────────────────────────────────
CAPITAL  = 200.0
RISK_PCT = 0.01
SPREAD   = 0.00025   # 0.025%
LEVERAGE = 1000
MIN_LOT  = 0.01
LOT_STEP = 0.01
# For XAUUSD: 1 lot = 1 oz gold (CFD-style, same math as BTC)

# ── XAUUSD DATA DOWNLOADER ────────────────────────────────────────────────────
def download_xauusd(api_key, days=730, csv_file=XAUUSD_CSV):
    print(f"Downloading XAUUSD 5m data ({days} days) from Twelve Data...")
    end_dt   = datetime.utcnow()
    start_dt = end_dt - timedelta(days=days)
    url      = "https://api.twelvedata.com/time_series"

    all_rows    = []
    current_end = end_dt
    req_n       = 0

    while current_end > start_dt:
        params = {
            "symbol":     "XAU/USD",
            "interval":   "5min",
            "outputsize": 5000,
            "apikey":     api_key,
            "end_date":   current_end.strftime("%Y-%m-%d %H:%M:%S"),
            "timezone":   "UTC",
            "format":     "JSON",
        }
        try:
            resp = requests.get(url, params=params, timeout=20)
            data = resp.json()
        except Exception as e:
            print(f"\n  ⚠ Request error: {e} — retrying in 10s...")
            time.sleep(10)
            continue

        if data.get("status") == "error" or "code" in data:
            msg = data.get("message", str(data))
            print(f"\n  ✗ API error: {msg}")
            if "api_key" in msg.lower() or "invalid" in msg.lower():
                print("  → Set TWELVEDATA_API_KEY to your free key from twelvedata.com")
            break

        values = data.get("values", [])
        if not values:
            break

        all_rows.extend(values)
        req_n += 1
        earliest = datetime.strptime(values[-1]["datetime"], "%Y-%m-%d %H:%M:%S")
        pct      = (end_dt - earliest) / (end_dt - start_dt) * 100
        print(f"  [{req_n:3d} req] {len(all_rows):>7,} candles | {pct:.1f}% | up to {earliest.strftime('%Y-%m-%d')}",
              end="\r")

        if earliest <= start_dt:
            break
        current_end = earliest - timedelta(minutes=5)
        time.sleep(9)   # free tier: 8 req/min → wait ~9s between calls

    print()
    if not all_rows:
        return None

    df = pd.DataFrame(all_rows)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.rename(columns={"datetime": "open_time",
                             "open": "Open", "high": "High",
                             "low": "Low",   "close": "Close",
                             "volume": "Volume"})
    for c in ["Open", "High", "Low", "Close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0) if "volume" in df.columns else 0.0
    df = df[["open_time", "Open", "High", "Low", "Close", "Volume"]]
    df = df[~df["open_time"].duplicated(keep="first")]
    df.sort_values("open_time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.dropna(inplace=True)
    df.to_csv(csv_file, index=False)
    print(f"  Saved → {csv_file}  ({len(df):,} candles)")
    return df


def load_yfinance_fallback():
    """Download GC=F (Gold Futures) 5m from yfinance — last ~70 days only."""
    import yfinance as yf
    print("Falling back to yfinance GC=F 5m (last ~70 days)...")
    df = yf.download("GC=F", period="60d", interval="5m", auto_adjust=True, progress=False)
    if hasattr(df.columns, "levels"):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[~df.index.duplicated(keep="first")]
    df.sort_index(inplace=True)
    df.dropna(inplace=True)
    df.index.name = "open_time"
    df.to_csv(XAUUSD_CSV)
    print(f"  {len(df):,} candles | {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}")
    print("  NOTE: Only 70 days of 5m data available free. Set TWELVEDATA_API_KEY for full 2-year history.")
    return df


def load_xauusd():
    if os.path.exists(XAUUSD_CSV):
        print(f"Loading {XAUUSD_CSV}...", end=" ")
        df = pd.read_csv(XAUUSD_CSV, parse_dates=["open_time"])
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        df = df[~df["open_time"].duplicated(keep="first")]
        df.sort_values("open_time", inplace=True)
        df.set_index("open_time", inplace=True)
        df.dropna(inplace=True)
        print(f"{len(df):,} candles | {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}")
        return df

    if TWELVEDATA_API_KEY != "YOUR_KEY_HERE":
        df = download_xauusd(TWELVEDATA_API_KEY, days=730)
        if df is not None and not df.empty:
            df.set_index("open_time", inplace=True)
            return df
        print("  Twelve Data download failed — falling back to yfinance.")

    return load_yfinance_fallback()


# ── LOAD + SLICE 2 YEARS ──────────────────────────────────────────────────────
df_full = load_xauusd()
cutoff  = df_full.index[-1] - pd.DateOffset(years=2)
df_full = df_full[df_full.index >= cutoff].copy()
print(f"Using: {df_full.index[0].strftime('%Y-%m-%d')} → {df_full.index[-1].strftime('%Y-%m-%d')} | {len(df_full):,} candles\n")

# ── INDICATOR BUILDER ─────────────────────────────────────────────────────────
def build_indicators(df):
    d = df.copy()
    c = d["Close"]

    for s in [5, 8, 9, 13, 20, 21, 50, 100, 200]:
        d[f"ema{s}"] = c.ewm(span=s, adjust=False).mean()

    e1 = c.ewm(span=9,  adjust=False).mean()
    e2 = e1.ewm(span=9, adjust=False).mean()
    e3 = e2.ewm(span=9, adjust=False).mean()
    d["tema9"]  = 3*e1 - 3*e2 + e3
    e1b = c.ewm(span=21, adjust=False).mean()
    e2b = e1b.ewm(span=21, adjust=False).mean()
    e3b = e2b.ewm(span=21, adjust=False).mean()
    d["tema21"] = 3*e1b - 3*e2b + e3b

    def wma(s, n):
        return s.rolling(n).apply(lambda x: np.dot(x, np.arange(1, len(x)+1)) /
                                  np.arange(1, len(x)+1).sum(), raw=True)
    d["hma9"]  = wma(2*wma(c,4)  - wma(c,9),  3)
    d["hma20"] = wma(2*wma(c,10) - wma(c,20), 4)

    def rsi(s, n):
        delta = s.diff()
        g = delta.clip(lower=0).ewm(com=n-1, adjust=False).mean()
        l = (-delta.clip(upper=0)).ewm(com=n-1, adjust=False).mean()
        return 100 - 100/(1 + g/l)
    d["rsi7"]  = rsi(c, 7)
    d["rsi14"] = rsi(c, 14)

    r = d["rsi14"]
    d["stochrsi_k"] = 100*(r - r.rolling(14).min())/(r.rolling(14).max() - r.rolling(14).min() + 1e-9)
    d["stochrsi_d"] = d["stochrsi_k"].rolling(3).mean()

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    d["macd"]     = ema12 - ema26
    d["macd_sig"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"]= d["macd"] - d["macd_sig"]

    bb_mid      = c.rolling(20).mean()
    bb_std      = c.rolling(20).std()
    d["bb_mid"] = bb_mid
    d["bb_up"]  = bb_mid + 2*bb_std
    d["bb_lo"]  = bb_mid - 2*bb_std
    d["bb_width"]=(d["bb_up"] - d["bb_lo"]) / (bb_mid + 1e-9)

    high_low = d["High"] - d["Low"]
    high_pc  = (d["High"] - c.shift()).abs()
    low_pc   = (d["Low"]  - c.shift()).abs()
    tr       = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
    atr14    = tr.ewm(span=14, adjust=False).mean()
    d["atr14"]  = atr14
    kc_mid      = c.ewm(span=20, adjust=False).mean()
    d["kc_up"]  = kc_mid + 1.5*atr14
    d["kc_lo"]  = kc_mid - 1.5*atr14

    d["don_hi"] = d["High"].rolling(20).max()
    d["don_lo"] = d["Low"].rolling(20).min()

    dm_plus  = ((d["High"]-d["High"].shift()).clip(lower=0)).where(
                (d["High"]-d["High"].shift()) > (d["Low"].shift()-d["Low"]), 0)
    dm_minus = ((d["Low"].shift()-d["Low"]).clip(lower=0)).where(
                (d["Low"].shift()-d["Low"]) > (d["High"]-d["High"].shift()), 0)
    atr14s   = tr.ewm(span=14, adjust=False).mean()
    di_plus  = 100*dm_plus.ewm(span=14, adjust=False).mean()/(atr14s+1e-9)
    di_minus = 100*dm_minus.ewm(span=14, adjust=False).mean()/(atr14s+1e-9)
    dx       = 100*(di_plus-di_minus).abs()/(di_plus+di_minus+1e-9)
    d["adx"]      = dx.ewm(span=14, adjust=False).mean()
    d["di_plus"]  = di_plus
    d["di_minus"] = di_minus

    tp20       = (d["High"]+d["Low"]+c)/3
    d["cci20"] = (tp20 - tp20.rolling(20).mean())/(0.015*tp20.rolling(20).std()+1e-9)

    hi14 = d["High"].rolling(14).max()
    lo14 = d["Low"].rolling(14).min()
    d["wpr"] = -100*(hi14-c)/(hi14-lo14+1e-9)

    d["tp"]      = (d["High"]+d["Low"]+c)/3
    d["cum_tpv"] = (d["tp"]*d["Volume"]).cumsum()
    d["cum_vol"] = d["Volume"].cumsum()
    d["vwap"]    = d["cum_tpv"]/(d["cum_vol"]+1e-9)

    d["obv"]     = (np.sign(c.diff())*d["Volume"]).cumsum()
    d["obv_ema"] = d["obv"].ewm(span=20, adjust=False).mean()

    ha_c         = (d["Open"]+d["High"]+d["Low"]+c)/4
    ha_o         = (d["Open"].shift()+c.shift())/2
    d["ha_bull"] = (ha_c > ha_o).astype(int)

    hl2          = (d["High"]+d["Low"])/2
    atr10        = tr.ewm(span=10, adjust=False).mean()
    d["st_up"]   = hl2 - 3*atr10
    d["st_dn"]   = hl2 + 3*atr10

    d["vol_ma20"]  = d["Volume"].rolling(20).mean()
    d["vol_ratio"] = d["Volume"]/(d["vol_ma20"]+1e-9)

    d["squeeze"]   = (d["bb_up"] < d["kc_up"]) & (d["bb_lo"] > d["kc_lo"])
    mid12          = (d["High"].rolling(12).max()+d["Low"].rolling(12).min())/2
    d["sqz_mom"]   = c - (mid12 + c.rolling(12).mean())/2

    d["sar_proxy"] = c.ewm(span=5, adjust=False).mean()

    return d.dropna()


# ── SIGNAL GENERATORS ─────────────────────────────────────────────────────────
def _sig(d, up, dn):
    s = pd.Series(0, index=d.index)
    s[up] = 1; s[dn] = -1
    return s

def sig_ema_9_21(d):
    return _sig(d,
        (d["ema9"]>d["ema21"])&(d["ema9"].shift()<=d["ema21"].shift()),
        (d["ema9"]<d["ema21"])&(d["ema9"].shift()>=d["ema21"].shift()))

def sig_ema_9_21_rsi(d):
    return _sig(d,
        (d["ema9"]>d["ema21"])&(d["ema9"].shift()<=d["ema21"].shift())&(d["rsi14"]<65),
        (d["ema9"]<d["ema21"])&(d["ema9"].shift()>=d["ema21"].shift())&(d["rsi14"]>35))

def sig_ema_5_13_50(d):
    bull = (d["ema5"]>d["ema13"])&(d["ema13"]>d["ema50"])
    bear = (d["ema5"]<d["ema13"])&(d["ema13"]<d["ema50"])
    return _sig(d, bull&~bull.shift().fillna(False), bear&~bear.shift().fillna(False))

def sig_ema_13_50(d):
    return _sig(d,
        (d["ema13"]>d["ema50"])&(d["ema13"].shift()<=d["ema50"].shift()),
        (d["ema13"]<d["ema50"])&(d["ema13"].shift()>=d["ema50"].shift()))

def sig_ema_21_50_200(d):
    bull = (d["ema21"]>d["ema50"])&(d["ema50"]>d["ema200"])
    bear = (d["ema21"]<d["ema50"])&(d["ema50"]<d["ema200"])
    return _sig(d, bull&~bull.shift().fillna(False), bear&~bear.shift().fillna(False))

def sig_tema_cross(d):
    return _sig(d,
        (d["tema9"]>d["tema21"])&(d["tema9"].shift()<=d["tema21"].shift()),
        (d["tema9"]<d["tema21"])&(d["tema9"].shift()>=d["tema21"].shift()))

def sig_hma_trend(d):
    return _sig(d,
        (d["hma9"]>d["hma20"])&(d["hma9"].shift()<=d["hma20"].shift()),
        (d["hma9"]<d["hma20"])&(d["hma9"].shift()>=d["hma20"].shift()))

def sig_macd_hist(d):
    return _sig(d,
        (d["macd_hist"]>0)&(d["macd_hist"].shift()<=0),
        (d["macd_hist"]<0)&(d["macd_hist"].shift()>=0))

def sig_macd_hist_ema(d):
    return _sig(d,
        (d["macd_hist"]>0)&(d["macd_hist"].shift()<=0)&(d["Close"]>d["ema50"]),
        (d["macd_hist"]<0)&(d["macd_hist"].shift()>=0)&(d["Close"]<d["ema50"]))

def sig_macd_sig_cross(d):
    return _sig(d,
        (d["macd"]>d["macd_sig"])&(d["macd"].shift()<=d["macd_sig"].shift()),
        (d["macd"]<d["macd_sig"])&(d["macd"].shift()>=d["macd_sig"].shift()))

def sig_macd_zero_cross(d):
    return _sig(d,
        (d["macd"]>0)&(d["macd"].shift()<=0),
        (d["macd"]<0)&(d["macd"].shift()>=0))

def sig_rsi_reversal(d):
    s = pd.Series(0, index=d.index)
    s[d["rsi14"]<25] = 1; s[d["rsi14"]>75] = -1
    return s

def sig_rsi_reversal_fast(d):
    s = pd.Series(0, index=d.index)
    s[d["rsi7"]<20] = 1; s[d["rsi7"]>80] = -1
    return s

def sig_rsi_momentum(d):
    return _sig(d,
        (d["rsi14"]>55)&(d["rsi14"].shift()<=55),
        (d["rsi14"]<45)&(d["rsi14"].shift()>=45))

def sig_stochrsi(d):
    return _sig(d,
        (d["stochrsi_k"]>d["stochrsi_d"])&(d["stochrsi_k"].shift()<=d["stochrsi_d"].shift())&(d["stochrsi_k"]<50),
        (d["stochrsi_k"]<d["stochrsi_d"])&(d["stochrsi_k"].shift()>=d["stochrsi_d"].shift())&(d["stochrsi_k"]>50))

def sig_stochrsi_extreme(d):
    return _sig(d,
        (d["stochrsi_k"]>d["stochrsi_d"])&(d["stochrsi_k"].shift()<=d["stochrsi_d"].shift())&(d["stochrsi_k"]<20),
        (d["stochrsi_k"]<d["stochrsi_d"])&(d["stochrsi_k"].shift()>=d["stochrsi_d"].shift())&(d["stochrsi_k"]>80))

def sig_bb_bounce(d):
    return _sig(d,
        (d["Close"]<d["bb_lo"])&(d["Close"].shift()>=d["bb_lo"].shift()),
        (d["Close"]>d["bb_up"])&(d["Close"].shift()<=d["bb_up"].shift()))

def sig_bb_breakout(d):
    return _sig(d,
        ~d["squeeze"]&d["squeeze"].shift().fillna(True)&(d["Close"]>d["bb_up"]),
        ~d["squeeze"]&d["squeeze"].shift().fillna(True)&(d["Close"]<d["bb_lo"]))

def sig_keltner_break(d):
    return _sig(d,
        (d["Close"]>d["kc_up"])&(d["Close"].shift()<=d["kc_up"].shift()),
        (d["Close"]<d["kc_lo"])&(d["Close"].shift()>=d["kc_lo"].shift()))

def sig_donchian_break(d):
    return _sig(d,
        (d["Close"]>=d["don_hi"].shift())&(d["vol_ratio"]>1.2),
        (d["Close"]<=d["don_lo"].shift())&(d["vol_ratio"]>1.2))

def sig_vwap_cross(d):
    return _sig(d,
        (d["Close"]>d["vwap"])&(d["Close"].shift()<=d["vwap"].shift()),
        (d["Close"]<d["vwap"])&(d["Close"].shift()>=d["vwap"].shift()))

def sig_vwap_bounce(d):
    bu = (d["Low"]<d["vwap"])&(d["Close"]>d["vwap"])&(d["Close"]>d["ema21"])
    bd = (d["High"]>d["vwap"])&(d["Close"]<d["vwap"])&(d["Close"]<d["ema21"])
    return _sig(d, bu&~bu.shift().fillna(False), bd&~bd.shift().fillna(False))

def sig_adx_ema(d):
    bull = (d["adx"]>25)&(d["di_plus"]>d["di_minus"])&(d["Close"]>d["ema50"])
    bear = (d["adx"]>25)&(d["di_minus"]>d["di_plus"])&(d["Close"]<d["ema50"])
    return _sig(d, bull&~bull.shift().fillna(False), bear&~bear.shift().fillna(False))

def sig_cci_reversal(d):
    return _sig(d,
        (d["cci20"]>-100)&(d["cci20"].shift()<=-100),
        (d["cci20"]<100)&(d["cci20"].shift()>=100))

def sig_cci_extreme(d):
    return _sig(d,
        (d["cci20"]>-200)&(d["cci20"].shift()<=-200),
        (d["cci20"]<200)&(d["cci20"].shift()>=200))

def sig_wpr(d):
    return _sig(d,
        (d["wpr"]>-80)&(d["wpr"].shift()<=-80),
        (d["wpr"]<-20)&(d["wpr"].shift()>=-20))

def sig_obv_trend(d):
    return _sig(d,
        (d["obv"]>d["obv_ema"])&(d["obv"].shift()<=d["obv_ema"].shift()),
        (d["obv"]<d["obv_ema"])&(d["obv"].shift()>=d["obv_ema"].shift()))

def sig_heikin_ashi(d):
    return _sig(d,
        (d["ha_bull"]==1)&(d["ha_bull"].shift()==0),
        (d["ha_bull"]==0)&(d["ha_bull"].shift()==1))

def sig_sar_proxy(d):
    return _sig(d,
        (d["Close"]>d["sar_proxy"])&(d["Close"].shift()<=d["sar_proxy"].shift()),
        (d["Close"]<d["sar_proxy"])&(d["Close"].shift()>=d["sar_proxy"].shift()))

def sig_squeeze_momentum(d):
    return _sig(d,
        (d["sqz_mom"]>0)&(d["sqz_mom"].shift()<=0)&~d["squeeze"],
        (d["sqz_mom"]<0)&(d["sqz_mom"].shift()>=0)&~d["squeeze"])

def sig_supertrend(d):
    bull = d["Close"]>d["st_up"]
    bear = d["Close"]<d["st_dn"]
    return _sig(d, bull&~bull.shift().fillna(False), bear&~bear.shift().fillna(False))

def sig_vol_breakout(d):
    return _sig(d,
        (d["Close"]>d["High"].rolling(10).max().shift())&(d["vol_ratio"]>1.5),
        (d["Close"]<d["Low"].rolling(10).min().shift())&(d["vol_ratio"]>1.5))

def sig_multi_consensus(d):
    lc = (d["ema9"]>d["ema21"])&(d["rsi14"]>50)&(d["macd_hist"]>0)&(d["Close"]>d["vwap"])
    sc = (d["ema9"]<d["ema21"])&(d["rsi14"]<50)&(d["macd_hist"]<0)&(d["Close"]<d["vwap"])
    return _sig(d, lc&~lc.shift().fillna(False), sc&~sc.shift().fillna(False))

def sig_rsi_macd_ema(d):
    lc = (d["rsi14"]>50)&(d["macd_hist"]>0)&(d["Close"]>d["ema100"])
    sc = (d["rsi14"]<50)&(d["macd_hist"]<0)&(d["Close"]<d["ema100"])
    return _sig(d, lc&~lc.shift().fillna(False), sc&~sc.shift().fillna(False))

def sig_price_action_vol(d):
    cp = (d["Close"]-d["Open"])/d["Open"]
    return _sig(d,
        (cp>0.001)&(d["vol_ratio"]>1.8)&(d["Close"]>d["ema21"]),
        (cp<-0.001)&(d["vol_ratio"]>1.8)&(d["Close"]<d["ema21"]))

def sig_ema_ribbon(d):
    bull = (d["ema5"]>d["ema8"])&(d["ema8"]>d["ema13"])&(d["ema13"]>d["ema21"])
    bear = (d["ema5"]<d["ema8"])&(d["ema8"]<d["ema13"])&(d["ema13"]<d["ema21"])
    return _sig(d, bull&~bull.shift().fillna(False), bear&~bear.shift().fillna(False))

def sig_atr_breakout(d):
    pc = d["Close"].shift()
    return _sig(d,
        d["Close"] > pc + d["atr14"],
        d["Close"] < pc - d["atr14"])

STRATEGIES = {
    "EMA_9_21":          sig_ema_9_21,
    "EMA_9_21+RSI":      sig_ema_9_21_rsi,
    "EMA_5_13_50":       sig_ema_5_13_50,
    "EMA_13_50":         sig_ema_13_50,
    "EMA_21_50_200":     sig_ema_21_50_200,
    "TEMA_Cross":        sig_tema_cross,
    "HullMA_9_20":       sig_hma_trend,
    "MACD_Hist":         sig_macd_hist,
    "MACD_Hist+EMA50":   sig_macd_hist_ema,
    "MACD_SigCross":     sig_macd_sig_cross,
    "MACD_ZeroCross":    sig_macd_zero_cross,
    "RSI_Reversal":      sig_rsi_reversal,
    "RSI_Fast_Reversal": sig_rsi_reversal_fast,
    "RSI_Momentum":      sig_rsi_momentum,
    "StochRSI_Cross":    sig_stochrsi,
    "StochRSI_Extreme":  sig_stochrsi_extreme,
    "BB_Bounce":         sig_bb_bounce,
    "BB_Breakout":       sig_bb_breakout,
    "Keltner_Break":     sig_keltner_break,
    "Donchian_Break":    sig_donchian_break,
    "VWAP_Cross":        sig_vwap_cross,
    "VWAP_Bounce":       sig_vwap_bounce,
    "ADX_EMA":           sig_adx_ema,
    "CCI_Reversal":      sig_cci_reversal,
    "CCI_Extreme":       sig_cci_extreme,
    "Williams_%R":       sig_wpr,
    "OBV_Trend":         sig_obv_trend,
    "Heikin_Ashi":       sig_heikin_ashi,
    "SAR_Proxy":         sig_sar_proxy,
    "Squeeze_Momentum":  sig_squeeze_momentum,
    "SuperTrend":        sig_supertrend,
    "Vol_Breakout":      sig_vol_breakout,
    "Multi_Consensus":   sig_multi_consensus,
    "RSI_MACD_EMA":      sig_rsi_macd_ema,
    "Price_Action_Vol":  sig_price_action_vol,
    "EMA_Ribbon":        sig_ema_ribbon,
    "ATR_Breakout":      sig_atr_breakout,
}

# ── LOT CALCULATOR ────────────────────────────────────────────────────────────
def calc_lots(capital, price, sl_pct):
    risk_usd       = capital * RISK_PCT
    sl_usd_per_lot = price * sl_pct       # $ loss per 1 lot if SL hit (1 lot = 1 oz)
    lots_ideal     = risk_usd / sl_usd_per_lot
    lots           = max(MIN_LOT, round(lots_ideal / LOT_STEP) * LOT_STEP)
    lots           = round(lots, 2)
    margin_req     = lots * price / LEVERAGE
    if margin_req > capital:
        lots = max(MIN_LOT, round((capital * LEVERAGE / price) / LOT_STEP) * LOT_STEP)
        lots = round(lots, 2)
    return lots

# ── BACKTEST ENGINE ───────────────────────────────────────────────────────────
def backtest(df, signals, sl_pct, rr):
    cap        = CAPITAL
    pos        = None
    entry_fill = sl_p = tp_p = lots = 0.0
    trades     = []
    tp_pct     = sl_pct * rr
    peak       = cap

    for i in range(1, len(df)):
        if cap <= 0:
            break
        price = float(df["Close"].iloc[i])
        ts    = df.index[i]
        sig   = int(signals.iloc[i])

        if pos is not None:
            hit_tp = (pos==1 and price>=tp_p) or (pos==-1 and price<=tp_p)
            hit_sl = (pos==1 and price<=sl_p) or (pos==-1 and price>=sl_p)
            if hit_tp or hit_sl:
                exit_px = tp_p if hit_tp else sl_p
                pnl     = lots * abs(exit_px - entry_fill) * (1 if hit_tp else -1)
                cap     = max(0.0, cap + pnl)
                peak    = max(peak, cap)
                trades.append({
                    "exit_time": ts,
                    "result":    "TP" if hit_tp else "SL",
                    "lots":      lots,
                    "pnl":       round(pnl, 4),
                    "capital":   round(cap, 4),
                    "peak":      round(peak, 4),
                })
                pos = None

        if pos is None and sig != 0 and cap > 0:
            lots       = calc_lots(cap, price, sl_pct)
            entry_fill = price*(1+SPREAD) if sig==1 else price*(1-SPREAD)
            sl_p       = entry_fill*(1-sl_pct) if sig==1 else entry_fill*(1+sl_pct)
            tp_p       = entry_fill*(1+tp_pct) if sig==1 else entry_fill*(1-tp_pct)
            pos        = sig

    return pd.DataFrame(trades)

# ── STATS ─────────────────────────────────────────────────────────────────────
def calc_stats(t):
    if t.empty or len(t) < 10:
        return None
    wins  = (t["result"]=="TP").sum()
    final = t["capital"].iloc[-1]
    tot   = (final - CAPITAL) / CAPITAL * 100
    # max drawdown
    caps  = pd.concat([pd.Series([CAPITAL]), t["capital"].reset_index(drop=True)])
    peak  = caps.cummax()
    dd    = (caps - peak) / peak * 100
    mdd   = dd.min()
    mdd_usd = (caps - peak).min()
    # monthly
    t = t.copy()
    t["month"] = pd.to_datetime(t["exit_time"]).dt.to_period("M")
    prev_cap = CAPITAL
    mo_rets  = []
    for m in sorted(t["month"].unique()):
        mt = t[t["month"]==m]
        end = mt["capital"].iloc[-1]
        mo_rets.append((end - prev_cap)/prev_cap*100)
        prev_cap = end
    mr = pd.Series(mo_rets)
    return {
        "trades":   len(t),
        "win%":     round(wins/len(t)*100, 1),
        "total%":   round(tot, 1),
        "avg_mo%":  round(mr.mean(), 2),
        "best_mo%": round(mr.max(), 2),
        "worst_mo%":round(mr.min(), 2),
        "mdd%":     round(mdd, 1),
        "mdd_$":    round(abs(mdd_usd), 2),
        "final":    round(final, 2),
    }

# ── RUN ───────────────────────────────────────────────────────────────────────
SL_GRID = [0.002, 0.003, 0.005, 0.008, 0.010]
RR_GRID = [1.5, 2.0, 2.5, 3.0, 3.5]

print("Building indicators...")
df = build_indicators(df_full)
print(f"Done. {len(df):,} candles.\n")

total   = len(STRATEGIES) * len(SL_GRID) * len(RR_GRID)
results = []
run_i   = 0
print(f"Testing {len(STRATEGIES)} strategies × {len(SL_GRID)} SL × {len(RR_GRID)} RR = {total} combos...\n")

for name, fn in STRATEGIES.items():
    sigs = fn(df)
    for sl in SL_GRID:
        for rr in RR_GRID:
            run_i += 1
            t = backtest(df, sigs, sl, rr)
            s = calc_stats(t)
            if s:
                s["label"] = f"{name}|SL{sl*100:.1f}%|RR{rr}"
                s["name"]  = name
                s["sl"]    = sl
                s["rr"]    = rr
                results.append(s)
            if run_i % 37 == 0:
                print(f"  [{run_i:4d}/{total}]...", end="\r")

print(f"  [{total}/{total}] done.   \n")

res = pd.DataFrame(results).sort_values("avg_mo%", ascending=False).reset_index(drop=True)

# ── RANKINGS ──────────────────────────────────────────────────────────────────
print("═"*120)
print(f"  XAUUSD 5m  |  $200 Capital  |  1:1000 Leverage  |  0.025% Spread  |  Min Lot 0.01")
print("═"*120)
print(f"  {'#':<4} {'STRATEGY':<40} {'TRADES':>7} {'WIN%':>6} {'AVG/MO%':>9} {'BEST/MO':>9} {'WORST/MO':>9} {'MDD%':>7} {'MDD$':>7} {'FINAL$':>9}")
print("═"*120)
for i, row in res.head(20).iterrows():
    mark = " ✓" if row["avg_mo%"] > 0 else "  "
    print(f"  {i+1:<4} {row['label']:<40} {int(row['trades']):>7} {row['win%']:>5.1f}% "
          f"{row['avg_mo%']:>9.2f}% {row['best_mo%']:>8.2f}% {row['worst_mo%']:>8.2f}% "
          f"{row['mdd%']:>7.1f}% ${row['mdd_$']:>6.2f} ${row['final']:>8.2f}{mark}")
print("═"*120)
print(f"\n  Profitable: {(res['avg_mo%']>0).sum()} / {len(res)}  |  Survived (final>0): {(res['final']>0).sum()} / {len(res)}\n")

# ── TOP 5 MONTHLY BREAKDOWN ───────────────────────────────────────────────────
print("═"*80)
print("  PER-MONTH BREAKDOWN — TOP 5 STRATEGIES")
print(f"  XAUUSD 5m | $200 | Spread 0.025% | Lot 0.01 min | 1:1000")
print("═"*80)

for rank in range(min(5, len(res))):
    row  = res.iloc[rank]
    sigs = STRATEGIES[row["name"]](df)
    t    = backtest(df, sigs, row["sl"], row["rr"])

    caps    = pd.concat([pd.Series([CAPITAL]), t["capital"].reset_index(drop=True)])
    peak    = caps.cummax()
    mdd     = ((caps - peak)/peak*100).min()
    mdd_usd = (caps - peak).min()

    t = t.copy()
    t["month"] = pd.to_datetime(t["exit_time"]).dt.to_period("M")
    prev_cap = CAPITAL
    mo_rows  = []
    for m in sorted(t["month"].unique()):
        mt  = t[t["month"]==m]
        end = mt["capital"].iloc[-1]
        mo_rows.append({
            "month":   str(m),
            "trades":  len(mt),
            "win%":    (mt["result"]=="TP").sum()/len(mt)*100,
            "ret%":    (end - prev_cap)/prev_cap*100,
            "capital": end,
            "avg_lot": mt["lots"].mean(),
        })
        prev_cap = end

    mo = pd.DataFrame(mo_rows)
    rets = mo["ret%"]

    print(f"\n  #{rank+1}  {row['label']}")
    print(f"  Trades: {int(row['trades'])} | Win: {row['win%']}% | "
          f"Final: ${row['final']:,.2f} | Total: {row['total%']:+.1f}%")
    print(f"  Max Drawdown: {mdd:.1f}%  (${abs(mdd_usd):.2f})  |  "
          f"Positive months: {(rets>0).sum()}/{len(rets)}")
    print(f"\n  {'Month':<10} {'Trades':>7} {'Win%':>6} {'Return%':>10} {'Capital':>10} {'AvgLot':>8}  Bar")
    print(f"  {'-'*72}")
    for _, r in mo.iterrows():
        ret = r["ret%"]
        bar = ("█"*min(int(abs(ret)/3), 30)) if ret>=0 else ("▓"*min(int(abs(ret)/3), 30))
        sgn = "+" if ret>=0 else ""
        print(f"  {r['month']:<10} {int(r['trades']):>7} {r['win%']:>5.1f}% "
              f"{sgn}{ret:>9.2f}% ${r['capital']:>9.2f} {r['avg_lot']:>8.3f}  {bar}")
    print(f"  {'-'*72}")
    print(f"  Avg: {rets.mean():+.2f}%/mo | Best: {rets.max():+.2f}% | Worst: {rets.min():+.2f}%")

print("\n" + "═"*80)
w = res.iloc[0]
print(f"  WINNER : {w['label']}")
print(f"  Avg/Mo : {w['avg_mo%']:+.2f}%  |  Best: {w['best_mo%']:+.2f}%  |  MDD: {w['mdd%']:.1f}% (${w['mdd_$']:.2f})")
print(f"  Final  : ${w['final']:,.2f}  from $200  ({w['total%']:+.1f}% over 2 years)")
print("═"*80)
