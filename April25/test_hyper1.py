"""
HYPER STRATEGY SEARCH — BTC/USDT 5m  |  2 Years  |  0.03% Spread
══════════════════════════════════════════════════════════════════
Tests 25+ distinct strategies, ranks by monthly return.
Capital: $100  |  Leverage: 1:1000  |  Risk: 1%/trade  |  Spread: 0.03%
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from data_loader import load_data

# ── CONFIG ────────────────────────────────────────────────────────────────────
CAPITAL   = 100.0
RISK_PCT  = 0.01
SPREAD    = 0.0003   # 0.03% one-way spread applied at entry
LEVERAGE  = 1000

# ── LOAD DATA (last 2 years) ──────────────────────────────────────────────────
df_full = load_data()
cutoff  = df_full.index[-1] - pd.DateOffset(years=2)
df_full = df_full[df_full.index >= cutoff].copy()
print(f"Using: {df_full.index[0].strftime('%Y-%m-%d')} → {df_full.index[-1].strftime('%Y-%m-%d')} | {len(df_full):,} candles\n")

# ── INDICATOR BUILDER ─────────────────────────────────────────────────────────
def build_indicators(df):
    d = df.copy()
    c = d["Close"]

    # EMAs
    for s in [5, 8, 9, 13, 20, 21, 50, 100, 200]:
        d[f"ema{s}"] = c.ewm(span=s, adjust=False).mean()

    # TEMA
    e1 = c.ewm(span=9, adjust=False).mean()
    e2 = e1.ewm(span=9, adjust=False).mean()
    e3 = e2.ewm(span=9, adjust=False).mean()
    d["tema9"] = 3*e1 - 3*e2 + e3
    e1b = c.ewm(span=21, adjust=False).mean()
    e2b = e1b.ewm(span=21, adjust=False).mean()
    e3b = e2b.ewm(span=21, adjust=False).mean()
    d["tema21"] = 3*e1b - 3*e2b + e3b

    # HMA (Hull MA)
    def hma(s, n):
        wma_half = s.rolling(n//2).apply(lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(), raw=True)
        wma_full = s.rolling(n).apply(lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(), raw=True)
        diff = 2 * wma_half - wma_full
        return diff.rolling(int(np.sqrt(n))).apply(lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(), raw=True)
    d["hma9"]  = hma(c, 9)
    d["hma20"] = hma(c, 20)

    # RSI
    def rsi(s, n):
        delta = s.diff()
        g = delta.clip(lower=0).ewm(com=n-1, adjust=False).mean()
        l = (-delta.clip(upper=0)).ewm(com=n-1, adjust=False).mean()
        return 100 - 100/(1 + g/l)
    d["rsi7"]  = rsi(c, 7)
    d["rsi14"] = rsi(c, 14)

    # Stochastic RSI (14-period rsi, 3-period smooth)
    r = d["rsi14"]
    rmin = r.rolling(14).min()
    rmax = r.rolling(14).max()
    d["stochrsi_k"] = 100 * (r - rmin) / (rmax - rmin + 1e-9)
    d["stochrsi_d"] = d["stochrsi_k"].rolling(3).mean()

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    d["macd"]     = ema12 - ema26
    d["macd_sig"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"]= d["macd"] - d["macd_sig"]

    # Bollinger Bands (20,2)
    bb_mid        = c.rolling(20).mean()
    bb_std        = c.rolling(20).std()
    d["bb_mid"]   = bb_mid
    d["bb_up"]    = bb_mid + 2 * bb_std
    d["bb_lo"]    = bb_mid - 2 * bb_std
    d["bb_width"] = (d["bb_up"] - d["bb_lo"]) / (bb_mid + 1e-9)

    # Keltner Channel (20 EMA ± 1.5*ATR)
    atr14         = (d["High"] - d["Low"]).ewm(span=14, adjust=False).mean()
    d["atr14"]    = atr14
    kc_mid        = c.ewm(span=20, adjust=False).mean()
    d["kc_up"]    = kc_mid + 1.5 * atr14
    d["kc_lo"]    = kc_mid - 1.5 * atr14

    # Donchian Channel (20)
    d["don_hi"]   = d["High"].rolling(20).max()
    d["don_lo"]   = d["Low"].rolling(20).min()

    # ADX + DI
    high_low      = d["High"] - d["Low"]
    high_pc       = (d["High"] - d["Close"].shift()).abs()
    low_pc        = (d["Low"]  - d["Close"].shift()).abs()
    tr            = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
    dm_plus       = ((d["High"] - d["High"].shift()).clip(lower=0)).where(
                     (d["High"] - d["High"].shift()) > (d["Low"].shift() - d["Low"]), 0)
    dm_minus      = ((d["Low"].shift() - d["Low"]).clip(lower=0)).where(
                     (d["Low"].shift() - d["Low"]) > (d["High"] - d["High"].shift()), 0)
    atr14s        = tr.ewm(span=14, adjust=False).mean()
    di_plus       = 100 * dm_plus.ewm(span=14, adjust=False).mean() / (atr14s + 1e-9)
    di_minus      = 100 * dm_minus.ewm(span=14, adjust=False).mean() / (atr14s + 1e-9)
    dx            = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-9)
    d["adx"]      = dx.ewm(span=14, adjust=False).mean()
    d["di_plus"]  = di_plus
    d["di_minus"] = di_minus

    # CCI (20)
    tp20          = (d["High"] + d["Low"] + c) / 3
    d["cci20"]    = (tp20 - tp20.rolling(20).mean()) / (0.015 * tp20.rolling(20).std() + 1e-9)

    # Williams %R (14)
    hi14          = d["High"].rolling(14).max()
    lo14          = d["Low"].rolling(14).min()
    d["wpr"]      = -100 * (hi14 - c) / (hi14 - lo14 + 1e-9)

    # VWAP (rolling daily reset)
    d["tp"]       = (d["High"] + d["Low"] + c) / 3
    d["cum_tpv"]  = (d["tp"] * d["Volume"]).cumsum()
    d["cum_vol"]  = d["Volume"].cumsum()
    d["vwap"]     = d["cum_tpv"] / (d["cum_vol"] + 1e-9)

    # Parabolic SAR (simplified vectorized approximation)
    # We use a simple price-acceleration proxy: close vs 5-bar shifted EMA
    d["sar_proxy"] = c.ewm(span=5, adjust=False).mean()

    # OBV
    d["obv"]       = (np.sign(c.diff()) * d["Volume"]).cumsum()
    d["obv_ema"]   = d["obv"].ewm(span=20, adjust=False).mean()

    # Heikin Ashi
    ha_c           = (d["Open"] + d["High"] + d["Low"] + c) / 4
    ha_o           = (d["Open"].shift() + d["Close"].shift()) / 2
    d["ha_close"]  = ha_c
    d["ha_open"]   = ha_o
    d["ha_bull"]   = (ha_c > ha_o).astype(int)

    # SuperTrend (10, 3)
    hl2           = (d["High"] + d["Low"]) / 2
    atr10         = tr.ewm(span=10, adjust=False).mean()
    d["st_up"]    = hl2 - 3 * atr10
    d["st_dn"]    = hl2 + 3 * atr10

    # Volume spike
    d["vol_ma20"]  = d["Volume"].rolling(20).mean()
    d["vol_ratio"] = d["Volume"] / (d["vol_ma20"] + 1e-9)

    # Squeeze (BB inside KC = squeeze)
    d["squeeze"]   = (d["bb_up"] < d["kc_up"]) & (d["bb_lo"] > d["kc_lo"])
    # Momentum for squeeze: 12-bar highest-high/lowest-low midpoint delta
    mid12          = (d["High"].rolling(12).max() + d["Low"].rolling(12).min()) / 2
    d["sqz_mom"]   = c - (mid12 + c.rolling(12).mean()) / 2

    return d.dropna()


# ── SIGNAL GENERATORS ─────────────────────────────────────────────────────────
# Each returns a pd.Series with values: +1 (long), -1 (short), 0 (no trade)

def sig_ema_9_21(d):
    cross_up   = (d["ema9"] > d["ema21"]) & (d["ema9"].shift() <= d["ema21"].shift())
    cross_dn   = (d["ema9"] < d["ema21"]) & (d["ema9"].shift() >= d["ema21"].shift())
    s = pd.Series(0, index=d.index)
    s[cross_up] = 1; s[cross_dn] = -1
    return s

def sig_ema_9_21_rsi(d):
    cross_up   = (d["ema9"] > d["ema21"]) & (d["ema9"].shift() <= d["ema21"].shift()) & (d["rsi14"] < 65)
    cross_dn   = (d["ema9"] < d["ema21"]) & (d["ema9"].shift() >= d["ema21"].shift()) & (d["rsi14"] > 35)
    s = pd.Series(0, index=d.index)
    s[cross_up] = 1; s[cross_dn] = -1
    return s

def sig_ema_5_13_50(d):
    bull = (d["ema5"] > d["ema13"]) & (d["ema13"] > d["ema50"])
    bear = (d["ema5"] < d["ema13"]) & (d["ema13"] < d["ema50"])
    cross_up = bull & ~bull.shift().fillna(False)
    cross_dn = bear & ~bear.shift().fillna(False)
    s = pd.Series(0, index=d.index)
    s[cross_up] = 1; s[cross_dn] = -1
    return s

def sig_ema_13_50(d):
    cross_up = (d["ema13"] > d["ema50"]) & (d["ema13"].shift() <= d["ema50"].shift())
    cross_dn = (d["ema13"] < d["ema50"]) & (d["ema13"].shift() >= d["ema50"].shift())
    s = pd.Series(0, index=d.index)
    s[cross_up] = 1; s[cross_dn] = -1
    return s

def sig_ema_21_50_200(d):
    bull = (d["ema21"] > d["ema50"]) & (d["ema50"] > d["ema200"])
    bear = (d["ema21"] < d["ema50"]) & (d["ema50"] < d["ema200"])
    cross_up = bull & ~bull.shift().fillna(False)
    cross_dn = bear & ~bear.shift().fillna(False)
    s = pd.Series(0, index=d.index)
    s[cross_up] = 1; s[cross_dn] = -1
    return s

def sig_tema_cross(d):
    cross_up = (d["tema9"] > d["tema21"]) & (d["tema9"].shift() <= d["tema21"].shift())
    cross_dn = (d["tema9"] < d["tema21"]) & (d["tema9"].shift() >= d["tema21"].shift())
    s = pd.Series(0, index=d.index)
    s[cross_up] = 1; s[cross_dn] = -1
    return s

def sig_hma_trend(d):
    cross_up = (d["hma9"] > d["hma20"]) & (d["hma9"].shift() <= d["hma20"].shift())
    cross_dn = (d["hma9"] < d["hma20"]) & (d["hma9"].shift() >= d["hma20"].shift())
    s = pd.Series(0, index=d.index)
    s[cross_up] = 1; s[cross_dn] = -1
    return s

def sig_macd_hist(d):
    flip_up = (d["macd_hist"] > 0) & (d["macd_hist"].shift() <= 0)
    flip_dn = (d["macd_hist"] < 0) & (d["macd_hist"].shift() >= 0)
    s = pd.Series(0, index=d.index)
    s[flip_up] = 1; s[flip_dn] = -1
    return s

def sig_macd_hist_ema(d):
    flip_up = (d["macd_hist"] > 0) & (d["macd_hist"].shift() <= 0) & (d["Close"] > d["ema50"])
    flip_dn = (d["macd_hist"] < 0) & (d["macd_hist"].shift() >= 0) & (d["Close"] < d["ema50"])
    s = pd.Series(0, index=d.index)
    s[flip_up] = 1; s[flip_dn] = -1
    return s

def sig_macd_sig_cross(d):
    cross_up = (d["macd"] > d["macd_sig"]) & (d["macd"].shift() <= d["macd_sig"].shift())
    cross_dn = (d["macd"] < d["macd_sig"]) & (d["macd"].shift() >= d["macd_sig"].shift())
    s = pd.Series(0, index=d.index)
    s[cross_up] = 1; s[cross_dn] = -1
    return s

def sig_macd_zero_cross(d):
    cross_up = (d["macd"] > 0) & (d["macd"].shift() <= 0)
    cross_dn = (d["macd"] < 0) & (d["macd"].shift() >= 0)
    s = pd.Series(0, index=d.index)
    s[cross_up] = 1; s[cross_dn] = -1
    return s

def sig_rsi_reversal(d):
    s = pd.Series(0, index=d.index)
    s[d["rsi14"] < 25] = 1
    s[d["rsi14"] > 75] = -1
    return s

def sig_rsi_reversal_fast(d):
    s = pd.Series(0, index=d.index)
    s[d["rsi7"] < 20] = 1
    s[d["rsi7"] > 80] = -1
    return s

def sig_rsi_momentum(d):
    cross_up = (d["rsi14"] > 55) & (d["rsi14"].shift() <= 55)
    cross_dn = (d["rsi14"] < 45) & (d["rsi14"].shift() >= 45)
    s = pd.Series(0, index=d.index)
    s[cross_up] = 1; s[cross_dn] = -1
    return s

def sig_stochrsi(d):
    cross_up = (d["stochrsi_k"] > d["stochrsi_d"]) & (d["stochrsi_k"].shift() <= d["stochrsi_d"].shift()) & (d["stochrsi_k"] < 50)
    cross_dn = (d["stochrsi_k"] < d["stochrsi_d"]) & (d["stochrsi_k"].shift() >= d["stochrsi_d"].shift()) & (d["stochrsi_k"] > 50)
    s = pd.Series(0, index=d.index)
    s[cross_up] = 1; s[cross_dn] = -1
    return s

def sig_stochrsi_extreme(d):
    cross_up = (d["stochrsi_k"] > d["stochrsi_d"]) & (d["stochrsi_k"].shift() <= d["stochrsi_d"].shift()) & (d["stochrsi_k"] < 20)
    cross_dn = (d["stochrsi_k"] < d["stochrsi_d"]) & (d["stochrsi_k"].shift() >= d["stochrsi_d"].shift()) & (d["stochrsi_k"] > 80)
    s = pd.Series(0, index=d.index)
    s[cross_up] = 1; s[cross_dn] = -1
    return s

def sig_bb_bounce(d):
    long_sig  = (d["Close"] < d["bb_lo"]) & (d["Close"].shift() >= d["bb_lo"].shift())
    short_sig = (d["Close"] > d["bb_up"]) & (d["Close"].shift() <= d["bb_up"].shift())
    s = pd.Series(0, index=d.index)
    s[long_sig] = 1; s[short_sig] = -1
    return s

def sig_bb_breakout(d):
    squeeze_end_up = ~d["squeeze"] & d["squeeze"].shift().fillna(True) & (d["Close"] > d["bb_up"])
    squeeze_end_dn = ~d["squeeze"] & d["squeeze"].shift().fillna(True) & (d["Close"] < d["bb_lo"])
    s = pd.Series(0, index=d.index)
    s[squeeze_end_up] = 1; s[squeeze_end_dn] = -1
    return s

def sig_keltner_break(d):
    break_up = (d["Close"] > d["kc_up"]) & (d["Close"].shift() <= d["kc_up"].shift())
    break_dn = (d["Close"] < d["kc_lo"]) & (d["Close"].shift() >= d["kc_lo"].shift())
    s = pd.Series(0, index=d.index)
    s[break_up] = 1; s[break_dn] = -1
    return s

def sig_donchian_break(d):
    break_up = (d["Close"] >= d["don_hi"].shift()) & (d["vol_ratio"] > 1.2)
    break_dn = (d["Close"] <= d["don_lo"].shift()) & (d["vol_ratio"] > 1.2)
    s = pd.Series(0, index=d.index)
    s[break_up] = 1; s[break_dn] = -1
    return s

def sig_vwap_cross(d):
    cross_up = (d["Close"] > d["vwap"]) & (d["Close"].shift() <= d["vwap"].shift())
    cross_dn = (d["Close"] < d["vwap"]) & (d["Close"].shift() >= d["vwap"].shift())
    s = pd.Series(0, index=d.index)
    s[cross_up] = 1; s[cross_dn] = -1
    return s

def sig_vwap_bounce(d):
    bounce_up = (d["Low"] < d["vwap"]) & (d["Close"] > d["vwap"]) & (d["Close"] > d["ema21"])
    bounce_dn = (d["High"] > d["vwap"]) & (d["Close"] < d["vwap"]) & (d["Close"] < d["ema21"])
    s = pd.Series(0, index=d.index)
    s[bounce_up & ~bounce_up.shift().fillna(False)] = 1
    s[bounce_dn & ~bounce_dn.shift().fillna(False)] = -1
    return s

def sig_adx_ema(d):
    bull = (d["adx"] > 25) & (d["di_plus"] > d["di_minus"]) & (d["Close"] > d["ema50"])
    bear = (d["adx"] > 25) & (d["di_minus"] > d["di_plus"]) & (d["Close"] < d["ema50"])
    cross_up = bull & ~bull.shift().fillna(False)
    cross_dn = bear & ~bear.shift().fillna(False)
    s = pd.Series(0, index=d.index)
    s[cross_up] = 1; s[cross_dn] = -1
    return s

def sig_cci_reversal(d):
    s = pd.Series(0, index=d.index)
    long_cross  = (d["cci20"] > -100) & (d["cci20"].shift() <= -100)
    short_cross = (d["cci20"] < 100) & (d["cci20"].shift() >= 100)
    s[long_cross]  = 1
    s[short_cross] = -1
    return s

def sig_cci_extreme(d):
    s = pd.Series(0, index=d.index)
    long_cross  = (d["cci20"] > -200) & (d["cci20"].shift() <= -200)
    short_cross = (d["cci20"] < 200) & (d["cci20"].shift() >= 200)
    s[long_cross]  = 1
    s[short_cross] = -1
    return s

def sig_wpr(d):
    long_sig  = (d["wpr"] > -80) & (d["wpr"].shift() <= -80)
    short_sig = (d["wpr"] < -20) & (d["wpr"].shift() >= -20)
    s = pd.Series(0, index=d.index)
    s[long_sig] = 1; s[short_sig] = -1
    return s

def sig_obv_trend(d):
    cross_up = (d["obv"] > d["obv_ema"]) & (d["obv"].shift() <= d["obv_ema"].shift())
    cross_dn = (d["obv"] < d["obv_ema"]) & (d["obv"].shift() >= d["obv_ema"].shift())
    s = pd.Series(0, index=d.index)
    s[cross_up] = 1; s[cross_dn] = -1
    return s

def sig_heikin_ashi(d):
    flip_bull = (d["ha_bull"] == 1) & (d["ha_bull"].shift() == 0)
    flip_bear = (d["ha_bull"] == 0) & (d["ha_bull"].shift() == 1)
    s = pd.Series(0, index=d.index)
    s[flip_bull] = 1; s[flip_bear] = -1
    return s

def sig_sar_proxy(d):
    cross_up = (d["Close"] > d["sar_proxy"]) & (d["Close"].shift() <= d["sar_proxy"].shift())
    cross_dn = (d["Close"] < d["sar_proxy"]) & (d["Close"].shift() >= d["sar_proxy"].shift())
    s = pd.Series(0, index=d.index)
    s[cross_up] = 1; s[cross_dn] = -1
    return s

def sig_squeeze_momentum(d):
    mom_up = (d["sqz_mom"] > 0) & (d["sqz_mom"].shift() <= 0) & ~d["squeeze"]
    mom_dn = (d["sqz_mom"] < 0) & (d["sqz_mom"].shift() >= 0) & ~d["squeeze"]
    s = pd.Series(0, index=d.index)
    s[mom_up] = 1; s[mom_dn] = -1
    return s

def sig_supertrend(d):
    # SuperTrend: price above st_up → bull, price below st_dn → bear
    bull = d["Close"] > d["st_up"]
    bear = d["Close"] < d["st_dn"]
    flip_bull = bull & ~bull.shift().fillna(False)
    flip_bear = bear & ~bear.shift().fillna(False)
    s = pd.Series(0, index=d.index)
    s[flip_bull] = 1; s[flip_bear] = -1
    return s

def sig_vol_breakout(d):
    # Price makes new 10-bar high/low + volume spike
    new_hi = d["Close"] > d["High"].rolling(10).max().shift()
    new_lo = d["Close"] < d["Low"].rolling(10).min().shift()
    vol_ok = d["vol_ratio"] > 1.5
    s = pd.Series(0, index=d.index)
    s[new_hi & vol_ok] = 1
    s[new_lo & vol_ok] = -1
    return s

def sig_multi_consensus(d):
    # Long: ema9>ema21, rsi14>50, macd_hist>0, close>vwap
    # Short: ema9<ema21, rsi14<50, macd_hist<0, close<vwap
    long_cond  = (d["ema9"] > d["ema21"]) & (d["rsi14"] > 50) & (d["macd_hist"] > 0) & (d["Close"] > d["vwap"])
    short_cond = (d["ema9"] < d["ema21"]) & (d["rsi14"] < 50) & (d["macd_hist"] < 0) & (d["Close"] < d["vwap"])
    flip_long  = long_cond  & ~long_cond.shift().fillna(False)
    flip_short = short_cond & ~short_cond.shift().fillna(False)
    s = pd.Series(0, index=d.index)
    s[flip_long]  = 1
    s[flip_short] = -1
    return s

def sig_rsi_macd_ema(d):
    long_cond  = (d["rsi14"] > 50) & (d["macd_hist"] > 0) & (d["Close"] > d["ema100"])
    short_cond = (d["rsi14"] < 50) & (d["macd_hist"] < 0) & (d["Close"] < d["ema100"])
    flip_long  = long_cond  & ~long_cond.shift().fillna(False)
    flip_short = short_cond & ~short_cond.shift().fillna(False)
    s = pd.Series(0, index=d.index)
    s[flip_long]  = 1
    s[flip_short] = -1
    return s

def sig_price_action_vol(d):
    # Big bullish candle (>0.2%) + volume spike + above EMA21
    candle_pct = (d["Close"] - d["Open"]) / d["Open"]
    bull_candle = (candle_pct > 0.002) & (d["vol_ratio"] > 1.8) & (d["Close"] > d["ema21"])
    bear_candle = (candle_pct < -0.002) & (d["vol_ratio"] > 1.8) & (d["Close"] < d["ema21"])
    s = pd.Series(0, index=d.index)
    s[bull_candle] = 1; s[bear_candle] = -1
    return s

def sig_ema_ribbon(d):
    # All short EMAs above all long EMAs = strong trend
    bull = (d["ema5"] > d["ema8"]) & (d["ema8"] > d["ema13"]) & (d["ema13"] > d["ema21"])
    bear = (d["ema5"] < d["ema8"]) & (d["ema8"] < d["ema13"]) & (d["ema13"] < d["ema21"])
    flip_bull = bull & ~bull.shift().fillna(False)
    flip_bear = bear & ~bear.shift().fillna(False)
    s = pd.Series(0, index=d.index)
    s[flip_bull] = 1; s[flip_bear] = -1
    return s

def sig_atr_breakout(d):
    prev_close = d["Close"].shift()
    break_up   = d["Close"] > prev_close + d["atr14"]
    break_dn   = d["Close"] < prev_close - d["atr14"]
    s = pd.Series(0, index=d.index)
    s[break_up] = 1; s[break_dn] = -1
    return s

# ── ALL STRATEGIES ────────────────────────────────────────────────────────────
STRATEGIES = {
    "EMA_9_21":           sig_ema_9_21,
    "EMA_9_21+RSI":       sig_ema_9_21_rsi,
    "EMA_5_13_50":        sig_ema_5_13_50,
    "EMA_13_50":          sig_ema_13_50,
    "EMA_21_50_200":      sig_ema_21_50_200,
    "TEMA_Cross":         sig_tema_cross,
    "HullMA_9_20":        sig_hma_trend,
    "MACD_Hist":          sig_macd_hist,
    "MACD_Hist+EMA50":    sig_macd_hist_ema,
    "MACD_SigCross":      sig_macd_sig_cross,
    "MACD_ZeroCross":     sig_macd_zero_cross,
    "RSI_Reversal":       sig_rsi_reversal,
    "RSI_Fast_Reversal":  sig_rsi_reversal_fast,
    "RSI_Momentum":       sig_rsi_momentum,
    "StochRSI_Cross":     sig_stochrsi,
    "StochRSI_Extreme":   sig_stochrsi_extreme,
    "BB_Bounce":          sig_bb_bounce,
    "BB_Breakout":        sig_bb_breakout,
    "Keltner_Break":      sig_keltner_break,
    "Donchian_Break":     sig_donchian_break,
    "VWAP_Cross":         sig_vwap_cross,
    "VWAP_Bounce":        sig_vwap_bounce,
    "ADX_EMA":            sig_adx_ema,
    "CCI_Reversal":       sig_cci_reversal,
    "CCI_Extreme":        sig_cci_extreme,
    "Williams_%R":        sig_wpr,
    "OBV_Trend":          sig_obv_trend,
    "Heikin_Ashi":        sig_heikin_ashi,
    "SAR_Proxy":          sig_sar_proxy,
    "Squeeze_Momentum":   sig_squeeze_momentum,
    "SuperTrend":         sig_supertrend,
    "Vol_Breakout":       sig_vol_breakout,
    "Multi_Consensus":    sig_multi_consensus,
    "RSI_MACD_EMA":       sig_rsi_macd_ema,
    "Price_Action_Vol":   sig_price_action_vol,
    "EMA_Ribbon":         sig_ema_ribbon,
    "ATR_Breakout":       sig_atr_breakout,
}

# ── BACKTEST ENGINE ────────────────────────────────────────────────────────────
def backtest(df, signals, sl_pct=0.003, rr=2.0, spread=SPREAD, capital=CAPITAL):
    cap    = capital
    pos    = None
    trades = []
    entry_fill = sl_p = tp_p = risk = 0.0
    entry_time = None

    for i in range(1, len(df)):
        if cap <= 0:
            break
        price = float(df["Close"].iloc[i])
        ts    = df.index[i]
        sig   = int(signals.iloc[i])

        if pos is not None:
            if pos == 1:
                hit_tp = price >= tp_p
                hit_sl = price <= sl_p
            else:
                hit_tp = price <= tp_p
                hit_sl = price >= sl_p

            if hit_tp or hit_sl:
                exit_px = tp_p if hit_tp else sl_p
                sl_dist = abs(entry_fill - sl_p)
                move    = abs(exit_px - entry_fill)
                pnl     = risk * (move / sl_dist) * (1 if hit_tp else -1)
                cap     = max(0.0, cap + pnl)
                trades.append({
                    "entry_time": entry_time,
                    "exit_time":  ts,
                    "dir":        pos,
                    "entry":      entry_fill,
                    "exit":       exit_px,
                    "result":     "TP" if hit_tp else "SL",
                    "pnl":        pnl,
                    "capital":    cap,
                })
                pos = None

        if pos is None and sig != 0 and cap > 0:
            risk  = cap * RISK_PCT
            tp_pct = sl_pct * rr
            if sig == 1:
                entry_fill = price * (1 + spread)
                sl_p       = entry_fill * (1 - sl_pct)
                tp_p       = entry_fill * (1 + tp_pct)
            else:
                entry_fill = price * (1 - spread)
                sl_p       = entry_fill * (1 + sl_pct)
                tp_p       = entry_fill * (1 - tp_pct)
            pos        = sig
            entry_time = ts

    return pd.DataFrame(trades)


def monthly_returns(trades_df, capital_start):
    if trades_df.empty:
        return pd.Series(dtype=float)
    t = trades_df.copy()
    t["month"] = pd.to_datetime(t["exit_time"]).dt.to_period("M")
    months = sorted(t["month"].unique())
    results = {}
    prev_cap = capital_start
    for m in months:
        mt = t[t["month"] == m]
        if mt.empty:
            continue
        end_cap = mt["capital"].iloc[-1]
        pct     = (end_cap - prev_cap) / prev_cap * 100
        results[str(m)] = round(pct, 2)
        prev_cap = end_cap
    return pd.Series(results)


def summary_stats(trades_df, name, sl_pct, rr):
    if trades_df.empty:
        return {"strategy": name, "trades": 0, "win_rate": 0,
                "total_ret%": 0, "avg_month%": 0, "best_month%": 0,
                "worst_month%": 0, "final_capital": CAPITAL,
                "sl_pct": sl_pct, "rr": rr}
    wins = (trades_df["result"] == "TP").sum()
    wr   = wins / len(trades_df) * 100
    fin  = trades_df["capital"].iloc[-1]
    tot  = (fin - CAPITAL) / CAPITAL * 100
    mr   = monthly_returns(trades_df, CAPITAL)
    return {
        "strategy":     name,
        "trades":       len(trades_df),
        "win_rate":     round(wr, 1),
        "total_ret%":   round(tot, 1),
        "avg_month%":   round(mr.mean(), 2) if len(mr) else 0,
        "best_month%":  round(mr.max(), 2)  if len(mr) else 0,
        "worst_month%": round(mr.min(), 2)  if len(mr) else 0,
        "final_capital":round(fin, 2),
        "sl_pct":       sl_pct,
        "rr":           rr,
    }


# ── RUN ALL STRATEGIES × PARAM GRID ──────────────────────────────────────────
print("Building indicators...")
df = build_indicators(df_full)
print(f"Indicators ready. {len(df):,} candles after dropna.\n")

SL_GRID = [0.002, 0.003, 0.005]
RR_GRID = [1.5, 2.0, 2.5]

all_results = []
total_runs   = len(STRATEGIES) * len(SL_GRID) * len(RR_GRID)
run_i        = 0

print(f"Running {len(STRATEGIES)} strategies × {len(SL_GRID)} SL × {len(RR_GRID)} RR = {total_runs} combinations...\n")

best_monthly_trades = {}  # strategy_name → trades_df for top performers

for name, sig_fn in STRATEGIES.items():
    signals = sig_fn(df)
    n_sigs  = (signals != 0).sum()
    for sl in SL_GRID:
        for rr in RR_GRID:
            run_i += 1
            label = f"{name}|SL{sl*100:.1f}%|RR{rr}"
            trades = backtest(df, signals, sl_pct=sl, rr=rr)
            stats  = summary_stats(trades, label, sl, rr)
            all_results.append(stats)
            if run_i % 30 == 0 or run_i == total_runs:
                print(f"  [{run_i:3d}/{total_runs}] done...", end="\r")

print()

# ── RANK AND DISPLAY ──────────────────────────────────────────────────────────
results_df = pd.DataFrame(all_results)
results_df.sort_values("avg_month%", ascending=False, inplace=True)
results_df.reset_index(drop=True, inplace=True)

print("\n" + "═"*110)
print(f"{'RANK':<5} {'STRATEGY':<42} {'TRADES':>7} {'WIN%':>6} {'AVG/MO%':>9} {'BEST/MO%':>10} {'WORST/MO%':>11} {'TOTAL%':>9} {'FINAL $':>9}")
print("═"*110)
for i, row in results_df.head(20).iterrows():
    print(f"{i+1:<5} {row['strategy']:<42} {int(row['trades']):>7} {row['win_rate']:>6.1f} "
          f"{row['avg_month%']:>9.2f} {row['best_month%']:>10.2f} {row['worst_month%']:>11.2f} "
          f"{row['total_ret%']:>9.1f} {row['final_capital']:>9.2f}")
print("═"*110)

# ── TOP 5 FULL PER-MONTH BREAKDOWN ────────────────────────────────────────────
print("\n\n" + "═"*80)
print("  PER-MONTH BREAKDOWN — TOP 5 STRATEGIES")
print("═"*80)

for rank in range(5):
    row    = results_df.iloc[rank]
    sname  = row["strategy"]
    sl_val = row["sl_pct"]
    rr_val = row["rr"]
    # Reconstruct signal and trades
    base_name = sname.split("|")[0]
    sig_fn    = STRATEGIES[base_name]
    signals   = sig_fn(df)
    trades    = backtest(df, signals, sl_pct=sl_val, rr=rr_val)
    mr        = monthly_returns(trades, CAPITAL)

    print(f"\n#{rank+1}  {sname}")
    print(f"    Trades: {int(row['trades'])} | Win%: {row['win_rate']}% | Total: {row['total_ret%']}% | Final: ${row['final_capital']}")
    print(f"    {'Month':<10} {'Return%':>9}  {'Bar'}")
    print(f"    {'-'*50}")
    for month, ret in mr.items():
        bar_len = int(abs(ret) / 5)
        bar     = ("█" * min(bar_len, 30)) if ret >= 0 else ("▓" * min(bar_len, 30))
        sign    = "+" if ret >= 0 else ""
        print(f"    {month:<10} {sign}{ret:>8.2f}%  {bar}")
    print(f"    Avg: {mr.mean():.2f}%/month | Best: {mr.max():.2f}% | Worst: {mr.min():.2f}%")

print("\n" + "═"*80)
print(f"  WINNER: {results_df.iloc[0]['strategy']}")
print(f"  Avg Monthly Return: {results_df.iloc[0]['avg_month%']:.2f}% | Best Month: {results_df.iloc[0]['best_month%']:.2f}%")
print("═"*80)
