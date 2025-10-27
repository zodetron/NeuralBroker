// === EMA + Volume Spike + Higher TF EMA + TP/Trailing SL Bot (Safe Fetch) ===
import fetch from 'node-fetch';
import dotenv from 'dotenv';
dotenv.config();

const API_KEY = process.env.ALPACA_API_KEY;
const SECRET_KEY = process.env.ALPACA_SECRET_KEY;

const headers = {
  'APCA-API-KEY-ID': API_KEY,
  'APCA-API-SECRET-KEY': SECRET_KEY,
};

let previousEMARelation = null;
let isRunning = false;
let entryPrice = null;
let entrySide = null;
let stopLoss = null;
let takeProfit = null;

// === EMA CALCULATION ===
function calculateEMA(prices, period) {
  const k = 2 / (period + 1);
  let emaArray = [];
  let ema = prices.slice(0, period).reduce((a, b) => a + b) / period;
  emaArray[period - 1] = ema;

  for (let i = period; i < prices.length; i++) {
    ema = prices[i] * k + ema * (1 - k);
    emaArray[i] = ema;
  }
  return emaArray;
}

// === SAFE FETCH FUNCTION ===
async function safeFetchJSON(url) {
  const res = await fetch(url, { headers });
  const contentType = res.headers.get('content-type');

  if (!contentType || !contentType.includes('application/json')) {
    const text = await res.text();
    throw new Error(`Invalid JSON response: ${text}`);
  }

  return await res.json();
}

// === GET LIVE PRICE ===
async function fetchLivePrice() {
  const url = 'https://data.alpaca.markets/v1beta3/crypto/us/latest/trades?symbols=BTC/USD';
  const data = await safeFetchJSON(url);
  if (!data.trades || !data.trades['BTC/USD']) throw new Error('No trade data received.');
  return data.trades['BTC/USD'].p;
}

// === GET HISTORICAL CANDLES ===
async function fetchCandles(timeframe = '5Min', limit = 50) {
  const now = new Date();
  const end = now.toISOString();
  const start = new Date(now.getTime() - limit * 5 * 60 * 1000).toISOString();

  const url = `https://data.alpaca.markets/v1beta3/crypto/us/bars?symbols=BTC/USD&timeframe=${timeframe}&start=${start}&end=${end}&limit=${limit}`;
  const data = await safeFetchJSON(url);

  const bars = data.bars['BTC/USD'];
  if (!bars || bars.length < 20) throw new Error('Not enough candle data for EMA calculation.');

  return bars.map(bar => ({
    close: bar.c,
    volume: bar.v
  }));
}

// === POSITION / ORDER FUNCTIONS ===
async function checkPosition() {
  const url = 'https://paper-api.alpaca.markets/v2/positions/BTCUSD';
  try {
    const data = await safeFetchJSON(url);
    return data;
  } catch (err) {
    if (err.message.includes('404')) return null;
    throw err;
  }
}

async function closePosition() {
  try {
    const res = await fetch('https://paper-api.alpaca.markets/v2/positions/BTCUSD', {
      method: 'DELETE',
      headers,
    });

    const contentType = res.headers.get('content-type');
    let data;
    if (contentType && contentType.includes('application/json')) {
      data = await res.json();
    } else {
      const text = await res.text();
      console.log("Response not JSON on closePosition:", text);
      data = text;
    }

    console.log(`🔴 Position closed:`, data.id || data.message || data);
    entryPrice = null;
    entrySide = null;
    stopLoss = null;
    takeProfit = null;
    return true;

  } catch (error) {
    console.error("Error closing position:", error.message);
    return false;
  }
}

async function placeOrder(side, qty = 0.01) {
  const order = {
    symbol: 'BTC/USD',
    qty: qty.toFixed(4),
    side,
    type: 'market',
    time_in_force: 'gtc',
  };

  try {
    const res = await fetch('https://paper-api.alpaca.markets/v2/orders', {
      method: 'POST',
      headers: { ...headers, 'Content-Type': 'application/json' },
      body: JSON.stringify(order),
    });

    const contentType = res.headers.get('content-type');
    let data;
    if (contentType && contentType.includes('application/json')) {
      data = await res.json();
    } else {
      const text = await res.text();
      console.log("Response not JSON on placeOrder:", text);
      data = text;
    }

    console.log(`📤 ${side.toUpperCase()} order placed:`, data.id || data.message || data);

    entryPrice = await fetchLivePrice();
    entrySide = side;
    stopLoss = side === 'buy' ? entryPrice * 0.99 : entryPrice * 1.01;
    takeProfit = side === 'buy' ? entryPrice * 1.02 : entryPrice * 0.98;
    console.log(`🔒 SL: ${stopLoss.toFixed(2)} | TP: ${takeProfit.toFixed(2)}`);

  } catch (err) {
    console.error("Order placement error:", err.message);
  }
}

// === STOP LOSS & TAKE PROFIT CHECK ===
async function checkStopLossAndTP() {
  if (!entryPrice || !entrySide) return;

  try {
    const livePrice = await fetchLivePrice();

    if (entrySide === 'buy') {
      if (livePrice <= stopLoss) await closePosition();
      else if (livePrice >= takeProfit) await closePosition();
      else if (livePrice >= entryPrice * 1.01 && livePrice * 0.99 > stopLoss) {
        stopLoss = livePrice * 0.99;
        console.log(`🔄 Trailing SL updated LONG: ${stopLoss.toFixed(2)}`);
      }
    }

    if (entrySide === 'sell') {
      if (livePrice >= stopLoss) await closePosition();
      else if (livePrice <= takeProfit) await closePosition();
      else if (livePrice <= entryPrice * 0.99 && livePrice * 1.01 < stopLoss) {
        stopLoss = livePrice * 1.01;
        console.log(`🔄 Trailing SL updated SHORT: ${stopLoss.toFixed(2)}`);
      }
    }

  } catch (err) {
    console.error("SL/TP check error:", err.message);
  }
}

// === MAIN TRADING LOGIC ===
async function checkEMAAndTrade() {
  if (isRunning) return;
  isRunning = true;

  try {
    
    const livePrice = await fetchLivePrice();
    const bars5m = await fetchCandles('5Min', 50);
    const bars15m = await fetchCandles('15Min', 50);

    const closes5m = bars5m.map(b => b.close);
    const volumes5m = bars5m.map(b => b.volume);
    const closes15m = bars15m.map(b => b.close);

    const ema12Arr = calculateEMA(closes5m, 100);
    const ema20Arr = calculateEMA(closes5m, 100);
    const ema50HTFArr = calculateEMA(closes15m, 100);


    const latestEMA12 = ema12Arr.at(-1);
    const latestEMA20 = ema20Arr.at(-1);
    const latestEMA50 = ema50HTFArr.at(-1);

    const avgVolume = volumes5m.slice(-11, -1).reduce((a, b) => a + b) / 10;
    const latestVolume = volumes5m.at(-1);
    const volumeRatio = latestVolume / avgVolume;

    console.log(`\n💰 Live BTC/USD: $${livePrice}`);
    console.log(`📈 EMA12: ${latestEMA12.toFixed(2)} | EMA20: ${latestEMA20.toFixed(2)} | EMA50(HTF): ${latestEMA50.toFixed(2)}`);
    console.log(`📊 Volume: ${latestVolume.toFixed(2)} | Avg: ${avgVolume.toFixed(2)} | Ratio: ${volumeRatio.toFixed(2)}`);

    const currentEMARelation =
      latestEMA12 > latestEMA20 ? 'above' :
      latestEMA12 < latestEMA20 ? 'below' : 'equal';

    if (currentEMARelation === previousEMARelation) {
      console.log("🔄 EMA relation unchanged, skipping");
      return;
    }

    const position = await checkPosition();

    if (currentEMARelation === 'above' && latestEMA12 > latestEMA50 && volumeRatio > 1.5) {
      console.log("✅ Bullish crossover + Volume spike + Uptrend → LONG signal");
      if (position) await closePosition();
      await placeOrder('buy');
      previousEMARelation = 'above';

    } else if (currentEMARelation === 'below' && latestEMA12 < latestEMA50 && volumeRatio > 1.5) {
      console.log("❌ Bearish crossover + Volume spike + Downtrend → SHORT signal");
      if (position) await closePosition();
      await placeOrder('sell');
      previousEMARelation = 'below';

    } else if (volumeRatio < 0.5) {
      console.log("⚠️ Volume too low, ignoring weak signal.");
    } else {
      console.log("⏸️ No strong trend confirmation or volume spike, skipping trade.");
    }

  } catch (error) {
    console.error("❌ Error:", error.message);
  } finally {
    isRunning = false;
  }
}

// === INTERVALS ===
checkEMAAndTrade();
setInterval(checkEMAAndTrade, 5 * 60 * 1000);
setInterval(checkStopLossAndTP, 15 * 1000);

console.log("🚀 EMA + Volume + Higher TF EMA + TP/Trailing SL Bot started...");
