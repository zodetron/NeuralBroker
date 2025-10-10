// === EMA + Volume Spike Bot with 1% SL ===
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

// === GET LIVE PRICE ===
async function fetchLivePrice() {
  const url = 'https://data.alpaca.markets/v1beta3/crypto/us/latest/trades?symbols=BTC/USD';
  const res = await fetch(url, { headers });
  const data = await res.json();
  return data.trades['BTC/USD'].p;
}

// === GET HISTORICAL CANDLES (includes volume) ===
async function fetchCandles() {
  const now = new Date();
  const end = now.toISOString();
  const start = new Date(now.getTime() - 30 * 5 * 60 * 1000).toISOString();

  const url = `https://data.alpaca.markets/v1beta3/crypto/us/bars?symbols=BTC/USD&timeframe=5Min&start=${start}&end=${end}&limit=30`;
  const res = await fetch(url, { headers });
  const data = await res.json();

  const bars = data.bars['BTC/USD'];
  if (!bars || bars.length < 20) throw new Error('Not enough candle data for EMA calculation.');

  return bars.map(bar => ({
    close: bar.c,
    volume: bar.v
  }));
}

// === POSITION / ORDER FUNCTIONS ===
async function checkPosition() {
  const res = await fetch('https://paper-api.alpaca.markets/v2/positions/BTCUSD', { headers });
  if (res.status === 404) return null;
  try { return await res.json(); }
  catch { return null; }
}

async function closePosition() {
  try {
    const res = await fetch('https://paper-api.alpaca.markets/v2/positions/BTCUSD', {
      method: 'DELETE',
      headers,
    });
    if (res.status === 404) return false;
    const data = await res.json();
    console.log(`🔴 Position closed:`, data.id || data.message || data);
    entryPrice = null;
    entrySide = null;
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
  const res = await fetch('https://paper-api.alpaca.markets/v2/orders', {
    method: 'POST',
    headers: { ...headers, 'Content-Type': 'application/json' },
    body: JSON.stringify(order),
  });
  const data = await res.json();
  console.log(`📤 ${side.toUpperCase()} order placed:`, data.id || data.message || data);
  entryPrice = await fetchLivePrice();
  entrySide = side;
}

// === STOP LOSS CHECK ===
async function checkStopLoss() {
  if (!entryPrice || !entrySide) return;
  try {
    const livePrice = await fetchLivePrice();
    if (entrySide === 'buy' && livePrice <= entryPrice * 0.99) {
      console.log(`🛑 SL hit LONG. Price: ${livePrice}, Entry: ${entryPrice}`);
      await closePosition();
    }
    if (entrySide === 'sell' && livePrice >= entryPrice * 1.01) {
      console.log(`🛑 SL hit SHORT. Price: ${livePrice}, Entry: ${entryPrice}`);
      await closePosition();
    }
  } catch (err) {
    console.error("SL check error:", err.message);
  }
}

// === MAIN TRADING LOGIC (EMA + Volume Filter) ===
async function checkEMAAndTrade() {
  if (isRunning) return;
  isRunning = true;

  try {
    const livePrice = await fetchLivePrice();
    const bars = await fetchCandles();

    const closes = bars.map(b => b.close);
    const volumes = bars.map(b => b.volume);

    const ema12Arr = calculateEMA(closes, 12);
    const ema20Arr = calculateEMA(closes, 20);

    const latestEMA12 = ema12Arr.at(-1);
    const latestEMA20 = ema20Arr.at(-1);

    // === Volume Spike Detection ===
    const avgVolume = volumes.slice(-11, -1).reduce((a, b) => a + b) / 10;
    const latestVolume = volumes.at(-1);
    const volumeRatio = latestVolume / avgVolume;

    console.log(`\n💰 Live BTC/USD: $${livePrice}`);
    console.log(`📈 EMA12: ${latestEMA12.toFixed(2)} | EMA20: ${latestEMA20.toFixed(2)}`);
    console.log(`📊 Volume: ${latestVolume.toFixed(2)} | Avg: ${avgVolume.toFixed(2)} | Ratio: ${volumeRatio.toFixed(2)}`);

    const currentEMARelation =
      latestEMA12 > latestEMA20 ? 'above' :
      latestEMA12 < latestEMA20 ? 'below' : 'equal';

    if (currentEMARelation === previousEMARelation) {
      console.log("🔄 EMA relation unchanged, skipping");
      return;
    }

    // === Trade Conditions ===
    const position = await checkPosition();

    if (currentEMARelation === 'above' && volumeRatio > 1.5) {
      console.log("✅ Bullish crossover + Volume spike → LONG signal");
      if (position) await closePosition();
      await placeOrder('buy');
      previousEMARelation = 'above';

    } else if (currentEMARelation === 'below' && volumeRatio > 1.5) {
      console.log("❌ Bearish crossover + Volume spike → SHORT signal");
      if (position) await closePosition();
      await placeOrder('sell');
      previousEMARelation = 'below';

    } else if (volumeRatio < 0.5) {
      console.log("⚠️ Volume too low, ignoring weak signal.");
    } else {
      console.log("⏸️ No strong volume confirmation, skipping trade.");
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
setInterval(checkStopLoss, 15 * 1000);

console.log("🚀 EMA + Volume Spike Bot started (1% SL, 5m EMA check, 15s SL monitor)...");

//looping thorugh SL AND auto-check for spiking with volume transformations
