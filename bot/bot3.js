//implemented 1% SL in bot 1
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

// Track entry price & side for SL
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

// === GET HISTORICAL CANDLES ===
async function fetchCandles() {
  const now = new Date();
  const end = now.toISOString();
  const start = new Date(now.getTime() - 30 * 5 * 60 * 1000).toISOString();

  const url = `https://data.alpaca.markets/v1beta3/crypto/us/bars?symbols=BTC/USD&timeframe=5Min&start=${start}&end=${end}&limit=30`;

  const res = await fetch(url, { headers });
  const data = await res.json();

  const bars = data.bars['BTC/USD'];
  if (!bars || bars.length < 20) {
    throw new Error('Not enough candle data for EMA calculation.');
  }

  return bars.map(bar => bar.c);
}

// === CHECK OPEN POSITION ===
async function checkPosition() {
  const res = await fetch('https://paper-api.alpaca.markets/v2/positions/BTC/USD', { headers });
  if (res.status === 404) return null;
  const data = await res.json();
  return data;
}

// === CLOSE POSITION ===
async function closePosition() {
  try {
    const res = await fetch('https://paper-api.alpaca.markets/v2/positions/BTC/USD', {
      method: 'DELETE',
      headers,
    });
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

// === PLACE ORDER ===
async function placeOrder(side, qty = 0.01) {
  const order = {
    symbol: 'BTC/USD',
    qty: qty.toFixed(4),
    side: side,
    type: 'market',
    time_in_force: 'gtc',
  };

  const res = await fetch('https://paper-api.alpaca.markets/v2/orders', {
    method: 'POST',
    headers: {
      ...headers,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(order),
  });

  const data = await res.json();
  console.log(`📤 ${side.toUpperCase()} order placed:`, data.id || data.message || data);

  // Save entry price for SL
  entryPrice = await fetchLivePrice();
  entrySide = side;
}

// === STOP LOSS CHECK ===
async function checkStopLoss() {
  if (!entryPrice || !entrySide) return;
  try {
    const livePrice = await fetchLivePrice();
    if (entrySide === 'buy' && livePrice <= entryPrice * 0.99) {
      console.log(`🛑 Stop-loss triggered for LONG. Price: ${livePrice}, Entry: ${entryPrice}`);
      await closePosition();
    }
    if (entrySide === 'sell' && livePrice >= entryPrice * 1.01) {
      console.log(`🛑 Stop-loss triggered for SHORT. Price: ${livePrice}, Entry: ${entryPrice}`);
      await closePosition();
    }
  } catch (err) {
    console.error("SL check error:", err.message);
  }
}

// === MAIN FUNCTION ===
async function checkEMAAndTrade() {
  if (isRunning) return;
  isRunning = true;

  try {
    const livePrice = await fetchLivePrice();
    const closes = await fetchCandles();

    const ema12Arr = calculateEMA(closes, 12);
    const ema20Arr = calculateEMA(closes, 20);

    const latestEMA12 = ema12Arr[ema12Arr.length - 1];
    const latestEMA20 = ema20Arr[ema20Arr.length - 1];

    console.log(`\n💰 Live BTC/USD Price: $${livePrice}`);
    console.log(`📈 EMA 12: $${latestEMA12.toFixed(2)}`);
    console.log(`📉 EMA 20: $${latestEMA20.toFixed(2)}`);

    const currentEMARelation =
      latestEMA12 > latestEMA20 ? 'above' :
      latestEMA12 < latestEMA20 ? 'below' : 'equal';

    if (currentEMARelation === previousEMARelation) {
      console.log("🔄 EMA relation unchanged, skipping action");
      return;
    }

    const position = await checkPosition();

    if (currentEMARelation === 'above') {
      console.log("✅ Bullish crossover: EMA 12 is above EMA 20");
      if (position) await closePosition();
      await placeOrder('buy');
      previousEMARelation = 'above';

    } else if (currentEMARelation === 'below') {
      console.log("❌ Bearish crossover: EMA 12 is below EMA 20");
      if (position) await closePosition();
      await placeOrder('sell');
      previousEMARelation = 'below';
    }

  } catch (error) {
    console.error("❌ Error:", error.message);
  } finally {
    isRunning = false;
  }
}

// Run EMA check every 5 minutes
checkEMAAndTrade();
setInterval(checkEMAAndTrade, 5 * 60 * 1000);

// Run SL check every 15 seconds
setInterval(checkStopLoss, 15 * 1000);

console.log("🚀 EMA Crossover Bot started with 1% Stop-Loss (SL checked every 15s)...");
