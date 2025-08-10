import fetch from 'node-fetch';
import dotenv from 'dotenv';
dotenv.config();

const API_KEY = process.env.ALPACA_API_KEY;
const SECRET_KEY = process.env.ALPACA_SECRET_KEY;

const headers = {
  'APCA-API-KEY-ID': API_KEY,
  'APCA-API-SECRET-KEY': SECRET_KEY,
};

// Track previous EMA state
let previousEMARelation = null; // 'above', 'below', or null
let isRunning = false;

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
  const url = 'https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT';
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
  const data = await res.json();
  return parseFloat(data.price);
}

// === GET HISTORICAL CANDLES ===
async function fetchCandles() {
  // Binance API endpoint for candlesticks (klines)
  // symbol=BTCUSDT, interval=5m, limit=30 bars
  const url = `https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=5m&limit=30`;

  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);

  const data = await res.json();
  // data is an array of arrays: [
  //   [
  //     1499040000000,      // Open time
  //     "0.01634790",       // Open
  //     "0.80000000",       // High
  //     "0.01575800",       // Low
  //     "0.01577100",       // Close
  //     "148976.11427815",  // Volume
  //     1499644799999,      // Close time
  //     "2434.19055334",    // Quote asset volume
  //     308,                // Number of trades
  //     "1756.87402397",    // Taker buy base asset volume
  //     "28.46694368",      // Taker buy quote asset volume
  //     "17928899.62484339" // Ignore
  //   ],
  //   ...
  // ]

  if (data.length < 20) {
    throw new Error('Not enough candle data for EMA calculation.');
  }

  // Extract close prices from each candle
  const closePrices = data.map(candle => parseFloat(candle[4]));

  return closePrices;
}


// === CHECK OPEN POSITION ===
async function checkPosition() {
  const res = await fetch('https://paper-api.alpaca.markets/v2/positions/BTC/USD', { headers });

  if (res.status === 404) return null; // No open position
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
      latestEMA12 < latestEMA20 ? 'below' : 
      'equal';

    // Skip if EMA relation hasn't changed
    if (currentEMARelation === previousEMARelation) {
      console.log("🔄 EMA relation unchanged, skipping action");
      return;
    }

    const position = await checkPosition();

    if (currentEMARelation === 'above') {
      console.log("✅ Bullish crossover: EMA 12 is above EMA 20");

      // Close any existing position first
      if (position) {
        await closePosition();
      }

      // Place buy order
      await placeOrder('buy');
      previousEMARelation = 'above';

    } else if (currentEMARelation === 'below') {
      console.log("❌ Bearish crossover: EMA 12 is below EMA 20");

      // Close any existing position first
      if (position) {
        await closePosition();
      }

      // Place sell order
      await placeOrder('sell');
      previousEMARelation = 'below';
    }

  } catch (error) {
    console.error("❌ Error:", error.message);
  } finally {
    isRunning = false;
  }
}

// Run immediately and then every 5 minutes
checkEMAAndTrade();
setInterval(checkEMAAndTrade, 5 * 60 * 1000);

console.log("🚀 EMA Crossover Bot started. Checking every 5 minutes...");