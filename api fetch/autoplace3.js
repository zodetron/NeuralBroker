import fetch from 'node-fetch';
import dotenv from 'dotenv';
dotenv.config();

const API_KEY = process.env.ALPACA_API_KEY;
const SECRET_KEY = process.env.ALPACA_SECRET_KEY;

const headers = {
  'APCA-API-KEY-ID': API_KEY,
  'APCA-API-SECRET-KEY': SECRET_KEY,
};

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

  return bars.map(bar => bar.c); // close prices
}

// === CHECK OPEN POSITION ===
async function getOpenPosition() {
  const url = 'https://paper-api.alpaca.markets/v2/positions/BTC/USD';
  const res = await fetch(url, { headers });

  if (res.status === 404) {
    return null; // No position
  }

  const data = await res.json();
  return data.side; // 'long' or 'short'
}

// === CLOSE EXISTING POSITION ===
async function closePosition() {
  const url = 'https://paper-api.alpaca.markets/v2/positions/BTC/USD';
  const res = await fetch(url, {
    method: 'DELETE',
    headers
  });

  if (!res.ok) {
    throw new Error('Failed to close position.');
  }

  console.log('❌ Existing position closed.');
}

// === PLACE ORDER ===
async function placeOrder(side) {
  const url = 'https://paper-api.alpaca.markets/v2/orders';
  const body = {
    symbol: 'BTC/USD',
    qty: 1,
    side,
    type: 'market',
    time_in_force: 'gtc'
  };

  const res = await fetch(url, {
    method: 'POST',
    headers: {
      ...headers,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(body)
  });

  if (!res.ok) {
    const error = await res.json();
    throw new Error(`Order failed: ${JSON.stringify(error)}`);
  }

  console.log(`✅ ${side.toUpperCase()} order placed successfully.`);
}

// === MAIN LOGIC ===
async function main() {
  try {
    const livePrice = await fetchLivePrice();
    const closes = await fetchCandles();

    const ema12Arr = calculateEMA(closes, 12);
    const ema20Arr = calculateEMA(closes, 20);

    const latestEMA12 = ema12Arr[ema12Arr.length - 1];
    const latestEMA20 = ema20Arr[ema20Arr.length - 1];

    console.log(`\n💰 BTC/USD Price: $${livePrice}`);
    console.log(`📈 EMA 12: ${latestEMA12.toFixed(2)} | 📉 EMA 20: ${latestEMA20.toFixed(2)}`);

    const currentPosition = await getOpenPosition();
    console.log(`📊 Current Position: ${currentPosition || 'None'}`);

    if (latestEMA12 > latestEMA20) {
      console.log('📍 Signal: BUY');
      if (currentPosition === 'short') {
        await closePosition();
        await placeOrder('buy');
      } else if (!currentPosition) {
        await placeOrder('buy');
      } else {
        console.log('📎 Holding BUY position. No action needed.');
      }

    } else if (latestEMA12 < latestEMA20) {
      console.log('📍 Signal: SELL');
      if (currentPosition === 'long') {
        await closePosition();
        await placeOrder('sell');
      } else if (!currentPosition) {
        await placeOrder('sell');
      } else {
        console.log('📎 Holding SELL position. No action needed.');
      }

    } else {
      console.log('⚖️ EMAs are equal. No action taken.');
    }

  } catch (error) {
    console.error('❌ Error:', error.message);
  }
}

main();
