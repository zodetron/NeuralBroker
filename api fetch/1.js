import fetch from 'node-fetch';
import dotenv from 'dotenv';
dotenv.config();

const API_KEY = process.env.ALPACA_API_KEY;
const SECRET_KEY = process.env.ALPACA_SECRET_KEY;
const DATA_URL = 'https://data.alpaca.markets/v1beta3/crypto';
const HEADERS = {
  'APCA-API-KEY-ID': API_KEY,
  'APCA-API-SECRET-KEY': SECRET_KEY,
};

// === EMA CALCULATION ===
function calculateEMA(prices, period) {
  const k = 2 / (period + 1);
  let emaArray = [];
  let ema = prices.slice(0, period).reduce((a, b) => a + b) / period; // SMA for first EMA
  emaArray[period - 1] = ema;

  for (let i = period; i < prices.length; i++) {
    ema = prices[i] * k + ema * (1 - k);
    emaArray[i] = ema;
  }

  return emaArray;
}

// === GET HISTORICAL CANDLES ===
async function getPastCandles() {
  const now = new Date();
  const end = now.toISOString();
  const start = new Date(now.getTime() - 30 * 5 * 60 * 1000).toISOString(); // 30 candles * 5 mins

  const url = `${DATA_URL}/bars/BTC/USD?timeframe=5Min&start=${start}&end=${end}&limit=30`;
  const res = await fetch(url, { headers: HEADERS });
  const data = await res.json();

  if (!data.bars || data.bars.length === 0) {
    throw new Error('Failed to fetch candle data.');
  }

  return data.bars;
}

// === GET LIVE PRICE ===
async function getLivePrice() {
  const url = `${DATA_URL}/latest/trades/BTC/USD`;
  const res = await fetch(url, { headers: HEADERS });
  const data = await res.json();

  if (!data.trade) {
    throw new Error('Failed to fetch live price.');
  }

  return data.trade.p;
}

// === MAIN FUNCTION ===
async function main() {
  try {
    const candles = await getPastCandles();
    const closes = candles.map(c => c.c);

    const ema12Arr = calculateEMA(closes, 12);
    const ema20Arr = calculateEMA(closes, 20);

    const latestEMA12 = ema12Arr[ema12Arr.length - 1];
    const latestEMA20 = ema20Arr[ema20Arr.length - 1];

    const livePrice = await getLivePrice();

    console.log(`\n📈 Live BTC/USD Price: $${livePrice.toFixed(2)}`);
    console.log(`📊 EMA 12: $${latestEMA12.toFixed(2)}`);
    console.log(`📊 EMA 20: $${latestEMA20.toFixed(2)}`);

    if (latestEMA12 > latestEMA20) {
      console.log('✅ Bullish crossover (EMA 12 > EMA 20)');
    } else if (latestEMA12 < latestEMA20) {
      console.log('❌ Bearish crossover (EMA 12 < EMA 20)');
    } else {
      console.log('⚠️ No crossover (EMAs are equal)');
    }

  } catch (error) {
    console.error('Error:', error.message);
  }
}

main();
