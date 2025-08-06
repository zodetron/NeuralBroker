// ema-strategy.js

require('dotenv').config();
const Alpaca = require('@alpacahq/alpaca-trade-api');
const { EMA } = require('technicalindicators');

const alpaca = new Alpaca({
  keyId: process.env.ALPACA_KEY_ID,
  secretKey: process.env.ALPACA_SECRET_KEY,
  paper: true,
  usePolygon: false,
  baseUrl: process.env.ALPACA_BASE_URL,
});

const SYMBOL = 'BTC/USD';
const TIMEFRAME = '5Min';
const LIMIT = 100;

async function fetchBars(symbol) {
  try {
    const bars = await alpaca.getCryptoBars({
      symbol,
      timeframe: "1Min",
      limit: 100, // fetch at least 20+ bars
    });

    console.log(`Fetched ${bars[symbol]?.length} bars for ${symbol}`); // ✅ Log here

    return bars[symbol];
  } catch (err) {
    throw new Error("Failed to fetch bars: " + err.message);
  }
}


async function getCurrentPosition() {
  try {
    const position = await alpaca.getPosition('BTC/USD');
    return position.side; // 'long' or 'short'
  } catch (err) {
    return null; // No current position
  }
}

async function closePosition() {
  try {
    await alpaca.closePosition('BTC/USD');
    console.log('✅ Closed position.');
  } catch (err) {
    console.log('ℹ️ No position to close.');
  }
}

async function placeOrder(side) {
  try {
    await alpaca.createOrder({
      symbol: 'BTC/USD',
      qty: 0.0001, // Adjust position size as needed
      side,
      type: 'market',
      time_in_force: 'gtc',
    });
    console.log(`📈 Placed ${side} order.`);
  } catch (err) {
    console.error('❌ Order failed:', err.message);
  }
}

console.log("Fetched bars:", bars[SYMBOL]?.length);


function calculateEMAs(closes) {
  const ema12 = EMA.calculate({ period: 12, values: closes });
  const ema20 = EMA.calculate({ period: 20, values: closes });

  return { ema12, ema20 };
}

async function runStrategy() {
  console.log('📈 Running EMA Strategy...');

  const bars = await fetchBarData();
  if (bars.length === 0) return;

  const closes = bars.map(bar => bar.close);
  const { ema12, ema20 } = calculateEMAs(closes);

  const len = ema12.length;
  const lastEma12 = ema12[len - 1];
  const prevEma12 = ema12[len - 2];
  const lastEma20 = ema20[len - 1];
  const prevEma20 = ema20[len - 2];

  const position = await getCurrentPosition();

  const bullishCross = prevEma12 < prevEma20 && lastEma12 > lastEma20;
  const bearishCross = prevEma12 > prevEma20 && lastEma12 < lastEma20;

  if (bullishCross && lastEma12 > lastEma20) {
    if (position !== 'long') {
      console.log('📊 Bullish crossover detected.');
      await closePosition();
      await placeOrder('buy');
    } else {
      console.log('📌 Already in long position.');
    }
  } else if (bearishCross && lastEma20 > lastEma12) {
    if (position !== 'short') {
      console.log('📉 Bearish crossover detected.');
      await closePosition();
      await placeOrder('sell');
    } else {
      console.log('📌 Already in short position.');
    }
  } else {
    console.log('🔁 No crossover detected.');
  }
}

runStrategy();
