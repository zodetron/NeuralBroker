require('dotenv').config();
const Alpaca = require('@alpacahq/alpaca-trade-api');
import Alpaca from '@alpacahq/alpaca-trade-api';
console.log('Alpaca SDK version:', Alpaca.version); // should log 4.x.x

const technicalindicators = require('technicalindicators');

const alpaca = new Alpaca({
  keyId: process.env.ALPACA_KEY_ID,
  secretKey: process.env.ALPACA_SECRET_KEY,
  paper: true,
  usePolygon: false
});

const SYMBOL = 'BTC/USD';
const TIMEFRAME = '5Min'; // or '15Min' or '1Hour'
const EMA_SHORT = 12;
const EMA_LONG = 20;
let lastSignal = null;

async function fetchBars() {
  const now = new Date();
  const start = new Date(now.getTime() - 1000 * 60 * 60 * 12); // 12 hours ago

  const barsIterable = await alpaca.getBarsV2(
    'BTC/USD',
    {
      start: start.toISOString(),
      end: now.toISOString(),
      timeframe: '5Min',
    },
    { feed: 'iex' } // Required for crypto
  );

  const bars = [];
  for await (let b of barsIterable) {
    bars.push(b);
  }

  return bars.map(bar => bar.Close); // array of closing prices
}



async function runStrategy() {
  console.log('📈 Running EMA Crossover Strategy...');
  try {
    const closes = await fetchBars();

    const ema12 = technicalindicators.EMA.calculate({ period: EMA_SHORT, values: closes });
    const ema20 = technicalindicators.EMA.calculate({ period: EMA_LONG, values: closes });

    const l = ema12.length;
    if (l < 2 || ema20.length < 2) {
      console.log('Not enough data for EMA calculation.');
      return;
    }

    const prev12 = ema12[l - 2];
    const curr12 = ema12[l - 1];
    const prev20 = ema20[l - 2];
    const curr20 = ema20[l - 1];

    console.log(`🔵 EMA12: ${curr12.toFixed(2)} | 🔴 EMA20: ${curr20.toFixed(2)}`);

    if (prev12 < prev20 && curr12 > curr20 && lastSignal !== 'BUY') {
      console.log('📤 Crossover detected: BUY SIGNAL');
      await closeExistingPosition();
      await submitOrder('buy');
      lastSignal = 'BUY';
    } else if (prev12 > prev20 && curr12 < curr20 && lastSignal !== 'SELL') {
      console.log('📥 Crossover detected: SELL SIGNAL');
      await closeExistingPosition();
      await submitOrder('sell');
      lastSignal = 'SELL';
    } else {
      console.log('⏸️ No crossover detected.');
    }

  } catch (err) {
    console.error('❌ Strategy failed:', err);
  }
}

async function closeExistingPosition() {
  try {
    await alpaca.closePosition('BTC/USD');
    console.log('✅ Closed existing BTC/USD position.');
  } catch (err) {
    if (err.statusCode === 404) {
      console.log('ℹ️ No open position to close.');
    } else {
      console.error('❌ Error closing position:', err.message);
    }
  }
}

async function submitOrder(side) {
  try {
    const price = (await alpaca.getLastTrade('BTC/USD')).price;
    const qty = 0.01; // Adjust based on your account equity

    await alpaca.createOrder({
      symbol: 'BTC/USD',
      qty,
      side,
      type: 'market',
      time_in_force: 'gtc'
    });

    console.log(`🚀 Submitted ${side.toUpperCase()} order for ${qty} BTC/USD at market price.`);
  } catch (err) {
    console.error('❌ Failed to submit order:', err.message);
  }
}

// Run every X minutes
setInterval(runStrategy, 60 * 1000); // every 1 minute
runStrategy(); // Initial run
