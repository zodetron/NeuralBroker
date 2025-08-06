require('dotenv').config();
const axios = require('axios');
const Alpaca = require('@alpacahq/alpaca-trade-api');
const { EMA } = require('technicalindicators');

// Alpaca trading client
const alpaca = new Alpaca({
  keyId: process.env.APCA_API_KEY_ID,
  secretKey: process.env.APCA_API_SECRET_KEY,
  paper: true,
  usePolygon: false,
});

// Config
const SYMBOL = 'BTC/USD';
const TIMEFRAME = '5Min';
const LIMIT = 100;
let lastSignal = null;

// Fetch crypto bars via REST
async function fetchBars() {
  const resp = await axios.get(
    `${process.env.APCA_API_DATA_URL}/v1beta3/crypto/us/bars`,
    {
      headers: {
        "APCA-API-KEY-ID": process.env.APCA_API_KEY_ID,
        "APCA-API-SECRET-KEY": process.env.APCA_API_SECRET_KEY,
      },
      params: {
        symbols: SYMBOL,
        timeframe: TIMEFRAME,
        limit: LIMIT,
      },
    }
  );
  return resp.data.bars[SYMBOL] || [];
}

// EMA strategy
async function runStrategy() {
  console.log('📈 Running EMA Strategy...');

  try {
    const bars = await fetchBars();
    if (bars.length < 20) {
      console.log("⚠️ Not enough bars.");
      return;
    }

    const closes = bars.map(b => b.c);
    const ema12 = EMA.calculate({ period: 12, values: closes });
    const ema20 = EMA.calculate({ period: 20, values: closes });

    const len = ema12.length;
    const prev12 = ema12[len - 2];
    const curr12 = ema12[len - 1];
    const prev20 = ema20[len - 2];
    const curr20 = ema20[len - 1];

    console.log(`EMA12: ${curr12.toFixed(2)} | EMA20: ${curr20.toFixed(2)}`);

    if (prev12 < prev20 && curr12 > curr20 && lastSignal !== 'BUY') {
      console.log('🔵 Buy signal detected');
      await alpaca.closePosition(SYMBOL).catch(() => {});
      await alpaca.createOrder({
        symbol: SYMBOL,
        qty: 0.001,
        side: 'buy',
        type: 'market',
        time_in_force: 'gtc',
      });
      lastSignal = 'BUY';
    } else if (prev12 > prev20 && curr12 < curr20 && lastSignal !== 'SELL') {
      console.log('🔴 Sell signal detected');
      await alpaca.closePosition(SYMBOL).catch(() => {});
      await alpaca.createOrder({
        symbol: SYMBOL,
        qty: 0.001,
        side: 'sell',
        type: 'market',
        time_in_force: 'gtc',
      });
      lastSignal = 'SELL';
    } else {
      console.log('➖ No signal');
    }

  } catch (e) {
    console.error('❌ Failure:', e.response?.data || e.message);
  }
}

// Run initially and on loop
runStrategy();
setInterval(runStrategy, 5 * 60 * 1000);
