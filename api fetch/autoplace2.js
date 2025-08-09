//only 1 open position at a time
import fetch from 'node-fetch';
import dotenv from 'dotenv';
dotenv.config();

const API_KEY = process.env.ALPACA_API_KEY;
const SECRET_KEY = process.env.ALPACA_SECRET_KEY;

const headers = {
  'APCA-API-KEY-ID': API_KEY,
  'APCA-API-SECRET-KEY': SECRET_KEY,
  'Content-Type': 'application/json',
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

// === FETCH CANDLES ===
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

// === FETCH LIVE PRICE ===
async function fetchLivePrice() {
  const url = 'https://data.alpaca.markets/v1beta3/crypto/us/latest/trades?symbols=BTC/USD';
  const res = await fetch(url, { headers });
  const data = await res.json();
  return data.trades['BTC/USD'].p;
}

// === FETCH CURRENT POSITION ===
async function checkPosition() {
  const res = await fetch('https://paper-api.alpaca.markets/v2/positions/BTC/USD', { headers });

  if (res.status === 404) return null;
  const data = await res.json();

  return {
    qty: parseFloat(data.qty),
    side: data.side, // 'long' or 'short'
  };
}

// === PLACE BUY/SELL ORDER ===
async function placeOrder(side) {
  const order = {
    symbol: 'BTC/USD',
    qty: 0.0001,
    side,
    type: 'market',
    time_in_force: 'gtc',
  };

  const res = await fetch('https://paper-api.alpaca.markets/v2/orders', {
    method: 'POST',
    headers,
    body: JSON.stringify(order),
  });

  if (!res.ok) {
    const err = await res.text();
    console.error(`❌ Order failed: ${err}`);
  } else {
    console.log(`✅ ${side.toUpperCase()} order placed successfully.`);
  }
}

// === CLOSE EXISTING POSITION ===
async function closePosition() {
  const res = await fetch('https://paper-api.alpaca.markets/v2/positions/BTC/USD', {
    method: 'DELETE',
    headers,
  });

  if (res.ok) {
    console.log("✅ Closed existing position.");
  } else {
    const err = await res.text();
    console.error(`❌ Failed to close position: ${err}`);
  }
}

// === MAIN BOT LOGIC ===
async function main() {
  try {
    const livePrice = await fetchLivePrice();
    const closes = await fetchCandles();

    const ema12Arr = calculateEMA(closes, 12);
    const ema20Arr = calculateEMA(closes, 20);

    const ema12 = ema12Arr[ema12Arr.length - 1];
    const ema20 = ema20Arr[ema20Arr.length - 1];

    console.log(`\n💰 BTC/USD Price: $${livePrice}`);
    console.log(`📈 EMA 12: ${ema12.toFixed(2)} | 📉 EMA 20: ${ema20.toFixed(2)}`);

    // === Determine Signal ===
    let signal = 'hold';
    if (ema12 > ema20) signal = 'buy';
    else if (ema12 < ema20) signal = 'sell';

    const position = await checkPosition();
    console.log("📊 Current Position:", position ? `${position.side.toUpperCase()} ${position.qty}` : "None");

    // === Trading Logic ===
    if (!position) {
      // No position open
      if (signal === 'buy' || signal === 'sell') {
        console.log(`📍 No position. Entering ${signal.toUpperCase()}.`);
        await placeOrder(signal);
      } else {
        console.log("⏸ Signal is HOLD. No action.");
      }
    } else {
      const currentSide = position.side; // 'long' or 'short'

      if ((currentSide === 'long' && signal === 'buy') || (currentSide === 'short' && signal === 'sell')) {
        console.log(`🔄 Holding current ${currentSide.toUpperCase()} position.`);
      } else if (signal === 'hold') {
        console.log("🟡 Signal is HOLD — keeping current position.");
      } else {
        console.log(`🔁 Switching from ${currentSide.toUpperCase()} to ${signal.toUpperCase()}.`);
        await closePosition();
        await placeOrder(signal);
      }
    }

  } catch (error) {
    console.error("❌ Error:", error.message);
  }
}

main();
