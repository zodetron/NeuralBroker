require("dotenv").config();


// require('dotenv').config();
console.log("API KEY:", process.env.APCA_API_KEY_ID);
console.log("SECRET:", process.env.APCA_API_SECRET_KEY);
console.log("URL:", process.env.APCA_API_BASE_URL);

const Alpaca = require("@alpacahq/alpaca-trade-api");
const { EMA } = require("technicalindicators");

const alpaca = new Alpaca({
  keyId: process.env.APCA_API_KEY_ID,
  secretKey: process.env.APCA_API_SECRET_KEY,
  paper: true,
});

const SYMBOL = "BTC/USD";
const TIMEFRAME = "1Min";
const LIMIT = 100;

async function fetchBars() {
  const resp = alpaca.getBarsV2(SYMBOL, {
    timeframe: TIMEFRAME,
    limit: LIMIT,
    adjustment: "raw",
  });

  const bars = [];
  for await (let b of resp) {
    bars.push(b);
  }
  return bars;
}

function calculateEMAs(closes) {
  const ema12 = EMA.calculate({ period: 12, values: closes });
  const ema20 = EMA.calculate({ period: 20, values: closes });

  // Pad shorter EMA to match index
  while (ema12.length < ema20.length) ema12.unshift(null);

  return { ema12, ema20 };
}

async function checkSignal() {
  console.log("📈 Running EMA Crossover Strategy...");

  try {
    const bars = await fetchBars();
    const closes = bars.map((b) => b.ClosePrice);

    const { ema12, ema20 } = calculateEMAs(closes);
    const len = ema12.length;

    if (ema12[len - 2] < ema20[len - 2] && ema12[len - 1] > ema20[len - 1]) {
      console.log("🔵 Buy Signal Detected");
      await closeAllPositions();
      await placeOrder("buy");
    } else if (ema12[len - 2] > ema20[len - 2] && ema12[len - 1] < ema20[len - 1]) {
      console.log("🔴 Sell Signal Detected");
      await closeAllPositions();
      await placeOrder("sell");
    } else {
      console.log("➖ No crossover");
    }
  } catch (err) {
    console.error("❌ Strategy failed:", err.message || err);
  }
}

async function placeOrder(side) {
  try {
    const order = await alpaca.createOrder({
      symbol: "BTC/USD",
      qty: 0.001, // adjust as needed
      side: side,
      type: "market",
      time_in_force: "gtc",
    });
    console.log(`✅ ${side.toUpperCase()} Order placed:`, order.id);
  } catch (err) {
    console.error("❌ Order failed:", err.message || err);
  }
}

async function closeAllPositions() {
  try {
    await alpaca.closeAllPositions();
    console.log("📤 All positions closed");
  } catch (err) {
    console.log("⚠️ No positions to close or error:", err.message);
  }
}

// Run every minute
checkSignal();
setInterval(checkSignal, 60 * 1000);
