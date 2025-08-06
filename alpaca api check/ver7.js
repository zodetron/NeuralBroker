require("dotenv").config();
const Alpaca = require("@alpacahq/alpaca-trade-api");
const { EMA } = require("technicalindicators");

const alpaca = new Alpaca({
  keyId: process.env.API_KEY,
  secretKey: process.env.SECRET_KEY,
  paper: true, // use true for paper trading
  usePolygon: false, // required for crypto
});

const SYMBOL = "BTC/USD"; // Crypto pair
let lastSignal = null; // "buy" or "sell"

async function fetchBars() {
  try {
    const response = await alpaca.getCryptoBarsV3(
      SYMBOL,
      { timeframe: "1Min", limit: 50 },
      { exchanges: ["CBSE"] }
    );

    const bars = response.bars[SYMBOL];

    if (!bars || bars.length < 20) {
      console.error("❌ Not enough bar data");
      return null;
    }

    return bars;
  } catch (err) {
    console.error("❌ Failed to fetch bars:", err.message);
    return null;
  }
}

function calculateEMAs(closes) {
  const ema12 = EMA.calculate({ period: 12, values: closes });
  const ema20 = EMA.calculate({ period: 20, values: closes });

  return {
    ema12: ema12[ema12.length - 1],
    ema12Prev: ema12[ema12.length - 2],
    ema20: ema20[ema20.length - 1],
    ema20Prev: ema20[ema20.length - 2],
  };
}

async function placeOrder(side) {
  try {
    await alpaca.createOrder({
      symbol: "BTC/USD",
      qty: 0.0001,
      side: side,
      type: "market",
      time_in_force: "gtc",
    });
    console.log(`✅ ${side.toUpperCase()} order placed`);
  } catch (err) {
    console.error("❌ Order failed:", err.message);
  }
}

async function runStrategy() {
  console.log("📈 Running EMA Strategy...");

  const bars = await fetchBars();
  if (!bars) return;

  const closes = bars.map((bar) => bar.c);

  const { ema12, ema12Prev, ema20, ema20Prev } = calculateEMAs(closes);

  if (
    ema12 === undefined ||
    ema20 === undefined ||
    ema12Prev === undefined ||
    ema20Prev === undefined
  ) {
    console.error("❌ EMA calculation failed");
    return;
  }

  console.log(`EMA12: ${ema12.toFixed(2)} | EMA20: ${ema20.toFixed(2)}`);

  // Buy Signal
  if (ema12Prev < ema20Prev && ema12 > ema20 && lastSignal !== "buy") {
    console.log("📘 Buy Signal Triggered");
    await placeOrder("buy");
    lastSignal = "buy";
  }

  // Sell Signal
  else if (ema20Prev < ema12Prev && ema20 > ema12 && lastSignal !== "sell") {
    console.log("📕 Sell Signal Triggered");
    await placeOrder("sell");
    lastSignal = "sell";
  } else {
    console.log("🟡 No trade signal");
  }
}

// Run every minute
runStrategy();
setInterval(runStrategy, 60 * 1000);
