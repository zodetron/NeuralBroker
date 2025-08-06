const { config } = require("dotenv");
const Alpaca = require("@alpacahq/alpaca-trade-api");
const { EMA } = require("technicalindicators");

config(); // Load .env

const alpaca = new Alpaca({
  keyId: process.env.API_KEY,
  secretKey: process.env.SECRET_KEY,
  paper: true,
});

const SYMBOL = "BTC/USD";
const TIMEFRAME = "15Min";
const EMA_FAST = 12;
const EMA_SLOW = 20;

async function fetchBars() {
  try {
    const bars = await alpaca.getCryptoBars([SYMBOL], {
      timeframe: TIMEFRAME,
      limit: 100,
    });

    const barList = bars[SYMBOL];

    if (!barList || barList.length < EMA_SLOW) {
      throw new Error("Not enough bars to calculate EMA.");
    }

    return barList;
  } catch (error) {
    throw new Error("Failed to fetch bars: " + error.message);
  }
}


function calculateEMA(values, period) {
  return EMA.calculate({ period, values });
}

function getPriceSeries(bars) {
  return bars.map((bar) => bar.close);
}

function detectCrossover(emaFast, emaSlow) {
  const len = Math.min(emaFast.length, emaSlow.length);
  if (len < 2) return null;

  const prevFast = emaFast[len - 2];
  const currFast = emaFast[len - 1];
  const prevSlow = emaSlow[len - 2];
  const currSlow = emaSlow[len - 1];

  if (prevFast < prevSlow && currFast > currSlow) {
    return "bullish";
  } else if (prevFast > prevSlow && currFast < currSlow) {
    return "bearish";
  } else {
    return null;
  }
}

let currentPosition = null;

async function runStrategy() {
  console.log("📈 Running EMA Strategy...");

  try {
    const bars = await fetchBars();
    const prices = getPriceSeries(bars);

    const emaFast = calculateEMA(prices, EMA_FAST);
    const emaSlow = calculateEMA(prices, EMA_SLOW);

    const crossover = detectCrossover(emaFast, emaSlow);
    const lastPrice = prices[prices.length - 1];

    if (crossover === "bullish" && currentPosition !== "long") {
      console.log(`🔵 Bullish crossover detected at $${lastPrice}. Opening BUY position.`);
      currentPosition = "long";
    } else if (crossover === "bearish" && currentPosition !== "short") {
      console.log(`🔴 Bearish crossover detected at $${lastPrice}. Opening SELL position.`);
      currentPosition = "short";
    } else {
      console.log(`No crossover detected. Current price: $${lastPrice}`);
    }
  } catch (err) {
    console.error("❌ Failure:", err.message);
  }
}

// Run once (or you can setInterval to poll every minute)
runStrategy();
