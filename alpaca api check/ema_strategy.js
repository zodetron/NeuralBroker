require("dotenv").config();
const Alpaca = require("@alpacahq/alpaca-trade-api");
const technicalIndicators = require("technicalindicators");

const alpaca = new Alpaca({
  keyId: process.env.ALPACA_KEY_ID,
  secretKey: process.env.ALPACA_SECRET_KEY,
  paper: true,
});

// SETTINGS
const SYMBOL = "BTC/USD";
const EMA_FAST = 12;
const EMA_SLOW = 20;
const POSITION_QTY = 1;

async function runStrategy() {
  try {
    // Step 1: Fetch historical data
    const bars = await alpaca.getCryptoBars(SYMBOL, {
      timeframe: "5Min",
      limit: 50, // more than 20
    });

    const closes = bars.map((bar) => bar.c); // closing prices

    // Step 2: Calculate EMAs
    const ema12 = technicalIndicators.EMA.calculate({
      period: EMA_FAST,
      values: closes,
    });

    const ema20 = technicalIndicators.EMA.calculate({
      period: EMA_SLOW,
      values: closes,
    });

    const len = Math.min(ema12.length, ema20.length);

    const latestEma12 = ema12[len - 1];
    const latestEma20 = ema20[len - 1];
    const prevEma12 = ema12[len - 2];
    const prevEma20 = ema20[len - 2];

    console.log(`EMA12: ${latestEma12} | EMA20: ${latestEma20}`);

    // Step 3: Detect crossover
    let signal = null;

    if (prevEma12 < prevEma20 && latestEma12 > latestEma20) {
      signal = "buy";
    } else if (prevEma12 > prevEma20 && latestEma12 < latestEma20) {
      signal = "sell";
    }

    if (!signal) {
      console.log("⏸️ No crossover signal.");
      return;
    }

    // Step 4: Check current position
    const positions = await alpaca.getPositions();
    const btcPos = positions.find((pos) => pos.symbol === "BTC/USD");

    // Step 5: Act based on signal
    if (signal === "buy") {
      if (btcPos && btcPos.side === "long") {
        console.log("✅ Already in a long position");
      } else {
        if (btcPos) await alpaca.closePosition("BTC/USD");
        await alpaca.createOrder({
          symbol: "BTC/USD",
          qty: POSITION_QTY,
          side: "buy",
          type: "market",
          time_in_force: "gtc",
        });
        console.log("📈 BUY order placed");
      }
    } else if (signal === "sell") {
      if (btcPos && btcPos.side === "short") {
        console.log("✅ Already in a short position");
      } else {
        if (btcPos) await alpaca.closePosition("BTC/USD");
        await alpaca.createOrder({
          symbol: "BTC/USD",
          qty: POSITION_QTY,
          side: "sell",
          type: "market",
          time_in_force: "gtc",
        });
        console.log("📉 SELL order placed");
      }
    }
  } catch (err) {
    console.error("🚨 Error:", err.message);
  }
}

runStrategy();
