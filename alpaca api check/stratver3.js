require("dotenv").config();
const axios = require("axios");
const { EMA } = require("technicalindicators");

const API_KEY = process.env.ALPACA_KEY_ID;
const API_SECRET = process.env.ALPACA_SECRET_KEY;

const symbol = "BTC/USD";
const timeframe = "1Min";
const limit = 100;

async function runEMAStrategy() {
  console.log("📈 Running EMA Strategy...");

  try {
    const response = await axios.get(
      `https://data.alpaca.markets/v1beta3/crypto/us/bars`, // ✅ CORRECT ENDPOINT
      {
        headers: {
          "APCA-API-KEY-ID": API_KEY,
          "APCA-API-SECRET-KEY": API_SECRET,
        },
        params: {
          symbols: symbol, // ✅ symbol must be BTC/USD, ETH/USD etc.
          timeframe: timeframe,
          limit: limit,
        },
      }
    );

    const bars = response.data.bars[symbol];

    if (!bars || bars.length === 0) {
      console.log("⚠️ No bars returned");
      return;
    }

    const closes = bars.map(bar => bar.c);

    const shortEMA = EMA.calculate({ period: 9, values: closes });
    const longEMA = EMA.calculate({ period: 21, values: closes });

    const prevShort = shortEMA[shortEMA.length - 2];
    const prevLong = longEMA[longEMA.length - 2];
    const currShort = shortEMA[shortEMA.length - 1];
    const currLong = longEMA[longEMA.length - 1];

    if (prevShort < prevLong && currShort > currLong) {
      console.log("🟢 Buy signal");
    } else if (prevShort > prevLong && currShort < currLong) {
      console.log("🔴 Sell signal");
    } else {
      console.log("⏸️ No crossover. No trade.");
    }
  } catch (err) {
    console.error("❌ Strategy failed:", err.response?.data || err.message);
  }
}

runEMAStrategy();
