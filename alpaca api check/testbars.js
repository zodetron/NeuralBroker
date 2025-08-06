import Alpaca from '@alpacahq/alpaca-trade-api';
import dotenv from 'dotenv';

dotenv.config();

const alpaca = new Alpaca({
  keyId: process.env.ALPACA_API_KEY,
  secretKey: process.env.ALPACA_SECRET_KEY,
  paper: true,
});

const SYMBOL = 'BTC/USD';

async function fetchBars() {
  try {
    console.log("📈 Running EMA Strategy...");

    const bars = await alpaca.getBars(
      '1Min', // timeframe
      [SYMBOL], // symbols as an array
      {
        start: new Date(Date.now() - 60 * 60 * 1000).toISOString(), // 1 hour ago
        end: new Date().toISOString(), // now
        adjustment: 'raw',
      }
    );

    const btcBars = bars[SYMBOL];
    if (!btcBars || btcBars.length < 20) {
      throw new Error("Not enough bars to calculate EMA.");
    }

    console.log(`✅ Fetched ${btcBars.length} bars for ${SYMBOL}`);
  } catch (err) {
    console.error("❌ Failure: Failed to fetch bars:", err.message);
  }
}

fetchBars();
