require('dotenv').config();
const alpaca = require('@alpacahq/alpaca');
const { EMA } = require('technicalindicators');

const client = alpaca({
  credentials: {
    key: process.env.ALPACA_KEY_ID,
    secret: process.env.ALPACA_SECRET_KEY,
    paper: true,
  },
  rate_limit: true,
});

const symbol = 'BTC/USD';
const timeframe = '1Min'; // or '5Min' or '15Min'

async function runStrategy() {
  console.log('📈 Running EMA Crossover Strategy...');

  try {
    // Fetch historical crypto bars
    const barsResponse = await client.data.getCryptoBars({
      symbol,
      timeframe,
      start: new Date(Date.now() - 1000 * 60 * 60 * 2).toISOString(), // last 2 hours
      limit: 100,
    });

    const bars = barsResponse.bars;

    if (!bars || bars.length < 21) {
      console.log('Not enough data to calculate EMAs');
      return;
    }

    const closes = bars.map(bar => bar.close);

    const ema12 = EMA.calculate({ period: 12, values: closes });
    const ema20 = EMA.calculate({ period: 20, values: closes });

    const latestEma12 = ema12[ema12.length - 1];
    const prevEma12 = ema12[ema12.length - 2];

    const latestEma20 = ema20[ema20.length - 1];
    const prevEma20 = ema20[ema20.length - 2];

    console.log(`📊 EMA12: ${latestEma12.toFixed(2)} | EMA20: ${latestEma20.toFixed(2)}`);

    const positions = await client.trading.getPositions();
    const btcPosition = positions.find(pos => pos.symbol === 'BTC/USD');

    // Check for crossover signals
    const bullishCrossover = prevEma12 < prevEma20 && latestEma12 > latestEma20;
    const bearishCrossover = prevEma12 > prevEma20 && latestEma12 < latestEma20;

    if (bullishCrossover) {
      console.log('📈 Buy signal detected.');

      if (btcPosition && parseFloat(btcPosition.qty) > 0) {
        console.log('✅ Buy position already open.');
      } else {
        if (btcPosition) {
          console.log('🔁 Closing short position first...');
          await client.trading.closePosition({ symbol: 'BTC/USD' });
        }

        console.log('🚀 Placing buy order...');
        await client.trading.submitOrder({
          symbol: 'BTC/USD',
          qty: 0.01, // adjust your lot size
          side: 'buy',
          type: 'market',
          time_in_force: 'gtc',
        });
      }
    }

    if (bearishCrossover) {
      console.log('📉 Sell signal detected.');

      if (btcPosition && parseFloat(btcPosition.qty) < 0) {
        console.log('✅ Sell position already open.');
      } else {
        if (btcPosition) {
          console.log('🔁 Closing long position first...');
          await client.trading.closePosition({ symbol: 'BTC/USD' });
        }

        console.log('🚀 Placing sell order...');
        await client.trading.submitOrder({
          symbol: 'BTC/USD',
          qty: 0.01, // adjust your lot size
          side: 'sell',
          type: 'market',
          time_in_force: 'gtc',
        });
      }
    }

    if (!bullishCrossover && !bearishCrossover) {
      console.log('⏸️ No crossover signal currently.');
    }
  } catch (error) {
    console.error('❌ Strategy failed:', error.message || error);
  }
}

// Run it once (for testing)
// You can also schedule it with node-cron or setInterval
runStrategy();
