// server.js
import express from 'express';
import dotenv from 'dotenv';
import AlpacaPkg from '@alpacahq/alpaca-trade-api';

dotenv.config();

const Alpaca = AlpacaPkg;
const app = express();
const PORT = process.env.PORT || 3000;

const alpaca = new Alpaca({
  keyId: process.env.ALPACA_API_KEY,
  secretKey: process.env.ALPACA_SECRET_KEY,
  paper: true,
});

app.use(express.json());

app.post('/webhook', async (req, res) => {
  const data = req.body;

  try {
    const symbol = data.ticker || 'BTC/USD'; // default fallback
    const side = data.side?.toLowerCase(); // 'buy' or 'sell'
    const qty = parseFloat(data.qty) || 1;

    if (!['buy', 'sell'].includes(side)) {
      return res.status(400).json({ error: 'Invalid side' });
    }

    const order = await alpaca.createOrder({
      symbol,
      qty,
      side,
      type: 'market',
      time_in_force: 'gtc',
    });

    console.log(`✅ Order placed: ${side} ${qty} ${symbol}`);
    res.status(200).json(order);
  } catch (err) {
    console.error('❌ Failed to place order:', err.message);
    res.status(500).json({ error: err.message });
  }
});

app.listen(PORT, () => {
  console.log(`🚀 Server listening on port ${PORT}`);
});
