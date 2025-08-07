import fetch from 'node-fetch';
import dotenv from 'dotenv';
dotenv.config();

const API_KEY = process.env.ALPACA_API_KEY;
const SECRET_KEY = process.env.ALPACA_SECRET_KEY;

const headers = {
  'APCA-API-KEY-ID': API_KEY,
  'APCA-API-SECRET-KEY': SECRET_KEY,
};

const url = 'https://data.alpaca.markets/v1beta3/crypto/us/latest/trades?symbols=BTC/USD';

async function fetchLivePrice() {
  try {
    const res = await fetch(url, { headers });
    const raw = await res.text();

    console.log("📡 Status:", res.status);
    console.log("🔁 Raw Response:", raw);

    const data = JSON.parse(raw);

    const price = data.trades['BTC/USD'].p;
    console.log(`💰 Live BTC/USD Price: $${price}`);
  } catch (error) {
    console.error("❌ Error:", error.message);
  }
}

fetchLivePrice();
