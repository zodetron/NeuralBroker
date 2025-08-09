import Alpaca from '@alpacahq/alpaca-trade-api';

// Initialize Alpaca client
const alpaca = new Alpaca({
  keyId: 'ALPACA_API_KEY',
  secretKey: 'ALPACA_SECRET_KEY',
  paper: true, // true for paper trading, false for live
});

// Fetch and display open positions
async function getOpenPositions() {
  try {
    const positions = await alpaca.getPositions();
    if (positions.length === 0) {
      console.log("No open positions.");
    } else {
      console.log("Open Positions:");
      positions.forEach(pos => {
        console.log(`Symbol: ${pos.symbol}`);
        console.log(`Qty: ${pos.qty}`);
        console.log(`Side: ${pos.side}`);
        console.log(`Entry Price: ${pos.avg_entry_price}`);
        console.log(`Current Price: ${pos.current_price}`);
        console.log(`Unrealized P/L: ${pos.unrealized_pl}`);
        console.log('---------------------------');
      });
    }
  } catch (err) {
    console.error("Error fetching positions:", err);
  }
}

getOpenPositions();
