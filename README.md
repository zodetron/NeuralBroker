#NeuralBroker

**NeuralBroker** is an intelligent auto-trading bot built for **BTC/USD** that executes entries and exits based on **EMA crossover** and **volume spike confirmation**.  
It combines technical precision with adaptive logic to make smarter trading decisions — fully automated and backtest-ready.

---

## Features

- **EMA Crossover Strategy**
  - Uses short-term (EMA 12) and long-term (EMA 20) exponential moving averages.
  - Auto-detects bullish and bearish crossovers.
-  **Volume Spike Filter**
  - Confirms trade signals only when volume surges beyond the recent average.
  - Helps avoid false signals during low volatility.
-  **Auto Entry & Exit**
  - Places buy/sell orders automatically on signal confirmation.
  - Includes stop-loss and take-profit logic.
-  **Backtesting Support**
  - Test strategy performance over historical BTC/USD data.
-  **Real-Time Alerts**
  - Optional terminal notifications for trade events.
-  **Logging & Analytics**
  - Records every trade, PnL, and strategy metric for analysis.

---

##  Strategy Logic

1. **EMA 12 / EMA 20 crossover:**
   - `EMA12 > EMA20` → Bullish crossover → *Potential Buy*
   - `EMA12 < EMA20` → Bearish crossover → *Potential Sell*

2. **Volume spike filter:**
   - Current volume must be **> 1.5x** the rolling 20-period average.

3. **Execution:**
   - Entry signal only triggers if both crossover and volume spike align.
   - Auto exit when opposite crossover occurs or trailing stop is hit.

---

##  Installation

```bash
# Clone the repository
git clone https://github.com/<your-user



