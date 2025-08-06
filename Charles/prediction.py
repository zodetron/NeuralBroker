import pandas as pd
from ta import add_all_ta_features
import joblib
import datetime

# Load your trained model (if saved)
# model = joblib.load("forex_ml_model.pkl")  # Uncomment if you've saved it

# Simulate live data (can replace with API later)
live = {
    'opening_price': 118000,
    'closing_price': 118100,
    'lots': 3,
    'direction': 1,  # 1 for buy, 0 for sell
    'hour': datetime.datetime.utcnow().hour,
    'weekday': datetime.datetime.utcnow().weekday(),
}

live_df = pd.DataFrame([live])

# Add fake indicators for testing (replace when using real OHLC data)
live_df = add_all_ta_features(
    live_df,
    open="opening_price",
    high="opening_price",
    low="opening_price",
    close="closing_price",
    volume="lots",
    fillna=True
)

# Use same features as training
features = [
    'opening_price', 'closing_price', 'lots', 'direction', 'hour', 'weekday',
    'momentum_rsi', 'trend_ema_fast', 'trend_macd', 'volatility_bbm'
]

# Ensure only existing columns are used
features = [col for col in features if col in live_df.columns]

# Predict
X_live = live_df[features]
prediction = model.predict(X_live)[0]

# Interpret result
if prediction == 1:
    print("📈 Model says: ENTER BUY trade (paper)")
else:
    print("📉 Model says: ENTER SELL trade (paper)")
