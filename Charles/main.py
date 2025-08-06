import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# STEP 1: Load CSV
df = pd.read_csv("maintrades.csv")

# STEP 2: Create target column (1 = profitable trade, 0 = loss)
if 'target' not in df.columns:
    df['target'] = (df['profit_usd'] > 0).astype(int)

# STEP 3: Drop any rows with missing values
df = df.dropna()

# STEP 4: Pick features for training
basic_features = ['opening_price', 'closing_price', 'lots', 'direction', 'hour', 'weekday']
ta_features = [col for col in df.columns if col.startswith(('momentum_', 'trend_', 'volatility_'))]
features = basic_features + ta_features

# STEP 5: Split into input (X) and target (y)
X = df[features]
y = df['target']

# STEP 6: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 7: Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# STEP 8: Evaluate performance
y_pred = model.predict(X_test)
print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

# STEP 9: Save model
joblib.dump(model, "forex_ml_model.pkl")
print("✅ Model saved as 'forex_ml_model.pkl'")
