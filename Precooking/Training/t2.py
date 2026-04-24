#training model and saving pkl file

import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# --------------------------------
# Load cleaned dataset
# --------------------------------
df = pd.read_csv("mutual_funds_cleaned.csv")
print("Dataset loaded:", df.shape)

# --------------------------------
# Features & Target
# --------------------------------
X = df.drop(
    ["return_3yr", "scheme_name", "fund_manager", "amc_name"],
    axis=1
)
y = df["return_3yr"]

# --------------------------------
# Train-Test Split
# --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------
# Train RandomForest Model
# --------------------------------
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# --------------------------------
# Evaluation
# --------------------------------
y_pred = model.predict(X_test)
print("R2 Score:", round(r2_score(y_test, y_pred), 3))
print("MAE:", round(mean_absolute_error(y_test, y_pred), 2))

# --------------------------------
# Save model
# --------------------------------
joblib.dump(model, "mf_return_predictor.pkl")
print("Model saved as mf_return_predictor.pkl")
