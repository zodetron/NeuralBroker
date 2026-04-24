#lightgbm REGRESSOR 
import pandas as pd
import numpy as np
import joblib

from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load cleaned dataset
df = pd.read_csv("mutual_funds_cleaned.csv")

# Features and target
X = df.drop(
    ["return_3yr", "scheme_name", "fund_manager", "amc_name"],
    axis=1
)
y = df["return_3yr"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# LightGBM Regressor
model = LGBMRegressor(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=-1,           # -1 lets LightGBM decide
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("LightGBM Results")
print("R2 Score:", round(r2, 3))
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))

# Save model
joblib.dump(model, "lightgbm_return_3yr.pkl")
print("Model saved successfully!")


# 097