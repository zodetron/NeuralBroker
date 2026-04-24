#ElasticNet Regression (Best Linear Baseline)

import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load cleaned dataset
df = pd.read_csv("mutual_funds_cleaned.csv")

# Features & target
X = df.drop(
    ["return_3yr", "scheme_name", "fund_manager", "amc_name"],
    axis=1
)
y = df["return_3yr"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features (VERY IMPORTANT for ElasticNet)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ElasticNet Model
model = ElasticNet(
    alpha=0.05,      # regularization strength
    l1_ratio=0.5,    # mix of L1 (Lasso) and L2 (Ridge)
    random_state=42,
    max_iter=5000
)

# Train
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)

# Evaluation
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("ElasticNet Results")
print("R2 Score:", round(r2, 3))
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))

# Save model & scaler
joblib.dump(model, "elasticnet_return_3yr.pkl")
joblib.dump(scaler, "elasticnet_scaler.pkl")

print("Model and scaler saved successfully!")


# 093