#3️⃣ RandomForest Regressor (Very Reliable)

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
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

# RandomForest Regressor
model = RandomForestRegressor(
    n_estimators=500,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("RandomForest Results")
print("R2 Score:", round(r2, 3))
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))

# Save model
joblib.dump(model, "random_forest_return_3yr.pkl")
print("Model saved successfully!")

# -----------------------------
# Feature Importance (Top 10)
# -----------------------------
importances = model.feature_importances_
features = X.columns

fi = pd.DataFrame({
    "feature": features,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\nTop 10 Important Features:")
print(fi.head(10))

plt.figure(figsize=(10, 5))
plt.barh(fi["feature"][:10], fi["importance"][:10])
plt.gca().invert_yaxis()
plt.title("Top 10 Feature Importances - RandomForest")
plt.xlabel("Importance Score")
plt.show()


# 096