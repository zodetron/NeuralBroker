# You already ran preprocessing

# You already trained & saved model as mf_return_predictor.pkl

# You have mutual_funds_cleaned.csv

#using pkl file , enter fund name and predict future


# import pandas as pd
# import joblib

# # -----------------------------
# # Load cleaned data & model
# # -----------------------------
# df = pd.read_csv("mutual_funds_cleaned.csv")
# model = joblib.load("mf_return_predictor.pkl")

# # -----------------------------
# # User Input
# # -----------------------------
# mf_name = input("Enter Mutual Fund Scheme Name: ").strip()

# # -----------------------------
# # Check if MF exists
# # -----------------------------
# if mf_name not in df["scheme_name"].values:
#     print("❌ Mutual Fund not found in dataset.")
#     print("Available examples:")
#     print(df["scheme_name"].head(5).tolist())
#     exit()

# # -----------------------------
# # Extract MF row
# # -----------------------------
# mf_row = df[df["scheme_name"] == mf_name]

# # Drop non-feature columns
# X_mf = mf_row.drop(
#     ["return_3yr", "scheme_name", "fund_manager", "amc_name"],
#     axis=1
# )

# # -----------------------------
# # Predict future return
# # -----------------------------
# predicted_return = model.predict(X_mf)[0]

# # -----------------------------
# # Output
# # -----------------------------
# print("\n📊 FUTURE PERFORMANCE PREDICTION")
# print("────────────────────────────────")
# print(f"Mutual Fund     : {mf_name}")
# print(f"Predicted 3-Year Return : {predicted_return:.2f} %")


import pandas as pd
import joblib

# --------------------------------
# Load data & model
# --------------------------------
df = pd.read_csv("mutual_funds_cleaned.csv")
model = joblib.load("mf_return_predictor.pkl")

# --------------------------------
# User Input
# --------------------------------
mf_name = input("Enter Mutual Fund Scheme Name: ").strip()

# --------------------------------
# Validate MF
# --------------------------------
if mf_name not in df["scheme_name"].values:
    print(" Mutual Fund not found.")
    print("Try one of these:")
    print(df["scheme_name"].head(10).tolist())
    exit()

# --------------------------------
# Prepare data for prediction
# --------------------------------
mf_row = df[df["scheme_name"] == mf_name]

X_mf = mf_row.drop(
    ["return_3yr", "scheme_name", "fund_manager", "amc_name"],
    axis=1
)

# --------------------------------
# Prediction
# --------------------------------
prediction = model.predict(X_mf)[0]

print("\n📈 FUTURE PERFORMANCE FORECAST")
print("──────────────────────────────")
print("Mutual Fund :", mf_name)
print(f"Predicted 3-Year Return : {prediction:.2f}%")
