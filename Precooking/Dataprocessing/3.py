import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
df = pd.read_csv("MF_India_AI.csv")
print("Initial shape:", df.shape)
df.columns = (
    df.columns.str.strip()
              .str.lower()
              .str.replace(" ", "_")
)
df.drop_duplicates(subset=["scheme_name", "amc_name"], inplace=True)
print("After duplicates removed:", df.shape)
df.rename(columns={
    "fund_size_cr": "fund_size",
    "fund_age_yr": "fund_age",
    "sd": "standard_deviation",
    "returns_1yr": "return_1yr",
    "returns_3yr": "return_3yr",
    "returns_5yr": "return_5yr"
}, inplace=True)
num_cols = [
    "min_sip", "min_lumpsum", "expense_ratio",
    "fund_size", "fund_age",
    "sortino", "alpha", "standard_deviation",
    "beta", "sharpe",
    "return_1yr", "return_3yr", "return_5yr",
    "rating"
]

cat_cols = [
    "category", "sub_category",
    "amc_name", "risk_level"
]
for col in num_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .replace(["nan", "None", "--", ""], np.nan)
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
df["risk_level"] = df["risk_level"].astype(int)
for col in num_cols:
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    df[col] = df[col].clip(lower, upper)
df["risk_adjusted_score"] = df["sharpe"] + df["sortino"]
df["cost_efficiency"] = df["return_3yr"] / (df["expense_ratio"] + 1e-5)
df["stability_score"] = 1 / (df["standard_deviation"] + 1e-5)
df["experience_factor"] = df["fund_age"] * df["alpha"]
df = pd.get_dummies(
    df,
    columns=["category", "sub_category"],
    drop_first=True
)

le = LabelEncoder()
df["amc_name_encoded"] = le.fit_transform(df["amc_name"])
scale_cols = [
    "min_sip", "min_lumpsum", "expense_ratio",
    "fund_size", "fund_age",
    "alpha", "beta", "sharpe", "sortino",
    "standard_deviation",
    "risk_adjusted_score",
    "cost_efficiency",
    "stability_score",
    "experience_factor"
]

scaler = StandardScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])
X = df.drop(
    ["return_3yr", "scheme_name", "fund_manager", "amc_name"],
    axis=1
)
y = df["return_3yr"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
df.to_csv("mutual_funds_cleaned.csv", index=False)
print("Cleaned dataset saved successfully!")
