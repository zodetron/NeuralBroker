import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Change filename if needed
df = pd.read_csv("MF_India_AI.csv")
# df = pd.read_excel("mutual_funds_india.xlsx")

print("Initial shape:", df.shape)
df.columns = (
    df.columns.str.strip()
              .str.lower()
              .str.replace(" ", "_")
              .str.replace("%", "")
)
df.drop_duplicates(subset=["scheme_name", "amc_name"], inplace=True)
print("After duplicates removed:", df.shape)
num_cols = [
    "alpha", "beta", "sharpe", "sortino",
    "standard_deviation", "expense_ratio",
    "fund_size", "fund_age",
    "return_1yr", "return_3yr", "return_5yr"
]

cat_cols = [
    "category", "sub_category", "amc_name", "risk_level"
]
for col in num_cols:
    # Convert to string and remove unwanted characters
    df[col] = (
        df[col]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .replace(["nan", "None", "--", ""], np.nan)
    )

    # Convert to numeric safely
    df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing values using median
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
# One-hot encode category & sub-category
df = pd.get_dummies(
    df,
    columns=["category", "sub_category"],
    drop_first=True
)

# Label encode AMC name
le = LabelEncoder()
df["amc_name_encoded"] = le.fit_transform(df["amc_name"])
scale_cols = [
    "alpha", "beta", "sharpe", "sortino",
    "standard_deviation", "expense_ratio",
    "fund_size", "fund_age",
    "risk_adjusted_score",
    "cost_efficiency",
    "stability_score",
    "experience_factor"
]

scaler = StandardScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])
X = df.drop(["return_3yr", "scheme_name", "amc_name"], axis=1)
y = df["return_3yr"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
df.to_csv("mutual_funds_cleaned.csv", index=False)
print("Cleaned dataset saved successfully!")
