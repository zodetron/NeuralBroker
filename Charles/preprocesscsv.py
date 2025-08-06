import pandas as pd

# Load your raw CSV
df = pd.read_csv("gptgive.csv")

# Step 1: Convert opening time to datetime
df['opening_time_utc'] = pd.to_datetime(df['opening_time_utc'], errors='coerce')

# Step 2: Add hour and weekday
df['hour'] = df['opening_time_utc'].dt.hour
df['weekday'] = df['opening_time_utc'].dt.weekday

# Step 3: Convert trade type to direction: 1 = Buy, 0 = Sell
df['direction'] = df['type'].str.lower().map({'buy': 1, 'sell': 0})

# Step 4: Define target column: 1 = profitable trade, 0 = loss/break-even
df['target'] = (df['profit_usd'] > 0).astype(int)

# Optional: Check how many missing rows exist (just for info)
print("Missing values summary:\n", df.isnull().sum())

# DO NOT drop missing rows yet — you'll clean later if needed
df.to_csv("processed_gptgive.csv", index=False)
print("✅ Preprocessing complete! File saved as 'processed_gptgive.csv'")
