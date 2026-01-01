import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


# 1. Load dataset

df = pd.read_csv("data/touristData.csv")

# Fix totalcount if needed
df["totalcount"] = (
    df["totalcount"].astype(str).str.replace(",", "").astype(float)
)

# Convert month to numeric if needed
if df["month"].dtype == object:
    df["month_num"] = pd.to_datetime(df["month"], format="%B").dt.month
else:
    df["month_num"] = df["month"]

# Create date column
df["date"] = pd.to_datetime(
    df["year"].astype(str) + "-" + df["month_num"].astype(str) + "-01"
)

# Sort for time-series
df = df.sort_values(by=["origincountry_encoded", "date"])


# 2. Feature engineering

df["occupancy_rate"] = df["totalcount"] / df["num_rooms"]
df["occupancy_rate"] = df["occupancy_rate"].clip(0, 1)

df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)

# 3. Feature & target selection

feature_cols = [
    "num_establishments",
    "num_rooms",
    "dollarrate",
    "airpassengerfaresindex",
    "consumerpriceindex",
    "apparent_temperature_mean_celcius",
    "rain_sum_mm",
    "sunshine_duration_hours",
    "month_sin",
    "month_cos"
]

target_cols = [
    "totalcount",        # arrivals
    "tourism_revenue",   # revenue
    "occupancy_rate"     # occupancy
]

X = df[feature_cols]
y = df[target_cols]


# 4. Train-test split (NO shuffle)

split_index = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]


# 5. Train Random Forest

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

model = MultiOutputRegressor(rf)
model.fit(X_train, y_train)


# 6. Evaluation

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("Random Forest Performance")
print("--------------------------")
print(f"RMSE: {rmse:.2f}")
print(f"MAE : {mae:.2f}")

# Per target
for i, col in enumerate(target_cols):
    print(f"\nTarget: {col}")
    print("RMSE:", np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i])))
    print("MAE :", mean_absolute_error(y_test.iloc[:, i], y_pred[:, i]))


# 7. Save model & features

joblib.dump(model, "backend/tourism_rf_model.pkl")
joblib.dump(feature_cols, "backend/model_features.pkl")

print("\nModel and feature list saved successfully.")
