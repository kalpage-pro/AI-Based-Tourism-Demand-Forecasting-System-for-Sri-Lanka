import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("data/touristData.csv")

df["totalcount"] = (
    df["totalcount"].astype(str).str.replace(",", "").astype(float)
)

df["month_num"] = pd.to_datetime(df["month"], format="%B").dt.month
df["date"] = pd.to_datetime(
    df["year"].astype(str) + "-" + df["month_num"].astype(str) + "-01"
)

df = df.sort_values(by=["origincountry_encoded", "date"])

# -----------------------------
# Feature engineering (MATCH TRAINING)
# -----------------------------
df["occupancy_rate"] = df["totalcount"] / df["num_rooms"]
df["occupancy_rate"] = df["occupancy_rate"].clip(0, 1)

df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)

# -----------------------------
# Load feature list from training
# -----------------------------
feature_cols = joblib.load("backend/model_features.pkl")

target_cols = [
    "totalcount",
    "tourism_revenue",
    "occupancy_rate"
]

# Ensure feature names match those used during training
feature_mapping = {
    'dollarrate': 'dollarRate',
    'airpassengerfaresindex': 'AirPassengerFaresIndex',
    'consumerpriceindex': 'consumerPriceIndex',
    'origincountry_encoded': 'originCountry_encoded',
    'month_august': 'month_August',
    'month_december': 'month_December',
    'month_february': 'month_February',
    'month_january': 'month_January',
    'month_july': 'month_July',
    'month_june': 'month_June',
    'month_march': 'month_March',
    'month_may': 'month_May',
    'month_november': 'month_November',
    'month_october': 'month_October',
    'month_september': 'month_September'
}

df.rename(columns=feature_mapping, inplace=True)

# Ensure all required columns are present
for col in feature_cols:
    if col not in df.columns:
        df[col] = 0  # Add missing columns with default values

X = df[feature_cols]
y = df[target_cols]

# -----------------------------
# Train-test split (same logic)
# -----------------------------
split_index = int(len(df) * 0.8)
X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

# -----------------------------
# Load model & predict
# -----------------------------
model = joblib.load("backend/tourism_rf_model.pkl")
y_pred = model.predict(X_test)

# Adjust prediction handling based on the shape of y_pred
if y_pred.ndim == 1:
    y_pred = np.expand_dims(y_pred, axis=1)  # Reshape to 2D if predictions are 1D

# -----------------------------
# Plot function
# -----------------------------
def plot_actual_vs_pred(actual, predicted, title, ylabel):
    plt.figure()
    plt.plot(actual.values, label="Actual")
    plt.plot(predicted, label="Predicted")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

# -----------------------------
# Generate plots
# -----------------------------
plot_actual_vs_pred(
    y_test["totalcount"],
    y_pred[:, 0] if y_pred.shape[1] > 0 else y_pred.flatten(),
    "Actual vs Predicted Tourist Arrivals",
    "Tourist Arrivals"
)

if y_pred.shape[1] > 1:
    plot_actual_vs_pred(
        y_test["tourism_revenue"],
        y_pred[:, 1],
        "Actual vs Predicted Tourism Revenue",
        "Revenue"
    )

if y_pred.shape[1] > 2:
    plot_actual_vs_pred(
        y_test["occupancy_rate"],
        y_pred[:, 2],
        "Actual vs Predicted Occupancy Rate",
        "Occupancy Rate"
    )
