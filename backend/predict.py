import pandas as pd
import joblib

# Load model and features
model = joblib.load("tourism_rf_model.pkl")
features = joblib.load("model_features.pkl")

# Load new input data (CSV, not API)
new_data = pd.read_csv("new_input.csv")

# Align columns
for col in features:
    if col not in new_data.columns:
        new_data[col] = 0

new_data = new_data[features]

# Predict
predictions = model.predict(new_data)

# Save results
new_data["predicted_tourist_arrivals"] = predictions
new_data.to_csv("predictions_output.csv", index=False)

print("Predictions saved successfully")
    