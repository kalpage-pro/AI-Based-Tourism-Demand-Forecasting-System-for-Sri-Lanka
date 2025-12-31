import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
from flask import Flask, jsonify, request

# Load dataset
df = pd.read_csv("C:\\Users\\SDJ\\Desktop\\2026 Assignment\\touristData.csv")

# Sort by time
df = df.sort_values("date")

# Define features and target
X = df.drop(columns=["totalCount", "date"])
y = df["totalCount"]

# Encode categorical features
X = pd.get_dummies(X, drop_first=True)

# Time-based split
split_index = int(len(df) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Train Random Forest model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R² Score:", r2)

# Plot feature importances
importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(10,6))
plt.barh(features, importances)
plt.title("Feature Importance for Tourism Demand")
plt.show()

# Save model
joblib.dump(model, "backend/tourism_rf_model.pkl")
print("Model saved successfully")

# Save model features to avoid mismatch errors
joblib.dump(X.columns.tolist(), "backend/model_features.pkl")
print("Model features saved successfully")

# Flask app setup
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON
        input_data = request.get_json()
        input_df = pd.DataFrame([input_data])
        input_df = pd.get_dummies(input_df, drop_first=True)

        # Align columns with training data
        missing_cols = set(X.columns) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0
        input_df = input_df[X.columns]

        # Make prediction
        prediction = model.predict(input_df)[0]
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)