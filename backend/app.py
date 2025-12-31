from flask import Flask, jsonify, request
import joblib
import pandas as pd

# Load the model and features
model = joblib.load("tourism_rf_model.pkl")
features = joblib.load("model_features.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return "Tourism Forecasting API is running"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON
        input_data = request.get_json()
        input_df = pd.DataFrame([input_data])

        # Align columns with training data
        missing_cols = set(features) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0
        input_df = input_df[features]

        # Make prediction
        prediction = model.predict(input_df)[0]
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)