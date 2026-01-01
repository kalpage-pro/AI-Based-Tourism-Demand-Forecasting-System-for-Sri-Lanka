import joblib
import numpy as np


# Load trained model & features

MODEL_PATH = "backend/tourism_rf_model.pkl"
FEATURES_PATH = "backend/model_features.pkl"

model = joblib.load(MODEL_PATH)
feature_cols = joblib.load(FEATURES_PATH)


# Prediction function

def predict_tourism(input_data: dict):
    """
    input_data: dictionary with feature names as keys
    returns: arrivals, revenue, occupancy
    """

    # Ensure correct feature order
    input_vector = [input_data[feature] for feature in feature_cols]

    input_array = np.array(input_vector).reshape(1, -1)

    prediction = model.predict(input_array)[0]

    return {
        "predicted_arrivals": int(prediction[0]),
        "predicted_revenue": float(prediction[1]),
        "predicted_occupancy_rate": round(float(prediction[2]), 2)
    }



# Test locally

if __name__ == "__main__":
    sample_input = {
        "num_establishments": 2200,
        "num_rooms": 40000,
        "dollarrate": 320.5,
        "airpassengerfaresindex": 145.2,
        "consumerpriceindex": 190.3,
        "apparent_temperature_mean_celcius": 29.5,
        "rain_sum_mm": 180.0,
        "sunshine_duration_hours": 190.0,
        "month_sin": 0.5,
        "month_cos": -0.86
    }

    result = predict_tourism(sample_input)
    print("Prediction Result:")
    print(result)
