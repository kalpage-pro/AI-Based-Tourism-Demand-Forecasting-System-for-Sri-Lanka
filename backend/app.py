from fastapi import FastAPI
from pydantic import BaseModel
from backend.predict import predict_tourism

app = FastAPI(
    title="AI-Based Tourism Demand Forecasting API",
    description="Predicts tourist arrivals, tourism revenue, and hotel occupancy",
    version="1.0"
)


# Input schema

class TourismInput(BaseModel):
    num_establishments: float
    num_rooms: float
    dollarrate: float
    airpassengerfaresindex: float
    consumerpriceindex: float
    apparent_temperature_mean_celcius: float
    rain_sum_mm: float
    sunshine_duration_hours: float
    month_sin: float
    month_cos: float



# Routes

@app.get("/")
def root():
    return {"message": "Tourism Forecasting API is running"}


@app.post("/predict")
def predict(data: TourismInput):
    input_data = data.dict()
    result = predict_tourism(input_data)
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
