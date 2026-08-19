# 🇱🇰 Sri Lanka Tourist Prediction System - Machine Learning

This directory contains the machine learning models and scripts for predicting tourist arrivals, tourism revenue, and room occupancy in Sri Lanka.

## 📁 Directory Structure

```
Ml/
├── touristData.csv          # Historical tourist data
├── requirements.txt         # Python dependencies
├── models/                  # Trained models (generated after training)
│   ├── tourist_arrivals_model.pkl
│   ├── revenue_model.pkl
│   └── rooms_model.pkl
└── src/
    ├── train_model.py       # Model training script
    ├── predict.py           # Prediction script
    └── data_processor.py    # Data preprocessing utilities
```

## 🚀 Setup

### 1. Install Python Dependencies

```bash
cd Ml
pip install -r requirements.txt
```

### 2. Prepare Your Data

Ensure `touristData.csv` has the following columns:
- `year` - Year (e.g., 2020)
- `month` - Month (1-12)
- `totalcount` - Total tourist arrivals
- `dollarrate` - USD to LKR exchange rate
- `revenue` (optional) - Tourism revenue
- `rooms` (optional) - Room occupancy

## 🤖 Training Models

Train all models using the historical data:

```bash
python src/train_model.py
```

This will:
- Load and preprocess the data from `touristData.csv`
- Create lag features (12 months)
- Train RandomForest models for:
  - Tourist Arrivals
  - Revenue (if data available)
  - Rooms (if data available)
- Save trained models to `models/` directory
- Display evaluation metrics (MAE, RMSE, R²)

## 🔮 Making Predictions

Make predictions using the trained models:

```bash
# Predict all metrics for January 2026
python src/predict.py --year 2026 --month 1 --dollar-rate 200

# Predict only tourist arrivals
python src/predict.py --year 2026 --month 1 --dollar-rate 200 --type tourist_arrivals

# Predict with specific dollar rate
python src/predict.py --year 2026 --month 6 --dollar-rate 320 --type all
```

### Parameters:
- `--year` (required): Year for prediction (2000-2050)
- `--month` (required): Month for prediction (1-12)
- `--dollar-rate` (optional): USD to LKR exchange rate (default: 200)
- `--type` (optional): Prediction type - `tourist_arrivals`, `revenue`, `rooms`, or `all` (default: all)

### Output:
The prediction script outputs JSON with the results:
```json
{
  "year": 2026,
  "month": 1,
  "dollar_rate": 200,
  "tourist_arrivals": 125000,
  "confidence_tourist_arrivals": 0.87,
  "revenue": 15625000,
  "confidence_revenue": 0.82,
  "rooms": 150000,
  "confidence_rooms": 0.80,
  "overall_accuracy": 0.83,
  "model_version": "1.0.0"
}
```

## 📊 Data Processing

Use the data processor for custom preprocessing:

```bash
python src/data_processor.py touristData.csv
```

## 🧪 Model Features

The models use the following features:

1. **Temporal Features:**
   - `month_num` - Month number (1-12)
   
2. **Lag Features (12 months):**
   - `totalcount_lag_1` to `totalcount_lag_12`
   - `dollarrate_lag_1` to `dollarrate_lag_12`

3. **Input Features:**
   - Current `dollarrate`

## 📈 Model Performance

After training, the models typically achieve:
- **Tourist Arrivals:** R² > 0.85
- **Revenue:** R² > 0.80
- **Rooms:** R² > 0.78

## 🔧 Customization

### Adjust Model Parameters

Edit `train_model.py` to modify RandomForest parameters:

```python
model = RandomForestRegressor(
    n_estimators=100,      # Number of trees
    max_depth=10,          # Maximum tree depth
    min_samples_split=5,   # Minimum samples to split
    random_state=42
)
```

### Add More Features

Edit `data_processor.py` to add custom features:

```python
# Example: Add seasonal indicators
df['is_festival_month'] = df['month_num'].isin([4, 5, 7, 10]).astype(int)
```

## 🇱🇰 Sri Lanka Tourism Insights

The models are specifically designed for Sri Lankan tourism patterns:

- **High Season:** December - March
- **Mid Season:** July - August  
- **Low Season:** April - June, September - November

Major festivals affecting tourism:
- Sinhala & Tamil New Year (April)
- Vesak (May)
- Esala Perahera (July/August)
- Deepavali (October/November)

## 📝 Notes

- Models are trained on historical data and may need retraining with new data
- Predictions are more accurate for near-term forecasts (1-6 months)
- External factors (global events, disasters, etc.) may affect accuracy
- Regular model retraining is recommended as new data becomes available

## 🆘 Troubleshooting

**Error: Model file not found**
- Run `python src/train_model.py` first to generate model files

**Error: Data file not found**
- Ensure `touristData.csv` exists in the Ml/ directory

**Poor predictions**
- Check if you have enough historical data (minimum 24 months recommended)
- Retrain models with updated data
- Verify input dollar rate is reasonable (100-400 LKR/USD)

## 📞 Integration with Backend

The backend Node.js application automatically calls these Python scripts via the ML Service. No manual intervention needed for API predictions.

---

**Made with ❤️ for Sri Lanka Tourism 🇱🇰**
