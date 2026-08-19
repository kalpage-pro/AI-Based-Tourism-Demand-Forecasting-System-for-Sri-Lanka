# XGBoost Tourism Forecasting - Training Guide

## Overview

This guide explains how to use the XGBoost training script (`train_xgboost.py`) to train tourism demand forecasting models for the Sri Lanka Tourism Prediction System.

## Quick Start

### Basic Command

```bash
# Navigate to the src directory
cd Ml/src

# Train model with your data
python train_xgboost.py <csv_path> <target_column>
```

### Example Commands

```bash
# Predict tourist arrivals using existing data
python train_xgboost.py ../touristData.csv totalcount

# Predict tourism revenue
python train_xgboost.py ../touristData.csv tourism_revenue

# Use sample data
python train_xgboost.py ../data/sample_tourism_data.csv tourist_arrivals

# Predict hotel occupancy
python train_xgboost.py ../data/sample_tourism_data.csv hotel_occupancy
```

---

## Sample CSV Structure

### Required Format

Your CSV file should have:
- Row headers (column names) in the first row
- Comma-separated or tab-separated values
- At least one numeric column to use as target
- At least 10 rows of data

### Example CSV (Tourism Data)

```csv
year,month,tourist_arrivals,hotel_occupancy,tourism_revenue,exchange_rate,inflation,source_market,region
2018,1,125000,72.5,15000000,153.5,2.1,India,South_Asia
2018,2,118000,68.3,14200000,154.2,2.3,India,South_Asia
2018,3,132000,75.8,16500000,155.0,2.4,India,South_Asia
...
```

### Common Columns

The script automatically recognizes these tourism-related columns:

| Category | Column Names |
|----------|-------------|
| Temporal | year, month, quarter, week, day |
| Arrivals | tourist_arrivals, arrivals, visitors, totalcount |
| Revenue | revenue, tourism_revenue, earnings |
| Accommodation | hotel_occupancy, occupancy_rate, num_rooms |
| Economic | exchange_rate, dollarrate, inflation, consumerpriceindex |
| Geographic | source_market, region, country |
| Weather | temperature, rainfall, rain_sum_mm, sunshine_duration_seconds |

---

## Expected Terminal Output

When you run the training script, you'll see output like this:

```
======================================================================
  AI-Based Tourism Demand Forecasting - XGBoost Training
======================================================================

📁 CSV File: ../touristData.csv
🎯 Target Column: totalcount
⏰ Started: 2024-01-15 14:30:25

[Step 1] Loading Dataset
--------------------------------------------------
   File path: C:\...\touristData.csv
   Rows loaded: 1,250
   Columns found: 15
✅ Dataset loaded successfully

[Step 2] Validating Dataset
--------------------------------------------------
   Available columns (15):
       1. year (int64, 1,250 non-null)
       2. month (object, 1,250 non-null)
       3. totalcount (int64, 1,250 non-null)
       ...

   Target column: 'totalcount'
   Target stats:
      - Valid values: 1,250
      - Min: 1,500.00
      - Max: 165,000.00
      - Mean: 78,542.50
✅ Dataset validation passed

[Step 3] Preprocessing Data
--------------------------------------------------
   Dropping non-feature columns: ['date']
   Encoding categorical columns: ['month', 'source_market']

   Preprocessing Summary:
      - Original rows: 1,250
      - Final rows: 1,250
      - Features used: 13

   Feature columns:
       1. year
       2. dollarrate
       3. apparent_temperature_mean_celcius
       ...
✅ Data preprocessing completed

[Step 4] Splitting Data
--------------------------------------------------
   Training set: 1,000 samples (80%)
   Test set:     250 samples (20%)
✅ Data split completed

[Step 4] Training XGBoost Model
--------------------------------------------------
   Model: XGBRegressor
   Training samples: 1,000
   Features: 13

   Hyperparameters:
      - n_estimators: 200
      - max_depth: 6
      - learning_rate: 0.1
      ...

   Training in progress...
   Training time: 2.45 seconds
✅ Model training completed

[Step 5] Evaluating Model
--------------------------------------------------

   📈 Model Performance Metrics:
   ----------------------------------------
   RMSE (Root Mean Squared Error): 5,234.56
   MAE (Mean Absolute Error):      3,892.12
   R² Score:                       0.9234 (92.34%)
   MAPE (Mean Absolute % Error):   8.52%
   ----------------------------------------

   📊 Interpretation:
      Excellent fit! The model explains 92.3% of variance.

   🎯 Top Feature Importances:
      - month: 0.2345
      - dollarrate: 0.1892
      - year: 0.1456
      - consumerpriceindex: 0.0982
      - origincountry_encoded: 0.0875
✅ Model evaluation completed

[Step 6] Saving Model
--------------------------------------------------
   Model saved to: ..\models\xgboost_totalcount_20240115_143028.pkl
   Latest model:   ..\models\xgboost_totalcount_latest.pkl
   File size: 245.32 KB
✅ Model saved successfully

[Step 7] Saving Metadata
--------------------------------------------------
   Metadata saved to: ..\models\metadata_totalcount_latest.json

   Metadata Contents:
      - Model: XGBoost Tourism Forecasting Model
      - Target: totalcount
      - Features: 13
      - R² Score: 0.9234
      - Training Date: 2024-01-15 14:30:28
✅ Metadata saved successfully

======================================================================
  Training Complete!
======================================================================

   🎉 Summary:
   --------------------------------------------------
   Target Column:    totalcount
   Training Samples: 1,000
   Test Samples:     250
   Features Used:    13
   R² Score:         0.9234 (92.34%)
   RMSE:             5,234.56
   MAE:              3,892.12
   --------------------------------------------------

   📂 Output Files:
      Model:    ..\models\xgboost_totalcount_20240115_143028.pkl
      Metadata: ..\models\metadata_totalcount_latest.json
      Features: ..\models\features_totalcount_latest.pkl

   ⏰ Completed: 2024-01-15 14:30:28
```

---

## Output Files

After training, the following files are created in the `models/` directory:

| File | Purpose |
|------|---------|
| `xgboost_{target}_{timestamp}.pkl` | Timestamped model file |
| `xgboost_{target}_latest.pkl` | Latest model (for easy access) |
| `metadata_{target}_latest.json` | Training metadata and metrics |
| `features_{target}_latest.pkl` | Feature column names (for prediction) |

### Metadata JSON Structure

```json
{
  "model_info": {
    "name": "XGBoost Tourism Forecasting Model",
    "type": "XGBRegressor",
    "target_column": "totalcount",
    "algorithm": "XGBoost (Extreme Gradient Boosting)",
    "problem_type": "regression"
  },
  "training_info": {
    "training_date": "2024-01-15T14:30:28",
    "training_timestamp": "2024-01-15 14:30:28",
    "source_file": "../touristData.csv",
    "model_file": "..\\models\\xgboost_totalcount_20240115_143028.pkl"
  },
  "features": {
    "count": 13,
    "names": ["year", "dollarrate", "apparent_temperature", "..."]
  },
  "metrics": {
    "rmse": 5234.56,
    "mae": 3892.12,
    "r2_score": 0.9234,
    "mape": 8.52,
    "test_samples": 250
  },
  "hyperparameters": {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "..."
  },
  "version": "1.0.0"
}
```

---

## Evaluation Metrics Explained

| Metric | Description | Good Value |
|--------|-------------|------------|
| **RMSE** | Root Mean Squared Error - Average prediction error | Lower is better |
| **MAE** | Mean Absolute Error - Average absolute difference | Lower is better |
| **R² Score** | How well model explains variance | > 0.7 is good, > 0.9 is excellent |
| **MAPE** | Mean Absolute Percentage Error | < 10% is good |

---

## Extending to Other Models

The script is designed to be easily extended. To add RandomForest or LightGBM:

### Option 1: Modify the script

```python
# In train_xgboost.py, add to imports:
from sklearn.ensemble import RandomForestRegressor
# or
from lightgbm import LGBMRegressor

# In train_model() function, add model selection:
def train_model(X_train, y_train, model_type='xgboost'):
    if model_type == 'xgboost':
        model = XGBRegressor(**XGBOOST_PARAMS)
    elif model_type == 'randomforest':
        model = RandomForestRegressor(n_estimators=200, random_state=42)
    elif model_type == 'lightgbm':
        model = LGBMRegressor(n_estimators=200, random_state=42)
    
    model.fit(X_train, y_train)
    return model
```

### Option 2: Add command-line argument

```bash
python train_xgboost.py ../touristData.csv totalcount --model randomforest
```

---

## Troubleshooting

### Common Errors

| Error | Solution |
|-------|----------|
| `FileNotFoundError` | Check the CSV path is correct |
| `Target column not found` | Check column name matches exactly |
| `Not enough data` | Need at least 10 rows |
| `ModuleNotFoundError: xgboost` | Run `pip install xgboost` |

### Quick Fixes

```bash
# Install missing packages
pip install xgboost joblib pandas numpy scikit-learn

# Check available columns in your CSV
python -c "import pandas as pd; print(pd.read_csv('your_file.csv').columns.tolist())"
```

---

## Integration with Web System

The trained model can be loaded in your Node.js backend using a Python microservice:

```python
# In your prediction service
import joblib

# Load model
model = joblib.load('models/xgboost_totalcount_latest.pkl')
features = joblib.load('models/features_totalcount_latest.pkl')

# Make prediction
prediction = model.predict(input_data[features])
```

---

## Contact

For issues or enhancements, update the training script in `Ml/src/train_xgboost.py`.
