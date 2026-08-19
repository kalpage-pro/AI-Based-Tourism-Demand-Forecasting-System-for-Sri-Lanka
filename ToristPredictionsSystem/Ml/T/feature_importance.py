#!/usr/bin/env python3
"""
Feature Importance Extraction Script
Extracts and returns feature importance scores from trained models
"""

import json
import argparse
import joblib
import numpy as np
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

APP_DIR = Path(__file__).parent
META_PATH = APP_DIR / "metadata.json"

def load_metadata():
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def load_model(model_file):
    model_path = APP_DIR / model_file
    if not model_path.exists():
        model_path = APP_DIR / Path(model_file).name
    return joblib.load(model_path)

def get_feature_importance(model, feature_names):
    """Extract feature importance from a model"""
    importance = None
    
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_).flatten()
    
    if importance is None:
        return None
    
    # Normalize to percentages
    importance = (importance / importance.sum()) * 100
    
    # Create sorted list
    feature_importance = [
        {"feature": name, "importance": round(float(imp), 2)}
        for name, imp in zip(feature_names, importance)
    ]
    
    return sorted(feature_importance, key=lambda x: x["importance"], reverse=True)

def get_feature_descriptions():
    """Return human-readable descriptions for features"""
    return {
        "year": "Year",
        "month_num": "Month Number",
        "month_sin": "Month (Cyclical Sin)",
        "month_cos": "Month (Cyclical Cos)",
        "quarter": "Quarter",
        "dollarrate": "USD/LKR Exchange Rate",
        "apparent_temperature": "Apparent Temperature (°C)",
        "sunshine": "Sunshine Hours",
        "rain": "Rainfall (mm)",
        "precipitation_hours": "Precipitation Hours",
        "num_establishments": "Number of Hotel Establishments",
        "num_rooms": "Number of Hotel Rooms",
        "airfare_index": "Airfare Price Index",
        "cpi": "Consumer Price Index",
        "arrivals_lag1": "Arrivals (Previous Month)",
        "arrivals_lag2": "Arrivals (2 Months Ago)",
        "arrivals_lag3": "Arrivals (3 Months Ago)",
        "arrivals_lag12": "Arrivals (Same Month Last Year)",
        "arrivals_roll3": "3-Month Rolling Average Arrivals",
        "arrivals_roll6": "6-Month Rolling Average Arrivals",
        "arrivals_std3": "3-Month Arrivals Std Dev",
        "arrivals_yoy": "Year-over-Year Arrivals Change",
        "revenue_lag1": "Revenue (Previous Month)",
        "hotel_occupancy_rate": "Hotel Occupancy Rate"
    }

def main():
    parser = argparse.ArgumentParser(description='Extract feature importance')
    parser.add_argument('--type', type=str, default='all',
                        choices=['all', 'arrivals', 'revenue', 'occupancy'],
                        help='Model type to analyze')
    args = parser.parse_args()
    
    metadata = load_metadata()
    model_files = metadata.get("model_files", {})
    descriptions = get_feature_descriptions()
    
    results = {}
    
    models_to_analyze = []
    if args.type == 'all':
        models_to_analyze = ['rf_arrivals', 'xgb_arrivals', 'rf_revenue', 'xgb_revenue', 'rf_occupancy', 'xgb_occupancy']
    else:
        models_to_analyze = [f'rf_{args.type}', f'xgb_{args.type}']
    
    for model_key in models_to_analyze:
        if model_key not in model_files:
            continue
            
        try:
            model = load_model(model_files[model_key])
            
            # Determine features for this model
            if 'revenue' in model_key:
                features = metadata.get('features_revenue', metadata.get('features_base', []))
            else:
                features = metadata.get('features_base', [])
            
            importance = get_feature_importance(model, features)
            
            if importance:
                # Add descriptions
                for item in importance:
                    item["description"] = descriptions.get(item["feature"], item["feature"])
                
                results[model_key] = {
                    "model": model_key,
                    "model_type": "Random Forest" if model_key.startswith('rf') else "XGBoost",
                    "target": model_key.split('_')[1],
                    "features": importance[:10],  # Top 10
                    "all_features": importance
                }
        except Exception as e:
            results[model_key] = {"error": str(e)}
    
    # Calculate average importance across models
    if results:
        all_features = {}
        for model_result in results.values():
            if "all_features" in model_result:
                for feat in model_result["all_features"]:
                    if feat["feature"] not in all_features:
                        all_features[feat["feature"]] = []
                    all_features[feat["feature"]].append(feat["importance"])
        
        avg_importance = [
            {
                "feature": feat,
                "importance": round(sum(vals) / len(vals), 2),
                "description": descriptions.get(feat, feat)
            }
            for feat, vals in all_features.items()
        ]
        avg_importance = sorted(avg_importance, key=lambda x: x["importance"], reverse=True)
        
        results["average"] = {
            "features": avg_importance[:10],
            "all_features": avg_importance
        }
    
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
