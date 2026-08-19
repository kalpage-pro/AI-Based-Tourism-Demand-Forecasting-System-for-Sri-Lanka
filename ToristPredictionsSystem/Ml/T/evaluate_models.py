#!/usr/bin/env python3
"""
Model Evaluation Script
Provides model comparison metrics and performance analysis
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

def get_model_info(model, model_key):
    """Extract model information and parameters"""
    info = {
        "model_key": model_key,
        "model_type": "Random Forest" if model_key.startswith('rf') else "XGBoost",
        "target": model_key.split('_')[1].capitalize()
    }
    
    # Get model parameters
    if hasattr(model, 'get_params'):
        params = model.get_params()
        # Select key parameters
        if 'rf' in model_key:
            info["parameters"] = {
                "n_estimators": params.get('n_estimators'),
                "max_depth": params.get('max_depth'),
                "min_samples_split": params.get('min_samples_split'),
                "min_samples_leaf": params.get('min_samples_leaf'),
                "max_features": str(params.get('max_features'))
            }
        else:
            info["parameters"] = {
                "n_estimators": params.get('n_estimators'),
                "max_depth": params.get('max_depth'),
                "learning_rate": params.get('learning_rate'),
                "subsample": params.get('subsample'),
                "colsample_bytree": params.get('colsample_bytree')
            }
    
    # Get number of features
    if hasattr(model, 'n_features_in_'):
        info["n_features"] = model.n_features_in_
    
    # Get number of trees (for ensemble models)
    if hasattr(model, 'n_estimators'):
        info["n_trees"] = model.n_estimators
    
    return info

def compare_models():
    """Compare all models"""
    metadata = load_metadata()
    model_files = metadata.get("model_files", {})
    best_params = metadata.get("best_params", {})
    
    comparison = {
        "arrivals": {},
        "revenue": {},
        "occupancy": {}
    }
    
    model_keys = ['rf_arrivals', 'xgb_arrivals', 'rf_revenue', 'xgb_revenue', 'rf_occupancy', 'xgb_occupancy']
    
    for model_key in model_keys:
        if model_key not in model_files:
            continue
        
        try:
            model = load_model(model_files[model_key])
            info = get_model_info(model, model_key)
            
            # Add best params from metadata
            if model_key in best_params:
                info["tuned_params"] = best_params[model_key]
            
            # Determine target category
            target = model_key.split('_')[1]
            model_type = "random_forest" if model_key.startswith('rf') else "xgboost"
            
            comparison[target][model_type] = info
            
        except Exception as e:
            target = model_key.split('_')[1]
            model_type = "random_forest" if model_key.startswith('rf') else "xgboost"
            comparison[target][model_type] = {"error": str(e)}
    
    # Add summary
    comparison["summary"] = {
        "total_models": len([m for m in model_keys if m in model_files]),
        "model_types": ["Random Forest", "XGBoost"],
        "prediction_targets": ["Tourist Arrivals", "Tourism Revenue", "Hotel Occupancy"],
        "recommendations": generate_recommendations(comparison)
    }
    
    return comparison

def get_model_metrics(model_key):
    """Get detailed metrics for a specific model"""
    metadata = load_metadata()
    model_files = metadata.get("model_files", {})
    best_params = metadata.get("best_params", {})
    
    if model_key not in model_files and model_key != 'all':
        return {"error": f"Model {model_key} not found"}
    
    if model_key == 'all':
        return compare_models()
    
    model = load_model(model_files[model_key])
    info = get_model_info(model, model_key)
    
    # Add training metrics if available (these would be stored during training)
    info["metrics"] = {
        "note": "Run with test data to calculate actual metrics",
        "available_metrics": ["RMSE", "MAE", "MAPE", "R2"]
    }
    
    if model_key in best_params:
        info["tuned_params"] = best_params[model_key]
    
    return info

def generate_recommendations(comparison):
    """Generate model selection recommendations"""
    recommendations = []
    
    recommendations.append({
        "scenario": "High Accuracy Priority",
        "recommendation": "XGBoost models generally provide better accuracy for complex patterns",
        "models": ["xgb_arrivals", "xgb_revenue", "xgb_occupancy"]
    })
    
    recommendations.append({
        "scenario": "Interpretability Priority", 
        "recommendation": "Random Forest models are easier to interpret and explain",
        "models": ["rf_arrivals", "rf_revenue", "rf_occupancy"]
    })
    
    recommendations.append({
        "scenario": "Balanced Approach",
        "recommendation": "Use ensemble predictions combining both model types",
        "technique": "Average predictions from RF and XGBoost for each target"
    })
    
    return recommendations

def main():
    parser = argparse.ArgumentParser(description='Model evaluation and comparison')
    parser.add_argument('--action', type=str, default='compare',
                        choices=['compare', 'metrics'],
                        help='Action to perform')
    parser.add_argument('--model', type=str, default='all',
                        help='Model key for metrics action')
    args = parser.parse_args()
    
    if args.action == 'compare':
        result = compare_models()
    else:
        result = get_model_metrics(args.model)
    
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
