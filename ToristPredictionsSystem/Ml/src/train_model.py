#!/usr/bin/env python3
"""
Sri Lanka Tourist Prediction System - Model Training
Train RandomForest models for predicting tourist arrivals, revenue, and room occupancy
"""

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data_processor import DataProcessor

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_FILE = BASE_DIR / "touristData.csv"
MODELS_DIR = BASE_DIR / "models"

# Create models directory if it doesn't exist
MODELS_DIR.mkdir(exist_ok=True)

def train_tourist_arrivals_model(X_train, X_test, y_train, y_test):
    """Train model for tourist arrivals prediction"""
    print("\n🚀 Training Tourist Arrivals Model...")
    
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"📊 Tourist Arrivals Model Performance:")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²: {r2:.4f}")
    
    # Save model
    model_path = MODELS_DIR / "tourist_arrivals_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Model saved to {model_path}")
    
    return model, {'mae': mae, 'rmse': rmse, 'r2': r2}

def train_revenue_model(X_train, X_test, y_train, y_test):
    """Train model for tourism revenue prediction"""
    print("\n🚀 Training Tourism Revenue Model...")
    
    if X_train is None:
        print("⚠️ Skipping revenue model - no valid data")
        return None, None
    
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"📊 Revenue Model Performance:")
    print(f"  MAE: ${mae:,.2f}")
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  R²: {r2:.4f}")
    
    # Save model
    model_path = MODELS_DIR / "revenue_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Model saved to {model_path}")
    
    return model, {'mae': mae, 'rmse': rmse, 'r2': r2}

def train_rooms_model(X_train, X_test, y_train, y_test):
    """Train model for rooms prediction (using tourist count as proxy)"""
    print("\n🚀 Training Rooms Model...")
    
    # Estimate rooms needed (assume 2 tourists per room on average)
    y_train_rooms = y_train / 2
    y_test_rooms = y_test / 2
    
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train_rooms)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test_rooms, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_rooms, y_pred))
    r2 = r2_score(y_test_rooms, y_pred)
    
    print(f"📊 Rooms Model Performance:")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²: {r2:.4f}")
    
    # Save model
    model_path = MODELS_DIR / "rooms_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Model saved to {model_path}")
    
    return model, {'mae': mae, 'rmse': rmse, 'r2': r2}

def main():
    print("🇱🇰 Sri Lanka Tourist Prediction System - Model Training")
    print("=" * 60)
    
    try:
        # Load and process data
        processor = DataProcessor(DATA_FILE)
        processor.load_data()
        processor.create_features()
        
        # Get train/test splits for tourist arrivals
        X_train, X_test, y_train, y_test, features = processor.get_train_test_split()
        
        # Save feature names
        feature_path = MODELS_DIR / "features.pkl"
        with open(feature_path, 'wb') as f:
            pickle.dump(features, f)
        print(f"✅ Feature names saved to {feature_path}")
        
        # Train tourist arrivals model
        arrivals_model, arrivals_metrics = train_tourist_arrivals_model(
            X_train, X_test, y_train, y_test
        )
        
        # Train revenue model
        X_train_rev, X_test_rev, y_train_rev, y_test_rev, _ = processor.get_revenue_split()
        revenue_model, revenue_metrics = train_revenue_model(
            X_train_rev, X_test_rev, y_train_rev, y_test_rev
        )
        
        # Train rooms model
        rooms_model, rooms_metrics = train_rooms_model(
            X_train, X_test, y_train, y_test
        )
        
        print("\n" + "=" * 60)
        print("✅ All models trained successfully!")
        print("=" * 60)
        
        # Save training metadata
        metadata = {
            'version': '1.0.0',
            'training_date': pd.Timestamp.now().isoformat(),
            'data_points': len(processor.df),
            'features': features,
            'metrics': {
                'tourist_arrivals': arrivals_metrics,
                'revenue': revenue_metrics if revenue_metrics else {},
                'rooms': rooms_metrics
            }
        }
        
        metadata_path = MODELS_DIR / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"✅ Metadata saved to {metadata_path}")
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
