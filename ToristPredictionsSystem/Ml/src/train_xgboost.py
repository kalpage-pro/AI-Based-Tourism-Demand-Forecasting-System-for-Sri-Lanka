#!/usr/bin/env python3
"""
==============================================================================
AI-Based Tourism Demand Forecasting System - XGBoost Training Pipeline
Sri Lanka Tourism Prediction Project
==============================================================================

This script provides a production-ready training pipeline for tourism demand 
forecasting using XGBoost. It can train models on any tourism-related CSV dataset
with flexible target column selection via command line.

Features:
    - Command-line interface for flexible usage
    - Automatic feature detection for tourism data
    - Robust data preprocessing (missing values, encoding)
    - XGBoost regression model training
    - Comprehensive evaluation metrics (RMSE, MAE, R²)
    - Model persistence (.pkl format)
    - Metadata logging (JSON format)

Usage:
    python train_xgboost.py <csv_path> <target_column>
    
Example:
    python train_xgboost.py ../touristData.csv totalcount
    python train_xgboost.py ../touristData.csv tourism_revenue

Author: Tourism Prediction System
Version: 1.0.0
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import pandas as pd
import numpy as np
import json
import os
import sys
import joblib
import warnings
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Any

# Machine Learning imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Default paths (relative to script location)
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Tourism-related column patterns for automatic detection
TOURISM_COLUMNS = {
    'temporal': ['year', 'month', 'quarter', 'date', 'day', 'week'],
    'arrivals': ['tourist_arrivals', 'arrivals', 'visitors', 'totalcount', 'tourist_count', 'num_tourists'],
    'revenue': ['revenue', 'tourism_revenue', 'earnings', 'income'],
    'accommodation': ['hotel_occupancy', 'occupancy_rate', 'room_occupancy', 'num_rooms', 'num_establishments'],
    'economic': ['exchange_rate', 'dollarrate', 'inflation', 'gdp', 'consumerpriceindex', 'airpassengerfaresindex'],
    'geographic': ['source_market', 'region', 'country', 'origincountry_encoded', 'destination'],
    'weather': ['temperature', 'rainfall', 'rain_sum_mm', 'sunshine_duration_seconds', 
                'apparent_temperature_mean_celcius', 'precipitation_hours']
}

# XGBoost default hyperparameters (optimized for tourism data)
XGBOOST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0
}

# Train/test split configuration
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def print_header(text: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_step(step_num: int, text: str) -> None:
    """Print a formatted step indicator."""
    print(f"\n[Step {step_num}] {text}")
    print("-" * 50)


def print_success(text: str) -> None:
    """Print a success message."""
    print(f"✅ {text}")


def print_warning(text: str) -> None:
    """Print a warning message."""
    print(f"⚠️  {text}")


def print_error(text: str) -> None:
    """Print an error message."""
    print(f"❌ {text}")


def print_info(text: str) -> None:
    """Print an info message."""
    print(f"📊 {text}")


# ==============================================================================
# DATA LOADING FUNCTION
# ==============================================================================

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file.
    
    This function reads a CSV file and returns it as a pandas DataFrame.
    It handles various CSV formats including comma and tab delimited files.
    
    Args:
        csv_path (str): Path to the CSV file (absolute or relative)
        
    Returns:
        pd.DataFrame: Loaded dataset
        
    Raises:
        FileNotFoundError: If the CSV file does not exist
        ValueError: If the file cannot be parsed as CSV
    """
    print_step(1, "Loading Dataset")
    
    # Convert to Path object for better handling
    file_path = Path(csv_path)
    
    # Check if file exists
    if not file_path.exists():
        # Try relative to BASE_DIR
        file_path = BASE_DIR / csv_path
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    print(f"   File path: {file_path}")
    
    # Try to detect the delimiter
    try:
        # First try tab-separated (common for tourism data)
        df = pd.read_csv(file_path, sep='\t')
        if len(df.columns) == 1:
            # If only one column, try comma-separated
            df = pd.read_csv(file_path, sep=',')
    except Exception:
        # Fallback to auto-detection
        df = pd.read_csv(file_path)
    
    print(f"   Rows loaded: {len(df):,}")
    print(f"   Columns found: {len(df.columns)}")
    print_success("Dataset loaded successfully")
    
    return df


# ==============================================================================
# DATASET VALIDATION FUNCTION
# ==============================================================================

def validate_dataset(df: pd.DataFrame, target_column: str) -> bool:
    """
    Validate the loaded dataset for training requirements.
    
    This function checks:
    - Dataset is not empty
    - Target column exists
    - Target column has numeric values
    - Sufficient data points for training
    
    Args:
        df (pd.DataFrame): The loaded dataset
        target_column (str): Name of the target column for prediction
        
    Returns:
        bool: True if validation passes
        
    Raises:
        ValueError: If validation fails with descriptive error message
    """
    print_step(2, "Validating Dataset")
    
    # Check if dataset is empty
    if df.empty:
        raise ValueError("Dataset is empty. Please provide a valid CSV file.")
    
    # Check minimum rows
    MIN_ROWS = 10
    if len(df) < MIN_ROWS:
        raise ValueError(f"Dataset has only {len(df)} rows. Minimum {MIN_ROWS} rows required for training.")
    
    # Display available columns
    print(f"   Available columns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        non_null = df[col].notna().sum()
        print(f"      {i:2d}. {col} ({dtype}, {non_null:,} non-null)")
    
    # Check if target column exists
    if target_column not in df.columns:
        print_error(f"Target column '{target_column}' not found in dataset!")
        print(f"\n   Available columns you can use as target:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            print(f"      - {col}")
        raise ValueError(f"Target column '{target_column}' does not exist in the dataset.")
    
    # Check if target column is numeric
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        print_warning(f"Target column '{target_column}' is not numeric. Will attempt conversion.")
        try:
            df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
        except Exception as e:
            raise ValueError(f"Cannot convert target column to numeric: {e}")
    
    # Check for valid target values
    valid_target_count = df[target_column].notna().sum()
    if valid_target_count < MIN_ROWS:
        raise ValueError(f"Target column has only {valid_target_count} valid values. Minimum {MIN_ROWS} required.")
    
    print(f"\n   Target column: '{target_column}'")
    print(f"   Target stats:")
    print(f"      - Valid values: {valid_target_count:,}")
    print(f"      - Min: {df[target_column].min():,.2f}")
    print(f"      - Max: {df[target_column].max():,.2f}")
    print(f"      - Mean: {df[target_column].mean():,.2f}")
    
    print_success("Dataset validation passed")
    return True


# ==============================================================================
# DATA PREPROCESSING FUNCTION
# ==============================================================================

def preprocess_data(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series, List[str], Dict[str, Any]]:
    """
    Preprocess the dataset for XGBoost training.
    
    This function handles:
    - Dropping rows with invalid target values
    - Removing non-feature columns (like dates, IDs)
    - Handling missing values
    - Encoding categorical variables (one-hot encoding)
    - Separating features and target
    
    Args:
        df (pd.DataFrame): The raw dataset
        target_column (str): Name of the target column
        
    Returns:
        Tuple containing:
            - X (pd.DataFrame): Feature matrix
            - y (pd.Series): Target vector
            - feature_names (List[str]): Names of features used
            - preprocessing_info (Dict): Info about preprocessing steps
    """
    print_step(3, "Preprocessing Data")
    
    # Create a copy to avoid modifying original
    data = df.copy()
    preprocessing_info = {
        'original_rows': len(data),
        'original_columns': len(data.columns),
        'dropped_columns': [],
        'encoded_columns': [],
        'missing_filled': {}
    }
    
    # Step 3.1: Drop rows with missing target values
    initial_rows = len(data)
    data = data.dropna(subset=[target_column])
    dropped_rows = initial_rows - len(data)
    if dropped_rows > 0:
        print(f"   Dropped {dropped_rows:,} rows with missing target values")
    
    # Step 3.2: Separate target from features
    y = data[target_column].copy()
    X = data.drop(columns=[target_column])
    
    # Step 3.3: Identify and drop non-useful columns
    columns_to_drop = []
    
    # Drop date/time columns (they need special handling, keeping year/month)
    date_patterns = ['date', 'datetime', 'timestamp', 'time', 'created', 'updated']
    for col in X.columns:
        col_lower = col.lower()
        # Drop explicit date columns but keep year/month
        if any(pattern in col_lower for pattern in date_patterns) and col_lower not in ['year', 'month', 'day']:
            columns_to_drop.append(col)
        # Drop ID columns
        elif col_lower in ['id', 'index', 'row_id', 'record_id']:
            columns_to_drop.append(col)
    
    if columns_to_drop:
        print(f"   Dropping non-feature columns: {columns_to_drop}")
        X = X.drop(columns=columns_to_drop, errors='ignore')
        preprocessing_info['dropped_columns'] = columns_to_drop
    
    # Step 3.4: Handle categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if categorical_cols:
        print(f"   Encoding categorical columns: {categorical_cols}")
        
        for col in categorical_cols:
            # Check cardinality
            n_unique = X[col].nunique()
            
            if n_unique <= 10:
                # One-hot encoding for low cardinality
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
                preprocessing_info['encoded_columns'].append({
                    'column': col,
                    'method': 'one-hot',
                    'categories': n_unique
                })
            else:
                # Label encoding for high cardinality
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                preprocessing_info['encoded_columns'].append({
                    'column': col,
                    'method': 'label-encoding',
                    'categories': n_unique
                })
    
    # Step 3.5: Handle missing values in features
    missing_cols = X.columns[X.isnull().any()].tolist()
    
    if missing_cols:
        print(f"   Handling missing values in: {missing_cols}")
        
        for col in missing_cols:
            missing_count = X[col].isnull().sum()
            
            if pd.api.types.is_numeric_dtype(X[col]):
                # Fill numeric with median
                fill_value = X[col].median()
                X[col] = X[col].fillna(fill_value)
                preprocessing_info['missing_filled'][col] = {
                    'method': 'median',
                    'value': float(fill_value),
                    'count': int(missing_count)
                }
            else:
                # Fill categorical with mode
                fill_value = X[col].mode()[0] if not X[col].mode().empty else 'Unknown'
                X[col] = X[col].fillna(fill_value)
                preprocessing_info['missing_filled'][col] = {
                    'method': 'mode',
                    'value': str(fill_value),
                    'count': int(missing_count)
                }
    
    # Step 3.6: Keep only numeric columns
    X = X.select_dtypes(include=[np.number])
    
    # Store feature names
    feature_names = X.columns.tolist()
    
    # Summary
    preprocessing_info['final_rows'] = len(X)
    preprocessing_info['final_features'] = len(feature_names)
    
    print(f"\n   Preprocessing Summary:")
    print(f"      - Original rows: {preprocessing_info['original_rows']:,}")
    print(f"      - Final rows: {preprocessing_info['final_rows']:,}")
    print(f"      - Features used: {preprocessing_info['final_features']}")
    print(f"\n   Feature columns:")
    for i, feat in enumerate(feature_names, 1):
        print(f"      {i:2d}. {feat}")
    
    print_success("Data preprocessing completed")
    
    return X, y, feature_names, preprocessing_info


# ==============================================================================
# MODEL TRAINING FUNCTION
# ==============================================================================

def train_model(X_train: pd.DataFrame, y_train: pd.Series, 
                params: Optional[Dict] = None) -> XGBRegressor:
    """
    Train an XGBoost regression model.
    
    This function creates and trains an XGBRegressor with optimized
    hyperparameters for tourism demand forecasting.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target values
        params (Dict, optional): Custom XGBoost parameters
        
    Returns:
        XGBRegressor: Trained model
    """
    print_step(4, "Training XGBoost Model")
    
    # Use default params if not provided
    model_params = params if params else XGBOOST_PARAMS.copy()
    
    print(f"   Model: XGBRegressor")
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Features: {X_train.shape[1]}")
    print(f"\n   Hyperparameters:")
    for key, value in model_params.items():
        print(f"      - {key}: {value}")
    
    # Initialize model
    model = XGBRegressor(**model_params)
    
    # Train model
    print(f"\n   Training in progress...")
    start_time = datetime.now()
    
    model.fit(
        X_train, 
        y_train,
        eval_set=[(X_train, y_train)],
        verbose=False
    )
    
    training_time = (datetime.now() - start_time).total_seconds()
    
    print(f"   Training time: {training_time:.2f} seconds")
    print_success("Model training completed")
    
    return model


# ==============================================================================
# MODEL EVALUATION FUNCTION
# ==============================================================================

def evaluate_model(model: XGBRegressor, X_test: pd.DataFrame, 
                   y_test: pd.Series, target_column: str) -> Dict[str, float]:
    """
    Evaluate the trained model using multiple metrics.
    
    This function calculates:
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)
    - R² Score (Coefficient of Determination)
    - MAPE (Mean Absolute Percentage Error)
    
    Args:
        model (XGBRegressor): Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Actual target values
        target_column (str): Name of target column (for display)
        
    Returns:
        Dict[str, float]: Dictionary of evaluation metrics
    """
    print_step(5, "Evaluating Model")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate MAPE (avoiding division by zero)
    mask = y_test != 0
    if mask.any():
        mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
    else:
        mape = np.nan
    
    metrics = {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2_score': float(r2),
        'mape': float(mape) if not np.isnan(mape) else None,
        'test_samples': len(y_test)
    }
    
    print(f"\n   📈 Model Performance Metrics:")
    print(f"   " + "-" * 40)
    print(f"   RMSE (Root Mean Squared Error): {rmse:,.2f}")
    print(f"   MAE (Mean Absolute Error):      {mae:,.2f}")
    print(f"   R² Score:                       {r2:.4f} ({r2*100:.2f}%)")
    if mape is not None and not np.isnan(mape):
        print(f"   MAPE (Mean Absolute % Error):   {mape:.2f}%")
    print(f"   " + "-" * 40)
    
    # Interpretation
    print(f"\n   📊 Interpretation:")
    if r2 >= 0.9:
        print(f"      Excellent fit! The model explains {r2*100:.1f}% of variance.")
    elif r2 >= 0.7:
        print(f"      Good fit. The model explains {r2*100:.1f}% of variance.")
    elif r2 >= 0.5:
        print(f"      Moderate fit. Consider adding more features or tuning hyperparameters.")
    else:
        print(f"      Poor fit. The model may need more data or different features.")
    
    # Feature importance
    print(f"\n   🎯 Top Feature Importances:")
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(5).iterrows():
        print(f"      - {row['feature']}: {row['importance']:.4f}")
    
    print_success("Model evaluation completed")
    
    return metrics


# ==============================================================================
# MODEL SAVING FUNCTION
# ==============================================================================

def save_model(model: XGBRegressor, target_column: str, 
               model_name: Optional[str] = None) -> str:
    """
    Save the trained model to a .pkl file.
    
    The model is saved in the models directory with a descriptive name
    that includes the target column and timestamp.
    
    Args:
        model (XGBRegressor): Trained model to save
        target_column (str): Target column name (used in filename)
        model_name (str, optional): Custom model name
        
    Returns:
        str: Path to the saved model file
    """
    print_step(6, "Saving Model")
    
    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    if model_name:
        filename = f"{model_name}.pkl"
    else:
        # Create descriptive filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_target = target_column.replace(" ", "_").lower()
        filename = f"xgboost_{clean_target}_{timestamp}.pkl"
    
    model_path = MODELS_DIR / filename
    
    # Save using joblib (better for scikit-learn compatible models)
    joblib.dump(model, model_path)
    
    # Also save a "latest" version for easy access
    latest_path = MODELS_DIR / f"xgboost_{target_column.replace(' ', '_').lower()}_latest.pkl"
    joblib.dump(model, latest_path)
    
    print(f"   Model saved to: {model_path}")
    print(f"   Latest model:   {latest_path}")
    print(f"   File size: {model_path.stat().st_size / 1024:.2f} KB")
    
    print_success("Model saved successfully")
    
    return str(model_path)


# ==============================================================================
# METADATA SAVING FUNCTION
# ==============================================================================

def save_metadata(target_column: str, feature_names: List[str], 
                  metrics: Dict[str, float], preprocessing_info: Dict,
                  csv_path: str, model_path: str) -> str:
    """
    Save training metadata to a JSON file.
    
    This metadata is essential for:
    - Reproducing the training
    - Understanding model performance
    - Using the model for prediction
    
    Args:
        target_column (str): Name of target column
        feature_names (List[str]): List of feature column names
        metrics (Dict[str, float]): Evaluation metrics
        preprocessing_info (Dict): Information about preprocessing steps
        csv_path (str): Path to original CSV file
        model_path (str): Path to saved model
        
    Returns:
        str: Path to the saved metadata file
    """
    print_step(7, "Saving Metadata")
    
    # Create metadata dictionary
    metadata = {
        'model_info': {
            'name': f"XGBoost Tourism Forecasting Model",
            'type': 'XGBRegressor',
            'target_column': target_column,
            'algorithm': 'XGBoost (Extreme Gradient Boosting)',
            'problem_type': 'regression'
        },
        'training_info': {
            'training_date': datetime.now().isoformat(),
            'training_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'source_file': str(csv_path),
            'model_file': str(model_path)
        },
        'features': {
            'count': len(feature_names),
            'names': feature_names
        },
        'metrics': metrics,
        'preprocessing': preprocessing_info,
        'hyperparameters': XGBOOST_PARAMS,
        'version': '1.0.0'
    }
    
    # Generate metadata filename
    clean_target = target_column.replace(" ", "_").lower()
    metadata_filename = f"metadata_{clean_target}_latest.json"
    metadata_path = MODELS_DIR / metadata_filename
    
    # Save to JSON
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"   Metadata saved to: {metadata_path}")
    print(f"\n   Metadata Contents:")
    print(f"      - Model: {metadata['model_info']['name']}")
    print(f"      - Target: {target_column}")
    print(f"      - Features: {len(feature_names)}")
    print(f"      - R² Score: {metrics['r2_score']:.4f}")
    print(f"      - Training Date: {metadata['training_info']['training_timestamp']}")
    
    print_success("Metadata saved successfully")
    
    return str(metadata_path)


# ==============================================================================
# FEATURE COLUMNS SAVING FUNCTION
# ==============================================================================

def save_feature_columns(feature_names: List[str], target_column: str) -> str:
    """
    Save feature column names for prediction use.
    
    This is critical for ensuring predictions use the same features
    in the same order as during training.
    
    Args:
        feature_names (List[str]): List of feature column names
        target_column (str): Target column name
        
    Returns:
        str: Path to saved features file
    """
    clean_target = target_column.replace(" ", "_").lower()
    features_path = MODELS_DIR / f"features_{clean_target}_latest.pkl"
    
    joblib.dump(feature_names, features_path)
    
    print(f"   Features saved to: {features_path}")
    
    return str(features_path)


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main(csv_path: str, target_column: str) -> Dict[str, Any]:
    """
    Main training pipeline function.
    
    This function orchestrates the entire training process:
    1. Load data
    2. Validate dataset
    3. Preprocess data
    4. Split into train/test
    5. Train model
    6. Evaluate model
    7. Save model
    8. Save metadata
    
    Args:
        csv_path (str): Path to the CSV file
        target_column (str): Name of target column to predict
        
    Returns:
        Dict containing model, metrics, and paths
    """
    print_header("AI-Based Tourism Demand Forecasting - XGBoost Training")
    print(f"\n📁 CSV File: {csv_path}")
    print(f"🎯 Target Column: {target_column}")
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Step 1: Load data
        df = load_data(csv_path)
        
        # Step 2: Validate dataset
        validate_dataset(df, target_column)
        
        # Step 3: Preprocess data
        X, y, feature_names, preprocessing_info = preprocess_data(df, target_column)
        
        # Step 4: Split data
        print_step(4, "Splitting Data")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=TEST_SIZE, 
            random_state=RANDOM_STATE
        )
        print(f"   Training set: {len(X_train):,} samples ({(1-TEST_SIZE)*100:.0f}%)")
        print(f"   Test set:     {len(X_test):,} samples ({TEST_SIZE*100:.0f}%)")
        print_success("Data split completed")
        
        # Step 5: Train model (step number adjusted in function)
        model = train_model(X_train, y_train)
        
        # Step 6: Evaluate model
        metrics = evaluate_model(model, X_test, y_test, target_column)
        
        # Step 7: Save model
        model_path = save_model(model, target_column)
        
        # Step 8: Save metadata
        metadata_path = save_metadata(
            target_column, feature_names, metrics, 
            preprocessing_info, csv_path, model_path
        )
        
        # Save feature names for prediction
        features_path = save_feature_columns(feature_names, target_column)
        
        # Final summary
        print_header("Training Complete!")
        print(f"\n   🎉 Summary:")
        print(f"   " + "-" * 50)
        print(f"   Target Column:    {target_column}")
        print(f"   Training Samples: {len(X_train):,}")
        print(f"   Test Samples:     {len(X_test):,}")
        print(f"   Features Used:    {len(feature_names)}")
        print(f"   R² Score:         {metrics['r2_score']:.4f} ({metrics['r2_score']*100:.2f}%)")
        print(f"   RMSE:             {metrics['rmse']:,.2f}")
        print(f"   MAE:              {metrics['mae']:,.2f}")
        print(f"   " + "-" * 50)
        print(f"\n   📂 Output Files:")
        print(f"      Model:    {model_path}")
        print(f"      Metadata: {metadata_path}")
        print(f"      Features: {features_path}")
        print(f"\n   ⏰ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return {
            'model': model,
            'metrics': metrics,
            'feature_names': feature_names,
            'model_path': model_path,
            'metadata_path': metadata_path,
            'features_path': features_path,
            'preprocessing_info': preprocessing_info
        }
        
    except FileNotFoundError as e:
        print_error(f"File Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print_error(f"Validation Error: {e}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ==============================================================================
# COMMAND LINE INTERFACE
# ==============================================================================

def print_usage():
    """Print usage instructions."""
    print("""
==============================================================================
  AI-Based Tourism Demand Forecasting System - XGBoost Training Script
==============================================================================

USAGE:
    python train_xgboost.py <csv_path> <target_column>

ARGUMENTS:
    csv_path        Path to the CSV file containing tourism data
    target_column   Name of the column to predict (e.g., totalcount, tourism_revenue)

EXAMPLES:
    python train_xgboost.py ../touristData.csv totalcount
    python train_xgboost.py ../touristData.csv tourism_revenue
    python train_xgboost.py data/tourism_data.csv tourist_arrivals

COMMON TARGET COLUMNS:
    - totalcount        (tourist arrivals count)
    - tourism_revenue   (revenue from tourism)
    - num_rooms         (hotel room occupancy)

NOTE:
    - The CSV file must contain the specified target column
    - The script will automatically detect and use relevant features
    - Output will be saved in the models/ directory
==============================================================================
""")


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) < 3:
        print_usage()
        
        if len(sys.argv) == 2 and sys.argv[1] in ['--help', '-h', 'help']:
            sys.exit(0)
        
        print_error("Missing required arguments!")
        print("Please provide both CSV path and target column.")
        print("\nExample: python train_xgboost.py ../touristData.csv totalcount")
        sys.exit(1)
    
    # Get arguments
    csv_path = sys.argv[1]
    target_column = sys.argv[2]
    
    # Run training
    result = main(csv_path, target_column)
