#!/usr/bin/env python3
"""
Sri Lanka Tourist Prediction System - Data Processor
Utility functions for data preprocessing and feature engineering
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

class DataProcessor:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.column_mapping = {}
        
    def detect_columns(self, df):
        """Detect and map column names to standard format"""
        columns = df.columns.str.lower().str.strip()
        
        # Possible column name variations
        mappings = {
            'year': ['year', 'yr', 'yyyy'],
            'month': ['month', 'mon', 'mm', 'month_name'],
            'totalcount': ['totalcount', 'total_count', 'arrivals', 'tourist_arrivals', 
                          'tourists', 'total_arrivals', 'count'],
            'tourism_revenue': ['tourism_revenue', 'revenue', 'total_revenue', 'earnings'],
            'dollarrate': ['dollarrate', 'dollar_rate', 'usd_rate', 'exchange_rate', 'rate'],
            'rooms': ['rooms', 'room_count', 'hotel_rooms', 'accommodation', 'num_rooms'],
            'num_establishments': ['num_establishments', 'establishments', 'hotel_count'],
            'temperature': ['apparent_temperature_mean_celcius', 'temperature', 'temp'],
            'sunshine': ['sunshine_duration_seconds', 'sunshine'],
            'rainfall': ['rain_sum_mm', 'rainfall', 'rain'],
            'precipitation_hours': ['precipitation_hours', 'rain_hours'],
            'airpassengerfaresindex': ['airpassengerfaresindex', 'fare_index'],
            'consumerpriceindex': ['consumerpriceindex', 'cpi', 'price_index']
        }
        
        column_mapping = {}
        
        for standard_name, variations in mappings.items():
            for col in df.columns:
                if col.lower().strip() in variations:
                    column_mapping[col] = standard_name
                    break
        
        return column_mapping
    
    def clean_numeric_columns(self, df):
        """Remove commas and clean numeric columns"""
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # Try to convert after removing commas
                    df[col] = df[col].astype(str).str.replace(',', '').str.strip()
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass
        
        return df
    
    def convert_month_names(self, df, month_col):
        """Convert month names to numbers if needed"""
        if df[month_col].dtype == 'object':
            month_map = {
                'january': 1, 'jan': 1,
                'february': 2, 'feb': 2,
                'march': 3, 'mar': 3,
                'april': 4, 'apr': 4,
                'may': 5,
                'june': 6, 'jun': 6,
                'july': 7, 'jul': 7,
                'august': 8, 'aug': 8,
                'september': 9, 'sep': 9, 'sept': 9,
                'october': 10, 'oct': 10,
                'november': 11, 'nov': 11,
                'december': 12, 'dec': 12
            }
            
            df[month_col] = df[month_col].str.lower().str.strip().map(month_map)
        
        return df
    
    def load_data(self):
        """Load and clean CSV data"""
        try:
            print(f"📂 Reading CSV from: {self.csv_path}")
            
            # Try to read as tab-delimited first
            try:
                self.df = pd.read_csv(
                    self.csv_path,
                    sep='\t',  # Tab delimiter
                    on_bad_lines='skip',
                    encoding='utf-8',
                    low_memory=False
                )
                print("✅ Detected tab-delimited format")
            except:
                # Fallback to comma-delimited
                self.df = pd.read_csv(
                    self.csv_path,
                    sep=',',
                    on_bad_lines='skip',
                    encoding='utf-8',
                    low_memory=False
                )
                print("✅ Detected comma-delimited format")
            
            print(f"✅ Loaded {len(self.df)} rows")
            print(f"📊 Original columns: {self.df.columns.tolist()}")
            
            # Clean numeric columns first
            self.df = self.clean_numeric_columns(self.df)
            
            # Detect and map columns
            self.column_mapping = self.detect_columns(self.df)
            print(f"🔄 Column mapping: {self.column_mapping}")
            
            # Rename columns to standard names
            self.df = self.df.rename(columns=self.column_mapping)
            
            # Check if we have required columns
            required_cols = ['year', 'month', 'totalcount']
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            
            if missing_cols:
                print(f"❌ Missing required columns: {missing_cols}")
                print(f"📊 Available columns after mapping: {self.df.columns.tolist()}")
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Convert month names to numbers if needed
            if 'month' in self.df.columns:
                self.df = self.convert_month_names(self.df, 'month')
            
            # Ensure numeric types
            self.df['year'] = pd.to_numeric(self.df['year'], errors='coerce')
            self.df['month'] = pd.to_numeric(self.df['month'], errors='coerce')
            self.df['totalcount'] = pd.to_numeric(self.df['totalcount'], errors='coerce')
            
            # Drop rows with missing critical values
            before_count = len(self.df)
            self.df = self.df.dropna(subset=['year', 'month', 'totalcount'])
            after_count = len(self.df)
            
            if before_count > after_count:
                print(f"⚠️ Dropped {before_count - after_count} rows with missing critical data")
            
            print(f"✅ After cleaning: {len(self.df)} rows")
            print(f"📊 Final columns: {self.df.columns.tolist()}")
            print(f"📊 Data sample:\n{self.df.head()}")
            
            return self.df
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def create_features(self):
        """Create date-based features and lag features"""
        if self.df is None:
            self.load_data()
        
        print("\n🔧 Creating features...")
        
        # Create date index
        self.df['date'] = pd.to_datetime(
            self.df['year'].astype(int).astype(str) + '-' + 
            self.df['month'].astype(int).astype(str) + '-01',
            errors='coerce'
        )
        
        # Drop rows with invalid dates
        self.df = self.df.dropna(subset=['date'])
        
        # Sort by date
        self.df = self.df.sort_values('date')
        
        # Create month number feature
        self.df['month_num'] = self.df['date'].dt.month
        
        # Create lagged features for totalcount
        for i in range(1, 13):
            self.df[f'totalcount_lag_{i}'] = self.df['totalcount'].shift(i)
        
        # Create lagged features for dollarrate if exists
        if 'dollarrate' in self.df.columns:
            for i in range(1, 13):
                self.df[f'dollarrate_lag_{i}'] = self.df['dollarrate'].shift(i)
        else:
            # Create default dollar rate column
            self.df['dollarrate'] = 320  # Default LKR to USD rate
            for i in range(1, 13):
                self.df[f'dollarrate_lag_{i}'] = 320
        
        # Drop rows with NaN values from lagging
        before_count = len(self.df)
        self.df = self.df.dropna()
        after_count = len(self.df)
        
        print(f"⚠️ Dropped {before_count - after_count} rows due to lag features")
        print(f"✅ Features created. Final shape: {self.df.shape}")
        
        return self.df
    
    def get_train_test_split(self, test_size=0.2):
        """Split data into train and test sets"""
        if self.df is None or 'totalcount_lag_1' not in self.df.columns:
            self.create_features()
        
        # Define features and target
        exclude_cols = ['totalcount', 'year', 'month', 'date']
        if 'tourism_revenue' in self.df.columns:
            exclude_cols.append('tourism_revenue')
        if 'origincountry_encoded' in self.df.columns:
            exclude_cols.append('origincountry_encoded')
        
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        
        X = self.df[feature_cols]
        y = self.df['totalcount']
        
        # Time-series split (use older data for training)
        split_index = int(len(self.df) * (1 - test_size))
        
        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]
        
        print(f"\n📊 Train set: {X_train.shape}, Test set: {X_test.shape}")
        print(f"📊 Features ({len(feature_cols)}): {feature_cols[:10]}... (showing first 10)")
        
        return X_train, X_test, y_train, y_test, feature_cols
    
    def get_revenue_split(self, test_size=0.2):
        """Split data for revenue prediction"""
        if self.df is None:
            self.create_features()
        
        # Check if tourism_revenue exists and has valid values
        if 'tourism_revenue' not in self.df.columns:
            print("⚠️ tourism_revenue column not found, skipping revenue model")
            return None, None, None, None, None
        
        # Remove rows where tourism_revenue is NaN or 0
        revenue_df = self.df[self.df['tourism_revenue'].notna() & (self.df['tourism_revenue'] > 0)].copy()
        
        if len(revenue_df) < 20:  # Minimum rows needed
            print(f"⚠️ Insufficient revenue data ({len(revenue_df)} rows), skipping revenue model")
            return None, None, None, None, None
        
        exclude_cols = ['tourism_revenue', 'year', 'month', 'date', 'totalcount']
        if 'origincountry_encoded' in revenue_df.columns:
            exclude_cols.append('origincountry_encoded')
            
        feature_cols = [col for col in revenue_df.columns if col not in exclude_cols]
        
        X = revenue_df[feature_cols]
        y = revenue_df['tourism_revenue']
        
        split_index = int(len(revenue_df) * (1 - test_size))
        
        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]
        
        print(f"\n📊 Revenue train set: {X_train.shape}, Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, feature_cols

if __name__ == "__main__":
    # Test the processor
    csv_path = Path(__file__).parent.parent / "touristData.csv"
    processor = DataProcessor(csv_path)
    processor.load_data()
    processor.create_features()
    X_train, X_test, y_train, y_test, features = processor.get_train_test_split()
    print(f"\n✅ Data processing test successful!")
    print(f"📊 Sample feature names: {features[:10]}")
