import pandas as pd
from pathlib import Path

csv_path = Path(__file__).parent.parent / "touristData.csv"

print("🔍 Inspecting CSV file...")
print("=" * 60)

# Read just the first few rows to see structure - Try tab delimiter first
try:
    df = pd.read_csv(csv_path, sep='\t', nrows=5, on_bad_lines='skip')
    print("✅ Detected tab-delimited format")
except:
    df = pd.read_csv(csv_path, nrows=5, on_bad_lines='skip')
    print("✅ Detected comma-delimited format")

print(f"\n📊 Column Names:")
print(df.columns.tolist())

print(f"\n📊 Data Types:")
print(df.dtypes)

print(f"\n📊 First 5 rows:")
print(df.head())

print(f"\n📊 CSV Shape: {df.shape}")
print("=" * 60)