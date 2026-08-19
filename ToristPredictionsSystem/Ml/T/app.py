import json
import math
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Tourism Prediction UI", layout="centered")
st.title("Tourism ML – Run & Test Prediction")

# ---------- Find metadata.json (current folder OR models/) ----------
APP_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()

CANDIDATES = [
    APP_DIR / "metadata.json",          # your screenshot: metadata.json beside app.py
    APP_DIR / "models" / "metadata.json"
]

META_PATH = next((p for p in CANDIDATES if p.exists()), None)

if META_PATH is None:
    st.error(
        "metadata.json not found.\n\n"
        "Expected one of these:\n"
        f"- {CANDIDATES[0]}\n"
        f"- {CANDIDATES[1]}"
    )
    st.stop()

MODELS_BASE = META_PATH.parent  # where the metadata.json lives (same folder)
st.caption(f"Using metadata from: {META_PATH}")

# ---------- Load metadata ----------
with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

FEATURES_BASE = meta["features_base"]
FEATURES_REV = meta["features_revenue"]
MODEL_FILES = meta["model_files"]  # keys -> file paths (strings)

TARGET_OPTIONS = {
    "Arrivals (total_arrivals)": ("arrivals", FEATURES_BASE, ["rf_arrivals", "xgb_arrivals"]),
    "Revenue (tourism_revenue)": ("revenue", FEATURES_REV, ["rf_revenue", "xgb_revenue"]),
    "Occupancy (hotel_occupancy_rate)": ("occupancy", FEATURES_BASE, ["rf_occupancy", "xgb_occupancy"]),
}

# ---------- Select target/model ----------
target_label = st.selectbox("Select what you want to predict", list(TARGET_OPTIONS.keys()))
target_key, required_features, allowed_models = TARGET_OPTIONS[target_label]

model_key = st.selectbox("Select model", allowed_models)

# model file path: handle both absolute and relative paths
raw_model_path = Path(MODEL_FILES[model_key])
model_path = raw_model_path if raw_model_path.is_absolute() else (MODELS_BASE / raw_model_path)

# Extra fallback: sometimes metadata stores just the filename (no folder)
if not model_path.exists():
    alt = APP_DIR / raw_model_path.name
    if alt.exists():
        model_path = alt

if not model_path.exists():
    st.error(
        f"Model file not found.\n\n"
        f"Tried:\n- {model_path}\n"
        f"Metadata gave: {MODEL_FILES[model_key]}"
    )
    st.stop()

model = joblib.load(model_path)
st.caption(f"Loaded model: {model_key}  |  File: {model_path}")

# ---------- Inputs ----------
st.subheader("Enter inputs")

month_num = st.number_input("month_num (1-12)", min_value=1, max_value=12, value=1, step=1)
year = st.number_input("year", min_value=1900, max_value=2100, value=2025, step=1)

quarter = (int((month_num - 1) // 3) + 1)
month_sin = math.sin(2 * math.pi * month_num / 12)
month_cos = math.cos(2 * math.pi * month_num / 12)

inputs = {
    "year": float(year),
    "month_num": float(month_num),
    "month_sin": float(month_sin),
    "month_cos": float(month_cos),
    "quarter": float(quarter),
}

st.info(f"Auto-calculated: quarter={quarter}, month_sin={month_sin:.4f}, month_cos={month_cos:.4f}")

skip = {"year", "month_num", "month_sin", "month_cos", "quarter"}

for feat in required_features:
    if feat in skip:
        continue

    default_val = 0.0
    if feat == "hotel_occupancy_rate":
        default_val = 0.5

    inputs[feat] = st.number_input(feat, value=float(default_val))

X = pd.DataFrame([[inputs[f] for f in required_features]], columns=required_features)

st.subheader("Input preview")
st.dataframe(X, use_container_width=True)

# ---------- Predict ----------
if st.button("Predict"):
    try:
        pred = model.predict(X)[0]
        st.success(f"✅ Prediction: {pred}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")