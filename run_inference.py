import pandas as pd
import numpy as np
import joblib
import os
import sys
# Set logging level to suppress unnecessary TensorFlow messages in the GitHub Action log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

# === 1. CONFIGURATION ===
INPUT_PATH = os.path.join("data", "realtime_weather_features.csv")
OUTPUT_PATH = os.path.join("data", "realtime_predictions.csv")

XGB_PATH = os.path.join("models", "fire_xgb_v4.pkl")
LSTM_PATH = os.path.join("models", "fire_lstm_v4.keras")
SCALER_PATH = os.path.join("models", "scaler_v4.pkl")

def run_ensemble_inference():
    # 1. Validation Check
    if not all(os.path.exists(p) for p in [XGB_PATH, LSTM_PATH, SCALER_PATH]):
        print("❌ Error: Ensemble assets (XGB, LSTM, or Scaler) missing in 'models/'.")
        return

    if not os.path.exists(INPUT_PATH):
        print(f"❌ Error: Input data not found at {INPUT_PATH}. Run fetch_data.py first.")
        return

    # 2. Load Assets
    print("📂 Loading Ensemble Assets (v4)...")
    try:
        xgb_model = joblib.load(XGB_PATH)
        lstm_model = load_model(LSTM_PATH)
        scaler = joblib.load(SCALER_PATH)
        df = pd.read_csv(INPUT_PATH)
    except Exception as e:
        print(f"❌ Failed to load assets or data: {e}")
        return

    if df.empty:
        print("⚠️ Warning: Input CSV is empty. No predictions to run.")
        return

    # 3. Data Pre-processing
    print(f"🛠️ Pre-processing features for {len(df)} grid points...")
    
    # Calculate Wind Magnitude (GFS gives U/V components)
    if 'u_component_of_wind_10m' in df.columns and 'v_component_of_wind_10m' in df.columns:
        df['wind'] = np.sqrt(df['u_component_of_wind_10m']**2 + df['v_component_of_wind_10m']**2)
    
    # Temperature Check: Convert Kelvin to Celsius if GFS returns Kelvin (K - 273.15)
    # GFS temperature is often in Kelvin. If values are > 200, it's definitely Kelvin.
    if df['temperature'].mean() > 200:
        df['temperature'] = df['temperature'] - 273.15

    # Dryness Index (Temp / Rainfall + 1)
    df['dryness_index'] = df['temperature'] / (df['rainfall'] + 1)

    # Required feature list (MUST match your training column order)
    features = ['ndvi', 'temperature', 'rainfall', 'wind', 'dryness_index', 'elevation', 'slope']
    
    # Handle missing features or NaNs
    for feat in features:
        if feat not in df.columns:
            df[feat] = 0 # Fallback for missing columns
    
    df[features] = df[features].fillna(df[features].mean())

    # 4. Ensemble Inference
    print("🔮 Running Ensemble Inference (XGBoost + LSTM)...")
    X_scaled = scaler.transform(df[features])

    # XGBoost Prediction
    prob_xgb = xgb_model.predict_proba(X_scaled)[:, 1]
    
    # LSTM Prediction
    X_lstm = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))
    prob_lstm = lstm_model.predict(X_lstm, verbose=0).flatten()

    # Blended Probability
    df['fire_prob'] = (prob_xgb + prob_lstm) / 2

    # 5. Severity Categorization
    def get_severity(p):
        if p > 0.8: return "CRITICAL"
        if p > 0.6: return "HIGH"
        if p > 0.4: return "MODERATE"
        return "LOW"

    df['severity'] = df['fire_prob'].apply(get_severity)
    df['forecast_date'] = (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d %H:%M')

    # Keep only necessary columns for the Dashboard to save space/speed
    output_cols = ['grid_id', 'lat', 'lon', 'fire_prob', 'severity', 'temperature', 'ndvi', 'forecast_date']
    # Filter only the columns that actually exist
    final_cols = [c for c in output_cols if c in df.columns]
    
    df[final_cols].to_csv(OUTPUT_PATH, index=False)
    
    print("-" * 30)
    print(f"✅ ENSEMBLE PREDICTION COMPLETE")
    print(f"🔥 Critical Alerts: {len(df[df['fire_prob'] > 0.8])}")
    print(f"💾 Results saved to: {OUTPUT_PATH}")
    print("-" * 30)

if __name__ == "__main__":
    run_ensemble_inference()
