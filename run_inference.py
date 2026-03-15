import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

# === 1. CONFIGURATION (OS-Agnostic Paths) ===
# Using os.path.join ensures these work on both Windows and Linux
INPUT_PATH = os.path.join("data", "realtime_weather_features.csv")
OUTPUT_PATH = os.path.join("data", "realtime_predictions.csv")

XGB_PATH = os.path.join("models", "fire_xgb_v4.pkl")
LSTM_PATH = os.path.join("models", "fire_lstm_v4.keras")
SCALER_PATH = os.path.join("models", "scaler_v4.pkl")

def run_ensemble_inference():
    # Check if model assets exist before starting
    if not all(os.path.exists(p) for p in [XGB_PATH, LSTM_PATH, SCALER_PATH]):
        print("❌ Error: Ensemble assets (XGB, LSTM, or Scaler) missing in 'models/'.")
        return

    if not os.path.exists(INPUT_PATH):
        print(f"❌ Error: Input data not found at {INPUT_PATH}. Run fetch_data.py first.")
        return

    # --- 2. LOAD ASSETS ---
    print("📂 Loading Ensemble Assets (v4)...")
    xgb_model = joblib.load(XGB_PATH)
    lstm_model = load_model(LSTM_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    # Load Realtime Data sampled from GEE
    df = pd.read_csv(INPUT_PATH)
    
    # --- 3. DATA PRE-PROCESSING ---
    print("🛠️ Pre-processing features...")
    
    # Calculate Wind Magnitude if components exist
    if 'u_component_of_wind_10m' in df.columns and 'v_component_of_wind_10m' in df.columns:
        df['wind'] = np.sqrt(df['u_component_of_wind_10m']**2 + df['v_component_of_wind_10m']**2)
    elif 'wind' not in df.columns:
        df['wind'] = 2.5  # Standard fallback if wind data is missing

    # Create 'dryness_index' (Simple heuristic: Temp / (Rainfall + 1))
    # Note: Ensure temperature is in Celsius as per your training logic
    df['dryness_index'] = df['temperature'] / (df['rainfall'] + 1)

    # Feature list must match training exactly
    features = ['ndvi', 'temperature', 'rainfall', 'wind', 'dryness_index', 'elevation', 'slope']
    
    # Handle missing values if any (GEE occasionally returns nulls on cloud-masked pixels)
    df[features] = df[features].fillna(df[features].mean())

    # Scale the features
    X_scaled = scaler.transform(df[features])

    # --- 4. ENSEMBLE INFERENCE (Soft Voting) ---
    print("🔮 Running Ensemble Inference (XGBoost + LSTM)...")
    
    # XGBoost Probability
    prob_xgb = xgb_model.predict_proba(X_scaled)[:, 1]
    
    # LSTM Probability (Reshaped to [Samples, Time_Steps, Features])
    X_lstm = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))
    prob_lstm = lstm_model.predict(X_lstm).flatten()

    # Ensemble Average (The 'Foresight' Blend)
    df['fire_prob'] = (prob_xgb + prob_lstm) / 2

    # --- 5. CATEGORIZATION & EXPORT ---
    def get_severity(p):
        if p > 0.8: return "CRITICAL"
        if p > 0.6: return "HIGH"
        if p > 0.4: return "MODERATE"
        return "LOW"

    df['severity'] = df['fire_prob'].apply(get_severity)
    
    # Set prediction date (Targeting 48 hours ahead)
    df['forecast_date'] = (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d')

    # Save Results
    df.to_csv(OUTPUT_PATH, index=False)
    
    print("-" * 30)
    print(f"✅ ENSEMBLE PREDICTION COMPLETE")
    print(f"📅 Target Forecast: {df['forecast_date'].iloc[0]}")
    print(f"🔥 High/Critical Alerts: {len(df[df['fire_prob'] > 0.6])}")
    print(f"💾 Results saved to: {OUTPUT_PATH}")
    print("-" * 30)

if __name__ == "__main__":
    run_ensemble_inference()
