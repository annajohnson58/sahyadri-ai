import ee
import pandas as pd
import os
import sys
import json
from datetime import datetime, timedelta

# === 1. HYBRID INITIALIZATION ===
def initialize_gee():
    """Authenticates GEE using either GitHub Secrets or local credentials."""
    try:
        if 'EE_JSON_KEY' in os.environ:
            print("🔐 GitHub Actions detected. Authenticating via Service Account...")
            ee_key_data = json.loads(os.environ['EE_JSON_KEY'])
            credentials = ee.ServiceAccountCredentials(
                ee_key_data['client_email'], 
                key_data=os.environ['EE_JSON_KEY']
            )
            ee.Initialize(credentials)
        else:
            print("💻 Local environment detected. Initializing via user account...")
            ee.Initialize()
        print("✅ Earth Engine initialized. Mode: SAHYADRI_AI_FORECAST")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        sys.exit(1)

def get_verified_gfs():
    """Finds the latest GFS image that includes the precipitation band."""
    for i in range(0, 3):
        target_date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
        next_day = (datetime.now() - timedelta(days=i-1)).strftime('%Y-%m-%d')
        
        col = ee.ImageCollection("NOAA/GFS0P25")\
                .filterDate(target_date, next_day)\
                .filter(ee.Filter.gt('forecast_hours', 0))\
                .sort('system:time_start', False)
        
        if col.size().getInfo() > 0:
            img = col.first()
            print(f"📡 Using GFS Forecast Cycle from: {target_date}")
            return img
            
    raise Exception("❌ No GFS forecast data found. Check your internet or GEE quota.")

def start_forecast_fetch():
    # Use os.path.join for Windows/Linux compatibility
    grid_csv = os.path.join('data', 'grid_locations.csv')
    output_csv = os.path.join('data', 'realtime_weather_features.csv')

    if not os.path.exists(grid_csv):
        print(f"❌ Error: {grid_csv} not found.")
        return

    df = pd.read_csv(grid_csv)
    
    # --- 2. FETCH AND ALIGN LAYERS ---
    gfs_image = get_verified_gfs()
    
    temp = gfs_image.select('temperature_2m_above_ground').rename('temperature')
    wind_u = gfs_image.select('u_component_of_wind_10m_above_ground').rename('u_component_of_wind_10m')
    wind_v = gfs_image.select('v_component_of_wind_10m_above_ground').rename('v_component_of_wind_10m')
    precip = gfs_image.select('total_precipitation_surface').rename('rainfall')

    # NDVI: MODIS 
    ndvi_col = ee.ImageCollection("MODIS/061/MOD13Q1").sort('system:time_start', False)
    ndvi = ee.Image(ee.Algorithms.If(
        ndvi_col.size().gt(0),
        ndvi_col.first().select('NDVI').multiply(0.0001),
        ee.Image(0.6) 
    )).rename('ndvi')

    # TERRAIN: Static NASADEM
    terrain = ee.Terrain.products(ee.Image("NASA/NASADEM_HGT/001")).select(['elevation', 'slope', 'aspect'])

    # COMBINE ALL BANDS
    combined = ee.Image.cat([temp, wind_u, wind_v, ndvi, terrain, precip])

    # --- 3. EXTRACT DATA ---
    features = [ee.Feature(ee.Geometry.Point([row.lon, row.lat]), {'grid_id': row.grid_id}) 
                for _, row in df.iterrows()]
    fc = ee.FeatureCollection(features)

    print(f"🚀 Sampling {len(df)} grid points in Kerala...")
    
    # Reduction logic
    sampled_fc = combined.reduceRegions(
        collection=fc, 
        reducer=ee.Reducer.first(), 
        scale=250 
    ).getInfo()

    # --- 4. SAVE LOCALLY FOR PIPELINE ---
    # We extract the properties from the GEE FeatureCollection and save to CSV
    extracted_data = [feat['properties'] for feat in sampled_fc['features']]
    new_df = pd.DataFrame(extracted_data)
    
    # Ensure directory exists
    os.makedirs('data', exist_ok=True)
    new_df.to_csv(output_csv, index=False)
    
    print(f"✅ Real-time features saved to: {output_csv}")

if __name__ == "__main__":
    initialize_gee()
    start_forecast_fetch()
