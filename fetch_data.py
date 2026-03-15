import ee
import pandas as pd
import os
import sys
import json
import time
from datetime import datetime, timedelta

# === 1. HYBRID INITIALIZATION ===
def initialize_gee():
    """Authenticates GEE using either GitHub Secrets or local credentials."""
    try:
        if 'EE_JSON_KEY' in os.environ:
            print("🔐 GitHub Actions detected. Authenticating via Service Account...")
            ee_key_data = json.loads(os.environ['EE_JSON_KEY'])
            # Using standard initialization for service accounts
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
        
        # Check size without .getInfo() to save quota/time
        count = col.size().getInfo()
        if count > 0:
            img = col.first()
            print(f"📡 Using GFS Forecast Cycle from: {target_date}")
            return img
            
    raise Exception("❌ No GFS forecast data found. Check your internet or GEE quota.")

def start_forecast_fetch():
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

    # --- 3. CHUNKED EXTRACTION (FIX FOR 5000 LIMIT) ---
    print(f"🚀 Sampling {len(df)} grid points in Kerala...")
    
    # Split the dataframe into chunks of 3000 to stay under the 5000 limit
    chunk_size = 3000
    all_extracted_data = []

    for i in range(0, len(df), chunk_size):
        chunk_df = df.iloc[i : i + chunk_size]
        print(f"  Processing chunk {i//chunk_size + 1}: Points {i} to {i + len(chunk_df)}...")

        features = [
            ee.Feature(ee.Geometry.Point([row.lon, row.lat]), {
                'grid_id': str(row.grid_id),
                'lat': float(row.lat),
                'lon': float(row.lon)
            }) 
            for _, row in chunk_df.iterrows()
        ]
        
        fc = ee.FeatureCollection(features)

        # Apply reduction
        sampled_fc = combined.reduceRegions(
            collection=fc, 
            reducer=ee.Reducer.first(), 
            scale=250 
        ).getInfo()

        # Extract properties from this chunk
        chunk_data = [feat['properties'] for feat in sampled_fc['features']]
        all_extracted_data.extend(chunk_data)

    # --- 4. SAVE LOCALLY ---
    new_df = pd.DataFrame(all_extracted_data)
    
    # Ensure directory exists
    os.makedirs('data', exist_ok=True)
    new_df.to_csv(output_csv, index=False)
    
    print(f"✅ Real-time features saved to: {output_csv}")

if __name__ == "__main__":
    initialize_gee()
    start_forecast_fetch()
