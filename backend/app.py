import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Flatten
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import DBSCAN
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import requests
import json
import os

app = FastAPI(title="AeroRakshak AI - ISRO Hackathon")

# CPCB AQI Breakpoint Table
CPCB_BREAKPOINTS = {
    'PM2.5': [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 150), 
               (91, 120, 151, 200), (121, 250, 201, 300), (251, 9999, 301, 500)],
    'PM10': [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 150),
             (251, 350, 151, 200), (351, 430, 201, 300), (431, 9999, 301, 500)],
    'NO2': [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 150),
            (181, 280, 151, 200), (281, 400, 201, 300), (401, 9999, 301, 500)],
    'SO2': [(0, 5, 0, 50), (6, 10, 51, 100), (11, 25, 101, 150),
            (26, 40, 151, 200), (41, 80, 201, 300), (81, 9999, 301, 500)],
    'CO': [(0, 1.0, 0, 50), (1.1, 2.0, 51, 100), (2.1, 4.0, 101, 150),
           (4.1, 10.0, 151, 200), (10.1, 17.0, 201, 300), (17.1, 9999, 301, 500)],
    'O3': [(0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 150),
           (169, 208, 151, 200), (209, 748, 201, 300), (749, 9999, 301, 500)]
}

CPCB_STATIONS = [
    {"id": 1, "name": "Delhi - Anand Vihar", "lat": 28.67, "lon": 77.32},
    {"id": 2, "name": "Delhi - Mandir Marg", "lat": 28.62, "lon": 77.21},
    {"id": 3, "name": "Delhi - Punjabi Bagh", "lat": 28.67, "lon": 77.01},
    {"id": 4, "name": "Delhi - RK Puram", "lat": 28.59, "lon": 77.19},
    {"id": 5, "name": "Delhi - Rohini", "lat": 28.85, "lon": 77.09},
    {"id": 6, "name": "Noida", "lat": 28.58, "lon": 77.32},
    {"id": 7, "name": "Ghaziabad", "lat": 28.66, "lon": 77.41},
    {"id": 8, "name": "Gurgaon", "lat": 28.46, "lon": 77.03},
    {"id": 9, "name": "Jaipur", "lat": 26.85, "lon": 75.79},
    {"id": 10, "name": "Lucknow", "lat": 26.85, "lon": 80.95},
    {"id": 11, "name": "Kanpur", "lat": 26.47, "lon": 80.33},
    {"id": 12, "name": "Agra", "lat": 27.18, "lon": 78.02},
    {"id": 13, "name": "Varanasi", "lat": 25.32, "lon": 83.01},
    {"id": 14, "name": "Prayagraj", "lat": 25.45, "lon": 81.84},
    {"id": 15, "name": "Patna", "lat": 25.60, "lon": 85.14},
    {"id": 16, "name": "Chandigarh", "lat": 30.75, "lon": 76.78},
    {"id": 17, "name": "Amritsar", "lat": 31.64, "lon": 74.87},
    {"id": 18, "name": "Ludhiana", "lat": 30.88, "lon": 75.85},
    {"id": 19, "name": "Jalandhar", "lat": 31.32, "lon": 75.57},
    {"id": 20, "name": "Bathinda", "lat": 30.22, "lon": 74.95},
    {"id": 21, "name": "Patiala", "lat": 30.34, "lon": 76.38},
    {"id": 22, "name": "Ambala", "lat": 30.38, "lon": 76.92},
    {"id": 23, "name": "Panipat", "lat": 29.40, "lon": 76.96},
    {"id": 24, "name": "Sonipat", "lat": 28.99, "lon": 77.02},
    {"id": 25, "name": "Meerut", "lat": 28.97, "lon": 77.71},
    {"id": 26, "name": "Muzaffarnagar", "lat": 29.46, "lon": 77.68},
    {"id": 27, "name": "Roorkee", "lat": 29.87, "lon": 77.88},
    {"id": 28, "name": "Dehradun", "lat": 30.32, "lon": 78.03},
    {"id": 29, "name": "Haridwar", "lat": 29.97, "lon": 78.16},
    {"id": 30, "name": "Moradabad", "lat": 28.84, "lon": 78.77},
    {"id": 31, "name": "Bareilly", "lat": 28.36, "lon": 79.42},
    {"id": 32, "name": "Etawah", "lat": 27.55, "lon": 79.08},
    {"id": 33, "name": "Jhansi", "lat": 25.45, "lon": 78.57},
    {"id": 34, "name": "Rewa", "lat": 24.88, "lon": 81.31},
    {"id": 35, "name": "Satna", "lat": 24.57, "lon": 80.83}
]

class AQIFeatures(BaseModel):
    pm25: float
    pm10: float
    no2: float
    so2: float
    co: float
    o3: float
    aod: float
    blh: float
    temperature: float
    rh: float
    wind: float
    pressure: float
    fire_count: int
    lat: float
    lon: float

def compute_aqi(pm25: float, pm10: float, no2: float, so2: float, co: float, o3: float) -> Tuple[float, str, Dict[str, float]]:
    def calculate_sub_index(pollutant: str, value: float) -> float:
        breakpoints = CPCB_BREAKPOINTS[pollutant]
        for (lb, hb, lq, hq) in breakpoints:
            if lb <= value <= hb:
                return lq + (hq - lq) * (value - lb) / (hb - lb)
        return 500.0
    
    sub_indices = {
        'PM2.5': calculate_sub_index('PM2.5', pm25),
        'PM10': calculate_sub_index('PM10', pm10),
        'NO2': calculate_sub_index('NO2', no2),
        'SO2': calculate_sub_index('SO2', so2),
        'CO': calculate_sub_index('CO', co),
        'O3': calculate_sub_index('O3', o3)
    }
    
    responsible_pollutant = max(sub_indices, key=sub_indices.get)
    aqi = sub_indices[responsible_pollutant]
    
    return aqi, responsible_pollutant, sub_indices

def generate_ipl_data() -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range(start='2024-10-01', end='2024-11-30', freq='D')
    n_days = len(dates)
    n_stations = len(CPCB_STATIONS)
    
    data = []
    for i, station in enumerate(CPCB_STATIONS):
        lat, lon = station['lat'], station['lon']
        
        for j, date in enumerate(dates):
            day_of_period = j
            day_of_year = date.dayofyear
            
            # Base pollution levels with seasonal trend
            seasonal_factor = 1.0 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
            base_pm25 = 80 + 40 * seasonal_factor + np.random.normal(0, 20)
            base_pm10 = 150 + 80 * seasonal_factor + np.random.normal(0, 40)
            
            # Stubble burning effect (Punjab/Haryana in Oct-Nov)
            burning_factor = 0.0
            if 30.0 <= lat <= 32.5 and 74.0 <= lon <= 77.0:
                if 30 <= day_of_year <= 320:
                    burning_factor = 40 + np.random.exponential(30)
            elif 28.0 <= lat <= 30.5 and 75.0 <= lon <= 78.0:
                if 30 <= day_of_year <= 320:
                    burning_factor = 25 + np.random.exponential(20)
            
            # Transport effect
            transport_factor = 0.0
            if 27.0 <= lat <= 29.0 and 76.0 <= lon <= 78.0:
                transport_factor = burning_factor * 0.3
            
            pm25 = max(0, base_pm25 + burning_factor + transport_factor + np.random.normal(0, 15))
            pm10 = max(0, base_pm10 + burning_factor * 1.5 + transport_factor + np.random.normal(0, 30))
            
            # Satellite data
            aod = 0.3 + 0.4 * (pm25 / 200) + np.random.normal(0, 0.1)
            no2 = 20 + 40 * (pm25 / 150) + np.random.exponential(10)
            so2 = 8 + 15 * (pm25 / 100) + np.random.exponential(5)
            co = 0.8 + 1.5 * (pm25 / 100) + np.random.exponential(0.5)
            o3 = 15 + 25 * (pm25 / 150) + np.random.exponential(10)
            hcho = 0.5 + 2.0 * (pm25 / 100) + np.random.exponential(0.5)
            
            # Meteorological data (ERA5)
            temp_base = 25 - 10 * np.sin(2 * np.pi * day_of_year / 365)
            blh = 200 + 800 * np.random.random()
            temperature = temp_base + np.random.normal(0, 5)
            rh = 40 + 40 * np.random.random()
            wind = 1 + 5 * np.random.random()
            pressure = 1010 + 20 * np.random.normal(0, 1)
            
            # Fire counts (MODIS/VIIRS)
            fire_count = 0
            if 30.0 <= lat <= 32.5 and 74.0 <= lon <= 77.0:
                if 30 <= day_of_year <= 320:
                    fire_count = np.random.poisson(50 * (1 + np.sin(2 * np.pi * day_of_year / 60)))
            elif 28.0 <= lat <= 30.5 and 75.0 <= lon <= 78.0:
                if 25 <= day_of_year <= 315:
                    fire_count = np.random.poisson(20)
            
            data.append({
                'date': date,
                'station_id': station['id'],
                'station_name': station['name'],
                'lat': lat,
                'lon': lon,
                'pm25': pm25,
                'pm10': pm10,
                'no2': no2,
                'so2': so2,
                'co': co,
                'o3': o3,
                'aod': aod,
                'hcho': hcho,
                'blh': blh,
                'temperature': temperature,
                'rh': rh,
                'wind': wind,
                'pressure': pressure,
                'fire_count': fire_count
            })
    
    df = pd.DataFrame(data)
    aqi_results = df.apply(lambda row: compute_aqi(
        row['pm25'], row['pm10'], row['no2'], row['so2'], row['co'], row['o3']
    ), axis=1)
    df['aqi'] = [x[0] for x in aqi_results]
    df['responsible_pollutant'] = [x[1] for x in aqi_results]
    
    return df

def compute_validation_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r = np.corrcoef(y_true, y_pred)[0, 1]
    r2 = r2_score(y_true, y_pred)
    nmb = np.sum(y_pred - y_true) / np.sum(y_true)
    nme = np.sum(np.abs(y_pred - y_true)) / np.sum(y_true)
    ioa = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((np.abs(y_pred - np.mean(y_true)) + np.abs(y_true - np.mean(y_true))) ** 2)
    
    return {
        'RMSE': float(rmse),
        'MAE': float(mae),
        'R': float(r),
        'R2': float(r2),
        'NMB': float(nmb),
        'NME': float(nme),
        'IOA': float(ioa)
    }

def train_models(df: pd.DataFrame) -> Tuple[RandomForestRegressor, object, Sequential]:
    feature_cols = ['aod', 'no2', 'so2', 'co', 'o3', 'blh', 'temperature', 'rh', 'wind', 'pressure', 'fire_count', 'lat', 'lon']
    
    df_sorted = df.sort_values('date')
    train_end = '2024-11-15'
    val_end = '2024-11-22'
    
    train_df = df_sorted[df_sorted['date'] <= train_end]
    val_df = df_sorted[(df_sorted['date'] > train_end) & (df_sorted['date'] <= val_end)]
    test_df = df_sorted[df_sorted['date'] > val_end]
    
    X_train = train_df[feature_cols].values
    y_train_pm25 = train_df['pm25'].values
    y_train_pm10 = train_df['pm10'].values
    y_train_aqi = train_df['aqi'].values
    
    X_val = val_df[feature_cols].values
    y_val_pm25 = val_df['pm25'].values
    y_val_pm10 = val_df['pm10'].values
    y_val_aqi = val_df['aqi'].values
    
    X_test = test_df[feature_cols].values
    y_test_pm25 = test_df['pm25'].values
    y_test_pm10 = test_df['pm10'].values
    y_test_aqi = test_df['aqi'].values
    
    # Random Forest
    rf_pm25 = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf_pm25.fit(X_train, y_train_pm25)
    
    rf_pm10 = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf_pm10.fit(X_train, y_train_pm10)
    
    rf_aqi = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf_aqi.fit(X_train, y_train_aqi)
    
    # XGBoost
    import xgboost as xgb
    
    xgb_pm25 = xgb.XGBRegressor(learning_rate=0.05, n_estimators=500, random_state=42, n_jobs=-1)
    xgb_pm25.fit(X_train, y_train_pm25)
    
    xgb_pm10 = xgb.XGBRegressor(learning_rate=0.05, n_estimators=500, random_state=42, n_jobs=-1)
    xgb_pm10.fit(X_train, y_train_pm10)
    
    xgb_aqi = xgb.XGBRegressor(learning_rate=0.05, n_estimators=500, random_state=42, n_jobs=-1)
    xgb_aqi.fit(X_train, y_train_aqi)
    
    # CNN-LSTM
    model_cnn_lstm = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(feature_cols.__len__(), 1)),
        LSTM(128, return_sequences=False),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model_cnn_lstm.compile(optimizer='adam', loss='mse')
    
    X_train_seq = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val_seq = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    X_test_seq = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    model_cnn_lstm.fit(X_train_seq, y_train_aqi, epochs=10, batch_size=32, 
                       validation_data=(X_val_seq, y_val_aqi), verbose=0)
    
    joblib.dump(rf_aqi, 'models/rf_model.pkl')
    joblib.dump(xgb_aqi, 'models/xgb_model.pkl')
    model_cnn_lstm.save('models/cnn_lstm_model.h5')
    
    models = {'rf': rf_aqi, 'xgb': xgb_aqi, 'cnn_lstm': model_cnn_lstm}
    metrics = {}
    
    for name, model in models.items():
        if name == 'cnn_lstm':
            pred_test = model.predict(X_test_seq).flatten()
        else:
            pred_test = model.predict(X_test)
        
        pm25_pred = rf_pm25.predict(X_test) if name == 'rf' else (xgb_pm25.predict(X_test) if name == 'xgb' else model.predict(X_test_seq).flatten())
        pm10_pred = rf_pm10.predict(X_test) if name == 'rf' else (xgb_pm10.predict(X_test) if name == 'xgb' else model.predict(X_test_seq).flatten())
        
        metrics[f'{name}_aqi'] = compute_validation_metrics(y_test_aqi, pred_test)
        metrics[f'{name}_pm25'] = compute_validation_metrics(y_test_pm25, pm25_pred)
        metrics[f'{name}_pm10'] = compute_validation_metrics(y_test_pm10, pm10_pred)
    
    return rf_aqi, xgb_aqi, model_cnn_lstm, metrics

def detect_hotspots(df: pd.DataFrame) -> List[Dict]:
    df_latest = df[df['date'] == df['date'].max()]
    hcho_values = df_latest['hcho'].values
    mean_india = hcho_values.mean()
    std_india = hcho_values.std()
    
    df_latest = df_latest.copy()
    df_latest['z_score'] = (df_latest['hcho'] - mean_india) / std_india
    
    coords = df_latest[['lat', 'lon', 'z_score']].values
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    df_latest['cluster'] = dbscan.fit_predict(coords[:, :2])
    
    cluster_labels = {
        -1: 'Background',
        0: 'IGP-West',
        1: 'IGP-Central',
        2: 'IGP-East',
        3: 'Punjab',
        4: 'Haryana',
        5: 'UP-Central',
        6: 'Bihar',
        7: 'Rajasthan'
    }
    
    features = []
    for cluster_id in df_latest['cluster'].unique():
        if cluster_id == -1:
            continue
        cluster_df = df_latest[df_latest['cluster'] == cluster_id]
        avg_lat = cluster_df['lat'].mean()
        avg_lon = cluster_df['lon'].mean()
        avg_z = cluster_df['z_score'].mean()
        station_count = len(cluster_df)
        
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(avg_lon), float(avg_lat)]
            },
            "properties": {
                "cluster_id": int(cluster_id),
                "region": cluster_labels.get(cluster_id, 'Unknown'),
                "avg_z_score": float(avg_z),
                "station_count": int(station_count),
                "min_aqi": float(cluster_df['aqi'].min()),
                "max_aqi": float(cluster_df['aqi'].max())
            }
        })
    
    return features

def get_weather_data() -> List[Dict]:
    cities = [
        {"name": "Delhi", "lat": 28.61, "lon": 77.23},
        {"name": "Mumbai", "lat": 19.07, "lon": 72.88},
        {"name": "Kolkata", "lat": 22.57, "lon": 88.36},
        {"name": "Chennai", "lat": 13.08, "lon": 80.27},
        {"name": "Bengaluru", "lat": 12.97, "lon": 77.59}
    ]
    
    weather_data = []
    for city in cities:
        try:
            url = f"https://api.open-meteo.com/v1/forecast?latitude={city['lat']}&longitude={city['lon']}&current_weather=true&hourly=pm2_5,pm10"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                current = data.get('current_weather', {})
                weather_data.append({
                    'city': city['name'],
                    'latitude': city['lat'],
                    'longitude': city['lon'],
                    'temperature': current.get('temperature', 25),
                    'windspeed': current.get('windspeed', 5),
                    'wind_direction': current.get('winddirection', 180),
                    'pressure': current.get('pressure', 1010)
                })
        except Exception:
            weather_data.append({
                'city': city['name'],
                'latitude': city['lat'],
                'longitude': city['lon'],
                'temperature': 25.0,
                'windspeed': 5.0,
                'wind_direction': 180,
                'pressure': 1010.0
            })
    
    return weather_data

# Global state
df_data = None
rf_model = None
xgb_model = None
cnn_model = None
validation_metrics = None
hotspot_features = None

@app.on_event("startup")
async def startup_event():
    global df_data, rf_model, xgb_model, cnn_model, validation_metrics, hotspot_features
    
    os.makedirs('models', exist_ok=True)
    
    df_data = generate_ipl_data()
    rf_model, xgb_model, cnn_model, validation_metrics = train_models(df_data)
    hotspot_features = detect_hotspots(df_data)

@app.post("/api/predict")
async def predict(features: AQIFeatures):
    feature_array = np.array([[
        features.aod, features.no2, features.so2, features.co, features.o3,
        features.blh, features.temperature, features.rh, features.wind,
        features.pressure, features.fire_count, features.lat, features.lon
    ]])
    
    rf_pred = rf_model.predict(feature_array)[0]
    xgb_pred = xgb_model.predict(feature_array)[0]
    cnn_pred = cnn_model.predict(feature_array.reshape(1, 13, 1)).flatten()[0]
    
    ensemble_pred = (rf_pred + xgb_pred + cnn_pred) / 3
    
    return {
        "predicted_aqi": float(ensemble_pred),
        "rf_aqi": float(rf_pred),
        "xgb_aqi": float(xgb_pred),
        "cnn_lstm_aqi": float(cnn_pred),
        "models_used": ["RandomForest", "XGBoost", "CNN-LSTM"]
    }

@app.get("/api/validation")
async def get_validation():
    return validation_metrics

@app.get("/api/hotspots")
async def get_hotspots():
    return {
        "type": "FeatureCollection",
        "features": hotspot_features
    }

@app.get("/api/stations")
async def get_stations():
    return {"stations": CPCB_STATIONS, "total": len(CPCB_STATIONS)}

@app.get("/api/weather")
async def get_weather():
    return get_weather_data()

@app.get("/")
async def root():
    return {"message": "AeroRakshak AI API", "version": "1.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)