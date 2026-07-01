import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# CPCB Station locations for mock data generation
CPCB_STATIONS = [
    {'station_name': 'New Delhi', 'latitude': 28.6139, 'longitude': 77.2090},
    {'station_name': 'Mumbai', 'latitude': 19.0760, 'longitude': 72.8777},
    {'station_name': 'Kolkata', 'latitude': 22.5726, 'longitude': 88.3639},
    {'station_name': 'Chennai', 'latitude': 13.0824, 'longitude': 80.2707},
    {'station_name': 'Bangalore', 'latitude': 12.9716, 'longitude': 77.5946},
    {'station_name': 'Hyderabad', 'latitude': 17.3850, 'longitude': 78.4867},
    {'station_name': 'Ahmedabad', 'latitude': 23.0225, 'longitude': 72.5714},
    {'station_name': 'Pune', 'latitude': 18.5204, 'longitude': 73.8567},
    {'station_name': 'Lucknow', 'latitude': 26.8467, 'longitude': 80.9462},
    {'station_name': 'Jaipur', 'latitude': 26.9124, 'longitude': 75.7873},
    {'station_name': 'Kanpur', 'latitude': 26.4499, 'longitude': 80.3311},
    {'station_name': 'Nagpur', 'latitude': 21.1458, 'longitude': 79.0882},
    {'station_name': 'Ghaziabad', 'latitude': 28.6680, 'longitude': 77.4240},
    {'station_name': 'Indore', 'latitude': 22.7167, 'longitude': 75.8577},
    {'station_name': 'Coimbatore', 'latitude': 11.0168, 'longitude': 76.9558},
    {'station_name': 'Kochi', 'latitude': 9.9312, 'longitude': 76.2673},
    {'station_name': 'Patna', 'latitude': 25.5941, 'longitude': 85.1376},
    {'station_name': 'Ludhiana', 'latitude': 30.8656, 'longitude': 75.8650},
    {'station_name': 'Agra', 'latitude': 27.1767, 'longitude': 78.0322},
]

def generate_mock_cpcb_data(start_date, end_date, num_dates=None):
    """
    Generate realistic mock CPCB ground station data.
    Columns: station_name, latitude, longitude, date, PM2.5, PM10, NO2, SO2, CO, O3, AQI
    """
    dates = pd.date_range(start=start_date, end=end_date, periods=num_dates or 90)
    
    rows = []
    for date in dates:
        for station in CPCB_STATIONS:
            # Generate realistic pollutant values based on location type
            month_factor = 1.0
            if date.month in [10, 11, 12]:  # Crop burning season
                month_factor = 1.5
            
            # Base values with seasonal variation
            pm25 = np.random.uniform(30, 150) * month_factor
            pm10 = pm25 * 1.5 + np.random.uniform(0, 50)
            no2 = np.random.uniform(20, 80) * month_factor
            so2 = np.random.uniform(5, 40)
            co = np.random.uniform(0.5, 3.0)
            o3 = np.random.uniform(10, 100)
            
            # Calculate AQI
            aqi = calculate_indian_aqi(pm25, pm10, no2, so2, co, o3)
            
            rows.append({
                'station_name': station['station_name'],
                'latitude': station['latitude'],
                'longitude': station['longitude'],
                'date': date,
                'PM2.5': round(pm25, 2),
                'PM10': round(pm10, 2),
                'NO2': round(no2, 2),
                'SO2': round(so2, 2),
                'CO': round(co, 2),
                'O3': round(o3, 2),
                'AQI': round(aqi, 0)
            })
    
    return pd.DataFrame(rows)

def calculate_indian_aqi(pm25, pm10, no2, so2, co, o3):
    """
    Calculate AQI as per Indian CPCB standards (National Air Quality Index).
    """
    breakpoints = {
        'pm25':  [0, 30, 60, 90, 120, 250, 500],
        'pm10':  [0, 50, 100, 250, 350, 430, 500],
        'no2':   [0, 40, 80, 180, 280, 400, 500],
        'so2':   [0, 40, 80, 380, 800, 1600, 500],
        'co':    [0, 1, 2, 10, 17, 34, 46],
        'o3':    [0, 50, 100, 168, 208, 748, 500],
    }
    
    aqi_ranges = [0, 50, 100, 200, 300, 400, 500]
    
    def sub_index(concentration, bp):
        for i in range(len(bp) - 1):
            if bp[i] <= concentration <= bp[i+1]:
                return ((aqi_ranges[i+1] - aqi_ranges[i]) / 
                        (bp[i+1] - bp[i])) * (concentration - bp[i]) + aqi_ranges[i]
        return 500 if concentration > bp[-1] else 0
    
    pollutants = {'pm25': pm25, 'pm10': pm10, 'no2': no2,
                  'so2': so2, 'co': co, 'o3': o3}
    
    sub_indices = {}
    for name, conc in pollutants.items():
        if pd.notna(conc):
            sub_indices[name] = sub_index(conc, breakpoints[name])
        else:
            sub_indices[name] = np.nan
            
    valid_vals = [v for v in sub_indices.values() if not np.isnan(v)]
    return max(valid_vals) if valid_vals else np.nan

def prepare_training_dataset(mock_satellite_data, seq_length=7):
    """
    Prepare synthetic inputs for CNN-LSTM.
    Returns:
       spatial_input: (samples, lat, lon, channels)
       temporal_input: (samples, seq_length, features)
       target_aqi: (samples, 1)
    """
    # Use the mock spatial grids
    hcho = mock_satellite_data['HCHO']
    no2 = mock_satellite_data['NO2']
    temp = mock_satellite_data['temp']
    
    lat_bins, lon_bins = hcho.shape
    
    # We will generate a mock dataset of 100 samples
    num_samples = 100
    
    # Spatial input: Stack the grids
    # Shape: (lat_bins, lon_bins, 3 channels)
    single_spatial = np.stack([hcho, no2, temp], axis=-1)
    
    # To make samples, add some noise to the base spatial grid
    spatial_input = []
    for _ in range(num_samples):
        noisy_spatial = single_spatial + np.random.normal(0, 0.0001, single_spatial.shape)
        spatial_input.append(noisy_spatial)
    
    spatial_input = np.array(spatial_input)
    
    # Temporal input: Mock historical meteorological/pollution averages
    num_temporal_features = 5
    temporal_input = np.random.normal(0, 1, (num_samples, seq_length, num_temporal_features))
    
    # Target AQI: Some function of the spatial hotspots + random noise
    # Let's say AQI is highly correlated with HCHO mean
    hcho_means = np.mean(spatial_input[:, :, :, 0], axis=(1, 2))
    target_aqi = (hcho_means * 100000) + np.random.normal(0, 20, num_samples)
    target_aqi = np.clip(target_aqi, 0, 500).reshape(-1, 1)
    
    return spatial_input, temporal_input, target_aqi
