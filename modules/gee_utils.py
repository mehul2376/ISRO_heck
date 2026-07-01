import numpy as np
import pandas as pd

INDIA_BBOX = [68.0, 6.0, 97.5, 37.5]

def initialize_gee():
    """Initialize Google Earth Engine"""
    try:
        ee.Initialize()
        return True
    except Exception as e:
        print(f"GEE Initialization failed: {e}")
        return False

def generate_synthetic_grid(shape, base_val, noise_std, hotspots=None):
    grid = np.full(shape, base_val) + np.random.normal(0, noise_std, shape)
    if hotspots:
        for (hr, hc, radius, intensity) in hotspots:
            for r in range(shape[0]):
                for c in range(shape[1]):
                    dist = np.sqrt((r - hr)**2 + (c - hc)**2)
                    if dist <= radius:
                        grid[r, c] += intensity * (1 - dist/radius)
    return np.maximum(0, grid)

def extract_satellite_data(collection_id, bands, start_date, end_date,
                           bbox=INDIA_BBOX, spatial_resolution=0.5):
    """
    Mock extraction of satellite data over India.
    Generates synthetic spatial grids (lat x lon) for environmental variables.
    """
    lat_bins = int(np.ceil((bbox[3] - bbox[1]) / spatial_resolution))
    lon_bins = int(np.ceil((bbox[2] - bbox[0]) / spatial_resolution))
    shape = (lat_bins, lon_bins)
    
    # Lat/Lon grids
    lat_vals = np.linspace(bbox[1], bbox[3], lat_bins)
    lon_vals = np.linspace(bbox[0], bbox[2], lon_bins)
    lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
    
    # Define some hotspots (Delhi/NCR roughly at lat index ~45, lon index ~18 for 0.5 res)
    # Just generic hotspots for mock data
    hcho_hotspots = [(int(lat_bins*0.7), int(lon_bins*0.3), 5, 0.005), 
                     (int(lat_bins*0.6), int(lon_bins*0.5), 8, 0.004)]
    
    mock_data = {
        'HCHO': generate_synthetic_grid(shape, 0.002, 0.0005, hcho_hotspots),
        'NO2': generate_synthetic_grid(shape, 0.00005, 0.00001),
        'SO2': generate_synthetic_grid(shape, 0.0001, 0.00002),
        'CO': generate_synthetic_grid(shape, 0.03, 0.005),
        'O3': generate_synthetic_grid(shape, 0.13, 0.01),
        'wind_u': np.random.normal(2, 1, shape),
        'wind_v': np.random.normal(-1, 1, shape),
        'temp': generate_synthetic_grid(shape, 300, 5), # Kelvin
        'blh': generate_synthetic_grid(shape, 1000, 200), # Boundary layer height
        'lat': lat_grid,
        'lon': lon_grid
    }
    return mock_data

def extract_hcho_timeseries(start_date, end_date, bbox=INDIA_BBOX):
    """Extract HCHO column density time series"""
    dates = pd.date_range(start_date, end_date)
    # Random walk for time series
    trend = np.cumsum(np.random.normal(0, 0.0001, len(dates))) + 0.005
    return pd.DataFrame({'date': dates, 'HCHO_mean': trend})

def extract_fire_counts(start_date, end_date, bbox=INDIA_BBOX):
    """Extract VIIRS fire count data"""
    num_fires = np.random.randint(50, 200)
    lats = np.random.uniform(bbox[1], bbox[3], num_fires)
    lons = np.random.uniform(bbox[0], bbox[2], num_fires)
    frp = np.random.uniform(10, 100, num_fires)
    return list(zip(lats, lons, frp))
