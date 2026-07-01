import numpy as np
from sklearn.cluster import DBSCAN

# Known regions for India (approximate centroids)
KNOWN_REGIONS = {
    'Punjab-Haryana': (30.5, 75.5),
    'Delhi-NCR': (28.6, 77.2),
    'UP-Bihar': (26.0, 82.0),
    'Indo-Gangetic Plain': (26.0, 80.0),
    'Mumbai': (19.0, 72.8),
    'Kolkata': (22.5, 88.3),
    'Chennai': (13.0, 80.2),
    'Bangalore': (12.9, 77.6),
    'Northeast India': (26.0, 93.0),
    'Central India': (23.0, 79.0),
    'Gujarat': (23.0, 72.5),
    'Rajasthan': (26.5, 74.0),
    'Odisha': (20.0, 84.0),
    'Telangana': (18.0, 79.5),
}

def detect_hcho_hotspots(hcho_grid, lat_grid, lon_grid,
                          threshold_percentile=90, min_cluster_size=5):
    """
    Detect HCHO hotspots using statistical anomaly detection + spatial clustering.
    """
    threshold = np.nanpercentile(hcho_grid, threshold_percentile)
    anomaly_mask = hcho_grid > threshold
    
    anomaly_lats = lat_grid[anomaly_mask]
    anomaly_lons = lon_grid[anomaly_mask]
    anomaly_values = hcho_grid[anomaly_mask]
    
    if len(anomaly_lats) == 0:
        return []
    
    coords = np.column_stack([anomaly_lats, anomaly_lons])
    clustering = DBSCAN(eps=1.0, min_samples=min_cluster_size).fit(coords)
    
    hotspots = []
    for cluster_id in set(clustering.labels_):
        if cluster_id == -1:
            continue
        cluster_mask = clustering.labels_ == cluster_id
        centroid_lat = np.mean(anomaly_lats[cluster_mask])
        centroid_lon = np.mean(anomaly_lons[cluster_mask])
        mean_intensity = np.mean(anomaly_values[cluster_mask])
        
        # Assign region name based on nearest known location
        min_dist = float('inf')
        region_name = 'Unknown Region'
        for name, (r_lat, r_lon) in KNOWN_REGIONS.items():
            dist = np.sqrt((centroid_lat - r_lat)**2 + (centroid_lon - r_lon)**2)
            if dist < min_dist:
                min_dist = dist
                region_name = name
        
        hotspots.append({
            'lat': float(centroid_lat),
            'lon': float(centroid_lon),
            'intensity': float(mean_intensity),
            'cluster_size': int(cluster_mask.sum()),
            'name': region_name
        })
        
    return hotspots


def correlate_fire_hcho(fire_grid, hcho_grid, lat_grid, lon_grid, wind_u, wind_v):
    """
    Correlate fire counts with HCHO concentrations considering wind transport.
    """
    valid = ~(np.isnan(fire_grid) | np.isnan(hcho_grid))
    if valid.sum() < 10:
        return {'correlation': 0, 'p_value': 1}
    
    from scipy.stats import pearsonr
    r, p = pearsonr(fire_grid[valid].flatten(), hcho_grid[valid].flatten())
    
    # Wind direction analysis
    wind_speed = np.sqrt(wind_u**2 + wind_v**2)
    wind_dir = np.degrees(np.arctan2(-wind_u, -wind_v))
    
    return {
        'correlation': float(r),
        'p_value': float(p),
        'mean_wind_direction': float(np.nanmean(wind_dir)),
        'mean_wind_speed': float(np.nanmean(wind_speed))
    }
