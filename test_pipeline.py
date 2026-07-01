import sys
import os
import pandas as pd
import numpy as np

print("Testing Pipeline Imports...")
try:
    from modules.gee_utils import extract_satellite_data
    from modules.data_processor import prepare_training_dataset
    from modules.model import build_cnn_model, build_lstm_model, build_cnn_lstm_model
    from modules.hotspot_detector import detect_hcho_hotspots
    print("[OK] All modules imported successfully!")
except Exception as e:
    print(f"[FAIL] Import failed: {e}")
    sys.exit(1)

print("\nTesting Data Generation...")
mock_data = extract_satellite_data(None, None, '2024-01-01', '2024-01-07', spatial_resolution=0.5)
if 'HCHO' in mock_data and mock_data['HCHO'].shape[0] > 0:
    print("[OK] Mock data generated successfully")
else:
    print("[FAIL] Mock data generation failed")

print("\nTesting Hotspot Detection...")
hotspots = detect_hcho_hotspots(mock_data['HCHO'], mock_data['lat'], mock_data['lon'])
print(f"[OK] Found {len(hotspots)} hotspots")

print("\nTesting Dataset Preparation...")
spatial_input, temporal_input, target_aqi = prepare_training_dataset(mock_data, seq_length=7)
print(f"[OK] Spatial Input Shape: {spatial_input.shape}")
print(f"[OK] Temporal Input Shape: {temporal_input.shape}")
print(f"[OK] Target AQI Shape: {target_aqi.shape}")

print("\nTesting Model Initialization...")
try:
    cnn = build_cnn_model(spatial_input.shape[1:])
    lstm = build_lstm_model(7, 5)
    hybrid = build_cnn_lstm_model(spatial_input.shape[1:], 7, 5)
    print("[OK] Models initialized successfully")
    
    print("\nTesting Model Training (1 epoch, CNN only)...")
    cnn.fit(spatial_input, target_aqi, epochs=1, batch_size=16, verbose=0)
    print("[OK] CNN Training successful")
except Exception as e:
    print(f"[FAIL] Model error: {e}")

print("\nPipeline test complete!")
