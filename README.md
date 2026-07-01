# Surface AQI Prediction & HCHO Hotspot Identification over India

Satellite-based Air Quality Monitoring using Sentinel-5P, MODIS/VIIRS & Deep Learning.

## Quick Start

### Using pip (standard Python)
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Using uv (recommended)
```bash
uv init
uv add streamlit pandas numpy plotly folium streamlit-folium scikit-learn tensorflow geopandas shapely earthengine-api geemap xarray scipy netCDF4
uv run streamlit run app.py
```

## Google Earth Engine Setup

The app runs in **Mock Data Mode** by default. To use real GEE data:

### Step 1: Authenticate Earth Engine
```bash
earthengine authenticate
```

### Step 2: Set your GEE Project
```bash
earthengine set_project <your-project-id>
```

### Step 3: Configure environment
```bash
# Copy .env.example to .env
copy .env.example .env

# Edit .env and set your GEE_PROJECT_ID
# Then set USE_MOCK_DATA=false
```

Or set environment variable directly:
```bash
set GEE_PROJECT_ID=<your-project-id>
set USE_MOCK_DATA=false
```

## CPCB Data

The app includes a mock CPCB data generator. To use real CPCB data:

1. Download CSV from [CPCB Open Data Portal](https://cpcb.nic.in/)
2. Upload via the **Data Explorer** page
3. Or place CSV in `data/raw/` directory with columns:
   - station_name, latitude, longitude, date, PM2.5, PM10, NO2, SO2, CO, O3, AQI

## Project Structure

```
project/
├── app.py                    # Main Streamlit dashboard
├── pages/
│   ├── 1_🗑️_Data_Explorer.py
│   ├── 2_🗺️_AQI_Prediction.py
│   ├── 3_🔥_HCHO_Hotspots.py
│   └── 4_📊_Time_Series.py
├── modules/
│   ├── gee_utils.py          # GEE data extraction
│   ├── data_processor.py     # Data preprocessing & CPCB AQI
│   ├── model.py              # CNN, LSTM, CNN-LSTM models
│   ├── hotspot_detector.py     # DBSCAN hotspot detection
│   └── visualization.py        # Maps & charts
├── data/
│   └── raw/                  # Downloaded data files
└── .streamlit/
    └── config.toml           # Streamlit theme
```

## Model Architecture

- **CNN**: Spatial feature extraction from pollutant grids
- **LSTM**: Temporal pattern learning from meteorological data
- **CNN-LSTM Hybrid**: Combines spatial and temporal features (primary model)

## Features

- Interactive AQI prediction maps over India
- HCHO hotspot detection with DBSCAN clustering
- Fire count overlay from VIIRS/MODIS
- Temporal trend analysis
- Model performance metrics (RMSE, MAE, R², Correlation)