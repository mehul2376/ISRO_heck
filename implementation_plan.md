# Surface AQI Prediction & HCHO Hotspot Identification over India

This project implements a web-based dashboard and analysis pipeline for surface AQI prediction and formaldehyde (HCHO) hotspot identification over India. It combines satellite-observed columnar pollutants (Sentinel-5P TROPOMI), fire counts (MODIS/VIIRS), meteorological data (ERA5), and ground-level monitoring stations (CPCB) using Deep Learning (CNN, LSTM, CNN-LSTM Hybrid) and spatial clustering (DBSCAN).

## User Review Required

> [!IMPORTANT]
> **Google Earth Engine (GEE) Authentication**: GEE requires interactive authentication (`ee.Authenticate()`) which cannot be done in a headless terminal. To make this application work out-of-the-box, we will implement a robust, highly realistic **Synthetic Geospatial Data Generator** as a fallback. It will generate:
> 1. Daily spatial grids (0.5° resolution) over India for all variables (HCHO, NO2, SO2, CO, O3, wind, temp, boundary layer height, fire counts).
> 2. Realistic pollution patterns (e.g., agricultural crop burning in Punjab/Haryana in Oct-Nov, transport along the Indo-Gangetic Plain, urban hotspots in Delhi, Mumbai, Bengaluru, Kolkata).
> 3. CPCB station datasets at actual coordinates for major Indian cities.
> If you have a pre-authenticated Earth Engine project, you can supply your project ID, and the application will attempt to use real GEE data.

> [!NOTE]
> **Deep Learning Libraries**: We will install TensorFlow and Keras to train the CNN, LSTM, and Hybrid models. Due to potential library/Python version mismatches on Python 3.13, we will write fallback Scikit-Learn regressors (Random Forest/Gradient Boosting) and simpler neural network representations using PyTorch or pure NumPy/SciPy if TensorFlow fails to install.

## Proposed Changes

We will create a brand new Streamlit project under the workspace `e:\Hackathon`.

---

### Configuration & Dependencies

#### [NEW] [requirements.txt](file:///e:/Hackathon/requirements.txt)
Defines all Python package dependencies for the project, including Streamlit, Earth Engine API, Folium, Streamlit-Folium, Plotly, Scikit-Learn, and TensorFlow.

#### [NEW] [.streamlit/config.toml](file:///e:/Hackathon/.streamlit/config.toml)
Configures Streamlit for custom theme colors (slate/blue palette, dark mode headers) and increases max upload size.

---

### Python Modules

#### [NEW] [gee_utils.py](file:///e:/Hackathon/modules/gee_utils.py)
- Initializes Google Earth Engine.
- Extracts Sentinel-5P (HCHO, NO2, SO2, CO, O3), MODIS/VIIRS fire counts, and ERA5 meteorological variables.
- Provides a high-fidelity synthetic data generator as a fallback for headless runs.

#### [NEW] [data_processor.py](file:///e:/Hackathon/modules/data_processor.py)
- Implements CPCB National AQI calculation (maximum of sub-indices for PM2.5, PM10, NO2, SO2, CO, O3).
- Interpolates satellite grids spatially to ground station locations.
- Aligns datasets temporally (daily scale) and creates training sequences for LSTMs.
- Scales features using standard normalization.

#### [NEW] [model.py](file:///e:/Hackathon/modules/model.py)
- Builds three deep learning models:
  1. **CNN**: Captures spatial correlation of pollution.
  2. **LSTM**: Captures temporal dependencies (using sequence lookback).
  3. **CNN-LSTM Hybrid**: Fuses spatial features extracted by CNN and temporal features extracted by LSTM (the best performing model).
- Trains and evaluates models using RMSE, Pearson correlation (R), MAE, and R².

#### [NEW] [hotspot_detector.py](file:///e:/Hackathon/modules/hotspot_detector.py)
- Detects HCHO hotspots using DBSCAN clustering of grid anomalies (above the 90th percentile).
- Associates hotspots with nearest known geographic regions (e.g., Punjab-Haryana, Delhi-NCR, Indo-Gangetic Plain).
- Analyzes pollution transport by correlating fire counts and wind vectors with HCHO concentrations.

#### [NEW] [visualization.py](file:///e:/Hackathon/modules/visualization.py)
- Renders predicted surface AQI overlays, wind vector arrows, fire count markers, and hotspot regions on a Folium map.
- Creates dual-axis time-series plots (AQI vs HCHO) and model performance charts using Plotly.

---

### Streamlit Application

#### [NEW] [app.py](file:///e:/Hackathon/app.py)
- Main Streamlit dashboard containing page headers, KPI cards (RMSE, R, Hotspots, Fires), and control sidebar (date selection, model selection, visualization toggle).
- Integrates the four analysis tabs:
  1. **🗺️ AQI Prediction Map**: Folium interactive map showing surface AQI and fire locations.
  2. **🔥 HCHO Hotspot Analysis**: DBSCAN hotspot locations, wind patterns, and fire transport correlation.
  3. **📈 Time Series**: Temporal trends for selected cities/regions.
  4. **🤖 Model Performance**: Real-time training option, loss curves, and model comparison.

---

## Verification Plan

### Automated Tests
We will write a test script in the workspace to verify the code:
- Run a pipeline check: `python -c "import modules.gee_utils, modules.data_processor, modules.model, modules.hotspot_detector; print('All modules imported successfully!')"`
- Generate mock data, train all three models, and output evaluation metrics.

### Manual Verification
- We will launch the Streamlit app locally: `streamlit run app.py` (which will execute via the `run_command` tool).
- Verify the interactive Folium maps render properly and response dynamically to sidebar adjustments (e.g., changing the hotspot percentile threshold or toggling wind vectors).
- Inspect the Plotly time series graphs and check for correct calculation of Indian AQI breakpoints.
