import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
import pandas as pd

st.set_page_config(page_title="AQI Prediction Map", page_icon="🗺️", layout="wide")

st.title("🗺️ Predicted Surface AQI over India")
st.markdown("AQI predicted by CNN-LSTM Hybrid model using Sentinel-5P, INSAT-3D AOD, and ERA5/MERRA-2 data.")

# Controls specific to this page
col1, col2 = st.columns(2)
with col1:
    model_choice = st.selectbox(
        "Deep Learning Model",
        ["CNN-LSTM Hybrid (Recommended)", "CNN Only", "LSTM Only"]
    )
with col2:
    date_to_view = st.date_input("Date", st.session_state.get('end_date', pd.to_datetime('today')))

# Create base map
m = folium.Map(location=[22.5, 79.0], zoom_start=5, tiles='CartoDB dark_matter')

if st.session_state.get('use_mock_data', True):
    # Generate mock heatmap data for India
    # Bounding box roughly 68.0 to 97.5 lon, 6.0 to 37.5 lat
    st.info("Displaying Mock Heatmap Data for the CNN-LSTM predictions.")
    
    # Randomly generate some AQI values over a grid
    lats = np.linspace(8.0, 35.0, 30)
    lons = np.linspace(70.0, 95.0, 30)
    
    heat_data = []
    for lat in lats:
        for lon in lons:
            # Create a fake "hotspot" around northern plains
            dist_to_delhi = np.sqrt((lat - 28.6)**2 + (lon - 77.2)**2)
            aqi_val = max(50, 400 - (dist_to_delhi * 30) + np.random.normal(0, 20))
            heat_data.append([lat, lon, aqi_val])
            
    # Normalize for heatmap visualization (Folium HeatMap expects values 0-1 or uses relative max)
    from folium.plugins import HeatMap
    
    # We will just add circles for AQI to show discrete points in the grid for a mock UI
    # In a real app we might use a raster overlay (e.g. LocalTileLayer or ImageOverlay)
    for point in heat_data:
        lat, lon, aqi = point
        if aqi > 0:
            if aqi <= 50: color = 'green'
            elif aqi <= 100: color = '#ffff00' # yellow
            elif aqi <= 200: color = 'orange'
            elif aqi <= 300: color = 'red'
            elif aqi <= 400: color = 'purple'
            else: color = 'maroon'
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=4,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
                weight=0
            ).add_to(m)

st_folium(m, width=1200, height=600, returned_objects=[])

# Legend below the map
st.markdown("""
**AQI Color Legend (Indian Standard):**
🟢 0-50 (Good) | 🟡 51-100 (Satisfactory) | 🟠 101-200 (Moderate) | 🔴 201-300 (Poor) | 🟣 301-400 (Very Poor) | 🟤 401+ (Severe)
""")