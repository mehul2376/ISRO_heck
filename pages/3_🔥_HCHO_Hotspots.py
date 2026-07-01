import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np

st.set_page_config(page_title="HCHO Hotspots", page_icon="🔥", layout="wide")

st.title("🔥 HCHO Hotspots & Source Transport")
st.markdown("Identification of Formaldehyde (HCHO) hotspots and correlation with MODIS/VIIRS fire counts.")

col1, col2 = st.columns(2)
with col1:
    hotspot_threshold = st.slider("HCHO Anomaly Threshold (Percentile)", 70, 99, 90)
with col2:
    show_wind = st.checkbox("Overlay Wind Vectors (ERA5/MERRA-2)", value=False)

m = folium.Map(location=[22.5, 79.0], zoom_start=5, tiles='CartoDB dark_matter')

if st.session_state.get('use_mock_data', True):
    st.info("Mock Mode: Showing synthetic hotspots and fire points.")
    
    # Mock Hotspots
    mock_hotspots = [
        {'name': 'Punjab-Haryana', 'lat': 30.5, 'lon': 75.5, 'intensity': 0.0084, 'cluster_size': 142},
        {'name': 'Delhi-NCR', 'lat': 28.6, 'lon': 77.2, 'intensity': 0.0076, 'cluster_size': 89},
        {'name': 'Indo-Gangetic Plain', 'lat': 26.5, 'lon': 80.5, 'intensity': 0.0065, 'cluster_size': 210}
    ]
    
    for hs in mock_hotspots:
        folium.Marker(
            location=[hs['lat'], hs['lon']],
            popup=f"<b>{hs['name']}</b><br>Intensity: {hs['intensity']:.4f} mol/m²<br>Cluster Size: {hs['cluster_size']} pixels",
            icon=folium.Icon(color='orange', icon='fire', prefix='fa')
        ).add_to(m)
        
        # Add a mock "anomaly zone" circle
        folium.Circle(
            location=[hs['lat'], hs['lon']],
            radius=hs['cluster_size'] * 1000,
            color='orange',
            fill=True,
            fill_opacity=0.3
        ).add_to(m)
        
    # Mock Fire Points
    for _ in range(50):
        lat = np.random.uniform(28.0, 32.0)
        lon = np.random.uniform(74.0, 78.0)
        folium.CircleMarker(
            location=[lat, lon],
            radius=2,
            color='red',
            fill=True,
            fill_opacity=0.8,
            tooltip="Active Fire (VIIRS)"
        ).add_to(m)

st_folium(m, width=1200, height=600, returned_objects=[])

st.markdown("""
### Transport Analysis
The identified hotspots highly correlate with active fire counts in the Northwestern regions. The wind patterns (if enabled) illustrate the transport pathway of pollutants towards the Indo-Gangetic Plain.
""")