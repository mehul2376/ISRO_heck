import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Page configuration
st.set_page_config(
    page_title="Surface AQI & HCHO Hotspot Analysis — India",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional layout
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 0.5rem 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    .badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.9rem;
        font-weight: 500;
        margin-left: 1rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 4px solid #667eea;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.95rem;
        color: #6c757d;
        font-weight: 500;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1a1a2e;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e9ecef;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f8f9fa;
        border-radius: 8px 8px 0 0;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: white;
        border-top: 3px solid #667eea;
    }
    .info-box {
        background: #f8f9fa;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# Header with title and mode badge
st.markdown("""
<div class="main-header">
    <h1>Surface AQI & HCHO Hotspot Analysis — India</h1>
    <p>Satellite-based AQI prediction and HCHO hotspot detection using Sentinel-5P, INSAT-3D, CPCB, FIRMS and ERA5</p>
    <span class="badge">Mock Mode</span>
</div>
""", unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Controls")
    
    st.subheader("Date Range")
    start_date = st.date_input("Start Date", pd.to_datetime("2024-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2024-12-31"))
    
    st.subheader("Data Settings")
    data_mode = st.radio("Data Mode", ["Mock Mode", "Live GEE"], horizontal=True)
    use_mock_data = data_mode == "Mock Mode"
    
    st.subheader("Layer Selection")
    show_aqi = st.checkbox("AQI Layer", value=True)
    show_hcho = st.checkbox("HCHO Layer", value=True)
    show_fire = st.checkbox("Fire Layer", value=True)
    show_wind = st.checkbox("Wind Arrows", value=True)
    
    st.subheader("Analysis Settings")
    pollutant = st.selectbox("AQI Pollutant", ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"], index=0)
    hcho_threshold = st.slider("HCHO Threshold (ppb)", 0.5, 3.0, 1.0, 0.1)
    
    # Store in session state
    st.session_state.update({
        'start_date': start_date,
        'end_date': end_date,
        'use_mock_data': use_mock_data,
        'show_aqi': show_aqi,
        'show_hcho': show_hcho,
        'show_fire': show_fire,
        'show_wind': show_wind,
        'pollutant': pollutant,
        'hcho_threshold': hcho_threshold
    })

# Metric Cards Row
st.markdown("### Key Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Highest AQI City</div>
        <div class="metric-value">Delhi</div>
        <div style="font-size:0.9rem;color:#6c757d;">AQI: 324</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Mean AQI (India)</div>
        <div class="metric-value">142</div>
        <div style="font-size:0.9rem;color:#6c757d;">Moderate Category</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card class="metric-card">
        <div class="metric-label">HCHO Hotspots</div>
        <div class="metric-value">18</div>
        <div style="font-size:0.9rem;color:#6c757d;">Above Threshold</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Fire-HCHO Correlation</div>
        <div class="metric-value">0.87</div>
        <div style="font-size:0.9rem;color:#6c757d;">Strong Positive</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", 
    "🗺️ AQI Map", 
    "🔥 HCHO Hotspots", 
    "💨 Fire & Wind Transport", 
    "📈 Model Performance"
])

# Overview Tab
with tab1:
    st.markdown('<div class="section-header">Project Overview</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        This dashboard provides comprehensive analysis of air quality and atmospheric composition over India using multi-satellite data integration and machine learning models.
        
        **Key Features:**
        - **Surface AQI Analysis**: CNN-LSTM model predictions using satellite and ground observations
        - **HCHO Hotspot Detection**: Sentinel-5P TROPOMI formaldehyde anomaly identification
        - **Fire Influence Analysis**: MODIS/VIIRS fire detection with transport modeling
        - **Model Performance Evaluation**: Comparison of deep learning architectures
        
        **Data Sources Integrated:**
        - 🛰️ **Sentinel-5P TROPOMI**: HCHO, NO2, SO2, CO, O3
        - 📡 **INSAT-3D**: Aerosol Optical Depth (AOD)
        - 🏭 **CPCB**: Ground truth AQI measurements
        - 🔥 **NASA FIRMS**: MODIS/VIIRS fire radiative power
        - 🌪️ **ERA5 Reanalysis**: Meteorological parameters (wind, temperature, humidity)
        """)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <strong>Study Period:</strong><br>
            January 2024 - December 2024<br><br>
            <strong>Spatial Coverage:</strong><br>
            68°E-97°E, 6°N-37°N (Indian Mainland)<br><br>
            <strong>Update Frequency:</strong><br>
            Daily composites<br><br>
            <strong>Model Resolution:</strong><br>
            0.05° × 0.05° (~5km)
        </div>
        """, unsafe_allow_html=True)

# AQI Map Tab
with tab2:
    st.markdown('<div class="section-header">Surface AQI Analysis</div>', unsafe_allow_html=True)
    
    col_map, col_info = st.columns([3, 2])
    
    with col_map:
        # AQI Map
        m = folium.Map(location=[22.5, 79.0], zoom_start=5, tiles='CartoDB positron')
        
        if st.session_state.get('use_mock_data', True):
            # Generate India-clipped AQI data
            # Create grid points within India boundary
            lats = np.linspace(8.0, 35.0, 40)
            lons = np.linspace(68.0, 97.0, 40)
            
            for lat in lats:
                for lon in lons:
                    # Simple India mask (approximate)
                    if (lat >= 8 and lat <= 35 and lon >= 68 and lon <= 97):
                        # Distance-weighted AQI with Delhi hotspot
                        dist_to_delhi = np.sqrt((lat - 28.6)**2 + (lon - 77.2)**2)
                        aqi = max(30, 400 - (dist_to_delhi * 8) + np.random.normal(0, 15))
                        
                        # AQI Color Scale (Indian Standard)
                        if aqi <= 50: color = '#00e400'  # Green - Good
                        elif aqi <= 100: color = '#ffff00'  # Yellow - Satisfactory
                        elif aqi <= 200: color = '#ff7e00'  # Orange - Moderate
                        elif aqi <= 300: color = '#ff0000'  # Red - Poor
                        elif aqi <= 400: color = '#8f3f97'  # Purple - Very Poor
                        else: color = '#7e0023'  # Maroon - Severe
                        
                        folium.CircleMarker(
                            location=[lat, lon],
                            radius=3,
                            color=color,
                            fill=True,
                            fill_color=color,
                            fill_opacity=0.7,
                            weight=0,
                            tooltip=f"AQI: {aqi:.0f}"
                        ).add_to(m)
        
        st_folium(m, width=700, height=500, returned_objects=[])
    
    with col_info:
        # AQI Legend
        st.markdown("""
        <div class="info-box">
            <h4>AQI Legend (NAQI India)</h4>
            <div style="display:flex;align-items:center;margin:0.5rem 0;">
                <div style="width:20px;height:20px;background:#00e400;margin-right:8px;"></div>
                <span>0-50: Good</span>
            </div>
            <div style="display:flex;align-items:center;margin:0.5rem 0;">
                <div style="width:20px;height:20px;background:#ffff00;margin-right:8px;"></div>
                <span>51-100: Satisfactory</span>
            </div>
            <div style="display:flex;align-items:center;margin:0.5rem 0;">
                <div style="width:20px;height:20px;background:#ff7e00;margin-right:8px;"></div>
                <span>101-200: Moderate</span>
            </div>
            <div style="display:flex;align-items:center;margin:0.5rem 0;">
                <div style="width:20px;height:20px;background:#ff0000;margin-right:8px;"></div>
                <span>201-300: Poor</span>
            </div>
            <div style="display:flex;align-items:center;margin:0.5rem 0;">
                <div style="width:20px;height:20px;background:#8f3f97;margin-right:8px;"></div>
                <span>301-400: Very Poor</span>
            </div>
            <div style="display:flex;align-items:center;margin:0.5rem 0;">
                <div style="width:20px;height:20px;background:#7e0023;margin-right:8px;"></div>
                <span>401+: Severe</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Top 10 Polluted Cities
        st.markdown("#### Top 10 Most Polluted Cities")
        top_cities = pd.DataFrame({
            'City': ['Delhi', 'Ghaziabad', 'Noida', 'Gurugram', 'Faridabad', 
                    'Lucknow', 'Kanpur', 'Varanasi', 'Patna', 'Agra'],
            'AQI': [324, 318, 312, 305, 298, 285, 278, 270, 265, 260],
            'Category': ['Severe']*4 + ['Very Poor']*2 + ['Poor']*4
        })
        st.dataframe(
            top_cities.style.background_gradient(subset=['AQI'], cmap='Reds'),
            use_container_width=True,
            hide_index=True
        )
        
        # Note about data
        st.markdown("""
        <div style="font-size:0.9rem;color:#6c757d;font-style:italic;margin-top:1rem;">
            Note: This is a prototype visualization using sample data. 
            In production, this would show real-time AQI predictions from the CNN-LSTM model.
        </div>
        """, unsafe_allow_html=True)

# HCHO Hotspots Tab
with tab3:
    st.markdown('<div class="section-header">HCHO Hotspot Analysis</div>', unsafe_allow_html=True)
    
    col_map, col_info = st.columns([3, 2])
    
    with col_map:
        # HCHO Hotspot Map
        m3 = folium.Map(location=[22.5, 79.0], zoom_start=5, tiles='CartoDB positron')
        
        if st.session_state.get('use_mock_data', True):
            # Generate HCHO heatmap
            lats = np.linspace(8.0, 35.0, 30)
            lons = np.linspace(68.0, 97.0, 30)
            heat_data = []
            
            for lat in lats:
                for lon in lons:
                    if lat >= 8 and lat <= 35 and lon >= 68 and lon <= 97:
                        # HCHO concentration with hotspots in industrial/agricultural regions
                        dist_to_delhi = np.sqrt((lat - 28.6)**2 + (lon - 77.2)**2)
                        dist_to_igp = np.sqrt((lat - 26.0)**2 + (lon - 82.0)**2)  # IGP center
                        hcho = max(0.1, 2.5 - (dist_to_delhi * 0.03) - (dist_to_igp * 0.02) + np.random.normal(0, 0.2))
                        heat_data.append([lat, lon, hcho])
            
            # Add heatmap layer
            from folium.plugins import HeatMap
            HeatMap(
                heat_data,
                radius=15,
                blur=10,
                max_zoom=10,
                gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 0.8: 'red', 1.0: 'darkred'}
            ).add_to(m3)
            
            # Add DBSCAN hotspot markers
            hotspots = [
                {'lat': 28.6, 'lon': 77.2, 'intensity': 2.45, 'region': 'Delhi-NCR', 'size': 25},
                {'lat': 30.3, 'lon': 72.8, 'intensity': 2.10, 'region': 'Punjab-Haryana', 'size': 22},
                {'lat': 24.8, 'lon': 85.3, 'intensity': 1.95, 'region': 'Indo-Gangetic Plain', 'size': 20},
                {'lat': 26.8, 'lon': 80.9, 'intensity': 1.85, 'region': 'Uttar Pradesh', 'size': 18},
                {'lat': 19.1, 'lon': 77.2, 'intensity': 1.75, 'region': 'Central India', 'size': 16},
                {'lat': 22.5, 'lon': 88.3, 'intensity': 1.60, 'region': 'West Bengal', 'size': 14}
            ]
            
            for hs in hotspots:
                folium.CircleMarker(
                    location=[hs['lat'], hs['lon']],
                    radius=hs['size']/4,
                    color='darkred',
                    fill=True,
                    fill_color='red',
                    fill_opacity=0.8,
                    weight=2,
                    tooltip=f"Region: {hs['region']}<br>HCHO: {hs['intensity']:.2f} ppb<br>Cluster Size: {hs['size']} km²"
                ).add_to(m3)
        
        st_folium(m3, width=700, height=500, returned_objects=[])
    
    with col_info:
        # Hotspot Details Table
        st.markdown("#### Detected HCHO Hotspots")
        hotspots_df = pd.DataFrame({
            'Region': ['Delhi-NCR', 'Punjab-Haryana', 'Indo-Gangetic Plain', 'Uttar Pradesh', 
                      'Central India', 'West Bengal', 'Assam', 'Gujarat'],
            'HCHO (ppb)': [2.45, 2.10, 1.95, 1.85, 1.75, 1.60, 1.55, 1.50],
            'Cluster Size (km²)': [25, 22, 20, 18, 16, 14, 12, 10],
            'Primary Source': ['Urban/Industrial', 'Agricultural Burning', 'Agricultural/Industrial', 
                             'Urban/Industrial', 'Industrial/Mining', 'Urban', 'Agricultural', 'Industrial'],
            'Confidence': ['High', 'High', 'Medium', 'Medium', 'Medium', 'Medium', 'Low', 'Medium']
        })
        
        # Style the dataframe
        def highlight_hcho(val):
            if isinstance(val, float):
                if val >= 2.0:
                    return 'background-color: #ffcccc'
                elif val >= 1.8:
                    return 'background-color: #ffe6cc'
                elif val >= 1.6:
                    return 'background-color: #ffffcc'
            return ''
        
        styled_df = hotspots_df.style.applymap(
            highlight_hcho, subset=['HCHO (ppb)']
        ).format({'HCHO (ppb)': '{:.2f}'})
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Known Source Regions
        st.markdown("""
        <div class="info-box">
            <h4>Known Source Regions</h4>
            <ul>
                <li><strong>Punjab-Haryana:</strong> Agricultural residue burning (Oct-Nov)</li>
                <li><strong>Delhi-NCR:</strong> Vehicular emissions, industrial, construction</li>
                <li><strong>Indo-Gangetic Plain:</strong> Domestic cooking, agricultural fires</li>
                <li><strong>Central India:</strong> Mining, industrial activities</li>
                <li><strong>Northeast India:</strong> Shifting cultivation, forest fires</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Fire & Wind Transport Tab
with tab4:
    st.markdown('<div class="section-header">Fire Influence and Transport Analysis</div>', unsafe_allow_html=True)
    
    # Controls for this tab
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 1])
    with col_ctrl1:
        show_fire_layer = st.checkbox("Show Fire Layer", value=True, key="fire_layer_cb")
    with col_ctrl2:
        show_wind_arrows = st.checkbox("Show Wind Arrows", value=True, key="wind_arrows_cb")
    with col_ctrl3:
        fire_date = st.date_input("Fire Date", pd.to_datetime("2024-06-15"))
    
    # Map and analysis columns
    col_map, col_charts = st.columns([2, 1])
    
    with col_map:
        # Fire and Wind Map
        m4 = folium.Map(location=[22.5, 79.0], zoom_start=5, tiles='CartoDB positron')
        
        if st.session_state.get('use_mock_data', True):
            # Add HCHO base layer (simplified)
            lats = np.linspace(8.0, 35.0, 20)
            lons = np.linspace(68.0, 97.0, 20)
            heat_data = []
            for lat in lats:
                for lon in lons:
                    if 8 <= lat <= 35 and 68 <= lon <= 97:
                        hcho = max(0.1, 2.0 - np.sqrt((lat-28.6)**2 + (lon-77.2)**2)*0.02 + np.random.normal(0,0.1))
                        heat_data.append([lat, lon, hcho])
            
            from folium.plugins import HeatMap
            HeatMap(heat_data, radius=12, blur=8, max_zoom=10, 
                   gradient={0.3: 'blue', 0.5: 'lime', 0.7: 'yellow', 0.9: 'red'}).add_to(m4)
            
            # Fire points overlay
            if show_fire_layer:
                fire_data = [
                    {'lat': 30.5, 'lon': 76.2, 'frp': 180, 'count': 12, 'date': '2024-06-15'},
                    {'lat': 29.8, 'lon': 75.8, 'frp': 150, 'count': 10, 'date': '2024-06-15'},
                    {'lat': 31.2, 'lon': 74.5, 'frp': 200, 'count': 15, 'date': '2024-06-14'},
                    {'lat': 28.9, 'lon': 80.1, 'frp': 90, 'count': 6, 'date': '2024-06-15'},
                    {'lat': 26.5, 'lon': 81.5, 'frp': 120, 'count': 8, 'date': '2024-06-15'},
                    {'lat': 24.2, 'lon': 85.8, 'frp': 75, 'count': 5, 'date': '2024-06-14'},
                    {'lat': 22.8, 'lon': 88.2, 'frp': 60, 'count': 4, 'date': '2024-06-15'}
                ]
                
                for fire in fire_data:
                    # Size based on FRP (Fire Radiative Power)
                    radius = max(4, min(12, 2 + fire['frp']/25))
                    folium.CircleMarker(
                        location=[fire['lat'], fire['lon']],
                        radius=radius,
                        color='darkred',
                        fill=True,
                        fill_color='red',
                        fill_opacity=0.7,
                        weight=1,
                        tooltip=f"FRP: {fire['frp']} MW<br>Count: {fire['count']}<br>Date: {fire['date']}"
                    ).add_to(m4)
            
            # Wind arrows (mock ERA5 u/v wind data)
            if show_wind_arrows:
                wind_points = [
                    {'lat': 20, 'lon': 75, 'u': 3.2, 'v': -1.5},  # Gujarat
                    {'lat': 25, 'lon': 80, 'u': 2.8, 'v': -0.8},  # MP
                    {'lat': 30, 'lon': 85, 'u': 1.5, 'v': 2.2},   # UP/Bihar border
                    {'lat': 35, 'lon': 78, 'u': 4.1, 'v': -0.5},  # Punjab
                    {'lat': 15, 'lon': 82, 'u': -1.2, 'v': 2.8}   # Andhra Pradesh
                ]
                
                for wp in wind_points:
                    # Calculate wind vector
                    speed = np.sqrt(wp['u']**2 + wp['v']**2)
                    if speed > 0.5:  # Only show significant winds
                        scale = 0.3  # Scale factor for arrow length
                        end_lat = wp['lat'] + wp['v'] * scale
                        end_lon = wp['lon'] + wp['u'] * scale
                        
                        folium.PolyLine(
                            locations=[[wp['lat'], wp['lon']], [end_lat, end_lon]],
                            color='blue',
                            weight=3,
                            opacity=0.8
                        ).add_to(m4)
                        
                        # Add arrowhead
                        folium.RegularPolygonMarker(
                            location=[end_lat, end_lon],
                            number_of_points=3,
                            radius=0,
                            rotation=15,
                            color='blue',
                            weight=2
                        ).add_to(m4)
        
        st_folium(m4, width=600, height=500, returned_objects=[])
    
    with col_charts:
        # Fire-HCHO Time Series
        st.markdown("#### Daily Fire Count vs Mean HCHO")
        if st.session_state.get('use_mock_data', True):
            dates = pd.date_range(end=st.session_state['end_date'], periods=10)
            fire_counts = np.array([8, 12, 15, 18, 22, 20, 16, 12, 9, 7])
            hcho_means = np.array([1.2, 1.4, 1.6, 1.9, 2.3, 2.1, 1.8, 1.5, 1.3, 1.1])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates, y=fire_counts,
                mode='lines+markers',
                name='Fire Count',
                line=dict(color='red', width=2),
                yaxis='y'
            ))
            fig.add_trace(go.Scatter(
                x=dates, y=hcho_means,
                mode='lines+markers',
                name='Mean HCHO (ppb)',
                line=dict(color='blue', width=2),
                yaxis='y2'
            ))
            
            fig.update_layout(
                xaxis_title="Date",
                yaxis=dict(title="Fire Count", side="left", color="red"),
                yaxis2=dict(title="HCHO (ppb)", side="right", overlaying="y", color="blue"),
                legend=dict(x=0.01, y=0.99),
                height=300,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Lag Correlation Bar Chart
        st.markdown("#### Fire-HCHO Lag Correlation")
        if st.session_state.get('use_mock_data', True):
            lags = [0, 1, 2, 3]
            correlations = [0.82, 0.87, 0.84, 0.79]  # Peak at 1-day lag
            
            fig = go.Figure(data=[
                go.Bar(
                    x=[f"Lag {lag} day" for lag in lags],
                    y=correlations,
                    marker_color=['#ff6b66', '#4ecdc4', '#45b7d1', '#96ceb4']
                )
            ])
            fig.update_layout(
                xaxis_title="Time Lag",
                yaxis_title="Pearson Correlation (R)",
                yaxis=dict(range=[0, 1]),
                height=250,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

# Model Performance Tab
with tab5:
    st.markdown('<div class="section-header">Model Performance Evaluation</div>', unsafe_allow_html=True)
    
    # Model Comparison Table
    st.markdown("#### Model Comparison Metrics")
    model_df = pd.DataFrame({
        'Model': ['CNN-LSTM (Primary)', 'CNN Only', 'LSTM Only', 'Random Forest', 'XGBoost'],
        'RMSE (↓)': [12.34, 15.67, 18.23, 14.56, 13.89],
        'MAE (↓)': [9.45, 11.23, 12.67, 10.34, 9.87],
        'R (↑)': [0.89, 0.78, 0.72, 0.81, 0.79],
        'R² (↑)': [0.79, 0.61, 0.52, 0.66, 0.62],
        'Training Time': ['2.1 hrs', '45 min', '1.2 hrs', '18 min', '25 min']
    })
    
    # Style the dataframe
    def highlight_min(s):
        is_min = s == s.min()
        return ['background-color: #d4edda' if v else '' for v in is_min]
    
    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: #d4edda' if v else '' for v in is_max]
    
    styled_model_df = model_df.style.apply(highlight_min, subset=['RMSE (↓)', 'MAE (↓)']) \
                                   .apply(highlight_max, subset=['R (↑)', 'R² (↑)']) \
                                   .format({
                                       'RMSE (↓)': '{:.2f}',
                                       'MAE (↓)': '{:.2f}',
                                       'R (↑)': '{:.2f}',
                                       'R² (↑)': '{:.2f}'
                                   })
    
    st.dataframe(styled_model_df, use_container_width=True, hide_index=True)
    
    # Actual vs Predicted Plot
    st.markdown("#### Actual vs Predicted AQI (Validation Set)")
    if st.session_state.get('use_mock_data', True):
        # Generate sample data
        np.random.seed(42)
        actual = np.random.randint(50, 400, 100)
        # Add realistic prediction error
        error = np.random.normal(0, 12, 100)
        predicted = np.clip(actual + error, 0, 500)
        
        fig = go.Figure()
        
        # Scatter plot
        fig.add_trace(go.Scatter(
            x=actual, y=predicted,
            mode='markers',
            marker=dict(
                size=8,
                color='rgba(102, 126, 234, 0.6)',
                line=dict(width=1, color='DarkSlateGrey')
            ),
            name='Predictions'
        ))
        
        # Perfect prediction line
        max_val = max(np.max(actual), np.max(predicted))
        min_val = min(np.min(actual), np.min(predicted))
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='Perfect Prediction'
        ))
        
        fig.update_layout(
            xaxis_title="Actual AQI",
            yaxis_title="Predicted AQI",
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Model Interpretation
    st.markdown("""
    <div class="info-box">
        <h4>Model Architecture Explanation</h4>
        <p><strong>CNN-LSTM (Primary Model):</strong> This hybrid architecture combines the strengths of both networks:</p>
        <ul>
            <li><strong>CNN Component:</strong> Extracts spatial features from satellite imagery (AOD, HCHO, etc.) and meteorological reanalysis data</li>
            <li><strong>LSTM Component:</strong> Captures temporal dependencies and trends in air quality patterns</li>
            <li><strong>Fusion Strategy:</strong> Features from CNN are fed into LSTM layers to create spatiotemporal representations</li>
        </ul>
        <p><strong>Performance Highlights:</strong></p>
        <ul>
            <li>CNN-LSTM achieves lowest RMSE (12.34) and highest R² (0.79) among all models tested</li>
            <li>Superior to standalone CNN (spatial only) or LSTM (temporal only) models</li>
            <li>Outperforms traditional ML models (Random Forest, XGBoost) in capturing complex spatiotemporal patterns</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)