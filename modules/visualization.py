import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap
import numpy as np

def create_aqi_choropleth(predicted_aqi, lat_grid, lon_grid):
    """
    Create an interactive AQI choropleth over India using Plotly.
    """
    fig = go.Figure(data=go.Heatmap(
        z=predicted_aqi,
        x=lon_grid[0, :],
        y=lat_grid[:, 0],
        colorscale=[
            [0, '#00e400'],     # Good - Green
            [0.1, '#ffff00'],   # Satisfactory - Yellow
            [0.2, '#ff7e00'],   # Moderate - Orange
            [0.4, '#ff0000'],   # Poor - Red
            [0.6, '#8f3f97'],   # Very Poor - Purple
            [0.8, '#7e0023'],   # Severe - Maroon
            [1.0, '#7e0023']
        ],
        zmin=0, zmax=500,
        colorbar=dict(
            title='AQI',
            tickvals=[0, 50, 100, 200, 300, 400, 500],
            titlefont=dict(size=14, color='#1a1a2e'),
            tickfont=dict(size=12, color='#1a1a2e')
        ),
        hovertemplate='Longitude: %{x}<br>Latitude: %{y}<br>AQI: %{z:.1f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text='Predicted Surface AQI over India',
            font=dict(size=20, color='#1a1a2e')
        ),
        xaxis=dict(
            title='Longitude',
            titlefont=dict(size=14, color='#1a1a2e'),
            tickfont=dict(size=12, color='#1a1a2e'),
            showgrid=True,
            gridwidth=1,
            gridcolor='#e0e0e0'
        ),
        yaxis=dict(
            title='Latitude',
            titlefont=dict(size=14, color='#1a1a2e'),
            tickfont=dict(size=12, color='#1a1a2e'),
            showgrid=True,
            gridwidth=1,
            gridcolor='#e0e0e0'
        ),
        height=700,
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        margin=dict(l=60, r=30, t=80, b=60)
    )
    return fig

def create_hcho_hotspot_map(hotspots, fire_coords, hcho_grid, lat_grid, lon_grid):
    """
    Create folium map showing HCHO hotspots with fire overlay.
    """
    m = folium.Map(
        location=[22.5, 79.0],
        zoom_start=5,
        tiles='CartoDB positron',
        control_scale=True
    )
    
    heat_data = []
    for i in range(lat_grid.shape[0]):
        for j in range(lon_grid.shape[1]):
            if not np.isnan(lat_grid[i,j]):
                heat_data.append([lat_grid[i,j], lon_grid[i,j], hcho_grid[i,j]])
    
    HeatMap(
        heat_data,
        radius=12,
        blur=10,
        max_zoom=10,
        gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'}
    ).add_to(m)
    
    if fire_coords is not None:
        fire_group = folium.FeatureGroup(name='Fire Points', show=True)
        for lat, lon, frp in fire_coords:
            folium.CircleMarker(
                location=[lat, lon],
                radius=max(3, min(8, 2 + frp/10)),
                color='darkred',
                fill=True,
                fill_color='red',
                fill_opacity=0.8,
                weight=1,
                tooltip=f"Fire Radiative Power: {frp:.1f}"
            ).add_to(fire_group)
        fire_group.add_to(m)
        
    if hotspots:
        hotspot_group = folium.FeatureGroup(name='HCHO Hotspots', show=True)
        for hs in hotspots:
            folium.Marker(
                location=[hs['lat'], hs['lon']],
                popup=folium.Popup(f"HCHO Intensity: {hs['intensity']:.2f}", max_width=200),
                icon=folium.Icon(color='orange', icon='fire', prefix='fa')
            ).add_to(hotspot_group)
        hotspot_group.add_to(m)
        
    folium.LayerControl(collapsed=False).add_to(m)
    return m