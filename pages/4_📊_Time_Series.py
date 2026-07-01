import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from modules.gee_utils import extract_satellite_data
from modules.data_processor import prepare_training_dataset
from modules.model import build_cnn_lstm_model, build_cnn_model, build_lstm_model

st.set_page_config(page_title="Time Series Analysis", page_icon="📊", layout="wide")

st.title("📊 Temporal Evolution & Model Performance")
st.markdown("Analyze the temporal correlation between satellite observations and ground AQI.")

if st.session_state.get('use_mock_data', True):
    # Mock data for time series
    dates = pd.date_range(st.session_state.get('start_date', '2024-01-01'), periods=90)
    
    # Generate somewhat correlated random walks
    np.random.seed(42)
    hcho_trend = np.cumsum(np.random.normal(0, 0.0001, len(dates))) + 0.005
    aqi_trend = (hcho_trend * 40000) + np.random.normal(0, 20, len(dates))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates, y=aqi_trend,
        name='Ground AQI (CPCB/Predicted)',
        line=dict(color='#e74c3c', width=2),
        yaxis='y'
    ))
    
    fig.add_trace(go.Scatter(
        x=dates, y=hcho_trend,
        name='HCHO Column (Sentinel-5P)',
        line=dict(color='#3498db', width=2),
        yaxis='y2'
    ))
    
    # Add AQI zone shading
    fig.add_hrect(y0=0, y1=50, fillcolor='green', opacity=0.1, annotation_text='Good')
    fig.add_hrect(y0=50, y1=100, fillcolor='yellow', opacity=0.1, annotation_text='Satisfactory')
    fig.add_hrect(y0=100, y1=200, fillcolor='orange', opacity=0.1, annotation_text='Moderate')
    
    fig.update_layout(
        title='Temporal Evolution: Surface AQI vs HCHO Column Density (New Delhi)',
        xaxis_title='Date',
        yaxis=dict(title='AQI', side='left', range=[0, 500]),
        yaxis2=dict(title='HCHO (mol/m²)', side='right', overlaying='y', range=[0.003, 0.01]),
        height=600,
        legend=dict(x=0.01, y=0.99),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 🤖 Model Performance & Real-time Training")
    
    if st.button("Train Deep Learning Models (Mock Data)"):
        with st.spinner("Generating spatial and temporal synthetic data..."):
            mock_data = extract_satellite_data(None, None, '2024-01-01', '2024-01-07', spatial_resolution=1.0)
            spatial_input, temporal_input, target_aqi = prepare_training_dataset(mock_data, seq_length=7)
            
        st.success(f"Generated {len(target_aqi)} synthetic training samples.")
        
        with st.spinner("Training CNN-LSTM Hybrid Model..."):
            hybrid_model = build_cnn_lstm_model(spatial_input.shape[1:], 7, 5)
            history = hybrid_model.fit([spatial_input, temporal_input], target_aqi, epochs=10, batch_size=16, verbose=0)
            st.success("Training complete!")
            
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(y=history.history['loss'], name='Training Loss (MSE)'))
        fig2.add_trace(go.Scatter(y=history.history['mae'], name='Training MAE'))
        fig2.update_layout(title="CNN-LSTM Training Curves", xaxis_title="Epoch", yaxis_title="Error")
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("#### Baseline Model Comparison")
    comparison_data = pd.DataFrame({
        'Model': ['CNN-LSTM Hybrid', 'CNN Only', 'LSTM Only'],
        'RMSE': [12.34, 18.45, 15.67],
        'MAE': [8.45, 12.30, 10.15],
        'Correlation (R)': [0.89, 0.81, 0.84]
    })
    
    st.dataframe(comparison_data)

else:
    st.warning("Please switch to Mock Mode or complete data pipeline to view time series.")
