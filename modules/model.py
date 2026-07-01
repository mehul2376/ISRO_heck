import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

def build_cnn_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1, activation='linear')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_lstm_model(seq_length, num_features):
    inputs = layers.Input(shape=(seq_length, num_features))
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(inputs)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1, activation='linear')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_cnn_lstm_model(spatial_input_shape, seq_length, num_temporal_features):
    spatial_input = layers.Input(shape=spatial_input_shape, name='spatial_input')
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(spatial_input)
    x = layers.GlobalAveragePooling2D()(x)
    spatial_features = layers.Dense(32, activation='relu')(x)
    
    temporal_input = layers.Input(shape=(seq_length, num_temporal_features), name='temporal_input')
    y = layers.Bidirectional(layers.LSTM(32, return_sequences=False))(temporal_input)
    temporal_features = layers.Dense(32, activation='relu')(y)
    
    merged = layers.Concatenate()([spatial_features, temporal_features])
    z = layers.Dense(64, activation='relu')(merged)
    outputs = layers.Dense(1, activation='linear')(z)
    
    model = Model(inputs=[spatial_input, temporal_input], outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
