import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

def build_lstm_sequences(df, target_col='PM25', lags=24, horizon=24):
    """Biến đổi dữ liệu sang dạng 3D [Samples, Time_Steps, Features]"""
    X, y, station_ids = [], [], []
    
    for sid in df['Station_No'].unique():
        st_data = df[df['Station_No'] == sid][target_col].values
        if len(st_data) < (lags + horizon):
            continue
            
        for i in range(len(st_data) - lags - horizon + 1):
            X.append(st_data[i : i + lags])
            y.append(st_data[i + lags : i + lags + horizon])
            station_ids.append(sid)
            
    X_arr = np.array(X).reshape((-1, lags, 1))
    y_arr = np.array(y)
    return X_arr, y_arr, np.array(station_ids)

def get_lstm_model(input_shape, horizon=24):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(horizon)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model