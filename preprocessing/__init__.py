from preprocessing.load_data import load_air_quality_data
from preprocessing.clean_data import clean_and_resample, handle_outliers_and_missing
from preprocessing.feature_engineering import create_features, scale_features
import os
__all__ = [
    'load_air_quality_data',
    'clean_and_resample',
    'handle_outliers_and_missing',
    'create_features',
    'scale_features',
    'run_full_preprocessing'
]

def run_full_preprocessing(file_path, cols_sensor):
    df = load_air_quality_data(file_path)
    df = clean_and_resample(df, cols_sensor)
    df = handle_outliers_and_missing(df, cols_sensor)
    df = create_features(df)
    
    final_features = cols_sensor + ['hour_sin', 'hour_cos', 'PM2.5_lag1', 'PM2.5_roll3']
    df, scaler = scale_features(df, final_features)
    
    return df, scaler