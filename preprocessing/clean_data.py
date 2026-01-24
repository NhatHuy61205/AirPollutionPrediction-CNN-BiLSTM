import pandas as pd
import numpy as np

def clean_and_resample(df, cols_sensor):
    """Loại bỏ giá trị vật lý vô lý và resample theo giờ."""
    df = df.copy()
    df.loc[(df['Temperature'] > 55) | (df['Temperature'] < -10), 'Temperature'] = np.nan
    df.loc[(df['Humidity'] > 100) | (df['Humidity'] < 0), 'Humidity'] = np.nan
    if 'PM2.5' in df.columns:
        df.loc[df['PM2.5'] <= 0, 'PM2.5'] = np.nan  
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df = df.groupby('Station_No')[cols_sensor].resample('h').mean().reset_index()
    return df

def handle_outliers_and_missing(df, cols_sensor):
    """Nội suy dữ liệu thiếu"""
    df = df.copy()
    
    for col in cols_sensor:
        df[col] = df.groupby('Station_No')[col].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both').ffill().bfill()
        )
    return df