import pandas as pd
import numpy as np

def clean_and_resample(df, cols_sensor):
    """Loại bỏ giá trị bất thường và resample theo giờ."""
    # Loại bỏ giá trị vật lý vô lý
    df.loc[(df['Temperature'] > 60) | (df['Temperature'] < 10), 'Temperature'] = np.nan
    df.loc[df['Humidity'] > 100, 'Humidity'] = np.nan
    
    # Resample theo giờ cho từng trạm
    df = df.set_index('date')
    df = df.groupby('Station_No')[cols_sensor].resample('h').mean().reset_index()
    return df

def handle_outliers_and_missing(df, cols_sensor):
    """Xử lý outliers bằng phương pháp IQR và nội suy dữ liệu thiếu."""
    def cap_outliers(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        return series.clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)

    for col in cols_sensor:
        # Giới hạn outlier
        df[col] = df.groupby('Station_No')[col].transform(cap_outliers)
        # Nội suy (Interpolate)
        df[col] = df.groupby('Station_No')[col].transform(
            lambda x: x.interpolate(method='linear').ffill().bfill()
        )
    return df