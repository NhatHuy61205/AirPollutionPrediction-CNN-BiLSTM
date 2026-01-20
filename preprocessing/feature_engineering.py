import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_features(df, target_col='PM2.5'):
    """Tạo đặc trưng thời gian, tuần hoàn và các biến trễ."""
    # Đặc trưng thời gian
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek

    # Đặc trưng tuần hoàn (Sin/Cos)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)

    # Đặc trưng trễ (Lag) và Trung bình trượt (Rolling)
    df[f'{target_col}_lag1'] = df.groupby('Station_No')[target_col].shift(1)
    df[f'{target_col}_roll3'] = df.groupby('Station_No')[target_col].transform(
        lambda x: x.rolling(window=3).mean()
    )

    # Lấp đầy NaN do dịch chuyển hàng
    df = df.ffill().bfill()
    return df

def scale_features(df, features_to_scale):
    """Chuẩn hóa dữ liệu về khoảng [0, 1]."""
    scaler = MinMaxScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    return df, scaler