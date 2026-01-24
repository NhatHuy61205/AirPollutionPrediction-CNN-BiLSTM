import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import joblib

def create_features(df, target_col="PM2.5"):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    for l in [1, 2, 3, 6, 12, 24]:
        df[f"{target_col}_lag{l}"] = df.groupby("Station_No")[target_col].shift(l)

    for w in [3, 6, 12]:
        group = df.groupby("Station_No")[target_col]
        df[f"{target_col}_roll{w}_mean"] = group.transform(lambda x: x.rolling(w).mean())
        df[f"{target_col}_roll{w}_std"] = group.transform(lambda x: x.rolling(w).std())

    for col in ['Temperature', 'Humidity', 'NO2', 'SO2']:
        if col in df.columns:
            df[f"{col}_lag1"] = df.groupby("Station_No")[col].shift(1)
            df[f"{col}_lag3"] = df.groupby("Station_No")[col].shift(3)

    return df.ffill().bfill()

def scale_features(df, features_to_scale, target_col="PM2.5", artifact_dir="artifacts"):
    """Chuẩn hóa dữ liệu và lưu scaler"""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()

    pm25_scaler = MinMaxScaler()
    df[[target_col]] = pm25_scaler.fit_transform(df[[target_col]])
    joblib.dump(pm25_scaler, artifact_dir / "pm25_scaler.pkl")

    other_features = [f for f in features_to_scale if f in df.columns and f != target_col]
    if other_features:
        feature_scaler = MinMaxScaler()
        df[other_features] = feature_scaler.fit_transform(df[other_features])
        joblib.dump(feature_scaler, artifact_dir / "feature_scaler.pkl")

    print(f"[INFO] Scalers saved to {artifact_dir}")
    return df