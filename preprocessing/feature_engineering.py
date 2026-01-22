import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import joblib


def create_features(df, target_col="PM2.5"):
    df["hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df[f"{target_col}_lag1"] = df.groupby("Station_No")[target_col].shift(1)
    df[f"{target_col}_roll3"] = df.groupby("Station_No")[target_col].transform(
        lambda x: x.rolling(window=3).mean()
    )

    df = df.ffill().bfill()
    return df


def scale_features(
    df,
    features_to_scale,
    target_col="PM2.5",
    artifact_dir="artifacts",
):
    """
    - Scale target PM2.5 bằng scaler RIÊNG
    - Scale các feature khác bằng scaler khác
    """

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    pm25_scaler = MinMaxScaler()
    df[[target_col]] = pm25_scaler.fit_transform(df[[target_col]])

    joblib.dump(pm25_scaler, artifact_dir / "pm25_scaler.pkl")

    other_features = [f for f in features_to_scale if f != target_col]
    if other_features:
        feature_scaler = MinMaxScaler()
        df[other_features] = feature_scaler.fit_transform(df[other_features])

        joblib.dump(feature_scaler, artifact_dir / "feature_scaler.pkl")

    print("[INFO] Saved artifacts:")
    print(" - pm25_scaler.pkl")
    if other_features:
        print(" - feature_scaler.pkl")

    return df
