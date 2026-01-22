from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

@dataclass
class XGBoostBaseConfig:
    time_col: str = "timestamp"
    target_col: str = "PM2.5"
    lags: int = 24
    horizon: int = 24
    rolling_windows: Tuple[int, ...] = (3, 6, 12, 24)
    add_hour_feature: bool = True
    xgb_params: Dict = None

    def __post_init__(self):
        if self.xgb_params is None:
            self.xgb_params = dict(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=8,
                tree_method="hist",  
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
            )

def _ensure_datetime(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out.sort_values(time_col).reset_index(drop=True)
    return out

def make_supervised_features(df: pd.DataFrame, cfg: XGBoostBaseConfig,
                             extra_feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
    df = _ensure_datetime(df, cfg.time_col)
    out = df.copy()

    if cfg.add_hour_feature:
        out["hour"] = out[cfg.time_col].dt.hour.astype(np.int16)

    for k in range(1, cfg.lags + 1):
        out[f"{cfg.target_col}_lag_{k}"] = out[cfg.target_col].shift(k)

    s = out[cfg.target_col].shift(1)
    for w in cfg.rolling_windows:
        out[f"{cfg.target_col}_rm_{w}"] = s.rolling(window=w, min_periods=w).mean()

    return out

def make_multi_horizon_targets(df_feat: pd.DataFrame, cfg: XGBoostBaseConfig) -> pd.DataFrame:
    out = df_feat.copy()
    for h in range(1, cfg.horizon + 1):
        out[f"y_t+{h}"] = out[cfg.target_col].shift(-h)
    return out

def build_xy(df: pd.DataFrame, cfg: XGBoostBaseConfig,
             extra_feature_cols: Optional[List[str]] = None):
    df_feat = make_supervised_features(df, cfg, extra_feature_cols)
    df_all = make_multi_horizon_targets(df_feat, cfg)

    feature_cols = []
    if cfg.add_hour_feature:
        feature_cols.append("hour")
    feature_cols += [f"{cfg.target_col}_lag_{k}" for k in range(1, cfg.lags + 1)]
    feature_cols += [f"{cfg.target_col}_rm_{w}" for w in cfg.rolling_windows]
    
    if extra_feature_cols:
        feature_cols += extra_feature_cols

    target_cols = [f"y_t+{h}" for h in range(1, cfg.horizon + 1)]
    
    df_model = df_all.dropna(subset=feature_cols + target_cols).copy()

    X = df_model[feature_cols]
    y = df_model[target_cols].to_numpy(dtype=np.float32)
    return X, y

def fit_model(X_train, y_train, cfg: XGBoostBaseConfig) -> MultiOutputRegressor:
    base = XGBRegressor(**cfg.xgb_params)
    model = MultiOutputRegressor(base, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def predict(model: MultiOutputRegressor, X) -> np.ndarray:
    return model.predict(X)