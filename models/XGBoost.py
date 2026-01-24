from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

@dataclass
class XGBoostBaseConfig:
    time_col: str = "date"
    target_col: str = "PM2.5"
    lags: int = 24
    horizon: int = 24
    rolling_windows: Tuple[int, ...] = (3, 6, 12, 24)
    add_hour_feature: bool = True
    xgb_params: Dict = None

    def __post_init__(self):
        if self.xgb_params is None:
            self.xgb_params = dict(
                n_estimators=1000,      
                learning_rate=0.02,     
                max_depth=9,            
                tree_method="hist",
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,          
                reg_lambda=2.0,         
                random_state=42,
                n_jobs=-1,
            )

def _ensure_datetime(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out.sort_values([ "Station_No", time_col]).reset_index(drop=True)
    return out

def build_xy(df: pd.DataFrame, cfg: XGBoostBaseConfig, extra_feature_cols: Optional[List[str]] = None):
    df_all = df.copy()
    
    target_cols = [f"y_t+{h}" for h in range(1, cfg.horizon + 1)]
    for h, col in enumerate(target_cols, 1):
        df_all[col] = df_all.groupby("Station_No")[cfg.target_col].shift(-h)
    
    feature_cols = [c for c in df.columns if any(x in c.lower() for x in ["lag", "roll", "hour", "day"])]
    
    if extra_feature_cols:
        for c in extra_feature_cols:
            if c in df_all.columns and c not in target_cols and c not in feature_cols:
                feature_cols.append(c)

    if cfg.target_col in feature_cols:
        feature_cols.remove(cfg.target_col)
    
    df_model = df_all.dropna(subset=feature_cols + target_cols).copy()
    
    if len(df_model) == 0:
        return None, None

    X = np.ascontiguousarray(df_model[feature_cols].values, dtype=np.float32)
    y = np.ascontiguousarray(df_model[target_cols].values, dtype=np.float32)
    
    return X, y


def fit_model(X_train, y_train, cfg: XGBoostBaseConfig) -> MultiOutputRegressor:
    base = XGBRegressor(**cfg.xgb_params)
    model = MultiOutputRegressor(base, n_jobs=1) 
    model.fit(X_train, y_train)
    return model

def predict(model: MultiOutputRegressor, X) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        X = X.values.astype(np.float32)
    return model.predict(X)