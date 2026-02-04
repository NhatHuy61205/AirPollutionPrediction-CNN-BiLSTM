from __future__ import annotations

from typing import Dict, Optional, Union
import numpy as np

ArrayLike = Union[np.ndarray, "list", "tuple"]

def _to_numpy(x: ArrayLike) -> np.ndarray:
    """Convert numpy / list / pandas / torch tensor -> np.ndarray(float64)."""
    if x is None:
        return None

    # torch tensor
    if hasattr(x, "detach") and callable(x.detach):
        x = x.detach().cpu().numpy()

    # pandas
    if hasattr(x, "to_numpy") and callable(x.to_numpy):
        x = x.to_numpy()

    x = np.asarray(x, dtype=np.float64)
    return x


def compute_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    eps: float = 1e-6,
    mask_nan: bool = True,
    per_horizon: bool = False,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Metrics:
      - Correlation (Pearson)
      - RMSE
      - MAPE
      - MAE

    Supports shapes:
      - (n,)
      - (n, H)

    Args:
      eps: tránh chia 0 cho MAPE
      mask_nan: loại bỏ các phần tử NaN/Inf trước khi tính
      per_horizon: nếu True và data là (n, H) -> trả thêm metric theo từng horizon (mảng H)
    """
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)

    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true{yt.shape} vs y_pred{yp.shape}")

    # flatten for global metrics
    yt_flat = yt.reshape(-1)
    yp_flat = yp.reshape(-1)

    if mask_nan:
        m = np.isfinite(yt_flat) & np.isfinite(yp_flat)
        yt_flat = yt_flat[m]
        yp_flat = yp_flat[m]

    if yt_flat.size == 0:
        return {"Correlation": np.nan, "RMSE": np.nan, "MAPE": np.nan, "MAE": np.nan}

    diff = yp_flat - yt_flat
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    mape = float(np.mean(np.abs(diff) / np.maximum(eps, np.abs(yt_flat))))

    # Pearson correlation
    yt_std = float(np.std(yt_flat))
    yp_std = float(np.std(yp_flat))
    if yt_std < eps or yp_std < eps:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(yt_flat, yp_flat)[0, 1])

    out: Dict[str, Union[float, np.ndarray]] = {
        "Correlation": corr,
        "RMSE": rmse,
        "MAPE": mape,
        "MAE": mae,
    }

    # per-horizon metrics
    if per_horizon and yt.ndim == 2:
        H = yt.shape[1]
        corr_h = np.full(H, np.nan, dtype=np.float64)
        rmse_h = np.full(H, np.nan, dtype=np.float64)
        mape_h = np.full(H, np.nan, dtype=np.float64)
        mae_h = np.full(H, np.nan, dtype=np.float64)

        for h in range(H):
            yth = yt[:, h]
            yph = yp[:, h]
            if mask_nan:
                mh = np.isfinite(yth) & np.isfinite(yph)
                yth = yth[mh]
                yph = yph[mh]
            if yth.size == 0:
                continue

            d = yph - yth
            mae_h[h] = np.mean(np.abs(d))
            rmse_h[h] = np.sqrt(np.mean(d * d))
            mape_h[h] = np.mean(np.abs(d) / np.maximum(eps, np.abs(yth)))

            ys = np.std(yth)
            ps = np.std(yph)
            if ys >= eps and ps >= eps:
                corr_h[h] = np.corrcoef(yth, yph)[0, 1]

        out["per_horizon"] = {
            "Correlation": corr_h,
            "RMSE": rmse_h,
            "MAPE": mape_h,
            "MAE": mae_h,
        }

    return out
def compute_metrics_real_scale(
    y_true_scaled: ArrayLike,
    y_pred_scaled: ArrayLike,
    scaler,
    eps: float = 1e-6,
    mask_nan: bool = True,
    target_index: int | None = None,  
    mape_threshold: float = 1.0,
) -> Dict[str, float]:
    """
    Compute metrics on REAL PM2.5 scale (µg/m³), consistent with the reference paper.

    - Nếu scaler được fit cho 1 cột (PM2.5): inverse_transform trực tiếp.
    - Nếu scaler được fit cho nhiều cột: phải cung cấp target_index (vị trí PM2.5 trong scaler).
    """

    yt = _to_numpy(y_true_scaled).reshape(-1)
    yp = _to_numpy(y_pred_scaled).reshape(-1)

    # --- detect scaler dimension ---
    n_feat = getattr(scaler, "n_features_in_", None)
    if n_feat is None:
        n_feat = len(getattr(scaler, "min_", []))

    if n_feat == 1:
        yt_real = scaler.inverse_transform(yt.reshape(-1, 1)).ravel()
        yp_real = scaler.inverse_transform(yp.reshape(-1, 1)).ravel()

    else:
        if target_index is None:
            raise ValueError(
                f"Scaler was fitted on {n_feat} features. "
                f"Please pass target_index (index of PM2.5 in the scaled feature list)."
            )

        ti = int(target_index)
        yt_real = (yt - scaler.min_[ti]) / scaler.scale_[ti]
        yp_real = (yp - scaler.min_[ti]) / scaler.scale_[ti]

    if mask_nan:
        m = np.isfinite(yt_real) & np.isfinite(yp_real)
        yt_real = yt_real[m]
        yp_real = yp_real[m]

    diff = yp_real - yt_real

    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    denom_mask = np.abs(yt_real) >= mape_threshold
    if np.any(denom_mask):
        mape = float(np.mean(np.abs(diff[denom_mask]) / np.abs(yt_real[denom_mask])))
    else:
        mape = float("nan")

    if np.std(yt_real) < eps or np.std(yp_real) < eps:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(yt_real, yp_real)[0, 1])

    return {"Correlation": corr, "RMSE": rmse, "MAPE": mape, "MAE": mae}