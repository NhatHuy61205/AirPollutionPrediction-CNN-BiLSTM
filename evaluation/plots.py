from __future__ import annotations
from typing import Optional, Sequence, Tuple, Union, Dict
import numpy as np
import matplotlib.pyplot as plt


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


def _mask_finite(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[m], y_pred[m]


# -----------------------
# 1) Actual vs Pred (time order)
# -----------------------
def plot_actual_vs_pred(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    title: str = "Actual vs Predicted",
    n_points: Optional[int] = 500,
    start: int = 0,
    horizon: Optional[int] = None,
    figsize: Tuple[int, int] = (12, 4),
    show: bool = True,
):
    """
    Plot actual vs predicted over sample index (giống time-series).
    Supports:
      - y shape (n,)  -> plot 1 line true & pred
      - y shape (n,H) -> choose horizon (0..H-1). If horizon=None -> plot t+1 (index 0)
    Args:
      n_points: giới hạn số điểm để biểu đồ không quá nặng (None = plot all)
      start: bắt đầu từ sample index nào
    """
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)

    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true{yt.shape} vs y_pred{yp.shape}")

    if yt.ndim == 2:
        h = 0 if horizon is None else int(horizon)
        yt = yt[:, h]
        yp = yp[:, h]

    yt, yp = _mask_finite(yt, yp)

    end = yt.shape[0] if n_points is None else min(yt.shape[0], start + n_points)
    idx = np.arange(start, end)

    plt.figure(figsize=figsize)
    plt.plot(idx, yt[start:end], label="Actual")
    plt.plot(idx, yp[start:end], label="Predicted")
    plt.title(title)
    plt.xlabel("Sample index")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()


# -----------------------
# 2) Scatter: Actual vs Pred (calibration look)
# -----------------------
def plot_scatter_actual_pred(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    title: str = "Scatter: Actual vs Predicted",
    horizon: Optional[int] = None,
    sample: Optional[int] = 5000,
    figsize: Tuple[int, int] = (6, 6),
    show: bool = True,
):
    """
    Scatter plot actual vs predicted.
    - sample: downsample number of points for speed (None=all)
    """
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)

    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true{yt.shape} vs y_pred{yp.shape}")

    if yt.ndim == 2:
        h = 0 if horizon is None else int(horizon)
        yt = yt[:, h]
        yp = yp[:, h]

    yt, yp = _mask_finite(yt, yp)

    if sample is not None and yt.size > sample:
        idx = np.random.RandomState(42).choice(yt.size, size=sample, replace=False)
        yt = yt[idx]
        yp = yp[idx]

    plt.figure(figsize=figsize)
    plt.scatter(yt, yp, s=8, alpha=0.5)
    # y=x reference
    mn = float(min(np.min(yt), np.min(yp)))
    mx = float(max(np.max(yt), np.max(yp)))
    plt.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    if show:
        plt.show()


# -----------------------
# 3) Residuals vs Pred
# -----------------------
def plot_residuals_vs_pred(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    title: str = "Residuals vs Predicted",
    horizon: Optional[int] = None,
    sample: Optional[int] = 5000,
    figsize: Tuple[int, int] = (7, 5),
    show: bool = True,
):
    """
    Residuals (pred-true) vs predicted.
    Helps detect heteroscedasticity / bias at high values.
    """
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)

    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true{yt.shape} vs y_pred{yp.shape}")

    if yt.ndim == 2:
        h = 0 if horizon is None else int(horizon)
        yt = yt[:, h]
        yp = yp[:, h]

    yt, yp = _mask_finite(yt, yp)
    res = yp - yt

    if sample is not None and yt.size > sample:
        idx = np.random.RandomState(42).choice(yt.size, size=sample, replace=False)
        yp = yp[idx]
        res = res[idx]

    plt.figure(figsize=figsize)
    plt.scatter(yp, res, s=8, alpha=0.5)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Residual (Pred - Actual)")
    plt.tight_layout()
    if show:
        plt.show()


# -----------------------
# 4) Error distribution (hist)
# -----------------------
def plot_error_distribution(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    title: str = "Error Distribution (Pred - Actual)",
    horizon: Optional[int] = None,
    bins: int = 50,
    figsize: Tuple[int, int] = (7, 4),
    show: bool = True,
):
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)

    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true{yt.shape} vs y_pred{yp.shape}")

    if yt.ndim == 2:
        h = 0 if horizon is None else int(horizon)
        yt = yt[:, h]
        yp = yp[:, h]

    yt, yp = _mask_finite(yt, yp)
    err = yp - yt

    plt.figure(figsize=figsize)
    plt.hist(err, bins=bins)
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel("Error")
    plt.ylabel("Count")
    plt.tight_layout()
    if show:
        plt.show()


# -----------------------
# 5) Horizon-wise metrics plot (RMSE/MAE/MAPE/Corr arrays)
# -----------------------
def plot_horizon_metrics(
    metrics_per_horizon: Dict[str, ArrayLike],
    title: str = "Metrics by Horizon",
    figsize: Tuple[int, int] = (10, 4),
    show: bool = True,
):
    """
    metrics_per_horizon example:
      {
        "RMSE": np.array(H),
        "MAE": np.array(H),
        "MAPE": np.array(H),
        "Correlation": np.array(H),
      }
    """
    plt.figure(figsize=figsize)
    for k, v in metrics_per_horizon.items():
        arr = _to_numpy(v).reshape(-1)
        x = np.arange(1, arr.size + 1)
        plt.plot(x, arr, label=k)

    plt.title(title)
    plt.xlabel("Horizon step (t+1 ... t+H)")
    plt.ylabel("Metric value")
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()


# -----------------------
# 6) Compare models on one metric (bar chart)
# -----------------------
def plot_model_comparison_bar(
    model_to_metric: Dict[str, float],
    metric_name: str = "RMSE",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 4),
    show: bool = True,
):
    """
    Example input:
      {"LGBM": 12.3, "CNN-BiLSTM": 10.9, "XGB": 11.7}
    """
    names = list(model_to_metric.keys())
    vals = [float(model_to_metric[n]) for n in names]

    plt.figure(figsize=figsize)
    plt.bar(names, vals)
    plt.title(title or f"Model Comparison - {metric_name}")
    plt.ylabel(metric_name)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    if show:
        plt.show()


# -----------------------
# 7) Plot by Station (overlay) - useful for station_split
# -----------------------
def plot_actual_vs_pred_by_station(
    station_ids: ArrayLike,
    y_true: ArrayLike,
    y_pred: ArrayLike,
    title: str = "Actual vs Predicted by Station",
    n_points_per_station: int = 200,
    horizon: Optional[int] = None,
    max_stations: int = 6,
    figsize: Tuple[int, int] = (12, 8),
    show: bool = True,
):
    """
    station_ids: (n,) station label aligned with y_true/y_pred samples.
    For multi-horizon y: pick horizon.
    Plot up to max_stations stations (first unique ones).
    """
    sids = _to_numpy(station_ids).reshape(-1)
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)

    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true{yt.shape} vs y_pred{yp.shape}")
    if sids.shape[0] != yt.shape[0]:
        raise ValueError(f"station_ids length {sids.shape[0]} != n_samples {yt.shape[0]}")

    if yt.ndim == 2:
        h = 0 if horizon is None else int(horizon)
        yt = yt[:, h]
        yp = yp[:, h]

    # stations
    uniq = []
    for x in sids.tolist():
        if x not in uniq:
            uniq.append(x)
        if len(uniq) >= max_stations:
            break

    nrows = len(uniq)
    plt.figure(figsize=figsize)

    for i, st in enumerate(uniq, start=1):
        mask = (sids == st)
        yts = yt[mask]
        yps = yp[mask]
        yts, yps = _mask_finite(yts, yps)

        end = min(yts.shape[0], n_points_per_station)
        idx = np.arange(end)

        ax = plt.subplot(nrows, 1, i)
        ax.plot(idx, yts[:end], label="Actual")
        ax.plot(idx, yps[:end], label="Predicted")
        ax.set_title(f"Station {st}")
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Value")
        ax.legend()

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if show:
        plt.show()
