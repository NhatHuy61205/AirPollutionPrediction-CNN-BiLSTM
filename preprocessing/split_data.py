from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

@dataclass
class SplitInfo:
    split_type: str
    output_dir: str
    train_path: str
    val_path: str
    test_path: str
    n_total: int
    n_train: int
    n_val: int
    n_test: int
    train_range: Tuple[pd.Timestamp, pd.Timestamp]
    val_range: Tuple[pd.Timestamp, pd.Timestamp]
    test_range: Tuple[pd.Timestamp, pd.Timestamp]


def _validate_ratios(train_size: float, val_size: float, test_size: float) -> None:
    s = train_size + val_size + test_size
    if abs(s - 1.0) > 1e-9:
        raise ValueError("train_size + val_size + test_size phải = 1.0")
    if not (0 < train_size < 1 and 0 < val_size < 1 and 0 < test_size < 1):
        raise ValueError("train/val/test size phải nằm trong (0, 1)")


def _load_and_prepare(
    input_csv_path: str,
    datetime_col: str,
    keep_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv_path)

    if datetime_col not in df.columns:
        raise ValueError(
            f"Không thấy cột thời gian '{datetime_col}'. Columns hiện có: {list(df.columns)}"
        )

    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    df = df.dropna(subset=[datetime_col]).sort_values(datetime_col).reset_index(drop=True)

    if keep_columns is not None:
        missing = [c for c in keep_columns if c not in df.columns]
        if missing:
            raise ValueError(f"keep_columns có cột không tồn tại: {missing}")
        df = df[keep_columns].copy()

    if len(df) < 10:
        raise ValueError("Dataset quá ít dòng để chia train/val/test hợp lý.")

    return df


def _save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str,
) -> Tuple[str, str, str]:
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    return train_path, val_path, test_path


def global_time_split_and_save(
    input_csv_path: str = "data/Air_Quality_Processed.csv",
    output_dir: str = "global_split",   
    datetime_col: str = "date",
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    keep_columns: Optional[list[str]] = None,
) -> SplitInfo:
    """
    Chia toàn bộ dataset theo timeline chung (không shuffle).
    Output luôn được lưu ở: <PROJECT_ROOT>/data/<output_dir>/
    """
    _validate_ratios(train_size, val_size, test_size)
    df = _load_and_prepare(input_csv_path, datetime_col, keep_columns)

    n = len(df)
    n_train = int(n * train_size)
    n_val = int(n * val_size)
    n_test = n - n_train - n_val

    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError("Một trong các tập bị rỗng. Hãy chỉnh lại tỉ lệ.")

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()

    
    out_dir = DATA_ROOT / output_dir
    train_path, val_path, test_path = _save_splits(train_df, val_df, test_df, str(out_dir))

    return SplitInfo(
        split_type="global_split",
        output_dir=str(out_dir),         
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        n_total=n,
        n_train=len(train_df),
        n_val=len(val_df),
        n_test=len(test_df),
        train_range=(train_df[datetime_col].min(), train_df[datetime_col].max()),
        val_range=(val_df[datetime_col].min(), val_df[datetime_col].max()),
        test_range=(test_df[datetime_col].min(), test_df[datetime_col].max()),
    )


def station_time_split_and_save(
    input_csv_path: str = "data/Air_Quality_Processed.csv",
    output_dir: str = "station_split",   
    datetime_col: str = "date",
    station_col: str = "Station_No",
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    keep_columns: Optional[list[str]] = None,
    require_min_points_per_station: int = 30,
) -> SplitInfo:
    """
    Chia theo timeline RIÊNG cho từng trạm:
      - Mỗi station: sort theo date rồi cắt train/val/test
      - Sau đó concat tất cả train lại, concat val, concat test

    Output luôn được lưu ở: <PROJECT_ROOT>/data/<output_dir>/
    """
    _validate_ratios(train_size, val_size, test_size)
    df = _load_and_prepare(input_csv_path, datetime_col, keep_columns)

    if station_col not in df.columns:
        raise ValueError(
            f"Không thấy cột trạm '{station_col}'. Columns hiện có: {list(df.columns)}"
        )

    trains, vals, tests = [], [], []

    for station_id, g in df.groupby(station_col, sort=False):
        g = g.sort_values(datetime_col).reset_index(drop=True)
        m = len(g)

        if m < require_min_points_per_station:
            continue

        m_train = int(m * train_size)
        m_val = int(m * val_size)
        m_test = m - m_train - m_val

        if m_train <= 0 or m_val <= 0 or m_test <= 0:
            continue

        trains.append(g.iloc[:m_train].copy())
        vals.append(g.iloc[m_train:m_train + m_val].copy())
        tests.append(g.iloc[m_train + m_val:].copy())

    if len(trains) == 0 or len(vals) == 0 or len(tests) == 0:
        raise ValueError(
            "Sau khi chia theo station, một trong các tập bị rỗng. "
            "Hãy giảm require_min_points_per_station hoặc kiểm tra dữ liệu."
        )

    train_df = pd.concat(trains, axis=0, ignore_index=True)
    val_df = pd.concat(vals, axis=0, ignore_index=True)
    test_df = pd.concat(tests, axis=0, ignore_index=True)

    # sắp lại theo thời gian để tiện debug/visualize
    train_df = train_df.sort_values(datetime_col).reset_index(drop=True)
    val_df = val_df.sort_values(datetime_col).reset_index(drop=True)
    test_df = test_df.sort_values(datetime_col).reset_index(drop=True)

    
    out_dir = DATA_ROOT / output_dir
    train_path, val_path, test_path = _save_splits(train_df, val_df, test_df, str(out_dir))

    return SplitInfo(
        split_type="station_split",
        output_dir=str(out_dir),         
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        n_total=len(df),
        n_train=len(train_df),
        n_val=len(val_df),
        n_test=len(test_df),
        train_range=(train_df[datetime_col].min(), train_df[datetime_col].max()),
        val_range=(val_df[datetime_col].min(), val_df[datetime_col].max()),
        test_range=(test_df[datetime_col].min(), test_df[datetime_col].max()),
    )

def make_all_splits(
    input_csv_path: str = "data/Air_Quality_Processed.csv",
    datetime_col: str = "date",
    station_col: str = "Station_No",
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    keep_columns: Optional[list[str]] = None,
) -> Dict[str, SplitInfo]:
    """
    Chạy 1 lần tạo cả 2 kiểu split:
      - <PROJECT_ROOT>/data/global_split/{train,val,test}.csv
      - <PROJECT_ROOT>/data/station_split/{train,val,test}.csv
    """
    info_global = global_time_split_and_save(
        input_csv_path=input_csv_path,
        output_dir="global_split",        
        datetime_col=datetime_col,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        keep_columns=keep_columns,
    )

    info_station = station_time_split_and_save(
        input_csv_path=input_csv_path,
        output_dir="station_split",       
        datetime_col=datetime_col,
        station_col=station_col,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        keep_columns=keep_columns,
    )

    return {"global": info_global, "station": info_station}
