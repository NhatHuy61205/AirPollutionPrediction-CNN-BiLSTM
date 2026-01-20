import pandas as pd
import os

def load_air_quality_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file tại: {file_path}")
    
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df = df.sort_values(by=['Station_No', 'date'])
    return df