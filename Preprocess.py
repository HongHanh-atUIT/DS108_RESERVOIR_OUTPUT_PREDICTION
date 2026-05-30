import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.preprocessing import PowerTransformer, StandardScaler
from typing import List, Dict, Optional
import math
import os


def shift_column(df: pd.DataFrame, column_name: str, periods: int,
                 new_column_name: str = None, inplace: bool = False) -> pd.DataFrame:
    
    if column_name not in df.columns:
        raise ValueError(f"Cột '{column_name}' không tồn tại trong DataFrame")

    if not inplace:
        df = df.copy()

    if new_column_name is None:
        if inplace:
            new_column_name = column_name
        else:
            new_column_name = f"{column_name}_shift{periods:+d}"

    df[new_column_name] = df[column_name].shift(periods=periods)

    if periods > 0:
        df = df.iloc[periods:]
    else:
        df = df.iloc[:periods]

    return df

def drop_unnecessary(df: pd.DataFrame, columns_to_drop: List[str],
                     inplace: bool = False, verbose: bool = True) -> pd.DataFrame:
    """
    Drop các cột không cần thiết khỏi DataFrame.
    """
    if not inplace:
        df = df.copy()

    existing_cols = [col for col in columns_to_drop if col in df.columns]
    not_found_cols = [col for col in columns_to_drop if col not in df.columns]

    if existing_cols:
        df = df.drop(columns=existing_cols)
        if verbose:
            print(f"Đã drop {len(existing_cols)} cột: {existing_cols}")
    else:
        if verbose:
            print("Không có cột nào trong danh sách cần drop tồn tại trong DataFrame.")

    if not_found_cols and verbose:
        print(f"Không tìm thấy {len(not_found_cols)} cột: {not_found_cols}")

    return df

def transform(df: pd.DataFrame,
              skew_cols: List[str],
              scale_cols: List[str] = None,
              use_standardize: bool = False,
              verbose: bool = True):
    """
    Giảm skewness bằng Yeo-Johnson và/hoặc chuẩn hóa bằng StandardScaler.

    Returns:
    --------
    (df_transformed, transform_info) : tuple
        transform_info chứa 'power_transformer', 'lambdas', 'standard_scaler' (nếu có)
    """
    df = df.copy()
    transform_info = {}

    if len(skew_cols) > 0:
        skew_cols_valid = [col for col in skew_cols if col in df.columns]

        if skew_cols_valid:
            pt = PowerTransformer(method='yeo-johnson', standardize=False)
            df[skew_cols_valid] = pt.fit_transform(df[skew_cols_valid])

            lambda_dict = dict(zip(skew_cols_valid, pt.lambdas_))
            transform_info['lambdas'] = lambda_dict
            transform_info['power_transformer'] = pt

            if verbose:
                print(f"Đã áp dụng Yeo-Johnson trên {len(skew_cols_valid)} cột:")
                for col, lam in lambda_dict.items():
                    print(f"   • {col:25} → λ = {lam:.4f}")
        else:
            print("Không có cột skew nào tồn tại.")

    if use_standardize:
        if scale_cols is not None:
            scale_cols = [col for col in scale_cols if col in df.columns]
            
            scaler = StandardScaler()
            df[scale_cols] = scaler.fit_transform(df[scale_cols])
            transform_info['standard_scaler'] = scaler

            if verbose:
                print(f"\nĐã áp dụng StandardScaler trên {len(scale_cols)} cột")

    if verbose:
        print(f"\nTransform hoàn tất!")
        print(f"   - Sử dụng Yeo-Johnson : {len(skew_cols)} cột")
        print(f"   - Sử dụng StandardScaler : {'Có' if use_standardize else 'Không'}")

    return df, transform_info


def add_lag_features(df: pd.DataFrame, column: str,
                     lags: List[int] = [1, 2, 3],
                     drop_na: bool = True,
                     inplace: bool = False) -> pd.DataFrame:
    """
    Thêm các lag features cho một cột.
    """
    if not inplace:
        df = df.copy()

    for lag in lags:
        new_col_name = f"{column}_lag{lag}"
        df[new_col_name] = df[column].shift(lag)

    if drop_na:
        df = df.dropna(subset=[f"{column}_lag{lag}" for lag in lags])

    print(f"Đã thêm {len(lags)} lag features cho cột '{column}': "
          f"{[f'{column}_lag{lag}' for lag in lags]}")
    return df


def add_time_feature(df: pd.DataFrame, time_col: str = 'time',
                     use_cyclic_encode: bool = True,
                     inplace: bool = False) -> pd.DataFrame:
    """
    Thêm các đặc trưng thời gian (hour, month, year) và tùy chọn sin-cos encoding.
    """
    if not inplace:
        df = df.copy()

    df[time_col] = pd.to_datetime(df[time_col])
    df['hour'] = df[time_col].dt.hour
    df['month'] = df[time_col].dt.month
    df['year'] = df[time_col].dt.year

    if use_cyclic_encode:
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df = df.drop(columns=['hour', 'month'])
        print("Đã thêm time features với Cyclic Encoding (sin/cos)")
    else:
        print("Đã thêm time features (hour, month, year)")

    return df


def add_flood(df: pd.DataFrame,
              flood_months: List[int] = [10, 11, 12, 1],
              qve_threshold: float = 750,
              inplace: bool = False) -> pd.DataFrame:
    """
    Thêm cột nhị phân 'is_flood' dựa trên tháng lũ và ngưỡng lưu lượng q_ve.
    """
    if not inplace:
        df = df.copy()

    if 'month' not in df.columns:
        if 'time' in df.columns:
            df['month'] = pd.to_datetime(df['time']).dt.month
        else:
            raise ValueError("DataFrame phải có cột 'month' hoặc 'time'")

    df['is_flood'] = (
        (df['month'].isin(flood_months)) &
        (df['q_ve'] >= qve_threshold)
    ).astype(int)

    print(f"Đã tạo cột 'is_flood' với flood_months={flood_months} "
          f"và q_ve_threshold={qve_threshold}")
    return df

def run_pipeline(df: pd.DataFrame,
                 cols_to_drop: List[str],
                 skew_cols: List[str],
                 scale_cols: List[str],
                 use_standardize: bool = True,
                 lag_column: str = 'sigma_qx',
                 lags: List[int] = [1, 2, 3],
                 flood_months: List[int] = [10, 11, 12, 1],
                 qve_threshold: float = 750,
                 fit_transformers: bool = True,
                 transform_info: dict = None,
                 verbose: bool = True) -> tuple:
    """
    Chạy toàn bộ pipeline preprocessing & feature engineering trên một DataFrame.
    Dùng fit_transformers=True cho train (fit + transform),
    fit_transformers=False + truyền transform_info từ train cho test (chỉ transform).
    """
    
    label = "TRAIN" if fit_transformers else "TEST"
    if verbose:
        print(f"\n{'─' * 55}")
        print(f"  Pipeline — {label}  (input shape: {df.shape})")
        print(f"{'─' * 55}")
 
    df = df.copy()
 
    # Bước 1: Drop cột không cần thiết
    if verbose:
        print("\n[1] Drop unnecessary columns")
    df = drop_unnecessary(df, cols_to_drop, verbose=verbose)
    
    # Bước 2: Flood flag
    if verbose:
        print("\n[2] Flood flag")
    df = add_flood(df, flood_months=flood_months, qve_threshold=qve_threshold)
    
    # Bước 3: Transform (Yeo-Johnson + StandardScaler)
    if verbose:
        print("\n[3] Transform (skewness + scale)")
    if fit_transformers:
        df, transform_info = transform(
            df=df,
            skew_cols=skew_cols,
            scale_cols=scale_cols,
            use_standardize=use_standardize,
            verbose=verbose,
        )
    else:
        if transform_info is None:
            raise ValueError("Phải truyền transform_info khi fit_transformers=False")
 
        # Áp dụng PowerTransformer đã fit
        if 'power_transformer' in transform_info:
            pt = transform_info['power_transformer']
            valid_cols = [c for c in transform_info['lambdas'] if c in df.columns]
            df[valid_cols] = pt.transform(df[valid_cols])
            if verbose:
                print(f"   Áp dụng Yeo-Johnson (đã fit) trên {len(valid_cols)} cột")
 
        # Áp dụng StandardScaler đã fit
        if use_standardize and 'standard_scaler' in transform_info:
            scaler = transform_info['standard_scaler']
            valid_scale = [c for c in scale_cols if c in df.columns]
            df[valid_scale] = scaler.transform(df[valid_scale])
            if verbose:
                print(f"   Áp dụng StandardScaler (đã fit) trên {len(valid_scale)} cột")
    
    # Bước 4: Lag features
    if verbose:
        print(f"\n[4] Lag features — cột '{lag_column}', lags={lags}")
    df = add_lag_features(df, column=lag_column, lags=lags)
    
    # Bước 5: Time features
    if verbose:
        print("\n[5] Time features")
    df = add_time_feature(df)
    
    # Bước 6: Drop cột time
    if 'time' in df.columns:
        df = df.drop(columns=['time'])
 
    if verbose:
        print(f"\n  Pipeline hoàn tất — output shape: {df.shape}")
 
    return df, transform_info

if __name__ == "__main__":
    # ── Config ───────────────────────────────────────────────
    START_TEST_TIME = "2025-01-01 00:00:00"
    SRC_FILE = "Data/Clean/clean_data_st2.csv"
 
    FEAT_COLS = [
        'time', 'ten_ho', 'h_tl', 'h_dbt', 'h_c', 'q_ve', 'sigma_qx', 'q_xt',
        'q_xm', 'n_cxs', 'n_cxm', 'temperature', 'humidity', 'precipitations',
        'wind_speeds', 'wind_direction', 'cloud_cover_mid', 'pressure',
        'soil_moisture_0_7cm',
    ]
    COLS_TO_DROP = ['h_dbt', 'h_c', 'ten_ho']
    SKEW_COLS = [
        'q_ve', 'sigma_qx', 'q_xt', 'n_cxs', 'n_cxm', 'precipitations',
        'wind_speeds', 'cloud_cover_mid', 'humidity', 'soil_moisture_0_7cm',
    ]
    SCALE_COLS = [
        'h_tl', 'q_ve', 'sigma_qx', 'q_xt', 'q_xm', 'n_cxs', 'n_cxm',
        'temperature', 'humidity', 'precipitations', 'wind_speeds',
        'wind_direction', 'cloud_cover_mid', 'pressure', 'soil_moisture_0_7cm',
    ]
    EDA_COLS = [
        'h_tl', 'q_ve', 'sigma_qx', 'q_xt', 'q_xm', 'n_cxs', 'n_cxm',
        'temperature', 'humidity', 'precipitations', 'wind_speeds',
        'wind_direction', 'cloud_cover_mid', 'pressure', 'soil_moisture_0_7cm', 'target',
    ]
    
    os.makedirs('Data/Final', exist_ok=True)
    output_train = 'Data/Final/st2_train_tree_final.csv'
    output_test = 'Data/Final/st2_test_tree_final.csv'
 
    # ── Bước 1: Load & Shift ─────────────────────────────────
    print("=" * 55)
    print("BƯỚC 1: LOAD & SHIFT DATA")
    print("=" * 55)
 
    clean_df = pd.read_csv(SRC_FILE)
    clean_df["time"] = pd.to_datetime(clean_df["time"])
    print(f"Loaded: {clean_df.shape}")
 
    train_df = clean_df[clean_df["time"] < START_TEST_TIME].copy()
    test_df  = clean_df[clean_df["time"] >= START_TEST_TIME].copy()
    print(f"Train: {train_df.shape} | Test: {test_df.shape}")
 
    train_df = shift_column(train_df, column_name="sigma_qx", periods=-1, new_column_name="target")
    test_df  = shift_column(test_df,  column_name="sigma_qx", periods=-1, new_column_name="target")
    print(f"Sau shift — Train: {train_df.shape} | Test: {test_df.shape}")
 
    # ── Bước 2: Fit pipeline trên Train ────────────────────────────────
    train_processed, info = run_pipeline(
        df=train_df,
        cols_to_drop=COLS_TO_DROP,
        skew_cols=SKEW_COLS,
        scale_cols=SCALE_COLS,
        use_standardize=False,
        fit_transformers=True,   # fit trên train
        verbose=True,
    )
    
    # ── Bước 3: Apply pipeline lên Test ──
    test_processed, _ = run_pipeline(
        df=test_df,
        cols_to_drop=COLS_TO_DROP,
        skew_cols=SKEW_COLS,
        scale_cols=SCALE_COLS,
        use_standardize=False,
        fit_transformers=False,  # chỉ transform, không fit lại
        transform_info=info,
        verbose=True,
    )
    
    train_processed.to_csv(output_train, index=False, encoding="utf-8-sig")
    test_processed.to_csv(output_test, index=False, encoding="utf-8-sig")
    
    print("\n" + "=" * 55)
    print("KẾT QUẢ CUỐI")
    print("=" * 55)
    print(f"Train processed : {train_processed.shape}")
    print(f"Test  processed : {test_processed.shape}")
    print(f"\nĐã lưu train xuống {output_train}")
    print(f"\nĐã lưu test xuống {output_test}")
    print("\nHoàn tất pipeline!")