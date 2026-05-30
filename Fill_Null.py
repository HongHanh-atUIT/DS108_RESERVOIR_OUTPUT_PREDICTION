"""
Sau khi tìm được phương pháp điền null thích hợp, lựa chọn phương pháp dùng Rolling Median + Seasonal Fill.
File này sẽ tự merge 2 df của evn và weather tạo raw_data và lưu lại.
Sau đó sẽ điền null trong raw_data tạo clean_data và lưu lại.
"""
import pandas as pd
import os
import shutil

# Ghép các cột ngày và giờ của dữ liệu evn để tạo biến time làm khóa để merge với weather
def add_time_column(df, date_col="ngay", time_col="gio", new_col="time"):
    df = df.copy()

    df[new_col] = pd.to_datetime(
        df[date_col].astype(str) + " " + df[time_col].astype(str),
        format="%Y-%m-%d %H:%M",
        errors="coerce"
    )
    
    # Di chuyển lên đầu
    cols = [new_col] + [col for col in df.columns if col != new_col]
    df = df[cols]
    
    df = df.drop(columns=["ngay", "gio"])
    return df

# Chuẩn hóa cột time của weather để giống với cột time của evn -> thống nhất format time
def convert_time(df, new_col = "time", old_col = "Time"):
    df[new_col] = pd.to_datetime(
        df[old_col],
        format="%Y-%m-%dT%H:%M",
        errors="coerce"
    )
    cols = [new_col] + [col for col in df.columns if col != new_col]
    df = df[cols]
    df = df.drop(columns=[old_col])
    return df

# Rename các cột bỏ phần đơn vị phía sau
def clean_column_names(df):
    new_columns = {}
    for col in df.columns:
        # Tách lấy phần trước dấu khoảng trắng đầu tiên
        new_name = col.split(' ')[0].lower()
        new_columns[col] = new_name
    
    df.rename(columns=new_columns, inplace=True)
    return df

# Merge 2 bảng theo kiểu outer join thông qua biến time
def merge(df1, df2, _how="outer", _on="time"):
    merged_df = pd.merge(df1, df2, on=_on, how=_how)
    merged_df = merged_df.sort_values("time").reset_index(drop=True)
    return merged_df

# Tính trước các giá trị trung bình trên tập train để tránh data leakage
def fit_rolling_median(train_df, SEASONAL_COLS):
    train = train_df.copy()
    train["time"] = pd.to_datetime(train["time"])
    train["week"] = train["time"].dt.isocalendar().week.astype(int)
    train["month"] = train["time"].dt.month

    seasonal_lookup = {}
    for col in SEASONAL_COLS:
        lookup = train.groupby(["week", "month"])[col].mean()
        seasonal_lookup[col] = lookup
    return seasonal_lookup

# Fill null, áp dụng lookup có từ tập train
def fill_rolling_median(df, SEASONAL_COLS, NON_SEASONAL_COLS, window_size, seasonal_lookup):
    result = df.copy()
    result["time"] = pd.to_datetime(result["time"])
    all_cols = SEASONAL_COLS + NON_SEASONAL_COLS

    # STEP 1: Rolling median
    rolling_med = result[all_cols].shift(1).rolling(window=window_size, min_periods=1).median()
    result[all_cols] = result[all_cols].fillna(rolling_med)

    # STEP 2: Seasonal fill
    if len(SEASONAL_COLS) > 0:
        result["week"] = result["time"].dt.isocalendar().week.astype(int)
        result["month"] = result["time"].dt.month
        for col in SEASONAL_COLS:
            lookup = seasonal_lookup[col]
            seasonal_values = result.set_index(["week", "month"]).index.map(lookup)
            result[col] = result[col].fillna(pd.Series(seasonal_values, index=result.index))
            result[col] = result[col].ffill()
        result = result.drop(columns=["week", "month"])

    # STEP 3: Non seasonal
    for col in NON_SEASONAL_COLS:
        result[col] = result[col].ffill()
    return result


def main():
    # Config
    evn_path_old = r"Data\evn_SongBaHa_2022_2025.csv"
    evn_path_new = "Data/Raw/evn_SongBaHa_2022_2025.csv"
    
    wt_path_old = "Data/sbh_weather_data.csv"
    wt_path_new = "Data/Raw/sbh_weather_data.csv"
    
    output_raw = "Data/Raw/raw_data_sbh.csv"
    output_clean = "Data/Clean/clean_data_sbh.csv"
    
    START_TEST_TIME = "2025-01-01 00:00:00"
    NON_SEASONAL_COLS = ['h_tl', 'h_dbt', 'h_c', 'q_ve', 'q_xt', 'n_cxs', 'n_cxm']
    SEASONAL_COLS = ['sigma_qx', 'q_xm']
    
    # Tạo folder và move file cũ
    os.makedirs('Data/Raw', exist_ok=True)
    os.makedirs('Data/Clean', exist_ok=True)

    if os.path.exists(evn_path_old) and os.path.exists(wt_path_old):
        shutil.move(evn_path_old, evn_path_new)
        shutil.move(wt_path_old, wt_path_new)
        print("[RAW DATA] Đã move các file thành công vào 'Data/Raw'")

    # Load dữ liệu
    evn = pd.read_csv(evn_path_new)
    wt = pd.read_csv(wt_path_new)
    
    # Gọi các hàm để merge, tạo thành merged_df
    print("[EVN] Thêm cột time....")
    evn = add_time_column(evn)
    print("[WT] Format cột time....")
    weather = convert_time(wt)
    print("[MERGE] Đang merge dữ liệu....")
    merged_df = merge(evn, weather)
    print("=" * 60)
    print("Hoàn tất quá trình merge!")
    print("[SAVE] Lưu merge làm dữ liệu thô ....")
    merged_df.to_csv(output_raw, index=False, encoding="utf-8-sig")
    print(f"[SAVE] Đã lưu dữ liệu thô xuống địa chỉ {output_raw}")
    print("[MERGE] Làm sạch tên cột....")
    merged_df = clean_column_names(merged_df)
    print("[MERGE] Columns list:")
    print(merged_df.columns.tolist())
    print("=" * 60)
    
    # Fill null
    print("[MERGE] Tiến hành fill missing values sử dụng Rolling Median với ws = 3...")
    merged_df["time"] = pd.to_datetime(merged_df["time"])
    train_df = merged_df[
        (merged_df["time"] < START_TEST_TIME)
    ].copy()
    
    seasonal_lookup = fit_rolling_median(train_df, SEASONAL_COLS)
    clean_df = fill_rolling_median(merged_df, SEASONAL_COLS=SEASONAL_COLS, NON_SEASONAL_COLS=NON_SEASONAL_COLS, 
                                    window_size= 3, seasonal_lookup = seasonal_lookup)
    
    clean_df["ten_ho"] = clean_df["ten_ho"].fillna(clean_df["ten_ho"].mode().values[0])    
    clean_df.to_csv(output_clean, index=False, encoding="utf-8-sig")
    print(f"[CLEAN] Fill null hoàn tất, lưu tại {output_clean}")
    
if __name__ == "__main__":
    main()