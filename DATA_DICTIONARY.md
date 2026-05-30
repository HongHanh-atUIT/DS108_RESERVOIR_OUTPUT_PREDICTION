# Data Dictionary

## Tổng quan

Dataset được tạo bằng cách merge dữ liệu vận hành hồ chứa và dữ liệu thời tiết theo cột `time`.

## Các cột chính

| Tên cột | Ý nghĩa | Kiểu dữ liệu | Ghi chú |
|---|---|---|---|
| time | Thời điểm quan sát | datetime | Khóa merge dữ liệu EVN và weather |
| ten_ho | Tên hồ chứa | categorical | Tên hồ chứa |
| h_tl | Mực nước thượng lưu | numeric | Biến thủy văn |
| h_dbt | Mực nước dâng bình thường | numeric | Biến thủy văn |
| h_c | Mực nước chết | numeric | Biến thủy văn |
| q_ve | Lưu lượng nước về hồ | numeric | Có thể dùng làm biến mục tiêu |
| sigma_qx | Tổng lưu lượng xả | numeric | Biến có yếu tố mùa vụ |
| q_xt | Lưu lượng xả qua đập tràn | numeric | Biến vận hành |
| q_xm | Lưu lượng xả qua máy phát điện | numeric | Biến có yếu tố mùa vụ |
| n_cxs | Số cửa xả sâu | numeric | Biến vận hành |
| n_cxm | Số cửa xả mặt | numeric | Biến vận hành |
| temperature | Nhiệt độ | numeric | Dữ liệu thời tiết |
| humidity | Độ ẩm | numeric | Dữ liệu thời tiết |
| precipitation | Lượng mưa | numeric | Dữ liệu thời tiết |
| wind_speed | Tốc độ gió | numeric | Dữ liệu thời tiết |
| wind_diẻction | Hướng gió | numeric | Dữ liệu thời tiết |
| clou_cover_mid | Độ che phủ mây | numeric | Dữ liệu thời tiết |
| pressure | Áp suất | numeric | Dữ liệu thời tiết |
| soil_moisture_0_7cm | Độ ẩm đất ở 7cm | numeric | Dữ liệu thời tiết |

## Target variable

Trong thực nghiệm chính, biến mục tiêu là:

- `q_ve`: lưu lượng nước về hồ.

## Missing values

Missing values được xử lý bằng:

1. Rolling median với cửa sổ 3 bước thời gian.
2. Seasonal mean theo tuần và tháng, fit trên tập train.
3. Forward fill cho các biến không theo mùa.

## Train/Test split

Dữ liệu được chia theo thời gian:

- Train: trước `2025-01-01 00:00:00`.
- Test: từ `2025-01-01 00:00:00` trở đi.

Cách chia này phù hợp với bài toán chuỗi thời gian và hạn chế data leakage.