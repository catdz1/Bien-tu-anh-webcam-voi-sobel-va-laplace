
## 1. Giới thiệu

Ứng dụng phát hiện biên từ video webcam theo thời gian thực, sử dụng hai thuật toán xử lý ảnh truyền thống là **Sobel** và **Laplacian**.

Project được phát triển trong khuôn khổ học phần **Xử lý ảnh và Thị giác máy tính (IT3243)**.

**Đề tài số 20:** Phát hiện biên từ ảnh webcam bằng Laplacian và Sobel.

Ứng dụng được xây dựng bằng Python, OpenCV, NumPy và Tkinter, cho phép người dùng quan sát trực tiếp kết quả xử lý ảnh, điều chỉnh tham số và đánh giá hiệu năng của hai thuật toán.

## 2. Mục tiêu

- Đọc video trực tiếp từ webcam.
- Chuyển ảnh màu BGR sang ảnh xám.
- Tiền xử lý và giảm nhiễu bằng Gaussian Blur.
- Tự cài đặt thuật toán Sobel bằng kernel 3×3.
- Tự cài đặt thuật toán Laplacian bằng kernel 3×3.
- Phát hiện biên trực tiếp trên từng khung hình webcam.
- Điều chỉnh các tham số xử lý ảnh.
- So sánh kết quả Sobel và Laplacian.
- Đo FPS và thời gian xử lý trung bình.
- Lưu ảnh và dữ liệu thực nghiệm.
- Xây dựng giao diện ứng dụng bằng Tkinter.

## 3. Công nghệ sử dụng

| Công nghệ | Vai trò |
|---|---|
| Python 3.12 | Ngôn ngữ lập trình |
| OpenCV | Đọc webcam và xử lý hình ảnh |
| NumPy | Tính toán ma trận và cài đặt thuật toán |
| Tkinter | Xây dựng giao diện desktop |
| Pillow | Hiển thị ảnh OpenCV trên Tkinter |
| Matplotlib | Vẽ biểu đồ thực nghiệm |
| CSV | Lưu dữ liệu thống kê |
| unittest | Kiểm thử thuật toán |

## 4. Thuật toán sử dụng

### 4.1. Tiền xử lý ảnh

Quy trình tiền xử lý bao gồm:

1. Đọc khung hình từ webcam.
2. Chuyển ảnh BGR sang ảnh xám.
3. Lọc nhiễu Gaussian.
4. Chuyển ảnh đã xử lý đến hai thuật toán phát hiện biên.

### 4.2. Thuật toán Sobel

Sobel sử dụng đạo hàm bậc nhất để phát hiện những vùng có sự thay đổi mạnh về cường độ sáng.

Kernel Sobel X:

```text
-1   0   1
-2   0   2
-1   0   1
```

Kernel Sobel Y:

```text
-1  -2  -1
 0   0   0
 1   2   1
```

Độ lớn gradient:

G = sqrt(Gx² + Gy²)

Chương trình tự cài đặt phép xử lý kernel bằng NumPy.

### 4.3. Thuật toán Laplacian

Laplacian sử dụng đạo hàm bậc hai để phát hiện sự thay đổi cường độ sáng.

Kernel Laplacian 4 láng giềng:

```text
 0   1   0
 1  -4   1
 0   1   0
```

Kernel Laplacian 8 láng giềng:

```text
 1   1   1
 1  -8   1
 1   1   1
```

Ứng dụng cho phép thay đổi giữa hai kernel để khảo sát kết quả.

## 5. Chức năng ứng dụng

### 5.1. Điều khiển webcam

- Bắt đầu webcam.
- Tạm dừng xử lý.
- Tiếp tục xử lý.
- Dừng webcam.

### 5.2. Điều chỉnh tham số

- Gaussian kernel: 3×3, 5×5, 7×7, 9×9.
- Ngưỡng Sobel: 0–255.
- Ngưỡng Laplacian: 0–255.
- Laplacian kernel: 4 hoặc 8 láng giềng.
- Chuyển đổi giữa ảnh biên mức xám và ảnh nhị phân.

### 5.3. Hiển thị hình ảnh

Giao diện hiển thị đồng thời bốn khung hình:

1. Webcam gốc.
2. Ảnh sau lọc Gaussian.
3. Kết quả phát hiện biên Sobel.
4. Kết quả phát hiện biên Laplacian.

### 5.4. Thống kê hiệu năng

- FPS trung bình.
- Thời gian xử lý Sobel.
- Thời gian xử lý Laplacian.
- Tổng thời gian xử lý pipeline.
- Số khung hình đã xử lý.

### 5.5. Lưu kết quả

Chương trình có khả năng lưu:

- Ảnh webcam gốc.
- Ảnh xám.
- Ảnh sau lọc Gaussian.
- Ảnh Sobel.
- Ảnh Laplacian.
- Ảnh biên nhị phân Sobel.
- Ảnh biên nhị phân Laplacian.
- File CSV chứa thông số và hiệu năng.

## 6. Cấu trúc thư mục

```text
DeTai20_BienWebcam/
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── edges.py
│   ├── pipeline.py
│   ├── metrics.py
│   ├── exporter.py
│   └── gui.py
│
├── experiments/
│   ├── __init__.py
│   └── evaluate.py
│
├── dataset/           
│   ├── low_light/
│   ├── clear_edges/
│   ├── complex/
│   └── noise/
│
├── results/           
│
├── tests/
│   └── test_edges.py
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```
## 7. Hướng dẫn cài đặt

### 7.1. Yêu cầu

- Windows 10 hoặc Windows 11.
- Python 3.12.
- Webcam.
- Visual Studio Code hoặc IDE hỗ trợ Python.

### 7.2. Clone repository

```bash
git clone https://github.com/catdz1/Bien-tu-anh-webcam-voi-sobel-va-laplace.git

```

### 7.3. Tạo môi trường ảo

```powershell
python -m venv .venv
```

Kích hoạt trong PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 7.4. Cài đặt thư viện

```powershell
python -m pip install -r requirements.txt
```

## 8. Chạy chương trình

Từ thư mục gốc project, chạy:

```powershell
python main.py
```

Ứng dụng Tkinter sẽ xuất hiện.

Nhấn **Bắt đầu / Tiếp tục** để mở webcam.

## 9. Chạy kiểm thử

Chạy kiểm thử thuật toán:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

## 10. Chạy thực nghiệm

Chuẩn bị ảnh trong thư mục `dataset/`, sau đó chạy:

```powershell
python -m experiments.evaluate
```

Chương trình xử lý từng ảnh một lần cho mỗi cấu hình (Gaussian G = 3, 5, 9 và Laplacian 4 hoặc 8 láng giềng),
đo thời gian nhiều lượt, sau đó nhị phân hóa lại cùng bản đồ biên ở nhiều ngưỡng T = 10, 20, 30.
Nhờ vậy mỗi ảnh có 3 × 2 × 3 = 18 cấu hình mà không phải tính lại tích chập cho từng ngưỡng.

Lưu ý khi đọc kết quả:

- Sobel đã chia 4, còn Laplacian giữ nguyên thang đo, nên cùng một ngưỡng T không có nghĩa là cùng độ nhạy.
  Mật độ biên giữa hai thuật toán vì thế chỉ so sánh trực tiếp được khi xét theo T (biểu đồ 05).
- Độ ổn định nhiễu dùng IoU giữa biên ảnh sạch và biên ảnh có nhiễu Gaussian (σ = 15).
  IoU là N/A khi cả hai mặt nạ rỗng, hoặc khi biên ảnh sạch có ít hơn 0,05 % số điểm ảnh
  (`MIN_EDGE_FRACTION`). Cột `*_iou_valid_count` trong `summary.csv` cho biết số ảnh hợp lệ.
- Kết quả chỉ mang tính minh họa trên bộ ảnh nhỏ; không nên kết luận chung từ vài ảnh.

Kết quả được ghi vào `results/evaluation_<thời gian>/`:

| Tệp | Nội dung |
|---|---|
| `details.csv` | Số liệu từng ảnh × cấu hình × ngưỡng |
| `summary.csv` | Trung bình theo cấu hình × ngưỡng, kèm số ảnh hợp lệ (`*_iou_valid_count`) |
| `01_algorithm_times.png` | Thời gian xử lý Sobel và Laplacian (T cơ sở) |
| `02_edge_density.png` | Mật độ điểm biên (T cơ sở) |
| `03_noise_stability.png` | IoU trước nhiễu Gaussian (T cơ sở) |
| `04_pipeline_fps.png` | Thông lượng toàn bộ pipeline (không tính webcam và GUI) |
| `05_density_vs_threshold_G*.png` | Mật độ biên theo ngưỡng T, mỗi Gaussian một ảnh |
| `06_noise_iou_vs_threshold_G*.png` | IoU khi có nhiễu theo ngưỡng T, mỗi Gaussian một ảnh |
| `comparisons/<nhóm>/<ảnh>/G*_L*_T*.png` | Ảnh gốc, ảnh sau Gaussian, biên nhị phân Sobel và Laplacian |

Trong biểu đồ, "N/A" là không có cặp hợp lệ; cột gạch chéo hoặc marker rỗng là chỉ một phần ảnh có IoU hợp lệ (ghi `n=`).
Các tham số `THRESHOLDS`, `BASELINE_THRESHOLD`, `SAVE_IMAGE_THRESHOLDS` nằm ở đầu `experiments/evaluate.py`.

## 11. Hạn chế và hướng phát triển

- Chất lượng phát hiện biên phụ thuộc vào ánh sáng và mức nhiễu.
- Ngưỡng Sobel và Laplacian cần được điều chỉnh theo ảnh.
- Chưa có bộ nhãn biên chuẩn để đánh giá Precision, Recall và F1-score.
- Hiệu năng phụ thuộc vào cấu hình máy tính.
- Có thể nghiên cứu tối ưu tốc độ xử lý và bổ sung phương pháp phát hiện biên khác.
