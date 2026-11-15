
# EDGE VISION – Webcam Edge Detection

## 1. Giới thiệu

**EDGE VISION** là ứng dụng phát hiện biên từ video webcam theo thời gian thực, sử dụng hai thuật toán xử lý ảnh truyền thống là **Sobel** và **Laplacian**.

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
DeTai20_WebcamEdges/
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

Lưu ý: Dataset và kết quả thực nghiệm được lưu cục bộ, không được đưa lên GitHub theo cấu hình `.gitignore` hiện tại.

## 7. Hướng dẫn cài đặt

### 7.1. Yêu cầu

- Windows 10 hoặc Windows 11.
- Python 3.12.
- Webcam.
- Visual Studio Code hoặc IDE hỗ trợ Python.

### 7.2. Clone repository

```bash
git clone https://github.com/catdz1/EdgeVision-Sobel-Laplacian.git
cd EdgeVision-Sobel-Laplacian
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

Chương trình sẽ xử lý ảnh, đo hiệu năng, đánh giá đặc điểm đường biên và xuất các kết quả:

- details.csv
- summary.csv
- Biểu đồ thời gian xử lý
- Biểu đồ mật độ biên
- Biểu đồ ổn định trước nhiễu
- Biểu đồ thông lượng pipeline

## 11. Trạng thái phát triển

- [x] Khởi tạo project.
- [x] Đọc webcam bằng OpenCV.
- [x] Chuyển ảnh xám và lọc Gaussian.
- [x] Tự cài đặt Sobel và Laplacian.
- [x] Điều chỉnh tham số xử lý.
- [x] Đo FPS và thời gian xử lý.
- [x] Hoàn thiện giao diện Tkinter.
- [x] Lưu ảnh và dữ liệu CSV.
- [x] Chạy thử chương trình đánh giá.
- [ ] Hoàn thiện bộ dữ liệu chính thức.
- [ ] Đánh giá đầy đủ trên nhiều điều kiện ánh sáng.
- [ ] Hoàn thiện báo cáo bài tập lớn.

## 12. Hạn chế và hướng phát triển

- Chất lượng phát hiện biên phụ thuộc vào ánh sáng và mức nhiễu.
- Ngưỡng Sobel và Laplacian cần được điều chỉnh theo ảnh.
- Chưa có bộ nhãn biên chuẩn để đánh giá Precision, Recall và F1-score.
- Hiệu năng phụ thuộc vào cấu hình máy tính.
- Có thể nghiên cứu tối ưu tốc độ xử lý và bổ sung phương pháp phát hiện biên khác.

## 13. Tác giả

Đề tài: Phát hiện biên từ ảnh webcam bằng Laplacian và Sobel.

Học phần: Xử lý ảnh và Thị giác máy tính.

Trường: Đại học Công nghệ Đông Á.

Thông tin sinh viên/nhóm: Bổ sung khi hoàn thiện báo cáo.
