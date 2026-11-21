# Thực nghiệm tự động
# python -m experiments.evaluate

import csv
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Chọn font hỗ trợ tiếng Việt nếu có (Windows thường có Segoe UI/Arial).
from matplotlib import font_manager
available_fonts = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = next(
    (f for f in ("Segoe UI", "Arial", "DejaVu Sans") if f in available_fonts),
    "DejaVu Sans",
)
plt.rcParams["axes.unicode_minus"] = False

from src.pipeline import process_frame, apply_threshold

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
OUT = ROOT / "results" / ("evaluation_" + datetime.now().strftime("%Y%m%d_%H%M%S"))

GAUSSIAN_SIZES = (3, 5, 9)
LAPLACIAN_KERNELS = (4, 8)
THRESHOLDS = (10, 20, 30)
BASELINE_THRESHOLD = 20  # Mốc so sánh cơ sở thuộc dải ngưỡng được khảo sát.
SAVE_IMAGE_THRESHOLDS = THRESHOLDS  # Lưu ảnh so sánh tại cả 3 ngưỡng T=10, 20, 30.
REPEATS = 20
WARMUP = 3
SIGMA = 15.0
# IoU của mặt nạ chỉ vài điểm ảnh không có ý nghĩa: coi là N/A khi biên ảnh sạch
# có ít hơn 0,05 % tổng số điểm ảnh (khoảng 150 điểm ảnh với ảnh 640x480).
MIN_EDGE_FRACTION = 0.0005

CONFIGS = [(g, l) for g in GAUSSIAN_SIZES for l in LAPLACIAN_KERNELS]
FIELDS = ["preprocess_ms", "sobel_ms", "laplacian_ms", "threshold_ms", "total_ms"]
NUMERIC_FIELDS = [
    "sobel_ms", "laplacian_ms", "total_ms", "total_std_ms", "pipeline_fps",
    "sobel_density_pct", "laplacian_density_pct",
    "sobel_components", "laplacian_components",
    "sobel_largest_fraction", "laplacian_largest_fraction",
    "sobel_noise_iou", "laplacian_noise_iou",
    "sobel_empty_clean", "laplacian_empty_clean",
    "sobel_empty_noisy", "laplacian_empty_noisy",
]


def edge_statistics(mask):
    binary = (mask > 0).astype(np.uint8)
    count = int(np.count_nonzero(binary))
    if count == 0:
        return 0.0, 0, 0.0
    number, labels, stats, centers = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    return 100 * count / binary.size, number - 1, float(max(areas) / count)


def mask_iou(a, b, min_fraction=MIN_EDGE_FRACTION):
    """IoU của hai mặt nạ biên.

    Trả về NaN (không xác định) khi:
    - cả hai mặt nạ rỗng, hoặc
    - biên ảnh sạch quá ít (< min_fraction tổng số điểm ảnh), vì IoU của
      vài điểm ảnh không phản ánh độ ổn định.
    """
    if a.shape != b.shape:
        raise ValueError("Hai mat na phai cung kich thuoc!")
    aa = a > 0
    bb = b > 0
    union = np.count_nonzero(aa | bb)
    if union == 0 or np.count_nonzero(aa) < min_fraction * aa.size:
        return float("nan")
    return float(np.count_nonzero(aa & bb) / union)


def valid_mean(values):
    """Bỏ qua NaN (IoU của hai mặt nạ rỗng)."""
    good = [float(v) for v in values if v is not None and np.isfinite(v)]
    return mean(good) if good else float("nan")


def add_noise(image, seed):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, SIGMA, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def read_images():
    supported = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = sorted(p for p in DATASET.rglob("*") if p.is_file() and p.suffix.lower() in supported)
    if not paths:
        raise RuntimeError("Chua co anh trong dataset/. Hay them anh truoc khi chay.")
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print("Bo qua anh loi:", path)
            continue
        if image.shape[:2] != (480, 640):
            image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
        yield path, image


def run_one(image, settings):
    # Chỉ đo thuật toán một lần cho mỗi G/L/ảnh; thời gian gần như độc lập ngưỡng.
    for _ in range(WARMUP):
        process_frame(image, **settings)
    timings = {field: [] for field in FIELDS}
    started = time.perf_counter()
    result = None
    for _ in range(REPEATS):
        result = process_frame(image, **settings)
        for field in FIELDS:
            timings[field].append(result["times"][field])
    elapsed = time.perf_counter() - started
    averages = {field: mean(values) for field, values in timings.items()}
    averages["pipeline_fps"] = REPEATS / elapsed if elapsed > 0 else 0.0
    averages["total_std_ms"] = stdev(timings["total_ms"]) if REPEATS > 1 else 0.0
    return result, averages


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_groups(rows, field_a, field_b, label_a, label_b, title, filename, ylabel,
                count_a=None, count_b=None):
    """Biểu đồ cột đôi. Cột NaN ghi "N/A"; cột thiếu mẫu hợp lệ có nét gạch chéo."""
    labels = [f"G{r['gaussian']}-L{r['laplacian_kernel']}" for r in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    partial = False
    for offset, field, label, count in ((-0.19, field_a, label_a, count_a),
                                        (0.19, field_b, label_b, count_b)):
        values = [r[field] for r in rows]
        bars = ax.bar(x + offset, [0 if not np.isfinite(v) else v for v in values],
                      0.38, label=label)
        for bar, row, v in zip(bars, rows, values):
            if not np.isfinite(v):
                ax.annotate("N/A", (bar.get_x() + bar.get_width() / 2, 0),
                            ha="center", va="bottom", fontsize=9, color="gray")
            elif count and row[count] < row["images"]:
                bar.set_hatch("//")
                bar.set_edgecolor("white")
                partial = True
                ax.annotate(f"n={row[count]}", (bar.get_x() + bar.get_width() / 2, v),
                            ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_xlabel("Cấu hình (G: Gaussian, L: Laplacian)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.18)  # chừa chỗ cho chú giải, không đè lên cột
    ax.legend(loc="upper left", ncol=2)
    if partial:
        fig.text(0.01, 0.01, "Cột gạch chéo: chỉ một phần ảnh có IoU hợp lệ (n = số ảnh hợp lệ); "
                 "N/A: không có cặp hợp lệ.", fontsize=8, color="gray")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / filename, dpi=160)
    plt.close(fig)


def plot_fps(rows):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([f"G{r['gaussian']}-L{r['laplacian_kernel']}" for r in rows],
           [r["pipeline_fps"] for r in rows])
    ax.set_title("Thông lượng xử lý toàn bộ quy trình (ngoại tuyến)")
    ax.set_xlabel("Cấu hình (G: Gaussian, L: Laplacian)")
    ax.set_ylabel("Số khung hình/giây (không tính webcam và GUI)")
    fig.tight_layout()
    fig.savefig(OUT / "04_pipeline_fps.png", dpi=160)
    plt.close(fig)


def plot_threshold_curves(rows, field_sobel, field_lap, ylabel, title, filename_prefix,
                          count_sobel=None, count_lap=None):
    """3 đường (Sobel, Laplacian 4, Laplacian 8) theo T cho từng Gaussian.

    Điểm NaN ghi "N/A" ở đáy biểu đồ; điểm thiếu mẫu hợp lệ vẽ marker rỗng.
    """
    by_key = {(r["gaussian"], r["laplacian_kernel"], r["threshold"]): r for r in rows}
    # Sobel không phụ thuộc việc chọn Laplacian 4 hay 8; dùng L4 một lần.
    series = (
        ("Sobel", "o", 4, field_sobel, count_sobel),
        ("Laplacian 4 láng giềng", "s", 4, field_lap, count_lap),
        ("Laplacian 8 láng giềng", "^", 8, field_lap, count_lap),
    )
    for g in GAUSSIAN_SIZES:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        partial = False
        for label, marker, kernel, field, count in series:
            records = [by_key[(g, kernel, t)] for t in THRESHOLDS]
            values = [r[field] for r in records]
            line, = ax.plot(THRESHOLDS, values, marker=marker, label=label)
            for t, r, v in zip(THRESHOLDS, records, values):
                if not np.isfinite(v):
                    ax.annotate("N/A", (t, 0), xycoords=("data", "axes fraction"),
                                xytext=(0, 4), textcoords="offset points", ha="center",
                                fontsize=8, color=line.get_color())
                elif count and r[count] < r["images"]:
                    partial = True
                    ax.plot([t], [v], marker=marker, markersize=10,
                            markerfacecolor="white", color=line.get_color())
                    ax.annotate(f"n={r[count]}", (t, v), xytext=(0, 8),
                                textcoords="offset points", ha="center", fontsize=8)
        ax.set_xlabel("Ngưỡng phát hiện biên T")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Gaussian {g}×{g} – {title}")
        ax.set_xticks(THRESHOLDS)
        ax.grid(alpha=0.2)
        ax.legend()
        if partial:
            fig.text(0.01, 0.01, "Marker rỗng: chỉ một phần ảnh có IoU hợp lệ (n = số ảnh hợp lệ). "
                     "N/A: không có cặp hợp lệ.", fontsize=8, color="gray")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(OUT / f"{filename_prefix}_G{g}.png", dpi=160)
        plt.close(fig)


def save_comparison(path, frame, result, sobel_mask, lap_mask, g, l, t):
    """Lưu ảnh với đúng mã ngưỡng, tránh ghi đè các cấu hình khác nhau."""
    def bgr(img):
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img

    top = np.hstack([frame, bgr(result["blurred"])])
    bottom = np.hstack([bgr(sobel_mask), bgr(lap_mask)])
    relative = path.relative_to(DATASET)
    folder = OUT / "comparisons" / relative.parent / relative.stem
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / f"G{g}_L{l}_T{t}.png"), np.vstack([top, bottom]))


def main():
    if not THRESHOLDS or not all(0 <= t <= 255 for t in THRESHOLDS):
        raise ValueError("THRESHOLDS phai la danh sach so tu 0 den 255.")
    images = list(read_images())
    if not images:
        raise RuntimeError("Khong doc duoc anh hop le trong dataset/")
    OUT.mkdir(parents=True, exist_ok=True)
    details = []

    # Cùng một ảnh nhiễu được dùng cho mọi G/L/T để so sánh công bằng.
    for image_id, (path, frame) in enumerate(images):
        relative = path.relative_to(DATASET)
        category = relative.parts[0] if len(relative.parts) > 1 else "other"
        noisy = add_noise(frame, seed=2026 + image_id)

        for g, l in CONFIGS:
            # T chỉ dùng để đo thời gian pipeline; ảnh mức xám Sobel/Laplacian
            # không phụ thuộc T, nên có thể ngưỡng hóa lại mà không tính lại kernel.
            settings = {"gaussian_size": g, "sobel_threshold": BASELINE_THRESHOLD,
                        "laplacian_threshold": BASELINE_THRESHOLD, "laplacian_kernel": l}
            result, avg = run_one(frame, settings)
            noisy_result = process_frame(noisy, **settings)

            for t in THRESHOLDS:
                sb = apply_threshold(result["sobel"], t)
                lb = apply_threshold(result["laplacian"], t)
                sb_noisy = apply_threshold(noisy_result["sobel"], t)
                lb_noisy = apply_threshold(noisy_result["laplacian"], t)

                sd, sc, sl = edge_statistics(sb)
                ld, lc, ll = edge_statistics(lb)
                row = {
                    "image": str(relative), "category": category,
                    "config": f"G{g}-L{l}-T{t}",
                    "gaussian": g, "laplacian_kernel": l, "threshold": t,
                    "sobel_threshold": t, "laplacian_threshold": t,
                    **avg,
                    "sobel_density_pct": sd, "laplacian_density_pct": ld,
                    "sobel_components": sc, "laplacian_components": lc,
                    "sobel_largest_fraction": sl, "laplacian_largest_fraction": ll,
                    "sobel_noise_iou": mask_iou(sb, sb_noisy),
                    "laplacian_noise_iou": mask_iou(lb, lb_noisy),
                    "sobel_empty_clean": int(not np.any(sb)),
                    "laplacian_empty_clean": int(not np.any(lb)),
                    "sobel_empty_noisy": int(not np.any(sb_noisy)),
                    "laplacian_empty_noisy": int(not np.any(lb_noisy)),
                }
                details.append(row)
                if t in SAVE_IMAGE_THRESHOLDS:
                    save_comparison(path, frame, result, sb, lb, g, l, t)

            print(f"{relative} | G={g} L={l} | T={list(THRESHOLDS)} | "
                  f"Sobel={avg['sobel_ms']:.2f} ms, Lap={avg['laplacian_ms']:.2f} ms")

    write_csv(OUT / "details.csv", details)

    groups = defaultdict(list)
    for row in details:
        groups[(row["gaussian"], row["laplacian_kernel"], row["threshold"])].append(row)

    summaries = []
    for g, l in CONFIGS:
        for t in THRESHOLDS:
            group = groups[(g, l, t)]
            summaries.append({
                "config": f"G{g}-L{l}-T{t}", "gaussian": g,
                "laplacian_kernel": l, "threshold": t, "images": len(group),
                **{field: valid_mean(r[field] for r in group) for field in NUMERIC_FIELDS},
                "sobel_iou_valid_count": sum(np.isfinite(r["sobel_noise_iou"]) for r in group),
                "laplacian_iou_valid_count": sum(np.isfinite(r["laplacian_noise_iou"]) for r in group),
                "sobel_both_empty_count": sum(
                    r["sobel_empty_clean"] and r["sobel_empty_noisy"] for r in group
                ),
                "laplacian_both_empty_count": sum(
                    r["laplacian_empty_clean"] and r["laplacian_empty_noisy"] for r in group
                ),
            })
    write_csv(OUT / "summary.csv", summaries)

    # Bốn biểu đồ cũ chỉ lấy cùng T cơ sở, tránh 30 cột chen chúc.
    baseline = BASELINE_THRESHOLD if BASELINE_THRESHOLD in THRESHOLDS else THRESHOLDS[0]
    baseline_rows = [r for r in summaries if r["threshold"] == baseline]
    plot_groups(baseline_rows, "sobel_ms", "laplacian_ms", "Sobel", "Laplacian",
                f"So sánh thời gian xử lý Sobel và Laplacian (T={baseline})",
                "01_algorithm_times.png", "Thời gian xử lý (ms/ảnh)")
    plot_groups(baseline_rows, "sobel_density_pct", "laplacian_density_pct", "Sobel", "Laplacian",
                f"So sánh mật độ điểm biên (T={baseline})", "02_edge_density.png", "Mật độ điểm biên (%)")
    plot_groups(baseline_rows, "sobel_noise_iou", "laplacian_noise_iou", "Sobel", "Laplacian",
                f"Độ ổn định trước nhiễu Gaussian (σ={SIGMA}, T={baseline})",
                "03_noise_stability.png", "IoU (chỉ tính các cặp hợp lệ)",
                count_a="sobel_iou_valid_count", count_b="laplacian_iou_valid_count")
    plot_fps(baseline_rows)

    plot_threshold_curves(summaries, "sobel_density_pct", "laplacian_density_pct",
                          "Mật độ điểm biên (%)", "Mật độ điểm biên theo ngưỡng T",
                          "05_density_vs_threshold")
    plot_threshold_curves(summaries, "sobel_noise_iou", "laplacian_noise_iou",
                          "IoU (bỏ qua cặp mặt nạ rỗng)", "Độ ổn định trước nhiễu theo ngưỡng T",
                          "06_noise_iou_vs_threshold",
                          count_sobel="sobel_iou_valid_count",
                          count_lap="laplacian_iou_valid_count")

    print("\nSaved results to:", OUT)
    print(f"Images: {len(images)} | configurations/image: {len(CONFIGS) * len(THRESHOLDS)}")
    print("Files: details.csv, summary.csv, 4 baseline plots, 6 threshold plots, comparisons/")
    print("NOTE: timing/FPS reused across T for same image, G and L; thresholded masks recomputed separately.")


if __name__ == "__main__":
    main()
