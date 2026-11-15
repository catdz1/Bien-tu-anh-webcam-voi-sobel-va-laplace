# Lưu ảnh và dữ liệu thực nghiệm.


import csv
from datetime import datetime
from pathlib import Path

import cv2


# Xac dinh thu muc goc cua project
BASE_DIR = Path(__file__).resolve().parent.parent
RESULT_DIR = BASE_DIR / "results"


def save_experiment(frame, result, stats, settings):
    """
    Luu 7 anh va thong so thuc nghiem vao CSV.

    Parameters:
        frame: Anh webcam goc.
        result: Ket qua cua process_frame().
        stats: Ket qua cua PerformanceMonitor.
        settings: Dictionary tham so xu ly.

    Returns:
        Duong dan thu muc luu anh.
    """

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    folder = RESULT_DIR / timestamp
    folder.mkdir(parents=True, exist_ok=True)

    images = {
        "original.png": frame,
        "gray.png": result["gray"],
        "gaussian.png": result["blurred"],
        "sobel.png": result["sobel"],
        "laplacian.png": result["laplacian"],
        "sobel_binary.png": result["sobel_binary"],
        "laplacian_binary.png":
            result["laplacian_binary"]
    }

    # Luu anh PNG
    for filename, image in images.items():
        path = folder / filename

        success = cv2.imwrite(
            str(path),
            image
        )

        if not success:
            raise OSError(
                f"Khong luu duoc anh: {path}"
            )

    # Luu thong tin thuc nghiem
    csv_path = RESULT_DIR / "experiments.csv"

    fields = [
        "timestamp",
        "width",
        "height",
        "gaussian_kernel",
        "sobel_threshold",
        "laplacian_threshold",
        "laplacian_kernel",
        "sample_frames",
        "avg_fps",
        "avg_preprocess_ms",
        "avg_sobel_ms",
        "avg_laplacian_ms",
        "avg_threshold_ms",
        "avg_total_ms"
    ]

    row = {
        "timestamp": timestamp,
        "width": frame.shape[1],
        "height": frame.shape[0],
        "gaussian_kernel": settings["gaussian_size"],
        "sobel_threshold": settings["sobel_threshold"],
        "laplacian_threshold": settings[
            "laplacian_threshold"
        ],
        "laplacian_kernel": settings[
            "laplacian_kernel"
        ],
        "sample_frames": stats["frames"],
        "avg_fps": round(stats["fps"], 3),
        "avg_preprocess_ms": round(
            stats["preprocess_ms"], 3
        ),
        "avg_sobel_ms": round(
            stats["sobel_ms"], 3
        ),
        "avg_laplacian_ms": round(
            stats["laplacian_ms"], 3
        ),
        "avg_threshold_ms": round(
            stats["threshold_ms"], 3
        ),
        "avg_total_ms": round(
            stats["total_ms"], 3
        )
    }

    # Ghi tiep vao CSV da co
    write_header = (
        not csv_path.exists()
        or csv_path.stat().st_size == 0
    )

    with csv_path.open(
        "a",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        if write_header:
            writer.writeheader()

        writer.writerow(row)

    return folder
