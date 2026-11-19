# Điều phối các bước xử lý, tạo ảnh biên nhị phân và đo thời gian.


import time
import cv2
import numpy as np

from src.preprocessing import preprocess_frame
from src.edges import (
    detect_sobel,
    apply_kernel,
    LAPLACIAN_KERNEL
)


# Laplacian 8 mat can
LAPLACIAN_8 = np.array([
    [1, 1, 1],
    [1, -8, 1],
    [1, 1, 1]
], dtype=np.float32)


def apply_threshold(image, threshold):
    """
    Chuyen ban do bien muc xam thanh anh nhi phan.

    Pixel > threshold: 255
    Pixel <= threshold: 0
    """
    if not 0 <= threshold <= 255:
        raise ValueError(
            "Threshold phai nam trong khoang 0-255!"
        )

    _, binary = cv2.threshold(
        image,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    return binary


def laplacian_with_kernel(gray, kernel_type=4):
    
    if kernel_type == 4:
        kernel = LAPLACIAN_KERNEL

    elif kernel_type == 8:
        kernel = LAPLACIAN_8

    else:
        raise ValueError(
           
        )

    response = apply_kernel(gray, kernel)

    magnitude = np.abs(response)

    result = np.clip(
        magnitude,
        0,
        255
    ).astype(np.uint8)

    return result


def process_frame(
    frame,
    gaussian_size=5,
    sobel_threshold=40,
    laplacian_threshold=40,
    laplacian_kernel=4
):
    

    # 1. Tien xu ly
    t0 = time.perf_counter()

    gray, blurred = preprocess_frame(
        frame,
        gaussian_size
    )

    t1 = time.perf_counter()

    # 2. Sobel tu cai dat
    sobel = detect_sobel(blurred)

    t2 = time.perf_counter()

    # 3. Laplacian tu cai dat
    laplacian = laplacian_with_kernel(
        blurred,
        laplacian_kernel
    )

    t3 = time.perf_counter()

    # 4. Nguong hoa ket qua
    sobel_binary = apply_threshold(
        sobel,
        sobel_threshold
    )

    laplacian_binary = apply_threshold(
        laplacian,
        laplacian_threshold
    )

    t4 = time.perf_counter()

    # 5. Tinh thoi gian theo mili giay
    times = {
        "preprocess_ms": (t1 - t0) * 1000,
        "sobel_ms": (t2 - t1) * 1000,
        "laplacian_ms": (t3 - t2) * 1000,
        "threshold_ms": (t4 - t3) * 1000,
        "total_ms": (t4 - t0) * 1000
    }

    return {
        "gray": gray,
        "blurred": blurred,
        "sobel": sobel,
        "laplacian": laplacian,
        "sobel_binary": sobel_binary,
        "laplacian_binary": laplacian_binary,
        "times": times
    }
