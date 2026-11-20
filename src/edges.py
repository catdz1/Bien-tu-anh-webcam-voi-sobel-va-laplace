# Chứa thuật toán Sobel, Laplacian.

import numpy as np

# Kernel Sobel theo truc X
SOBEL_X = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)


# Kernel Sobel theo truc Y
SOBEL_Y = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
], dtype=np.float32)


# Kernel Laplacian 4 mat na
LAPLACIAN_KERNEL = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
], dtype=np.float32)


# Kernel Laplacian 8 láng giềng
LAPLACIAN_KERNEL_8 = np.array([
    [1, 1, 1],
    [1, -8, 1],
    [1, 1, 1]
], dtype=np.float32)



def apply_kernel(image, kernel):
    

    if image is None or image.ndim != 2:
        raise ValueError("Dau vao phai la anh xam 2D!")

    kernel = np.asarray(kernel, dtype=np.float32)

    if kernel.ndim != 2:
        raise ValueError("Kernel phai la ma tran 2D!")

    kh, kw = kernel.shape

    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("Kich thuoc kernel phai la so le!")

    # Chuyen anh sang float de tranh tran so
    image_float = image.astype(np.float32)

    # Padding kieu reflect, tuong duong BORDER_REFLECT_101
    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(
        image_float,
        ((pad_h, pad_h), (pad_w, pad_w)),
        mode="reflect"
    )

    height, width = image.shape

    result = np.zeros(
        (height, width),
        dtype=np.float32
    )

    # Tinh tong tich tung phan tu kernel va anh
    # Xu ly ca ma tran de tang toc so voi lap tung pixel
    for i in range(kh):
        for j in range(kw):
            result += (
                kernel[i, j]
                * padded[i:i + height, j:j + width]
            )

    return result


def detect_sobel(gray):
   

    gx = apply_kernel(gray, SOBEL_X)
    gy = apply_kernel(gray, SOBEL_Y)

    # Do lon gradient
    magnitude = np.sqrt(gx * gx + gy * gy)

    # Chuan hoa ve khoang 0-255
    # Su dung cung thang do cho moi khung hinh
    magnitude = magnitude / (4.0 * 255.0) * 255.0

    edges = np.clip(
        magnitude, 0, 255
    ).astype(np.uint8)

    return edges



def detect_laplacian(gray, kernel_type=4):
    

    # Lua chon kernel
    if kernel_type == 4:
        kernel = LAPLACIAN_KERNEL

    elif kernel_type == 8:
        kernel = LAPLACIAN_KERNEL_8

    else:
        raise ValueError(
            "Laplacian kernel phai la 4 hoac 8!"
        )

    # Ap dung phep tich chap kernel
    response = apply_kernel(gray, kernel)

    # Lay gia tri tuyet doi cua dao ham bac hai
    magnitude = np.abs(response)

    # Gioi han cuong do ve 0-255
    edges = np.clip(
        magnitude, 0, 255
    ).astype(np.uint8)

    return edges


def detect_edges(gray):
   
    sobel_edges = detect_sobel(gray)
    laplacian_edges = detect_laplacian(gray)

    return sobel_edges, laplacian_edges
