
import cv2


def convert_to_grayscale(frame):
    """
    Chuyen anh mau BGR sang anh xam.

    Parameters:
        frame: Anh mau tu webcam.

    Returns:
        Anh xam 1 kenh.
    """
    if frame is None or frame.size == 0:
        raise ValueError("Anh dau vao khong hop le!")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return gray


def apply_gaussian_blur(gray, kernel_size=5):
    """
    Loc nhieu Gaussian tren anh xam.

    Parameters:
        gray: Anh xam.
        kernel_size: Kich thuoc kernel, phai la so le duong.

    Returns:
        Anh da loc Gaussian.
    """
    if gray is None or gray.size == 0:
        raise ValueError("Anh dau vao khong hop le!")

    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(
            "Kernel size phai la so le duong!"
        )

    blurred = cv2.GaussianBlur(
        gray,
        (kernel_size, kernel_size),
        0
    )

    return blurred


def preprocess_frame(frame, kernel_size=5):
    """
    Thuc hien tien xu ly anh:
    1. Chuyen anh xam.
    2. Loc Gaussian.
    """
    gray = convert_to_grayscale(frame)

    blurred = apply_gaussian_blur(
        gray,
        kernel_size
    )

    return gray, blurred
