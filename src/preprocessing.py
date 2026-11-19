# Chuyen anh xam va loc gaussian

import cv2

def convert_to_grayscale(frame):
   
    if frame is None or frame.size == 0:
        raise ValueError("Anh dau vao khong hop le!")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return gray


def apply_gaussian_blur(gray, kernel_size=5):
  
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
   
    gray = convert_to_grayscale(frame)

    blurred = apply_gaussian_blur(
        gray,
        kernel_size
    )

    return gray, blurred
