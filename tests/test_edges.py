# python -m unittest discover -s tests -p "test_*.py" -v

import unittest

import cv2
import numpy as np

from src.edges import (
    detect_laplacian,
    LAPLACIAN_KERNEL_8,
    apply_kernel,
    SOBEL_X,
    SOBEL_Y,
    LAPLACIAN_KERNEL,
    detect_edges
)


class TestEdgeAlgorithms(unittest.TestCase):

    def setUp(self):
        # Tao anh xam mau co the lap lai
        rng = np.random.default_rng(42)

        self.image = rng.integers(
            0, 256,
            size=(64, 64),
            dtype=np.uint8
        )

    def test_sobel_x(self):
        custom = apply_kernel(
            self.image,
            SOBEL_X
        )

        reference = cv2.Sobel(
            self.image,
            cv2.CV_32F,
            1, 0,
            ksize=3,
            borderType=cv2.BORDER_REFLECT_101
        )

        np.testing.assert_allclose(
            custom, reference, atol=1e-5
        )

    def test_sobel_y(self):
        custom = apply_kernel(
            self.image,
            SOBEL_Y
        )

        reference = cv2.Sobel(
            self.image,
            cv2.CV_32F,
            0, 1,
            ksize=3,
            borderType=cv2.BORDER_REFLECT_101
        )

        np.testing.assert_allclose(
            custom, reference, atol=1e-5
        )

    def test_laplacian(self):
        custom = apply_kernel(
            self.image,
            LAPLACIAN_KERNEL
        )

        reference = cv2.Laplacian(
            self.image,
            cv2.CV_32F,
            ksize=1,
            borderType=cv2.BORDER_REFLECT_101
        )

        np.testing.assert_allclose(
            custom, reference, atol=1e-5
        )

    
    def test_laplacian_8_neighbors(self):
        # Ket qua thuat toan tu cai dat
        custom = detect_laplacian(
            self.image,
            kernel_type=8
        )

        # Ket qua tham chieu tu OpenCV
        response = cv2.filter2D(
            self.image,
            cv2.CV_32F,
            LAPLACIAN_KERNEL_8,
            borderType=cv2.BORDER_REFLECT_101
        )

        expected = np.clip(
            np.abs(response), 0, 255
        ).astype(np.uint8)

        np.testing.assert_array_equal(
            custom, expected
        )

    def test_invalid_laplacian_kernel(self):
        with self.assertRaises(ValueError):
            detect_laplacian(
                self.image,
                kernel_type=6
            )


    def test_output_shape_and_type(self):
        sobel, laplacian = detect_edges(
            self.image
        )

        self.assertEqual(
            sobel.shape,
            self.image.shape
        )

        self.assertEqual(
            laplacian.shape,
            self.image.shape
        )

        self.assertEqual(sobel.dtype, np.uint8)
        self.assertEqual(laplacian.dtype, np.uint8)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            apply_kernel(
                np.zeros((5, 5, 3), dtype=np.uint8),
                SOBEL_X
            )


if __name__ == "__main__":
    unittest.main()
