
import unittest

import cv2
import numpy as np

from src.edges import (
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
