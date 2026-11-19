# Thống kê FPS và thời gian xử lý.


from collections import deque
from statistics import mean


class PerformanceMonitor:
    
    def __init__(self, max_samples=120):
        self.max_samples = max_samples
        self.reset()

    def reset(self):
        self.preprocess_times = deque(
            maxlen=self.max_samples
        )

        self.sobel_times = deque(
            maxlen=self.max_samples
        )

        self.laplacian_times = deque(
            maxlen=self.max_samples
        )

        self.threshold_times = deque(
            maxlen=self.max_samples
        )

        self.total_times = deque(
            maxlen=self.max_samples
        )

        self.fps_values = deque(
            maxlen=self.max_samples
        )

        self.frame_count = 0

    def update(self, times, fps=None):
        self.preprocess_times.append(
            times["preprocess_ms"]
        )

        self.sobel_times.append(
            times["sobel_ms"]
        )

        self.laplacian_times.append(
            times["laplacian_ms"]
        )

        self.threshold_times.append(
            times["threshold_ms"]
        )

        self.total_times.append(
            times["total_ms"]
        )

        if fps is not None and fps > 0:
            self.fps_values.append(fps)

        self.frame_count += 1

    @staticmethod
    def average(values):
        return mean(values) if values else 0.0

    def get_stats(self):
        return {
            "frames": self.frame_count,
            "preprocess_ms": self.average(
                self.preprocess_times
            ),
            "sobel_ms": self.average(
                self.sobel_times
            ),
            "laplacian_ms": self.average(
                self.laplacian_times
            ),
            "threshold_ms": self.average(
                self.threshold_times
            ),
            "total_ms": self.average(
                self.total_times
            ),
            "fps": (
                len(self.fps_values)
                / sum(1.0 / f for f in self.fps_values)
                if self.fps_values
                else 0.0
            )
        }
