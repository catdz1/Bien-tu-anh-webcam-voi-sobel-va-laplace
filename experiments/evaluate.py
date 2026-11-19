# python -m experiments.evaluate
# chạy kiểm thử và đánh giá tự động

import csv
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.pipeline import process_frame

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / 'dataset'
OUT = ROOT / 'results' / ('evaluation_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
REPEATS = 20
WARMUP = 3
SIGMA = 15.0
CONFIGS = [(g, l) for g in (3, 5, 9) for l in (4, 8)]
FIELDS = ['preprocess_ms', 'sobel_ms', 'laplacian_ms', 'threshold_ms', 'total_ms']


def edge_statistics(mask):
    binary = (mask > 0).astype(np.uint8)
    count = int(np.count_nonzero(binary))
    if count == 0:
        return 0.0, 0, 0.0
    number, labels, stats, centers = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    return 100 * count / binary.size, number - 1, float(max(areas) / count)


def mask_iou(a, b):
    aa = a > 0
    bb = b > 0
    union = np.count_nonzero(aa | bb)
    return float(np.count_nonzero(aa & bb) / union) if union else 1.0


def add_noise(image, seed):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, SIGMA, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def read_images():
    supported = {'.jpg', '.jpeg', '.png', '.bmp'}
    paths = sorted(p for p in DATASET.rglob('*') if p.is_file() and p.suffix.lower() in supported)
    if not paths:
        raise RuntimeError('Chua co anh trong dataset/. Hay them anh truoc khi chay.')
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print('Bo qua anh loi:', path)
            continue
        if image.shape[:2] != (480, 640):
            image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
        yield path, image


def run_one(image, settings):
    for _ in range(WARMUP):
        process_frame(image, **settings)
    timings = {field: [] for field in FIELDS}
    started = time.perf_counter()
    result = None
    for _ in range(REPEATS):
        result = process_frame(image, **settings)
        for field in FIELDS:
            timings[field].append(result['times'][field])
    elapsed = time.perf_counter() - started
    averages = {field: mean(values) for field, values in timings.items()}
    averages['pipeline_fps'] = REPEATS / elapsed if elapsed > 0 else 0.0
    averages['total_std_ms'] = stdev(timings['total_ms']) if REPEATS > 1 else 0.0
    return result, averages


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_groups(rows, field_a, field_b, label_a, label_b, title, filename, ylabel):
    labels = [r['config'] for r in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.19, [r[field_a] for r in rows], 0.38, label=label_a)
    ax.bar(x + 0.19, [r[field_b] for r in rows], 0.38, label=label_b)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=160)
    plt.close(fig)


def plot_fps(rows):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([r['config'] for r in rows], [r['pipeline_fps'] for r in rows])
    ax.set_title('Offline full-pipeline throughput')
    ax.set_ylabel('Processed frames / second (no webcam or GUI)')
    fig.tight_layout()
    fig.savefig(OUT / '04_pipeline_fps.png', dpi=160)
    plt.close(fig)


def main():
    images = list(read_images())
    if not images:
        raise RuntimeError('Khong doc duoc anh hop le trong dataset/')
    OUT.mkdir(parents=True, exist_ok=True)
    details = []
    for image_id, (path, frame) in enumerate(images):
        category = path.relative_to(DATASET).parts[0] if len(path.relative_to(DATASET).parts) > 1 else 'other'
        noisy = add_noise(frame, seed=2026 + image_id)
        for g, l in CONFIGS:
            settings = {'gaussian_size': g, 'sobel_threshold': 40,
                        'laplacian_threshold': 40, 'laplacian_kernel': l}
            result, avg = run_one(frame, settings)
            noisy_result = process_frame(noisy, **settings)
            sd, sc, sl = edge_statistics(result['sobel_binary'])
            ld, lc, ll = edge_statistics(result['laplacian_binary'])
            row = {
                'image': str(path.relative_to(DATASET)), 'category': category,
                'config': f'G{g}-L{l}', 'gaussian': g, 'laplacian_kernel': l,
                'sobel_threshold': 40, 'laplacian_threshold': 40,
                **avg,
                'sobel_density_pct': sd, 'laplacian_density_pct': ld,
                'sobel_components': sc, 'laplacian_components': lc,
                'sobel_largest_fraction': sl, 'laplacian_largest_fraction': ll,
                'sobel_noise_iou': mask_iou(result['sobel_binary'], noisy_result['sobel_binary']),
                'laplacian_noise_iou': mask_iou(result['laplacian_binary'], noisy_result['laplacian_binary'])
            }
            details.append(row)
            # Save one 2x2 contact sheet for each input/configuration.
            def bgr(img):
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
            top = np.hstack([frame, bgr(result['blurred'])])
            bottom = np.hstack([bgr(result['sobel_binary']), bgr(result['laplacian_binary'])])
            folder = OUT / 'comparisons' / category
            folder.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(folder / f'{path.stem}_G{g}_L{l}.png'), np.vstack([top, bottom]))
            print(f"{path.name} | G={g}, L={l} | Sobel={avg['sobel_ms']:.2f} ms | Lap={avg['laplacian_ms']:.2f} ms")

    write_csv(OUT / 'details.csv', details)
    groups = defaultdict(list)
    for row in details:
        groups[row['config']].append(row)
    numeric = ['sobel_ms', 'laplacian_ms', 'total_ms', 'total_std_ms',
               'pipeline_fps', 'sobel_density_pct', 'laplacian_density_pct',
               'sobel_components', 'laplacian_components',
               'sobel_largest_fraction', 'laplacian_largest_fraction',
               'sobel_noise_iou', 'laplacian_noise_iou']
    summaries = []
    for g, l in CONFIGS:
        name = f'G{g}-L{l}'
        group = groups[name]
        summaries.append({'config': name, 'images': len(group),
                          **{field: mean(r[field] for r in group) for field in numeric}})
    write_csv(OUT / 'summary.csv', summaries)

    plot_groups(summaries, 'sobel_ms', 'laplacian_ms', 'Sobel', 'Laplacian',
                'Algorithm runtime (custom NumPy)', '01_algorithm_times.png', 'Milliseconds / image')
    plot_groups(summaries, 'sobel_density_pct', 'laplacian_density_pct', 'Sobel', 'Laplacian',
                'Edge-pixel density', '02_edge_density.png', 'White pixels (%)')
    plot_groups(summaries, 'sobel_noise_iou', 'laplacian_noise_iou', 'Sobel', 'Laplacian',
                'Stability under Gaussian noise (sigma=15)', '03_noise_stability.png', 'IoU (higher = more stable)')
    plot_fps(summaries)
    print('\nSaved results to:', OUT)
    print('Files: details.csv, summary.csv, 4 plots, comparisons/')


if __name__ == '__main__':
    main()
