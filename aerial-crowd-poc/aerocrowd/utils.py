"""Utility helpers for tiling, overlays, and simple statistics."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class Tile:
    image: np.ndarray
    origin: Tuple[int, int]  # (x, y)


def _tile_positions(length: int, tile_size: int, overlap: int) -> List[int]:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if length <= tile_size:
        return [0]
    stride = max(1, tile_size - overlap)
    positions = list(range(0, max(1, length - tile_size + 1), stride))
    if not positions:
        positions = [0]
    last = positions[-1]
    if last + tile_size < length:
        positions.append(max(0, length - tile_size))
    # Deduplicate and ensure sorted
    result = sorted(dict.fromkeys(positions))
    return result


def tile_image(image: np.ndarray, tile_size: int, overlap: int) -> List[Tile]:
    """Split an image into overlapping tiles."""
    if tile_size <= 0:
        raise ValueError("tile_size must be > 0")
    height, width = image.shape[:2]
    if tile_size >= max(height, width):
        return [Tile(image=image, origin=(0, 0))]

    xs = _tile_positions(width, tile_size, overlap)
    ys = _tile_positions(height, tile_size, overlap)
    tiles: List[Tile] = []
    for y in ys:
        for x in xs:
            x_end = min(x + tile_size, width)
            y_end = min(y + tile_size, height)
            tile = image[y:y_end, x:x_end].copy()
            tiles.append(Tile(image=tile, origin=(x, y)))
    return tiles


def stitch_density_map(
    tiles: Sequence[Tuple[np.ndarray, Tuple[int, int]]],
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """Stitch a list of density tiles back into a single map via averaging."""
    height, width = image_shape
    density = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)
    for tile, (x, y) in tiles:
        h, w = tile.shape[:2]
        density[y : y + h, x : x + w] += tile.astype(np.float32)
        counts[y : y + h, x : x + w] += 1.0
    mask = counts > 0
    density[mask] /= counts[mask]
    return density


def offset_bboxes(
    bboxes: Iterable[Tuple[float, float, float, float, float]],
    dx: float,
    dy: float,
) -> List[Tuple[float, float, float, float, float]]:
    result = []
    for x1, y1, x2, y2, score in bboxes:
        result.append((x1 + dx, y1 + dy, x2 + dx, y2 + dy, score))
    return result


def draw_bboxes(
    image: np.ndarray,
    bboxes: Sequence[Tuple[float, float, float, float, float]],
    color: Tuple[int, int, int],
    label: str,
) -> np.ndarray:
    out = image.copy()
    for x1, y1, x2, y2, score in bboxes:
        pt1 = (int(round(x1)), int(round(y1)))
        pt2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(out, pt1, pt2, color, 2)
        text = f"{label} {score:.2f}"
        cv2.putText(out, text, (pt1[0], max(pt1[1] - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return out


def draw_points(
    image: np.ndarray,
    points: Sequence[Tuple[float, float]],
    color: Tuple[int, int, int],
    radius: int = 3,
) -> np.ndarray:
    out = image.copy()
    for x, y in points:
        cv2.circle(out, (int(round(x)), int(round(y))), radius, color, -1, cv2.LINE_AA)
    return out


def overlay_density_heatmap(image: np.ndarray, density: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    if density.ndim == 3:
        density = cv2.cvtColor(density, cv2.COLOR_BGR2GRAY)
    density = density.astype(np.float32)
    if density.shape[:2] != image.shape[:2]:
        density = cv2.resize(density, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
    norm = density - density.min()
    max_val = float(norm.max())
    if max_val > 0:
        norm /= max_val
    heatmap = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    blended = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return blended


def annotate_preview(
    base_image: np.ndarray,
    overlays: Sequence[Tuple[str, np.ndarray]],
    compare: bool = False,
    cols: int = 2,
) -> np.ndarray:
    if not overlays:
        return base_image
    if not compare:
        # Return first overlayed image (already annotated)
        return overlays[0][1]

    images = []
    for label, img in overlays:
        annotated = img.copy()
        cv2.putText(
            annotated,
            label,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        images.append(annotated)
    h, w = images[0].shape[:2]
    cols = max(1, cols)
    rows = math.ceil(len(images) / cols)
    padded = images + [np.zeros_like(images[0]) for _ in range(rows * cols - len(images))]
    grid_rows = []
    for r in range(rows):
        row_imgs = padded[r * cols : (r + 1) * cols]
        grid_rows.append(np.concatenate(row_imgs, axis=1))
    grid = np.concatenate(grid_rows, axis=0)
    return grid


def classify_texture_strata(textures: Sequence[float]) -> List[str]:
    if not textures:
        return []
    values = np.asarray(textures, dtype=float)
    if len(values) < 3:
        thresholds = [float(np.median(values))]
        labels = []
        for val in values:
            labels.append("dense" if val > thresholds[0] else "sparse")
        return labels
    q1, q2 = np.quantile(values, [0.33, 0.66])
    labels = []
    for val in values:
        if val <= q1:
            labels.append("sparse")
        elif val <= q2:
            labels.append("medium")
        else:
            labels.append("dense")
    return labels


def compute_stratified_ci(
    strata_counts: Dict[str, List[float]],
    strata_sizes: Dict[str, int],
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Return (estimate per tile, lower, upper) for stratified sampling."""
    total_population = sum(max(0, size) for size in strata_sizes.values())
    if total_population == 0:
        return (0.0, 0.0, 0.0)
    z = 1.96 if math.isclose(confidence, 0.95) else 1.96
    estimate = 0.0
    variance = 0.0
    for name, samples in strata_counts.items():
        N_h = max(1, strata_sizes.get(name, len(samples)))
        n_h = len(samples)
        if n_h == 0:
            continue
        weight = N_h / total_population
        mean_h = float(np.mean(samples))
        estimate += weight * mean_h
        if n_h > 1:
            s2 = float(np.var(samples, ddof=1))
        else:
            s2 = 0.0
        f_h = min(1.0, n_h / N_h)
        variance += (weight ** 2) * (1 - f_h) * (s2 / max(n_h, 1))
    std = math.sqrt(max(variance, 0.0))
    lower = max(estimate - z * std, 0.0)
    upper = max(estimate + z * std, 0.0)
    return estimate, lower, upper


def random_sample_by_strata(
    labels: Sequence[str],
    counts: Sequence[float],
    k_per_stratum: int,
    rng: Optional[random.Random] = None,
) -> Tuple[Dict[str, List[float]], Dict[str, int]]:
    if rng is None:
        rng = random.Random()
    strata_samples: Dict[str, List[float]] = {}
    strata_sizes: Dict[str, int] = {}
    for label, count in zip(labels, counts):
        strata_sizes[label] = strata_sizes.get(label, 0) + 1
        strata_samples.setdefault(label, [])
    for label in strata_samples.keys():
        indices = [i for i, lbl in enumerate(labels) if lbl == label]
        strata_sizes[label] = len(indices)
        sample_indices = rng.sample(indices, min(k_per_stratum, len(indices))) if indices else []
        strata_samples[label] = [counts[i] for i in sample_indices]
    return strata_samples, strata_sizes


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image


def _test_tiling_roundtrip() -> None:
    rng = np.random.default_rng(0)
    image = (rng.random((64, 80, 3)) * 255).astype(np.uint8)
    tiles = tile_image(image, tile_size=32, overlap=8)
    density_tiles = []
    for tile in tiles:
        density_tiles.append((np.ones(tile.image.shape[:2], dtype=np.float32), tile.origin))
    stitched = stitch_density_map(density_tiles, image.shape[:2])
    assert stitched.shape == image.shape[:2]
    assert np.allclose(stitched[stitched > 0], 1.0)


def _test_stratified_ci() -> None:
    rng = random.Random(42)
    labels = ["sparse", "medium", "dense", "sparse", "dense", "medium"]
    counts = [2, 5, 10, 3, 9, 6]
    samples, sizes = random_sample_by_strata(labels, counts, k_per_stratum=2, rng=rng)
    est, low, high = compute_stratified_ci(samples, sizes)
    assert est >= 0
    assert low <= high


if __name__ == "__main__":
    _test_tiling_roundtrip()
    _test_stratified_ci()
    print("utils tests passed")
