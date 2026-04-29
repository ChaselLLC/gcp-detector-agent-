from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def as_float32_points(points: Sequence[Sequence[float]], expected_count: int | None = None) -> np.ndarray:
    array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if expected_count is not None and len(array) != expected_count:
        raise ValueError(f"Expected {expected_count} points, got {len(array)}")
    return array


def order_quad_clockwise(points: Sequence[Sequence[float]]) -> np.ndarray:
    quad = as_float32_points(points, expected_count=4)

    sums = quad[:, 0] + quad[:, 1]
    diffs = quad[:, 1] - quad[:, 0]
    ordered = np.vstack(
        [
            quad[np.argmin(sums)],
            quad[np.argmin(diffs)],
            quad[np.argmax(sums)],
            quad[np.argmax(diffs)],
        ]
    ).astype(np.float32)

    unique_count = len({(float(point[0]), float(point[1])) for point in ordered})
    if unique_count != 4:
        center = quad.mean(axis=0)
        angles = np.arctan2(quad[:, 1] - center[1], quad[:, 0] - center[0])
        ordered = quad[np.argsort(angles)]
        start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
        ordered = np.roll(ordered, -start, axis=0)
        if ordered[1, 0] < ordered[3, 0]:
            ordered = ordered[[0, 3, 2, 1]]

    return ordered.astype(np.float32)


def quad_to_list(points: Sequence[Sequence[float]]) -> list[list[float]]:
    quad = as_float32_points(points)
    return [[float(point[0]), float(point[1])] for point in quad]


def polygon_to_bbox_xyxy(points: Sequence[Sequence[float]]) -> tuple[int, int, int, int]:
    polygon = as_float32_points(points)
    x_min = int(math.floor(float(np.min(polygon[:, 0]))))
    y_min = int(math.floor(float(np.min(polygon[:, 1]))))
    x_max = int(math.ceil(float(np.max(polygon[:, 0]))))
    y_max = int(math.ceil(float(np.max(polygon[:, 1]))))
    return x_min, y_min, x_max, y_max


def expand_bbox_xyxy(
    bbox_xyxy: Sequence[int],
    margin: int,
    image_width: int | None = None,
    image_height: int | None = None,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(value) for value in bbox_xyxy]
    x1 -= margin
    y1 -= margin
    x2 += margin
    y2 += margin
    if image_width is not None and image_height is not None:
        return clip_box_xyxy(x1, y1, x2, y2, image_width, image_height)
    return x1, y1, x2, y2


def clip_box_xyxy(x1: int, y1: int, x2: int, y2: int, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    clipped_x1 = max(0, min(int(x1), image_width))
    clipped_y1 = max(0, min(int(y1), image_height))
    clipped_x2 = max(0, min(int(x2), image_width))
    clipped_y2 = max(0, min(int(y2), image_height))
    return clipped_x1, clipped_y1, clipped_x2, clipped_y2


def quad_edge_lengths(quad_xy: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    quad = as_float32_points(quad_xy, expected_count=4)
    top = float(np.linalg.norm(quad[1] - quad[0]))
    right = float(np.linalg.norm(quad[2] - quad[1]))
    bottom = float(np.linalg.norm(quad[2] - quad[3]))
    left = float(np.linalg.norm(quad[3] - quad[0]))
    return top, right, bottom, left


def polygon_center(points: Sequence[Sequence[float]]) -> tuple[float, float]:
    polygon = as_float32_points(points)
    center = polygon.mean(axis=0)
    return float(center[0]), float(center[1])


def apply_homography_to_point(point_xy: Sequence[float], homography: np.ndarray) -> tuple[float, float]:
    point = np.array([float(point_xy[0]), float(point_xy[1]), 1.0], dtype=np.float64)
    mapped = homography @ point
    if abs(float(mapped[2])) < 1e-9:
        raise ValueError("Homography produced an invalid homogeneous coordinate")
    mapped /= float(mapped[2])
    return float(mapped[0]), float(mapped[1])


def apply_homography_to_points(points_xy: Sequence[Sequence[float]], homography: np.ndarray) -> np.ndarray:
    points = as_float32_points(points_xy)
    hom_points = np.concatenate([points.astype(np.float64), np.ones((len(points), 1), dtype=np.float64)], axis=1)
    mapped = hom_points @ homography.T
    valid = np.abs(mapped[:, 2]) >= 1e-9
    if not np.all(valid):
        raise ValueError("Homography produced invalid homogeneous coordinates")
    mapped = mapped[:, :2] / mapped[:, 2:3]
    return mapped.astype(np.float32)


def homography_to_list(homography: np.ndarray | None) -> list[list[float]] | None:
    if homography is None:
        return None
    array = np.asarray(homography, dtype=np.float64).reshape(3, 3)
    return [[float(value) for value in row] for row in array]


def is_point_inside_image(point_xy: Sequence[float], image_shape: Sequence[int]) -> bool:
    x_value = float(point_xy[0])
    y_value = float(point_xy[1])
    image_height = int(image_shape[0])
    image_width = int(image_shape[1])
    return 0.0 <= x_value < float(image_width) and 0.0 <= y_value < float(image_height)


def box_iou_xyxy(box1: Sequence[float], box2: Sequence[float]) -> float:
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h
    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))
    union = area1 + area2 - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def box_ios_xyxy(box1: Sequence[float], box2: Sequence[float]) -> float:
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h
    smaller_area = min(
        max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1])),
        max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1])),
    )
    if smaller_area <= 0.0:
        return 0.0
    return intersection / smaller_area


def flatten_polygon(points_xy: Iterable[Sequence[float]]) -> list[float]:
    flattened: list[float] = []
    for point in points_xy:
        flattened.extend((float(point[0]), float(point[1])))
    return flattened
