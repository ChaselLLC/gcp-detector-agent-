from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from gcp_geometry import (
    apply_homography_to_point,
    as_float32_points,
    clip_box_xyxy,
    expand_bbox_xyxy,
    order_quad_clockwise,
    polygon_to_bbox_xyxy,
    quad_edge_lengths,
    quad_to_list,
)

DEFAULT_PERSPECTIVE_MIN_SIDE = 96
DEGENERATE_SIDE_THRESHOLD = 2.0


@dataclass
class RoiCropResult:
    success: bool
    crop_mode: str
    roi_image: np.ndarray | None
    roi_bbox_xyxy: tuple[int, int, int, int] | None
    src_polygon_xy: list[list[float]]
    global_to_roi_h: np.ndarray | None
    roi_to_global_h: np.ndarray | None
    source_bbox_image: np.ndarray | None = None
    error: str | None = None


def map_local_point_to_global(point_xy: Sequence[float], roi_to_global_h: np.ndarray) -> tuple[float, float]:
    return apply_homography_to_point(point_xy, roi_to_global_h)


def _axis_aligned_crop(
    image_bgr: np.ndarray,
    polygon_xy: Sequence[Sequence[float]],
    roi_pad: int,
) -> RoiCropResult:
    image_height, image_width = image_bgr.shape[:2]
    bbox_xyxy = expand_bbox_xyxy(
        polygon_to_bbox_xyxy(polygon_xy),
        roi_pad,
        image_width=image_width,
        image_height=image_height,
    )
    x1, y1, x2, y2 = bbox_xyxy
    if x2 <= x1 or y2 <= y1:
        return RoiCropResult(
            success=False,
            crop_mode="axis_aligned",
            roi_image=None,
            roi_bbox_xyxy=None,
            src_polygon_xy=quad_to_list(polygon_xy),
            global_to_roi_h=None,
            roi_to_global_h=None,
            source_bbox_image=None,
            error="Axis-aligned ROI is empty after clipping",
        )

    crop = image_bgr[y1:y2, x1:x2].copy()
    global_to_roi_h = np.array([[1.0, 0.0, -x1], [0.0, 1.0, -y1], [0.0, 0.0, 1.0]], dtype=np.float32)
    roi_to_global_h = np.array([[1.0, 0.0, x1], [0.0, 1.0, y1], [0.0, 0.0, 1.0]], dtype=np.float32)
    return RoiCropResult(
        success=True,
        crop_mode="axis_aligned",
        roi_image=crop,
        roi_bbox_xyxy=bbox_xyxy,
        src_polygon_xy=quad_to_list(polygon_xy),
        global_to_roi_h=global_to_roi_h,
        roi_to_global_h=roi_to_global_h,
        source_bbox_image=crop.copy(),
    )


def crop_roi_from_polygon(
    image_bgr: np.ndarray,
    polygon_xy: Sequence[Sequence[float]],
    roi_pad: int = 8,
    min_size: int = DEFAULT_PERSPECTIVE_MIN_SIDE,
) -> RoiCropResult:
    if image_bgr is None or image_bgr.size == 0:
        return RoiCropResult(
            success=False,
            crop_mode="perspective",
            roi_image=None,
            roi_bbox_xyxy=None,
            src_polygon_xy=[],
            global_to_roi_h=None,
            roi_to_global_h=None,
            source_bbox_image=None,
            error="Input image is empty",
        )

    try:
        quad = order_quad_clockwise(polygon_xy)
    except Exception as exc:
        return RoiCropResult(
            success=False,
            crop_mode="perspective",
            roi_image=None,
            roi_bbox_xyxy=None,
            src_polygon_xy=[],
            global_to_roi_h=None,
            roi_to_global_h=None,
            source_bbox_image=None,
            error=f"Invalid polygon: {exc}",
        )

    image_height, image_width = image_bgr.shape[:2]
    bbox_xyxy = clip_box_xyxy(*polygon_to_bbox_xyxy(quad), image_width, image_height)
    x1, y1, x2, y2 = bbox_xyxy
    source_bbox_image = image_bgr[y1:y2, x1:x2].copy() if (x2 > x1 and y2 > y1) else None
    top, right, bottom, left = quad_edge_lengths(quad)
    inner_width = max(1.0, float(max(top, bottom)))
    inner_height = max(1.0, float(max(left, right)))

    if min(top, right, bottom, left) < DEGENERATE_SIDE_THRESHOLD:
        return _axis_aligned_crop(image_bgr, quad, roi_pad)

    scale = max(1.0, float(min_size) / min(inner_width, inner_height))
    scaled_inner_width = max(1, int(math.ceil(inner_width * scale)))
    scaled_inner_height = max(1, int(math.ceil(inner_height * scale)))

    roi_width = scaled_inner_width + (2 * roi_pad)
    roi_height = scaled_inner_height + (2 * roi_pad)
    dst_quad = np.array(
        [
            [float(roi_pad), float(roi_pad)],
            [float(roi_pad + scaled_inner_width - 1), float(roi_pad)],
            [float(roi_pad + scaled_inner_width - 1), float(roi_pad + scaled_inner_height - 1)],
            [float(roi_pad), float(roi_pad + scaled_inner_height - 1)],
        ],
        dtype=np.float32,
    )

    try:
        global_to_roi_h = cv2.getPerspectiveTransform(quad.astype(np.float32), dst_quad)
        roi_to_global_h = cv2.getPerspectiveTransform(dst_quad, quad.astype(np.float32))
        roi_image = cv2.warpPerspective(image_bgr, global_to_roi_h, (roi_width, roi_height))
    except cv2.error as exc:
        fallback_result = _axis_aligned_crop(image_bgr, quad, roi_pad)
        if fallback_result.success:
            fallback_result.error = f"Perspective crop failed, fallback used: {exc}"
        return fallback_result

    if roi_image is None or roi_image.size == 0:
        fallback_result = _axis_aligned_crop(image_bgr, quad, roi_pad)
        if fallback_result.success:
            fallback_result.error = "Perspective crop produced an empty ROI, fallback used"
        return fallback_result

    return RoiCropResult(
        success=True,
        crop_mode="perspective",
        roi_image=roi_image,
        roi_bbox_xyxy=bbox_xyxy,
        src_polygon_xy=quad_to_list(quad),
        global_to_roi_h=global_to_roi_h.astype(np.float32),
        roi_to_global_h=roi_to_global_h.astype(np.float32),
        source_bbox_image=source_bbox_image,
    )


def map_global_points_to_local(points_xy: Sequence[Sequence[float]], global_to_roi_h: np.ndarray) -> np.ndarray:
    points = as_float32_points(points_xy)
    mapped = cv2.perspectiveTransform(points.reshape(-1, 1, 2), global_to_roi_h.astype(np.float32))
    return mapped.reshape(-1, 2)
