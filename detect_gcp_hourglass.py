import csv
import math
import os
import re
import shutil
import subprocess
import sys
import traceback

import cv2
import matplotlib.pyplot as plt
import numpy as np


# =========================
# Config
# =========================

# Default input. A directory triggers batch mode; a single image triggers single-image mode.
input_path = "test_images"

# Batch output root and naming.
BATCH_OUTPUT_ROOT = "test_results"
BATCH_OUTPUT_PREFIX = "test_results_v"

# Environment variables used when the script recursively runs itself in batch mode.
ENV_SINGLE_IMAGE = "GCP_SINGLE_IMAGE"
ENV_OUTPUT_DIR = "GCP_OUTPUT_DIR"
ENV_DISABLE_WINDOWS = "GCP_DISABLE_WINDOWS"

# Supported image extensions.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Optional GUI windows; disabled by default.
SHOW_WINDOWS = False

# -------------------------
# Red mask extraction
# -------------------------
LOW_RED_1 = np.array([0, 60, 45], dtype=np.uint8)
HIGH_RED_1 = np.array([10, 255, 255], dtype=np.uint8)
LOW_RED_2 = np.array([156, 60, 45], dtype=np.uint8)
HIGH_RED_2 = np.array([180, 255, 255], dtype=np.uint8)

# -------------------------
# Hourglass branch
# -------------------------
HOURGLASS_CLOSE_KERNEL = 3
HOURGLASS_CLOSE_ITER = 1
HOURGLASS_OPEN_KERNEL = 3
HOURGLASS_OPEN_ITER = 0

HOURGLASS_CORE_MIN_AREA = 80
HOURGLASS_FRAGMENT_MIN_AREA = 18
HOURGLASS_CORE_MIN_DIM = 10

HOURGLASS_MERGE_MAX_GAP = 5
HOURGLASS_MERGE_EXPAND = 5
HOURGLASS_MERGE_MAX_CENTER_DIST_FACTOR = 0.55
HOURGLASS_MERGE_MIN_FILL_RATIO = 0.08
HOURGLASS_MERGE_TOUCHING_GAP = 2
HOURGLASS_MERGE_MIN_BBOX_OVERLAP_RATIO = 0.45
HOURGLASS_MERGE_TOUCHING_CENTER_FACTOR = 0.35
HOURGLASS_MERGE_CONTACT_KERNEL = 5
HOURGLASS_MERGE_MIN_CONTACT_RATIO = 0.02

# -------------------------
# Digit branch
# -------------------------
DIGIT_LOOSE_S_MIN = 35
DIGIT_LOOSE_V_MIN = 25
DIGIT_RED_DOMINANCE_MIN = 10
DIGIT_RED_MIN = 60

DIGIT_CLOSE_KERNEL = 3
DIGIT_CLOSE_ITER = 1
DIGIT_DILATE_KERNEL = 3
DIGIT_DILATE_ITER = 1

HOURGLASS_RESIDUAL_DILATE_KERNEL = 5
HOURGLASS_RESIDUAL_DILATE_ITER = 1

DIGIT_MIN_AREA = 10
DIGIT_MAX_AREA_RATIO_TO_HOURGLASS = 0.55
DIGIT_MIN_WIDTH = 4
DIGIT_MIN_HEIGHT = 4
DIGIT_MAX_WIDTH_RATIO_TO_IMAGE = 0.75
DIGIT_MAX_HEIGHT_RATIO_TO_IMAGE = 0.75
DIGIT_MIN_FILL_RATIO = 0.03
DIGIT_MAX_FILL_RATIO = 0.85
DIGIT_MIN_CENTER_DIST_FACTOR = 0.05
DIGIT_MAX_CENTER_DIST_FACTOR = 3.00
DIGIT_MAX_GAP_TO_HOURGLASS = 110
DIGIT_MERGE_MAX_GAP = 8
DIGIT_TIGHT_OVERLAP_MAX = 0.80
DIGIT_MIN_OUTSIDE_RATIO = 0.18

NUMBER_PADDING = 5

# -------------------------
# Center localization
# -------------------------
CENTER_ROI_SIZE = 48
ROI_LOCAL_CLOSE_KERNEL = 3
ROI_LOCAL_CLOSE_ITER = 1

WAIST_SEARCH_HALF_WIDTH = 14
WAIST_CENTER_PENALTY = 0.40

GFTT_MAX_CORNERS = 12
GFTT_QUALITY_LEVEL = 0.01
GFTT_MIN_DISTANCE = 3
GFTT_BLOCK_SIZE = 3
GFTT_USE_HARRIS = False
GFTT_HARRIS_K = 0.04

SUBPIX_WIN_SIZE = 5
SUBPIX_ZERO_ZONE = -1
SUBPIX_MAX_ITER = 50
SUBPIX_EPS = 0.001

SPLIT_EROSION_KERNEL = 3
SPLIT_MAX_EROSION_ITERS = 18
SPLIT_MIN_COMPONENT_AREA = 40
SPLIT_MIN_BALANCE_RATIO = 0.22
SPLIT_CENTER_PENALTY = 0.35

INNER_TIP_MAX_ANGLE_DEG = 155.0
INNER_TIP_MIN_SHARP_SCORE = 0.05
INNER_TIP_ORTHO_PENALTY = 0.15
INNER_TIP_PROJ_WEIGHT = 0.25
INNER_TIP_SHARP_WEIGHT = 4.0
INNER_TIP_AXIS_ALIGNMENT_MIN = 0.30

CORE_SEED_DIST_WEIGHT = 0.55
CORE_SEED_FORWARD_WEIGHT = 0.55
CORE_SEED_SIDE_PENALTY = 0.25
CORE_SUPPORT_MIN_DIST_RATIO = 0.45
CORE_SUPPORT_GROW_BACK_RATIO = -0.25
CORE_SUPPORT_GROW_FRONT_RATIO = 1.50
CORE_SUPPORT_SIDE_BASE_RATIO = 0.55
CORE_SUPPORT_SIDE_GROWTH = 0.25
CORE_SUPPORT_EXPAND_STEPS = 12
CORE_WEDGE_CONTOUR_BACK_RATIO = 0.30
CORE_WEDGE_CONTOUR_FRONT_RATIO = 1.60
CORE_WEDGE_CONTOUR_FRONT_RATIO_SQUARE = 1.40
CORE_WEDGE_SQUARE_MAX_ASPECT_RATIO = 1.05
CORE_WEDGE_SQUARE_MIN_DIM = 60
CORE_SPLIT_REFINE_MAX_DISTANCE = 1.20
CORE_SPLIT_REFINE_BLEND = 0.85
CORE_SPLIT_STABILIZE_MIN_ASPECT = 1.00
CORE_SPLIT_STABILIZE_MAX_ASPECT = 1.22
CORE_SPLIT_STABILIZE_MIN_DIM = 60
CORE_SPLIT_STABILIZE_MIN_BALANCE = 0.72
CORE_SPLIT_STABILIZE_MIN_RESIDUAL = 1.10
CORE_SPLIT_STABILIZE_MAX_DISTANCE = 0.45
CORE_WEDGE_MIN_SIDE_POINTS = 6
CORE_WEDGE_CLOSE_KERNEL = 5
CORE_WEDGE_MAX_MEAN_RESIDUAL = 6.0
CORE_HIGH_RESIDUAL_RELAX_START = 2.5
CORE_HIGH_RESIDUAL_MAX_BLEND = 0.45
CORE_FORCE_SPLIT_MAX_ASPECT_RATIO = 0.82
CORE_FORCE_SPLIT_MIN_BALANCE = 0.90
CORE_FORCE_SPLIT_MAX_ITER = 3

CENTER_CORE_RATIO = 0.36
CENTER_NARROW_RATIO = 0.22
CENTER_CONSISTENCY_RATIO = 0.14

# -------------------------
# Visualization
# -------------------------
NUMBER_BOX_COLOR = (0, 255, 0)
NUMBER_BOX_THICKNESS = 2
CROSSHAIR_COLOR = (255, 0, 0)
CROSSHAIR_SIZE = 10
CROSSHAIR_THICKNESS = 1


def get_script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def ensure_positive_odd(value: int) -> int:
    value = max(1, int(value))
    if value % 2 == 0:
        value += 1
    return value


def safe_imwrite(path: str, image: np.ndarray) -> bool:
    try:
        if image is None or image.size == 0:
            print(f"[警告] 无法保存 {path}：图像为空。")
            return False
        ok = cv2.imwrite(path, image)
        if not ok:
            print(f"[警告] cv2.imwrite 保存失败：{path}")
        return ok
    except Exception as exc:
        print(f"[警告] 保存图像失败：{path}，原因：{exc}")
        return False


def clip_box_xyxy(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int):
    x1 = max(0, min(int(x1), img_w))
    y1 = max(0, min(int(y1), img_h))
    x2 = max(0, min(int(x2), img_w))
    y2 = max(0, min(int(y2), img_h))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def bbox_xywh_to_xyxy(bbox):
    x, y, w, h = bbox
    return x, y, x + w, y + h


def bbox_gap(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    a_left, a_top, a_right, a_bottom = x1, y1, x1 + w1, y1 + h1
    b_left, b_top, b_right, b_bottom = x2, y2, x2 + w2, y2 + h2
    gap_x = max(0, max(b_left - a_right, a_left - b_right))
    gap_y = max(0, max(b_top - a_bottom, a_top - b_bottom))
    return int(gap_x), int(gap_y)


def bbox_intersection_ratio(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    left = max(x1, x2)
    top = max(y1, y2)
    right = min(x1 + w1, x2 + w2)
    bottom = min(y1 + h1, y2 + h2)
    inter_w = max(0, right - left)
    inter_h = max(0, bottom - top)
    inter_area = inter_w * inter_h
    area1 = max(1, w1 * h1)
    return float(inter_area) / float(area1)


def expand_bbox_xywh(bbox, margin, img_w=None, img_h=None):
    x, y, w, h = bbox
    x1 = x - margin
    y1 = y - margin
    x2 = x + w + margin
    y2 = y + h + margin
    if img_w is not None and img_h is not None:
        x1, y1, x2, y2 = clip_box_xyxy(x1, y1, x2, y2, img_w, img_h)
    return x1, y1, x2, y2


def union_bbox_from_components(components):
    if not components:
        return None
    xs = [item["bbox"][0] for item in components]
    ys = [item["bbox"][1] for item in components]
    rights = [item["bbox"][0] + item["bbox"][2] for item in components]
    bottoms = [item["bbox"][1] + item["bbox"][3] for item in components]
    x = min(xs)
    y = min(ys)
    right = max(rights)
    bottom = max(bottoms)
    return int(x), int(y), int(right - x), int(bottom - y)


def weighted_centroid_from_components(components):
    if not components:
        return None
    total_area = sum(max(1.0, float(item["area"])) for item in components)
    if total_area <= 0:
        return None
    cx = sum(float(item["centroid"][0]) * float(item["area"]) for item in components) / total_area
    cy = sum(float(item["centroid"][1]) * float(item["area"]) for item in components) / total_area
    return float(cx), float(cy)


def extract_red_mask(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, LOW_RED_1, HIGH_RED_1)
    mask2 = cv2.inRange(hsv, LOW_RED_2, HIGH_RED_2)
    return cv2.bitwise_or(mask1, mask2)


def extract_loose_digit_red_mask(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    low1 = np.array([0, DIGIT_LOOSE_S_MIN, DIGIT_LOOSE_V_MIN], dtype=np.uint8)
    high1 = np.array([10, 255, 255], dtype=np.uint8)
    low2 = np.array([156, DIGIT_LOOSE_S_MIN, DIGIT_LOOSE_V_MIN], dtype=np.uint8)
    high2 = np.array([180, 255, 255], dtype=np.uint8)
    loose_hsv = cv2.bitwise_or(cv2.inRange(hsv, low1, high1), cv2.inRange(hsv, low2, high2))
    b, g, r = cv2.split(image_bgr)
    red_dominance = (
        (r.astype(np.int16) - g.astype(np.int16) >= int(DIGIT_RED_DOMINANCE_MIN))
        & (r.astype(np.int16) - b.astype(np.int16) >= int(DIGIT_RED_DOMINANCE_MIN))
        & (r >= int(DIGIT_RED_MIN))
    )
    red_dominance = red_dominance.astype(np.uint8) * 255
    return cv2.bitwise_or(loose_hsv, red_dominance)


def build_hourglass_branch_mask(mask_red_raw: np.ndarray) -> np.ndarray:
    close_ksize = ensure_positive_odd(HOURGLASS_CLOSE_KERNEL)
    open_ksize = ensure_positive_odd(HOURGLASS_OPEN_KERNEL)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
    mask = cv2.morphologyEx(
        mask_red_raw,
        cv2.MORPH_CLOSE,
        kernel_close,
        iterations=max(1, int(HOURGLASS_CLOSE_ITER)),
    )
    if int(HOURGLASS_OPEN_ITER) > 0:
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel_open,
            iterations=int(HOURGLASS_OPEN_ITER),
        )
    return mask


def build_digit_branch_mask(image_bgr: np.ndarray, mask_red_raw: np.ndarray) -> np.ndarray:
    del image_bgr
    seed_mask = mask_red_raw.copy()
    close_ksize = ensure_positive_odd(DIGIT_CLOSE_KERNEL)
    dilate_ksize = ensure_positive_odd(DIGIT_DILATE_KERNEL)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_ksize, dilate_ksize))
    mask = cv2.morphologyEx(
        seed_mask,
        cv2.MORPH_CLOSE,
        kernel_close,
        iterations=max(1, int(DIGIT_CLOSE_ITER)),
    )
    mask = cv2.dilate(mask, kernel_dilate, iterations=max(1, int(DIGIT_DILATE_ITER)))
    return mask


def build_digit_loose_mask(image_bgr: np.ndarray) -> np.ndarray:
    mask = extract_loose_digit_red_mask(image_bgr)
    close_ksize = ensure_positive_odd(DIGIT_CLOSE_KERNEL)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel_close,
        iterations=max(1, int(DIGIT_CLOSE_ITER)),
    )
    return mask


def compute_component_overlap_ratio(component_mask: np.ndarray, reference_mask: np.ndarray) -> float:
    component_area = int(np.count_nonzero(component_mask))
    if component_area <= 0:
        return 0.0
    overlap = int(np.count_nonzero((component_mask > 0) & (reference_mask > 0)))
    return float(overlap) / float(component_area)


def attach_component_masks(components, labels):
    enriched = []
    for item in components:
        copied = dict(item)
        copied["component_mask"] = (labels == int(item["label"])).astype(np.uint8) * 255
        enriched.append(copied)
    return enriched


def analyze_components(mask: np.ndarray, branch_name: str):
    binary = (mask > 0).astype(np.uint8) * 255
    labels = np.zeros_like(binary, dtype=np.int32)
    components = []
    try:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        img_h, img_w = binary.shape[:2]
        for label_id in range(1, num_labels):
            x, y, w, h, area = stats[label_id]
            bbox_area = max(1, int(w) * int(h))
            fill_ratio = float(area) / float(bbox_area)
            aspect_ratio = float(max(w, h)) / float(max(1, min(w, h)))
            cx, cy = float(centroids[label_id][0]), float(centroids[label_id][1])
            touches_border = (x <= 0) or (y <= 0) or (x + w >= img_w) or (y + h >= img_h)
            components.append(
                {
                    "branch": branch_name,
                    "label": int(label_id),
                    "area": int(area),
                    "bbox": (int(x), int(y), int(w), int(h)),
                    "bbox_area": int(bbox_area),
                    "centroid": (cx, cy),
                    "fill_ratio": float(fill_ratio),
                    "aspect_ratio": float(aspect_ratio),
                    "touches_border": bool(touches_border),
                    "is_hourglass_core": False,
                    "is_hourglass_union": False,
                    "is_digit_candidate": False,
                    "filter_reason": "",
                    "score": None,
                }
            )
        components.sort(key=lambda item: item["area"], reverse=True)
    except Exception as exc:
        print(f"[警告] {branch_name} 支路连通域分析失败：{exc}")
    return labels, components


def build_mask_from_components(labels: np.ndarray, components, output_shape):
    mask = np.zeros(output_shape[:2], dtype=np.uint8)
    if labels is None or not components:
        return mask
    for item in components:
        mask[labels == int(item["label"])] = 255
    return mask


def component_score_hourglass_core(component, max_area, image_shape):
    img_h, img_w = image_shape[:2]
    cx, cy = component["centroid"]
    img_center_x = (img_w - 1) / 2.0
    img_center_y = (img_h - 1) / 2.0
    diag = float(np.hypot(img_w, img_h)) + 1e-6
    center_dist = float(np.hypot(cx - img_center_x, cy - img_center_y))
    area_score = float(component["area"]) / float(max(1, max_area))
    center_score = max(0.0, 1.0 - center_dist / diag)
    fill_ratio = float(component["fill_ratio"])
    fill_score = 1.0 - min(1.0, abs(fill_ratio - 0.35) / 0.35)
    return 0.60 * area_score + 0.25 * center_score + 0.15 * fill_score


def select_hourglass_core(components, image_shape):
    if not components:
        return None
    max_area = max(item["area"] for item in components) if components else 1
    valid_candidates = []
    for item in components:
        _, _, w, h = item["bbox"]
        reasons = []
        if item["area"] < HOURGLASS_CORE_MIN_AREA:
            reasons.append("area_too_small_for_core")
        if min(w, h) < HOURGLASS_CORE_MIN_DIM:
            reasons.append("bbox_dim_too_small_for_core")
        if reasons:
            item["filter_reason"] = ",".join(reasons)
            item["score"] = None
            continue
        item["score"] = float(component_score_hourglass_core(item, max_area, image_shape))
        valid_candidates.append(item)
    if not valid_candidates:
        core = max(components, key=lambda item: item["area"])
        core["is_hourglass_core"] = True
        if not core["filter_reason"]:
            core["filter_reason"] = "fallback_largest_component_as_core"
        return core
    core = max(valid_candidates, key=lambda item: item["score"])
    core["is_hourglass_core"] = True
    return core

def merge_hourglass_components(core_component, components, labels, image_shape):
    if core_component is None:
        return []
    img_h, img_w = image_shape[:2]
    merged = [core_component]
    core_component["is_hourglass_union"] = True
    contact_ksize = ensure_positive_odd(HOURGLASS_MERGE_CONTACT_KERNEL)
    contact_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (contact_ksize, contact_ksize))
    changed = True
    while changed:
        changed = False
        union_bbox = union_bbox_from_components(merged)
        union_centroid = weighted_centroid_from_components(merged)
        if union_bbox is None or union_centroid is None:
            break
        union_max_dim = max(union_bbox[2], union_bbox[3], 1)
        expanded_union_xyxy = expand_bbox_xywh(union_bbox, HOURGLASS_MERGE_EXPAND, img_w, img_h)
        union_mask = build_mask_from_components(labels, merged, image_shape)
        dilated_union_mask = cv2.dilate(union_mask, contact_kernel, iterations=1)
        for item in components:
            if item["is_hourglass_union"]:
                continue
            reasons = []
            if item["area"] < HOURGLASS_FRAGMENT_MIN_AREA:
                reasons.append("fragment_area_too_small")
            if item["fill_ratio"] < HOURGLASS_MERGE_MIN_FILL_RATIO:
                reasons.append("fragment_fill_too_sparse")
            gap_x, gap_y = bbox_gap(item["bbox"], union_bbox)
            gap = max(gap_x, gap_y)
            cx, cy = item["centroid"]
            centroid_dist = float(np.hypot(cx - union_centroid[0], cy - union_centroid[1]))
            overlap_ratio = bbox_intersection_ratio(item["bbox"], union_bbox)
            component_mask = (labels == int(item["label"])).astype(np.uint8) * 255
            contact_ratio = compute_component_overlap_ratio(component_mask, dilated_union_mask)
            item_box_xyxy = bbox_xywh_to_xyxy(item["bbox"])
            inside_expanded = not (
                item_box_xyxy[2] < expanded_union_xyxy[0]
                or item_box_xyxy[0] > expanded_union_xyxy[2]
                or item_box_xyxy[3] < expanded_union_xyxy[1]
                or item_box_xyxy[1] > expanded_union_xyxy[3]
            )
            if gap > HOURGLASS_MERGE_MAX_GAP:
                reasons.append(f"gap_too_large:{gap}")
            if centroid_dist > HOURGLASS_MERGE_MAX_CENTER_DIST_FACTOR * union_max_dim:
                reasons.append(f"center_dist_too_large:{centroid_dist:.1f}")
            if not inside_expanded:
                reasons.append("outside_expanded_union_bbox")
            if contact_ratio < HOURGLASS_MERGE_MIN_CONTACT_RATIO:
                reasons.append(f"contact_too_low:{contact_ratio:.2f}")
            touching_and_close = (
                gap <= HOURGLASS_MERGE_TOUCHING_GAP
                and centroid_dist <= HOURGLASS_MERGE_TOUCHING_CENTER_FACTOR * union_max_dim
            )
            overlap_good = overlap_ratio >= HOURGLASS_MERGE_MIN_BBOX_OVERLAP_RATIO
            contact_good = contact_ratio >= HOURGLASS_MERGE_MIN_CONTACT_RATIO
            if not (touching_and_close or overlap_good or contact_good):
                reasons.append(f"insufficient_attach_signal:{overlap_ratio:.2f}/{contact_ratio:.2f}")
            if reasons:
                item["filter_reason"] = ",".join(reasons)
                continue
            item["is_hourglass_union"] = True
            item["filter_reason"] = "merged_into_hourglass_union"
            merged.append(item)
            changed = True
    return merged


def build_digit_residual_mask(digit_branch_mask: np.ndarray, hourglass_union_mask: np.ndarray) -> np.ndarray:
    dilate_ksize = ensure_positive_odd(HOURGLASS_RESIDUAL_DILATE_KERNEL)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_ksize, dilate_ksize))
    dilated_hourglass = cv2.dilate(
        hourglass_union_mask,
        kernel,
        iterations=max(1, int(HOURGLASS_RESIDUAL_DILATE_ITER)),
    )
    return cv2.bitwise_and(digit_branch_mask, cv2.bitwise_not(dilated_hourglass))


def score_digit_candidate(component, hourglass_bbox, hourglass_centroid, hourglass_area, image_shape):
    img_h, img_w = image_shape[:2]
    x, y, w, h = component["bbox"]
    cx, cy = component["centroid"]
    hourglass_max_dim = max(hourglass_bbox[2], hourglass_bbox[3], 1)
    center_dist = float(np.hypot(cx - hourglass_centroid[0], cy - hourglass_centroid[1]))
    gap_x, gap_y = bbox_gap(component["bbox"], hourglass_bbox)
    gap = float(max(gap_x, gap_y))
    area_ratio = float(component["area"]) / float(max(1, hourglass_area))
    area_score = 1.0 - min(1.0, abs(area_ratio - 0.08) / 0.20)
    fill_score = 1.0 - min(1.0, abs(component["fill_ratio"] - 0.22) / 0.25)
    target_dist = 0.75 * hourglass_max_dim
    dist_score = 1.0 - min(1.0, abs(center_dist - target_dist) / max(1.0, target_dist))
    gap_score = 1.0 - min(1.0, gap / float(max(1.0, DIGIT_MAX_GAP_TO_HOURGLASS)))
    outside_score = float(component.get("outside_ratio", 0.0))
    residual_score = float(component.get("residual_support_ratio", 0.0))
    overlap_penalty = float(component.get("tight_overlap_ratio", 0.0))
    border_penalty = 0.10 if (x <= 0 or y <= 0 or x + w >= img_w or y + h >= img_h) else 0.0
    return (
        0.22 * area_score
        + 0.18 * fill_score
        + 0.14 * dist_score
        + 0.10 * gap_score
        + 0.22 * outside_score
        + 0.18 * residual_score
        - 0.20 * overlap_penalty
        - border_penalty
    )


def select_digit_components(digit_components, digit_labels, residual_mask, hourglass_union_mask, hourglass_union_components, image_shape):
    if not hourglass_union_components:
        for item in digit_components:
            item["filter_reason"] = "hourglass_not_available"
        return []
    hourglass_bbox = union_bbox_from_components(hourglass_union_components)
    hourglass_centroid = weighted_centroid_from_components(hourglass_union_components)
    hourglass_area = sum(item["area"] for item in hourglass_union_components)
    img_h, img_w = image_shape[:2]
    hourglass_max_dim = max(hourglass_bbox[2], hourglass_bbox[3], 1)
    tight_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tight_hourglass = cv2.dilate(hourglass_union_mask, tight_kernel, iterations=1)
    valid = []
    for item in digit_components:
        x, y, w, h = item["bbox"]
        cx, cy = item["centroid"]
        reasons = []
        component_mask = item.get("component_mask")
        if component_mask is None:
            component_mask = (digit_labels == int(item["label"])).astype(np.uint8) * 255
        tight_overlap_ratio = compute_component_overlap_ratio(component_mask, tight_hourglass)
        residual_support_ratio = compute_component_overlap_ratio(component_mask, residual_mask)
        outside_ratio = 1.0 - compute_component_overlap_ratio(component_mask, hourglass_union_mask)
        item["tight_overlap_ratio"] = float(tight_overlap_ratio)
        item["residual_support_ratio"] = float(residual_support_ratio)
        item["outside_ratio"] = float(outside_ratio)
        if item["area"] < DIGIT_MIN_AREA:
            reasons.append("digit_area_too_small")
        if item["area"] > DIGIT_MAX_AREA_RATIO_TO_HOURGLASS * hourglass_area:
            reasons.append("digit_area_too_large")
        if w < DIGIT_MIN_WIDTH or h < DIGIT_MIN_HEIGHT:
            reasons.append("digit_bbox_too_small")
        if w > DIGIT_MAX_WIDTH_RATIO_TO_IMAGE * img_w or h > DIGIT_MAX_HEIGHT_RATIO_TO_IMAGE * img_h:
            reasons.append("digit_bbox_too_large")
        if item["fill_ratio"] < DIGIT_MIN_FILL_RATIO:
            reasons.append("digit_fill_too_sparse")
        if item["fill_ratio"] > DIGIT_MAX_FILL_RATIO:
            reasons.append("digit_fill_too_dense")
        center_dist = float(np.hypot(cx - hourglass_centroid[0], cy - hourglass_centroid[1]))
        min_center_dist = DIGIT_MIN_CENTER_DIST_FACTOR * hourglass_max_dim
        max_center_dist = DIGIT_MAX_CENTER_DIST_FACTOR * hourglass_max_dim
        if center_dist < min_center_dist:
            reasons.append(f"digit_too_close_to_hourglass:{center_dist:.1f}")
        if center_dist > max_center_dist:
            reasons.append(f"digit_too_far_from_hourglass:{center_dist:.1f}")
        gap_x, gap_y = bbox_gap(item["bbox"], hourglass_bbox)
        gap = max(gap_x, gap_y)
        if gap > DIGIT_MAX_GAP_TO_HOURGLASS:
            reasons.append(f"digit_gap_too_large:{gap}")
        centroid_inside_hourglass_bbox = (
            hourglass_bbox[0] <= cx <= hourglass_bbox[0] + hourglass_bbox[2]
            and hourglass_bbox[1] <= cy <= hourglass_bbox[1] + hourglass_bbox[3]
        )
        if tight_overlap_ratio > DIGIT_TIGHT_OVERLAP_MAX and outside_ratio < DIGIT_MIN_OUTSIDE_RATIO:
            reasons.append(f"mostly_hourglass_overlap:{tight_overlap_ratio:.2f}")
        if centroid_inside_hourglass_bbox and residual_support_ratio < 0.02 and outside_ratio < DIGIT_MIN_OUTSIDE_RATIO:
            reasons.append("inside_hourglass_without_residual_support")
        if reasons:
            item["filter_reason"] = ",".join(reasons)
            continue
        item["score"] = float(
            score_digit_candidate(item, hourglass_bbox, hourglass_centroid, hourglass_area, image_shape)
        )
        valid.append(item)
    if not valid:
        return []
    residual_supported = [item for item in valid if float(item.get("residual_support_ratio", 0.0)) >= 0.05]
    if residual_supported:
        valid = residual_supported
    max_valid_area = max(item["area"] for item in valid)
    area_floor = max(int(DIGIT_MIN_AREA), int(0.25 * max_valid_area))
    filtered_valid = []
    for item in valid:
        if item["area"] >= area_floor:
            filtered_valid.append(item)
        else:
            item["filter_reason"] = f"below_relative_area_floor:{area_floor}"
    if filtered_valid:
        valid = filtered_valid
    digit_core = max(valid, key=lambda item: item["score"])
    digit_core["is_digit_candidate"] = True
    digit_core["filter_reason"] = "selected_as_digit_core"
    selected = [digit_core]
    core_bbox = digit_core["bbox"]
    for item in valid:
        if item["label"] == digit_core["label"]:
            continue
        gap_x, gap_y = bbox_gap(item["bbox"], core_bbox)
        gap = max(gap_x, gap_y)
        if gap <= DIGIT_MERGE_MAX_GAP:
            item["is_digit_candidate"] = True
            item["filter_reason"] = "merged_into_digit_union"
            selected.append(item)
        elif not item["filter_reason"]:
            item["filter_reason"] = f"not_selected_digit_gap:{gap}"
    return selected


def crop_number_region(image_bgr: np.ndarray, digit_components, padding: int):
    img_h, img_w = image_bgr.shape[:2]
    if not digit_components:
        return np.zeros((32, 32, 3), dtype=np.uint8), None
    bbox = union_bbox_from_components(digit_components)
    if bbox is None:
        return np.zeros((32, 32, 3), dtype=np.uint8), None
    try:
        x, y, w, h = bbox
        x1 = x - padding
        y1 = y - padding
        x2 = x + w + padding
        y2 = y + h + padding
        x1, y1, x2, y2 = clip_box_xyxy(x1, y1, x2, y2, img_w, img_h)
        if x2 <= x1 or y2 <= y1:
            return np.zeros((32, 32, 3), dtype=np.uint8), None
        crop = image_bgr[y1:y2, x1:x2].copy()
        return crop, (x1, y1, x2, y2)
    except Exception as exc:
        print(f"[警告] 数字区域裁剪失败：{exc}")
        return np.zeros((32, 32, 3), dtype=np.uint8), None


def compute_mask_centroid(binary_mask: np.ndarray):
    try:
        moments = cv2.moments(binary_mask, binaryImage=True)
        if moments["m00"] > 1e-8:
            return float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])
    except Exception as exc:
        print(f"[警告] 掩膜质心计算失败：{exc}")
    return None


def compute_pca_angle_from_mask(binary_mask: np.ndarray):
    pts = cv2.findNonZero(binary_mask)
    if pts is None or len(pts) < 5:
        return 0.0
    pts = pts.reshape(-1, 2).astype(np.float32)
    mean = np.mean(pts, axis=0)
    centered = pts - mean
    cov = np.dot(centered.T, centered) / max(1, len(pts) - 1)
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
        major_vec = eigvecs[:, np.argmax(eigvals)]
        angle_rad = float(np.arctan2(major_vec[1], major_vec[0]))
        return float(np.degrees(angle_rad))
    except Exception:
        return 0.0


def rotate_roi(mask_roi: np.ndarray, angle_deg: float):
    h, w = mask_roi.shape[:2]
    center = ((w - 1) / 2.0, (h - 1) / 2.0)
    mat = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)
    rotated = cv2.warpAffine(mask_roi, mat, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    inv_mat = cv2.invertAffineTransform(mat)
    return rotated, mat, inv_mat


def quadratic_subpixel_minimum(values, idx):
    if idx <= 0 or idx >= len(values) - 1:
        return float(idx)
    y1 = float(values[idx - 1])
    y2 = float(values[idx])
    y3 = float(values[idx + 1])
    denom = y1 - 2.0 * y2 + y3
    if abs(denom) < 1e-8:
        return float(idx)
    delta = 0.5 * (y1 - y3) / denom
    delta = float(np.clip(delta, -1.0, 1.0))
    return float(idx) + delta


def find_nearest_points_between_components(mask_a: np.ndarray, mask_b: np.ndarray):
    pts_a = cv2.findNonZero(mask_a)
    pts_b = cv2.findNonZero(mask_b)
    if pts_a is None or pts_b is None:
        return None
    pts_a = pts_a.reshape(-1, 2)
    pts_b = pts_b.reshape(-1, 2)
    if len(pts_a) > 500:
        pts_a = pts_a[:: max(1, len(pts_a) // 500)]
    if len(pts_b) > 500:
        pts_b = pts_b[:: max(1, len(pts_b) // 500)]
    diff = pts_a[:, None, :] - pts_b[None, :, :]
    dist2 = np.sum(diff.astype(np.float32) ** 2, axis=2)
    idx_a, idx_b = np.unravel_index(int(np.argmin(dist2)), dist2.shape)
    p_a = pts_a[idx_a].astype(np.float32)
    p_b = pts_b[idx_b].astype(np.float32)
    midpoint = (p_a + p_b) / 2.0
    return {
        "point_a": (float(p_a[0]), float(p_a[1])),
        "point_b": (float(p_b[0]), float(p_b[1])),
        "midpoint": (float(midpoint[0]), float(midpoint[1])),
        "distance": float(np.sqrt(dist2[idx_a, idx_b])),
    }

def locate_split_midpoint_in_roi(roi_mask: np.ndarray):
    if roi_mask is None or roi_mask.size == 0:
        return None, {"split_iter": None, "balance_ratio": None}
    kernel_size = ensure_positive_odd(SPLIT_EROSION_KERNEL)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    current = roi_mask.copy()
    h, w = current.shape[:2]
    center = np.array([(w - 1) / 2.0, (h - 1) / 2.0], dtype=np.float32)
    diag = float(np.hypot(w, h)) + 1e-6
    best = None
    for erosion_iter in range(1, int(SPLIT_MAX_EROSION_ITERS) + 1):
        current = cv2.erode(current, kernel, iterations=1)
        binary = (current > 0).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        valid_labels = [
            label_id for label_id in range(1, num_labels)
            if int(stats[label_id, cv2.CC_STAT_AREA]) >= int(SPLIT_MIN_COMPONENT_AREA)
        ]
        if len(valid_labels) < 2:
            continue
        valid_labels = sorted(valid_labels, key=lambda lid: int(stats[lid, cv2.CC_STAT_AREA]), reverse=True)[:2]
        area_a = int(stats[valid_labels[0], cv2.CC_STAT_AREA])
        area_b = int(stats[valid_labels[1], cv2.CC_STAT_AREA])
        balance_ratio = float(min(area_a, area_b)) / float(max(area_a, area_b))
        mask_a = (labels == valid_labels[0]).astype(np.uint8) * 255
        mask_b = (labels == valid_labels[1]).astype(np.uint8) * 255
        nearest = find_nearest_points_between_components(mask_a, mask_b)
        if nearest is None:
            continue
        midpoint = np.array(nearest["midpoint"], dtype=np.float32)
        center_dist = float(np.linalg.norm(midpoint - center))
        score = balance_ratio - SPLIT_CENTER_PENALTY * (center_dist / diag)
        candidate = {
            "midpoint": nearest["midpoint"],
            "split_iter": erosion_iter,
            "balance_ratio": balance_ratio,
            "distance": nearest["distance"],
            "score": score,
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate
        if balance_ratio >= float(SPLIT_MIN_BALANCE_RATIO):
            break
    if best is None:
        return None, {"split_iter": None, "balance_ratio": None}
    return best["midpoint"], {
        "split_iter": int(best["split_iter"]),
        "balance_ratio": float(best["balance_ratio"]),
        "split_distance": float(best["distance"]),
    }


def split_hourglass_into_lobes(hourglass_mask: np.ndarray):
    if hourglass_mask is None or hourglass_mask.size == 0:
        return None
    kernel_size = ensure_positive_odd(SPLIT_EROSION_KERNEL)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    current = hourglass_mask.copy()
    for erosion_iter in range(1, int(SPLIT_MAX_EROSION_ITERS) + 1):
        current = cv2.erode(current, kernel, iterations=1)
        binary = (current > 0).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        valid_labels = [
            label_id for label_id in range(1, num_labels)
            if int(stats[label_id, cv2.CC_STAT_AREA]) >= int(SPLIT_MIN_COMPONENT_AREA)
        ]
        if len(valid_labels) < 2:
            continue
        valid_labels = sorted(valid_labels, key=lambda lid: int(stats[lid, cv2.CC_STAT_AREA]), reverse=True)[:2]
        area_a = int(stats[valid_labels[0], cv2.CC_STAT_AREA])
        area_b = int(stats[valid_labels[1], cv2.CC_STAT_AREA])
        balance_ratio = float(min(area_a, area_b)) / float(max(area_a, area_b))
        seed_centroid_a = np.array(centroids[valid_labels[0]], dtype=np.float32)
        seed_centroid_b = np.array(centroids[valid_labels[1]], dtype=np.float32)
        ys, xs = np.where(hourglass_mask > 0)
        if len(xs) == 0:
            return None
        points = np.stack([xs, ys], axis=1).astype(np.float32)
        dist_a = np.sum((points - seed_centroid_a[None, :]) ** 2, axis=1)
        dist_b = np.sum((points - seed_centroid_b[None, :]) ** 2, axis=1)
        lobe_a = np.zeros_like(hourglass_mask, dtype=np.uint8)
        lobe_b = np.zeros_like(hourglass_mask, dtype=np.uint8)
        select_a = dist_a <= dist_b
        pts_a = points[select_a].astype(np.int32)
        pts_b = points[~select_a].astype(np.int32)
        if len(pts_a) == 0 or len(pts_b) == 0:
            continue
        lobe_a[pts_a[:, 1], pts_a[:, 0]] = 255
        lobe_b[pts_b[:, 1], pts_b[:, 0]] = 255
        return {
            "lobe_a": lobe_a,
            "lobe_b": lobe_b,
            "centroid_a": tuple(map(float, seed_centroid_a)),
            "centroid_b": tuple(map(float, seed_centroid_b)),
            "split_iter": int(erosion_iter),
            "balance_ratio": float(balance_ratio),
        }
    return None


def line_intersection(point_a, vec_a, point_b, vec_b):
    mat = np.array(
        [
            [float(vec_a[0]), -float(vec_b[0])],
            [float(vec_a[1]), -float(vec_b[1])],
        ],
        dtype=np.float32,
    )
    rhs = np.array(
        [
            float(point_b[0]) - float(point_a[0]),
            float(point_b[1]) - float(point_a[1]),
        ],
        dtype=np.float32,
    )
    try:
        scale_a, _ = np.linalg.solve(mat, rhs)
    except np.linalg.LinAlgError:
        return None
    return np.array(point_a, dtype=np.float32) + float(scale_a) * np.array(vec_a, dtype=np.float32)


def fit_line_through_points(points: np.ndarray):
    if points is None or len(points) < 2:
        return None
    pts = points.reshape(-1, 1, 2).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
    direction = np.array([vx, vy], dtype=np.float32)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm < 1e-6:
        return None
    direction /= direction_norm
    anchor = np.array([x0, y0], dtype=np.float32)
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    residuals = np.abs((points.astype(np.float32) - anchor[None, :]) @ normal)
    return {
        "anchor": anchor,
        "direction": direction,
        "mean_residual": float(np.mean(residuals)) if len(residuals) > 0 else 0.0,
    }


def choose_core_seed(lobe_mask: np.ndarray, other_centroid):
    if lobe_mask is None or lobe_mask.size == 0:
        return None
    dist_map = cv2.distanceTransform((lobe_mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    ys, xs = np.where(lobe_mask > 0)
    if len(xs) == 0:
        return None

    moments = cv2.moments(lobe_mask, binaryImage=True)
    if moments["m00"] <= 1e-8:
        return None

    lobe_centroid = np.array(
        [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
        dtype=np.float32,
    )
    other_centroid = np.array(other_centroid, dtype=np.float32)
    axis = other_centroid - lobe_centroid
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-6:
        return None
    axis /= axis_norm
    normal = np.array([-axis[1], axis[0]], dtype=np.float32)

    points = np.stack([xs, ys], axis=1).astype(np.float32)
    rel = points - lobe_centroid[None, :]
    forward = rel @ axis
    side = np.abs(rel @ normal)
    dist_values = dist_map[ys, xs].astype(np.float32)

    forward_norm = (forward - np.min(forward)) / max(1e-6, float(np.max(forward) - np.min(forward)))
    side_norm = side / max(1e-6, float(np.percentile(side, 90)))
    dist_norm = dist_values / max(1e-6, float(np.max(dist_values)))

    scores = (
        CORE_SEED_DIST_WEIGHT * dist_norm
        + CORE_SEED_FORWARD_WEIGHT * forward_norm
        - CORE_SEED_SIDE_PENALTY * np.clip(side_norm, 0.0, 1.5)
    )
    best_idx = int(np.argmax(scores))
    seed_point = points[best_idx]
    seed_dist = float(dist_values[best_idx])

    return {
        "seed_point": (float(seed_point[0]), float(seed_point[1])),
        "seed_dist": seed_dist,
        "dist_map": dist_map,
        "lobe_centroid": (float(lobe_centroid[0]), float(lobe_centroid[1])),
        "other_centroid": (float(other_centroid[0]), float(other_centroid[1])),
        "axis": (float(axis[0]), float(axis[1])),
        "normal": (float(normal[0]), float(normal[1])),
    }


def expand_core_support(lobe_mask: np.ndarray, seed_info: dict):
    if lobe_mask is None or lobe_mask.size == 0 or seed_info is None:
        return None

    dist_map = seed_info["dist_map"]
    seed_x, seed_y = seed_info["seed_point"]
    axis = np.array(seed_info["axis"], dtype=np.float32)
    normal = np.array(seed_info["normal"], dtype=np.float32)
    seed = np.array([seed_x, seed_y], dtype=np.float32)
    seed_dist = float(seed_info["seed_dist"])

    threshold = max(1.0, CORE_SUPPORT_MIN_DIST_RATIO * seed_dist)
    core_mask = ((dist_map >= threshold) & (lobe_mask > 0)).astype(np.uint8) * 255
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats((core_mask > 0).astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return None

    seed_ix = int(round(seed_x))
    seed_iy = int(round(seed_y))
    seed_ix = max(0, min(seed_ix, labels.shape[1] - 1))
    seed_iy = max(0, min(seed_iy, labels.shape[0] - 1))
    seed_label = int(labels[seed_iy, seed_ix])
    if seed_label <= 0:
        return None

    support_mask = (labels == seed_label).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for _ in range(int(CORE_SUPPORT_EXPAND_STEPS)):
        dilated = cv2.dilate(support_mask, kernel, iterations=1)
        added = (dilated > 0) & (lobe_mask > 0) & (support_mask == 0)
        if not np.any(added):
            break

        ys, xs = np.where(added)
        rel = np.stack([xs, ys], axis=1).astype(np.float32) - seed[None, :]
        forward = rel @ axis
        side = np.abs(rel @ normal)
        side_limit = CORE_SUPPORT_SIDE_BASE_RATIO * seed_dist + CORE_SUPPORT_SIDE_GROWTH * np.maximum(forward, 0.0)
        allow = (
            (forward >= CORE_SUPPORT_GROW_BACK_RATIO * seed_dist)
            & (forward <= CORE_SUPPORT_GROW_FRONT_RATIO * seed_dist)
            & (side <= side_limit)
        )
        if np.any(allow):
            support_mask[ys[allow], xs[allow]] = 255

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (ensure_positive_odd(CORE_WEDGE_CLOSE_KERNEL), ensure_positive_odd(CORE_WEDGE_CLOSE_KERNEL)),
    )
    support_mask = cv2.morphologyEx(
        support_mask,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=1,
    )
    support_mask = cv2.bitwise_and(support_mask, lobe_mask)
    if np.count_nonzero(support_mask) <= 0:
        return None
    return support_mask


def estimate_wedge_refined_apex(
    support_mask: np.ndarray,
    seed_info: dict,
    contour_back_ratio: float = CORE_WEDGE_CONTOUR_BACK_RATIO,
    contour_front_ratio: float = CORE_WEDGE_CONTOUR_FRONT_RATIO,
):
    if support_mask is None or support_mask.size == 0 or seed_info is None:
        return None, {}

    contours, _ = cv2.findContours(support_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, {}

    contour = max(contours, key=cv2.contourArea)[:, 0, :].astype(np.float32)
    if len(contour) < 2:
        return None, {}

    seed = np.array(seed_info["seed_point"], dtype=np.float32)
    axis = np.array(seed_info["axis"], dtype=np.float32)
    normal = np.array(seed_info["normal"], dtype=np.float32)
    seed_dist = float(seed_info["seed_dist"])

    rel = contour - seed[None, :]
    forward = rel @ axis
    side = rel @ normal
    keep = (
        (forward >= float(contour_back_ratio) * seed_dist)
        & (forward <= float(contour_front_ratio) * seed_dist)
    )
    contour = contour[keep]
    side = side[keep]
    if len(contour) < 2:
        return None, {}

    left_points = contour[side <= 0]
    right_points = contour[side > 0]
    if len(left_points) < int(CORE_WEDGE_MIN_SIDE_POINTS) or len(right_points) < int(CORE_WEDGE_MIN_SIDE_POINTS):
        return None, {}

    left_line = fit_line_through_points(left_points)
    right_line = fit_line_through_points(right_points)
    if left_line is None or right_line is None:
        return None, {}

    apex = line_intersection(
        left_line["anchor"],
        left_line["direction"],
        right_line["anchor"],
        right_line["direction"],
    )
    if apex is None:
        return None, {}

    mean_residual = 0.5 * (float(left_line["mean_residual"]) + float(right_line["mean_residual"]))
    debug = {
        "left_residual": float(left_line["mean_residual"]),
        "right_residual": float(right_line["mean_residual"]),
        "mean_residual": float(mean_residual),
        "left_count": int(len(left_points)),
        "right_count": int(len(right_points)),
    }
    return (float(apex[0]), float(apex[1])), debug


def locate_core_inner_midpoint(hourglass_mask: np.ndarray):
    split_result = split_hourglass_into_lobes(hourglass_mask)
    if split_result is None:
        return None, {"core_valid": False, "split_iter": None, "balance_ratio": None}

    bbox = cv2.boundingRect(cv2.findNonZero(hourglass_mask))
    bbox_x, bbox_y, bbox_w, bbox_h = bbox
    aspect_ratio = float(bbox_w) / float(max(1, bbox_h))
    contour_front_ratio = float(CORE_WEDGE_CONTOUR_FRONT_RATIO)
    if aspect_ratio < float(CORE_WEDGE_SQUARE_MAX_ASPECT_RATIO) and min(bbox_w, bbox_h) >= int(CORE_WEDGE_SQUARE_MIN_DIM):
        contour_front_ratio = float(CORE_WEDGE_CONTOUR_FRONT_RATIO_SQUARE)

    seed_a = choose_core_seed(split_result["lobe_a"], split_result["centroid_b"])
    seed_b = choose_core_seed(split_result["lobe_b"], split_result["centroid_a"])
    if seed_a is None or seed_b is None:
        return None, {
            "core_valid": False,
            "split_iter": split_result["split_iter"],
            "balance_ratio": split_result["balance_ratio"],
        }

    support_a = expand_core_support(split_result["lobe_a"], seed_a)
    support_b = expand_core_support(split_result["lobe_b"], seed_b)
    if support_a is None or support_b is None:
        return None, {
            "core_valid": False,
            "split_iter": split_result["split_iter"],
            "balance_ratio": split_result["balance_ratio"],
        }

    apex_a, apex_debug_a = estimate_wedge_refined_apex(
        support_a,
        seed_a,
        contour_back_ratio=CORE_WEDGE_CONTOUR_BACK_RATIO,
        contour_front_ratio=contour_front_ratio,
    )
    apex_b, apex_debug_b = estimate_wedge_refined_apex(
        support_b,
        seed_b,
        contour_back_ratio=CORE_WEDGE_CONTOUR_BACK_RATIO,
        contour_front_ratio=contour_front_ratio,
    )
    if apex_a is None or apex_b is None:
        return None, {
            "core_valid": False,
            "split_iter": split_result["split_iter"],
            "balance_ratio": split_result["balance_ratio"],
        }

    midpoint = (
        0.5 * (float(apex_a[0]) + float(apex_b[0])),
        0.5 * (float(apex_a[1]) + float(apex_b[1])),
    )
    split_midpoint, split_debug = locate_split_midpoint_in_roi(hourglass_mask)
    if split_midpoint is not None:
        split_distance = float(np.hypot(midpoint[0] - split_midpoint[0], midpoint[1] - split_midpoint[1]))
    else:
        split_distance = None

    mean_residual = 0.5 * (
        float(apex_debug_a.get("mean_residual", 999.0)) + float(apex_debug_b.get("mean_residual", 999.0))
    )

    if (
        split_midpoint is not None
        and aspect_ratio < float(CORE_WEDGE_SQUARE_MAX_ASPECT_RATIO)
        and min(bbox_w, bbox_h) >= int(CORE_WEDGE_SQUARE_MIN_DIM)
        and split_distance is not None
        and split_distance <= float(CORE_SPLIT_REFINE_MAX_DISTANCE)
    ):
        refined_with_split = True
        midpoint = (
            (1.0 - float(CORE_SPLIT_REFINE_BLEND)) * float(midpoint[0]) + float(CORE_SPLIT_REFINE_BLEND) * float(split_midpoint[0]),
            (1.0 - float(CORE_SPLIT_REFINE_BLEND)) * float(midpoint[1]) + float(CORE_SPLIT_REFINE_BLEND) * float(split_midpoint[1]),
        )
        split_distance = float(np.hypot(midpoint[0] - split_midpoint[0], midpoint[1] - split_midpoint[1]))
        refined_with_split_mode = "square_refine"
    else:
        refined_with_split = False
        refined_with_split_mode = None

    if (
        split_midpoint is not None
        and float(CORE_SPLIT_STABILIZE_MIN_ASPECT) <= aspect_ratio <= float(CORE_SPLIT_STABILIZE_MAX_ASPECT)
        and min(bbox_w, bbox_h) >= int(CORE_SPLIT_STABILIZE_MIN_DIM)
        and float(split_result["balance_ratio"]) >= float(CORE_SPLIT_STABILIZE_MIN_BALANCE)
        and mean_residual >= float(CORE_SPLIT_STABILIZE_MIN_RESIDUAL)
        and split_distance is not None
        and split_distance <= float(CORE_SPLIT_STABILIZE_MAX_DISTANCE)
    ):
        midpoint = (float(split_midpoint[0]), float(split_midpoint[1]))
        split_distance = 0.0
        refined_with_split = True
        refined_with_split_mode = "corridor_stabilized"

    dist_map = cv2.distanceTransform((hourglass_mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    if mean_residual > float(CORE_HIGH_RESIDUAL_RELAX_START):
        blend = min(
            float(CORE_HIGH_RESIDUAL_MAX_BLEND),
            (mean_residual - float(CORE_HIGH_RESIDUAL_RELAX_START)) / max(1.0, float(CORE_HIGH_RESIDUAL_RELAX_START)),
        )
        bbox_center_y = float(bbox_y + bbox_h / 2.0)
        midpoint = (
            float(midpoint[0]),
            (1.0 - blend) * float(midpoint[1]) + blend * bbox_center_y,
        )
    narrow_value = sample_distance_transform(dist_map, midpoint)

    force_split = bool(
        aspect_ratio <= float(CORE_FORCE_SPLIT_MAX_ASPECT_RATIO)
        and float(split_result["balance_ratio"]) >= float(CORE_FORCE_SPLIT_MIN_BALANCE)
        and int(split_result["split_iter"]) <= int(CORE_FORCE_SPLIT_MAX_ITER)
    )
    core_valid = bool(mean_residual <= float(CORE_WEDGE_MAX_MEAN_RESIDUAL) and not force_split)

    debug = {
        "core_valid": core_valid,
        "force_split": force_split,
        "split_iter": int(split_result["split_iter"]),
        "balance_ratio": float(split_result["balance_ratio"]),
        "support_area_a": int(np.count_nonzero(support_a)),
        "support_area_b": int(np.count_nonzero(support_b)),
        "seed_a": tuple(map(float, seed_a["seed_point"])),
        "seed_b": tuple(map(float, seed_b["seed_point"])),
        "apex_a": tuple(map(float, apex_a)),
        "apex_b": tuple(map(float, apex_b)),
        "mean_residual": float(mean_residual),
        "split_distance": split_distance,
        "narrow_value": float(narrow_value),
        "aspect_ratio": float(aspect_ratio),
        "refined_with_split": refined_with_split,
        "refined_with_split_mode": refined_with_split_mode,
    }
    if split_debug.get("split_distance") is not None and debug["split_distance"] is None:
        debug["split_distance"] = float(split_debug["split_distance"])
    return midpoint, debug


def compute_contour_vertex_angle(contour_points: np.ndarray, index: int, step: int) -> float:
    num_points = len(contour_points)
    if num_points < 3:
        return 180.0
    point = contour_points[index].astype(np.float32)
    prev_point = contour_points[(index - step) % num_points].astype(np.float32)
    next_point = contour_points[(index + step) % num_points].astype(np.float32)
    vec_prev = prev_point - point
    vec_next = next_point - point
    norm_prev = float(np.linalg.norm(vec_prev))
    norm_next = float(np.linalg.norm(vec_next))
    if norm_prev < 1e-6 or norm_next < 1e-6:
        return 180.0
    cosine = float(np.clip(np.dot(vec_prev, vec_next) / (norm_prev * norm_next), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def find_inward_tip_on_lobe(lobe_mask: np.ndarray, other_centroid):
    contours, _ = cv2.findContours(lobe_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    contour_points = contour[:, 0, :].astype(np.float32)
    if len(contour_points) < 8:
        return None
    moments = cv2.moments(lobe_mask, binaryImage=True)
    if moments["m00"] <= 1e-8:
        return None
    lobe_centroid = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]], dtype=np.float32)
    other_centroid = np.array(other_centroid, dtype=np.float32)
    axis = other_centroid - lobe_centroid
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-6:
        return None
    axis = axis / axis_norm
    step = max(2, len(contour_points) // 80)
    best = None
    for idx, point in enumerate(contour_points):
        angle_deg = compute_contour_vertex_angle(contour_points, idx, step)
        sharp_score = max(0.0, (INNER_TIP_MAX_ANGLE_DEG - angle_deg) / max(1.0, INNER_TIP_MAX_ANGLE_DEG - 60.0))
        if sharp_score < INNER_TIP_MIN_SHARP_SCORE:
            continue
        rel = point - lobe_centroid
        projection = float(np.dot(rel, axis))
        orthogonal = float(abs(rel[0] * axis[1] - rel[1] * axis[0]))
        distance_to_other = float(np.linalg.norm(point - other_centroid))
        score = (
            -distance_to_other
            + INNER_TIP_SHARP_WEIGHT * sharp_score
            + INNER_TIP_PROJ_WEIGHT * projection
            - INNER_TIP_ORTHO_PENALTY * orthogonal
        )
        candidate = {"point": (float(point[0]), float(point[1])), "angle_deg": float(angle_deg), "score": float(score)}
        if best is None or candidate["score"] > best["score"]:
            best = candidate
    return best


def sample_distance_transform(dist_map: np.ndarray, point_xy) -> float:
    if dist_map is None or dist_map.size == 0 or point_xy is None:
        return 0.0
    h, w = dist_map.shape[:2]
    x = int(round(float(point_xy[0])))
    y = int(round(float(point_xy[1])))
    x1 = max(0, x - 1)
    y1 = max(0, y - 1)
    x2 = min(w, x + 2)
    y2 = min(h, y + 2)
    patch = dist_map[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    return float(np.max(patch))


def locate_inner_tip_midpoint(hourglass_mask: np.ndarray):
    split_result = split_hourglass_into_lobes(hourglass_mask)
    if split_result is None:
        return None, {"split_iter": None, "balance_ratio": None}
    tip_a = find_inward_tip_on_lobe(split_result["lobe_a"], split_result["centroid_b"])
    tip_b = find_inward_tip_on_lobe(split_result["lobe_b"], split_result["centroid_a"])
    if tip_a is None or tip_b is None:
        return None, {
            "split_iter": split_result["split_iter"],
            "balance_ratio": split_result["balance_ratio"],
        }

    point_a = np.array(tip_a["point"], dtype=np.float32)
    point_b = np.array(tip_b["point"], dtype=np.float32)
    midpoint = (point_a + point_b) / 2.0

    x, y, w, h = cv2.boundingRect(cv2.findNonZero(hourglass_mask))
    center_x = x + w / 2.0
    center_y = y + h / 2.0
    dx = abs(float(midpoint[0]) - center_x) / max(1.0, CENTER_CORE_RATIO * w)
    dy = abs(float(midpoint[1]) - center_y) / max(1.0, CENTER_CORE_RATIO * h)
    midpoint_in_core = max(dx, dy) <= 1.0

    pair_axis = np.array(split_result["centroid_b"], dtype=np.float32) - np.array(split_result["centroid_a"], dtype=np.float32)
    pair_axis_norm = float(np.linalg.norm(pair_axis))
    axis_alignment = 0.0
    split_side_a = None
    split_side_b = None
    same_side = False
    if pair_axis_norm >= 1e-6:
        pair_axis /= pair_axis_norm
        tip_vec = point_b - point_a
        tip_vec_norm = float(np.linalg.norm(tip_vec))
        if tip_vec_norm >= 1e-6:
            axis_alignment = abs(float(np.dot(tip_vec / tip_vec_norm, pair_axis)))
        normal = np.array([-pair_axis[1], pair_axis[0]], dtype=np.float32)
        split_midpoint, _ = locate_split_midpoint_in_roi(hourglass_mask)
        if split_midpoint is not None:
            split_midpoint = np.array(split_midpoint, dtype=np.float32)
            split_side_a = float(np.dot(point_a - split_midpoint, normal))
            split_side_b = float(np.dot(point_b - split_midpoint, normal))
            same_side = split_side_a * split_side_b >= 0.0
            split_distance = float(np.linalg.norm(midpoint - split_midpoint))
        else:
            split_distance = None
    else:
        split_distance = None

    d1 = float(np.linalg.norm(midpoint - point_a))
    d2 = float(np.linalg.norm(midpoint - point_b))
    symmetry_gap = abs(d1 - d2) / max(1e-6, d1 + d2)
    dist_map = cv2.distanceTransform((hourglass_mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    narrow_value = sample_distance_transform(dist_map, midpoint)
    narrow_limit = max(1.0, CENTER_NARROW_RATIO * min(w, h))
    is_valid = bool(
        midpoint_in_core
        and axis_alignment >= INNER_TIP_AXIS_ALIGNMENT_MIN
        and narrow_value <= narrow_limit
        and symmetry_gap <= 0.08
        and not same_side
    )

    debug = {
        "split_iter": split_result["split_iter"],
        "balance_ratio": split_result["balance_ratio"],
        "tip_a": tip_a["point"],
        "tip_b": tip_b["point"],
        "tip_a_angle_deg": tip_a["angle_deg"],
        "tip_b_angle_deg": tip_b["angle_deg"],
        "midpoint_in_core": bool(midpoint_in_core),
        "axis_alignment": float(axis_alignment),
        "narrow_value": float(narrow_value),
        "symmetry_gap": float(symmetry_gap),
        "same_side": bool(same_side),
        "split_side_a": split_side_a,
        "split_side_b": split_side_b,
        "split_distance": split_distance,
        "inner_tip_valid": bool(is_valid),
    }
    return (float(midpoint[0]), float(midpoint[1])), debug


def score_center_candidate(point_xy, method_name: str, hourglass_mask: np.ndarray, bbox, reference_points, tip_pair=None):
    if point_xy is None or bbox is None:
        return -1e9
    x, y, w, h = bbox
    center_x = x + w / 2.0
    center_y = y + h / 2.0
    dx = abs(float(point_xy[0]) - center_x) / max(1.0, CENTER_CORE_RATIO * w)
    dy = abs(float(point_xy[1]) - center_y) / max(1.0, CENTER_CORE_RATIO * h)
    core_score = max(0.0, 1.0 - max(dx, dy))
    dist_map = cv2.distanceTransform((hourglass_mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    narrow_value = sample_distance_transform(dist_map, point_xy)
    narrow_limit = max(1.0, CENTER_NARROW_RATIO * min(w, h))
    narrow_score = max(0.0, 1.0 - narrow_value / narrow_limit)
    consistency_score = 0.0
    if reference_points:
        threshold = max(2.0, CENTER_CONSISTENCY_RATIO * min(w, h))
        distances = [
            float(np.hypot(point_xy[0] - ref[0], point_xy[1] - ref[1]))
            for ref in reference_points
            if ref is not None and ref != point_xy
        ]
        if distances:
            consistency_score = max(0.0, 1.0 - min(distances) / threshold)
    symmetry_score = 0.0
    if tip_pair is not None:
        d1 = float(np.hypot(point_xy[0] - tip_pair[0][0], point_xy[1] - tip_pair[0][1]))
        d2 = float(np.hypot(point_xy[0] - tip_pair[1][0], point_xy[1] - tip_pair[1][1]))
        denom = max(1e-6, d1 + d2)
        symmetry_score = max(0.0, 1.0 - abs(d1 - d2) / denom)
    method_bonus = {
        "inner_tip_midpoint": 0.12,
        "split_midpoint_global": 0.04,
        "waist_subpixel": 0.02,
        "center_constrained_corner_fallback": 0.0,
    }.get(method_name, 0.0)
    return 0.38 * core_score + 0.28 * narrow_score + 0.20 * consistency_score + 0.14 * symmetry_score + method_bonus

def detect_waist_point_in_rotated_roi(rotated_roi: np.ndarray):
    if rotated_roi is None or rotated_roi.size == 0:
        return None, {"waist_column": None, "waist_thickness": None}
    fg = rotated_roi > 0
    h, w = fg.shape[:2]
    center_x = (w - 1) / 2.0
    search_left = max(0, int(np.floor(center_x - WAIST_SEARCH_HALF_WIDTH)))
    search_right = min(w - 1, int(np.ceil(center_x + WAIST_SEARCH_HALF_WIDTH)))
    columns = []
    thicknesses = []
    center_ys = []
    for x in range(search_left, search_right + 1):
        ys = np.where(fg[:, x])[0]
        if len(ys) == 0:
            thickness = np.inf
            center_y = np.nan
        else:
            thickness = float(ys[-1] - ys[0] + 1)
            center_y = float((ys[0] + ys[-1]) / 2.0)
        columns.append(x)
        thicknesses.append(thickness)
        center_ys.append(center_y)
    if not columns:
        return None, {"waist_column": None, "waist_thickness": None}
    finite_mask = np.isfinite(thicknesses)
    if not np.any(finite_mask):
        return None, {"waist_column": None, "waist_thickness": None}
    filtered_thickness = np.array(thicknesses, dtype=np.float32)
    finite_values = filtered_thickness[np.isfinite(filtered_thickness)]
    fill_value = float(np.max(finite_values)) if len(finite_values) > 0 else 9999.0
    filtered_thickness[~np.isfinite(filtered_thickness)] = fill_value + 5.0
    smooth_kernel = np.array([1.0, 2.0, 1.0], dtype=np.float32)
    smooth_kernel /= np.sum(smooth_kernel)
    smoothed = np.convolve(filtered_thickness, smooth_kernel, mode="same")
    penalties = np.array([WAIST_CENTER_PENALTY * abs(x - center_x) for x in columns], dtype=np.float32)
    scores = smoothed + penalties
    best_local_idx = int(np.argmin(scores))
    best_x_sub = quadratic_subpixel_minimum(smoothed, best_local_idx)
    x_sub = float(columns[0]) + best_x_sub
    if best_local_idx >= len(center_ys) - 1:
        y_mid = float(center_ys[best_local_idx])
    elif not np.isfinite(center_ys[best_local_idx]):
        y_mid = float((h - 1) / 2.0)
    else:
        frac = float(x_sub - columns[best_local_idx])
        y0 = center_ys[best_local_idx]
        y1 = center_ys[min(best_local_idx + 1, len(center_ys) - 1)]
        if np.isfinite(y0) and np.isfinite(y1):
            y_mid = float(y0 + frac * (y1 - y0))
        else:
            y_mid = float(y0 if np.isfinite(y0) else (h - 1) / 2.0)
    return (float(x_sub), float(y_mid)), {
        "waist_column": float(x_sub),
        "waist_thickness": float(filtered_thickness[best_local_idx]),
    }


def detect_center_constrained_corner(roi_mask: np.ndarray):
    if roi_mask is None or roi_mask.size == 0:
        return None
    roi_gray = np.uint8(roi_mask.copy())
    corners = None
    try:
        corners = cv2.goodFeaturesToTrack(
            roi_gray,
            maxCorners=int(GFTT_MAX_CORNERS),
            qualityLevel=float(GFTT_QUALITY_LEVEL),
            minDistance=float(GFTT_MIN_DISTANCE),
            blockSize=int(GFTT_BLOCK_SIZE),
            useHarrisDetector=bool(GFTT_USE_HARRIS),
            k=float(GFTT_HARRIS_K),
        )
    except Exception:
        corners = None
    if corners is None or len(corners) == 0:
        return None
    corners = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    h, w = roi_mask.shape[:2]
    center = np.array([(w - 1) / 2.0, (h - 1) / 2.0], dtype=np.float32)
    dists = np.linalg.norm(corners - center[None, :], axis=1)
    best_idx = int(np.argmin(dists))
    point = corners[best_idx].reshape(1, 1, 2).astype(np.float32)
    try:
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            int(SUBPIX_MAX_ITER),
            float(SUBPIX_EPS),
        )
        win_size = (max(1, int(SUBPIX_WIN_SIZE)), max(1, int(SUBPIX_WIN_SIZE)))
        zero_zone = (int(SUBPIX_ZERO_ZONE), int(SUBPIX_ZERO_ZONE))
        refined = cv2.cornerSubPix(np.float32(roi_gray), point, win_size, zero_zone, criteria)
        if refined is not None and len(refined) > 0:
            return float(refined[0, 0, 0]), float(refined[0, 0, 1])
    except Exception:
        pass
    return float(point[0, 0, 0]), float(point[0, 0, 1])


def locate_hourglass_intersection(hourglass_union_mask: np.ndarray):
    img_h, img_w = hourglass_union_mask.shape[:2]
    debug = {
        "bbox": None,
        "roi_bbox": None,
        "pca_angle_deg": None,
        "split_iter": None,
        "split_balance_ratio": None,
        "core_inner_midpoint": None,
        "core_apex_a": None,
        "core_apex_b": None,
        "core_mean_residual": None,
        "inner_tip_midpoint": None,
        "split_candidate": None,
        "waist_candidate": None,
        "waist_point_rotated": None,
        "method": None,
    }
    pts = cv2.findNonZero(hourglass_union_mask)
    if pts is None or len(pts) == 0:
        print("[警告] 主靶标 union mask 为空，交点退化为图像中心。")
        return (img_w / 2.0, img_h / 2.0), True, debug

    bbox = cv2.boundingRect(pts)
    x, y, w, h = bbox
    debug["bbox"] = bbox
    bbox_mask = hourglass_union_mask[y : y + h, x : x + w].copy()

    core_midpoint_local, core_debug = locate_core_inner_midpoint(bbox_mask)
    if core_midpoint_local is not None:
        core_point = (float(x + core_midpoint_local[0]), float(y + core_midpoint_local[1]))
        debug["core_inner_midpoint"] = core_point
        debug["core_apex_a"] = (
            None
            if core_debug.get("apex_a") is None
            else (float(x + core_debug["apex_a"][0]), float(y + core_debug["apex_a"][1]))
        )
        debug["core_apex_b"] = (
            None
            if core_debug.get("apex_b") is None
            else (float(x + core_debug["apex_b"][0]), float(y + core_debug["apex_b"][1]))
        )
        debug["core_mean_residual"] = core_debug.get("mean_residual")
        debug["split_iter"] = core_debug.get("split_iter")
        debug["split_balance_ratio"] = core_debug.get("balance_ratio")
        debug["split_distance"] = core_debug.get("split_distance")
        if bool(core_debug.get("core_valid")):
            debug["method"] = "core_inner_midpoint"
            return core_point, False, debug

    candidates = []
    force_split = bool(core_debug.get("force_split")) if core_debug else False
    inner_tip_midpoint_local, inner_tip_debug = locate_inner_tip_midpoint(bbox_mask)
    if inner_tip_midpoint_local is not None and not force_split:
        inner_tip_point = (float(x + inner_tip_midpoint_local[0]), float(y + inner_tip_midpoint_local[1]))
        debug["inner_tip_midpoint"] = inner_tip_point
        if debug["split_iter"] is None:
            debug["split_iter"] = inner_tip_debug.get("split_iter")
        if debug["split_balance_ratio"] is None:
            debug["split_balance_ratio"] = inner_tip_debug.get("balance_ratio")
        debug["axis_alignment"] = inner_tip_debug.get("axis_alignment")
        debug["same_side_pair"] = inner_tip_debug.get("same_side")
        debug["inner_tip_valid"] = inner_tip_debug.get("inner_tip_valid")
        debug["split_side_a"] = inner_tip_debug.get("split_side_a")
        debug["split_side_b"] = inner_tip_debug.get("split_side_b")
        if debug["split_distance"] is None:
            debug["split_distance"] = inner_tip_debug.get("split_distance")
        tip_pair = None
        if inner_tip_debug.get("tip_a") is not None and inner_tip_debug.get("tip_b") is not None:
            tip_pair = (
                (float(x + inner_tip_debug["tip_a"][0]), float(y + inner_tip_debug["tip_a"][1])),
                (float(x + inner_tip_debug["tip_b"][0]), float(y + inner_tip_debug["tip_b"][1])),
            )
        candidates.append({"method": "inner_tip_midpoint", "point": inner_tip_point, "tip_pair": tip_pair})

    global_split_midpoint, split_debug = locate_split_midpoint_in_roi(bbox_mask)
    if global_split_midpoint is not None:
        split_point = (float(x + global_split_midpoint[0]), float(y + global_split_midpoint[1]))
        debug["split_candidate"] = split_point
        if debug["split_iter"] is None:
            debug["split_iter"] = split_debug.get("split_iter")
            debug["split_balance_ratio"] = split_debug.get("balance_ratio")
        candidates.append({"method": "split_midpoint_global", "point": split_point, "tip_pair": None})
        if force_split:
            debug["method"] = "split_midpoint_global"
            return split_point, True, debug

    center_x = x + w / 2.0
    center_y = y + h / 2.0
    half = CENTER_ROI_SIZE / 2.0
    roi_x1 = int(np.floor(center_x - half))
    roi_y1 = int(np.floor(center_y - half))
    roi_x2 = int(np.ceil(center_x + half))
    roi_y2 = int(np.ceil(center_y + half))
    roi_x1, roi_y1, roi_x2, roi_y2 = clip_box_xyxy(roi_x1, roi_y1, roi_x2, roi_y2, img_w, img_h)
    debug["roi_bbox"] = (roi_x1, roi_y1, roi_x2, roi_y2)

    if roi_x2 > roi_x1 and roi_y2 > roi_y1:
        roi_mask = hourglass_union_mask[roi_y1:roi_y2, roi_x1:roi_x2].copy()
        if roi_mask.size > 0:
            close_ksize = ensure_positive_odd(ROI_LOCAL_CLOSE_KERNEL)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
            roi_mask = cv2.morphologyEx(
                roi_mask,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=max(1, int(ROI_LOCAL_CLOSE_ITER)),
            )
            angle_deg = compute_pca_angle_from_mask(hourglass_union_mask)
            debug["pca_angle_deg"] = float(angle_deg)
            rotated_roi, _, inv_mat = rotate_roi(roi_mask, angle_deg)
            waist_point_rotated, _ = detect_waist_point_in_rotated_roi(rotated_roi)
            if waist_point_rotated is not None:
                debug["waist_point_rotated"] = waist_point_rotated
                local_x, local_y = waist_point_rotated
                src_x = inv_mat[0, 0] * local_x + inv_mat[0, 1] * local_y + inv_mat[0, 2]
                src_y = inv_mat[1, 0] * local_x + inv_mat[1, 1] * local_y + inv_mat[1, 2]
                waist_point = (float(roi_x1 + src_x), float(roi_y1 + src_y))
                debug["waist_candidate"] = waist_point
                candidates.append({"method": "waist_subpixel", "point": waist_point, "tip_pair": None})

            if candidates:
                reference_points = [item["point"] for item in candidates]
                for item in candidates:
                    item["score"] = score_center_candidate(
                        item["point"],
                        item["method"],
                        hourglass_union_mask,
                        bbox,
                        reference_points,
                        tip_pair=item.get("tip_pair"),
                    )
                best_candidate = max(candidates, key=lambda item: item["score"])
                inner_tip_candidate = next((item for item in candidates if item["method"] == "inner_tip_midpoint"), None)
                if inner_tip_candidate is not None:
                    consistency_threshold = max(2.0, CENTER_CONSISTENCY_RATIO * min(w, h))
                    consistent_with_aux = any(
                        item["method"] != "inner_tip_midpoint"
                        and float(np.hypot(inner_tip_candidate["point"][0] - item["point"][0], inner_tip_candidate["point"][1] - item["point"][1]))
                        <= consistency_threshold
                        for item in candidates
                    )
                    if bool(debug.get("inner_tip_valid")) and (consistent_with_aux or inner_tip_candidate["score"] >= best_candidate["score"] - 0.02):
                        best_candidate = inner_tip_candidate
                debug["method"] = best_candidate["method"]
                debug["candidate_scores"] = {item["method"]: float(item["score"]) for item in candidates}
                return best_candidate["point"], False, debug

            corner_xy = detect_center_constrained_corner(roi_mask)
            if corner_xy is not None:
                debug["method"] = "center_constrained_corner_fallback"
                return (float(roi_x1 + corner_xy[0]), float(roi_y1 + corner_xy[1])), True, debug

            roi_centroid = compute_mask_centroid(roi_mask)
            if roi_centroid is not None:
                debug["method"] = "roi_centroid_fallback"
                return (float(roi_x1 + roi_centroid[0]), float(roi_y1 + roi_centroid[1])), True, debug

    union_centroid = compute_mask_centroid(hourglass_union_mask)
    if union_centroid is not None:
        debug["method"] = "union_centroid_fallback"
        return union_centroid, True, debug

    debug["method"] = "bbox_center_fallback"
    return (center_x, center_y), True, debug


def draw_crosshair(image_bgr: np.ndarray, point_xy, size=10, color=(255, 0, 0), thickness=1):
    if image_bgr is None or point_xy is None:
        return image_bgr
    img_h, img_w = image_bgr.shape[:2]
    cx = int(round(float(point_xy[0])))
    cy = int(round(float(point_xy[1])))
    x1 = max(0, cx - int(size))
    x2 = min(img_w - 1, cx + int(size))
    y1 = max(0, cy - int(size))
    y2 = min(img_h - 1, cy + int(size))
    cv2.line(image_bgr, (x1, cy), (x2, cy), color, int(thickness), lineType=cv2.LINE_AA)
    cv2.line(image_bgr, (cx, y1), (cx, y2), color, int(thickness), lineType=cv2.LINE_AA)
    return image_bgr


def save_debug_overview(input_bgr: np.ndarray, clean_mask_img: np.ndarray, final_bgr: np.ndarray, output_path: str):
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
        axes[0].imshow(cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Input")
        axes[0].axis("off")
        axes[1].imshow(clean_mask_img, cmap="gray")
        axes[1].set_title("Clean Red Mask")
        axes[1].axis("off")
        axes[2].imshow(cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB))
        axes[2].set_title("Final Result")
        axes[2].axis("off")
        plt.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"[警告] 保存 debug_overview 失败：{exc}")


def print_component_report(title: str, components):
    print(f"\n========== {title} ==========")
    if not components:
        print("无连通域。")
        return
    for item in components:
        extra_parts = []
        if "tight_overlap_ratio" in item:
            extra_parts.append(f"tight_overlap={item['tight_overlap_ratio']:.3f}")
        if "outside_ratio" in item:
            extra_parts.append(f"outside_ratio={item['outside_ratio']:.3f}")
        if "residual_support_ratio" in item:
            extra_parts.append(f"residual_support={item['residual_support_ratio']:.3f}")
        extra_text = ", " + ", ".join(extra_parts) if extra_parts else ""
        print(
            f"label={item['label']}, "
            f"area={item['area']}, "
            f"bbox={item['bbox']}, "
            f"centroid=({item['centroid'][0]:.2f}, {item['centroid'][1]:.2f}), "
            f"fill_ratio={item['fill_ratio']:.3f}, "
            f"is_hourglass_core={item['is_hourglass_core']}, "
            f"is_hourglass_union={item['is_hourglass_union']}, "
            f"is_digit_candidate={item['is_digit_candidate']}, "
            f"reason={item['filter_reason'] if item['filter_reason'] else 'None'}"
            f"{extra_text}"
        )


def maybe_show_windows(image_dict: dict):
    if not SHOW_WINDOWS:
        return
    try:
        for name, image in image_dict.items():
            if image is not None and image.size > 0:
                cv2.imshow(name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as exc:
        print(f"[警告] 显示窗口失败：{exc}")


def resolve_input_image_path(input_name: str):
    checked_paths = []
    if os.path.isabs(input_name):
        candidate = os.path.abspath(input_name)
        checked_paths.append(candidate)
        if os.path.exists(candidate):
            return candidate, checked_paths
        return None, checked_paths
    script_dir = get_script_dir()
    candidate_script_dir = os.path.abspath(os.path.join(script_dir, input_name))
    candidate_cwd = os.path.abspath(input_name)
    for candidate in [candidate_script_dir, candidate_cwd]:
        if candidate not in checked_paths:
            checked_paths.append(candidate)
        if os.path.exists(candidate):
            return candidate, checked_paths
    return None, checked_paths


def build_missing_input_message(input_name: str, checked_paths):
    lines = [f"输入图像不存在：{input_name}", "已检查以下路径："]
    for path in checked_paths:
        lines.append(f"- {path}")
    script_dir = get_script_dir()
    test_images_dir = os.path.join(script_dir, "test_images")
    if os.path.isdir(test_images_dir):
        test_files = [name for name in os.listdir(test_images_dir) if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS]
        lines.append(f"检测到测试目录：{test_images_dir}")
        lines.append(f"其中共有 {len(test_files)} 张可用图片。")
        if test_files:
            lines.append(f"示例文件：{test_files[0]}")
            lines.append("若要单图运行，请将目标图片复制/重命名为脚本目录下的 gcp_crop.jpg")
    return "\n".join(lines)


def is_supported_image_file(path: str) -> bool:
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def collect_images_from_directory(directory_path: str):
    image_paths = []
    for name in sorted(os.listdir(directory_path)):
        path = os.path.join(directory_path, name)
        if is_supported_image_file(path):
            image_paths.append(path)
    return image_paths


def get_next_batch_output_dir() -> str:
    root_dir = get_script_dir()
    batch_root_dir = os.path.join(root_dir, BATCH_OUTPUT_ROOT)
    os.makedirs(batch_root_dir, exist_ok=True)
    pattern = re.compile(r"^test_results_v(\d+)$", re.IGNORECASE)
    versions = []
    for name in os.listdir(batch_root_dir):
        path = os.path.join(batch_root_dir, name)
        if not os.path.isdir(path):
            continue
        match = pattern.match(name)
        if match:
            versions.append(int(match.group(1)))
    next_version = (max(versions) + 1) if versions else 1
    output_dir = os.path.join(batch_root_dir, f"{BATCH_OUTPUT_PREFIX}{next_version}")
    os.makedirs(output_dir, exist_ok=False)
    return output_dir


def extract_summary_value(text: str, pattern: str) -> str:
    match = re.search(pattern, text)
    if not match:
        return ""
    return match.group(1).strip()


def build_machine_summary_line(bbox, point_xy, method: str, used_fallback: bool) -> str:
    bbox_text = "None" if bbox is None else str(bbox)
    point_text = f"{point_xy[0]:.3f},{point_xy[1]:.3f}"
    method_text = method if method else "None"
    fallback_text = "True" if used_fallback else "False"
    return f"SUMMARY|bbox={bbox_text}|point={point_text}|method={method_text}|used_fallback={fallback_text}"


def extract_machine_summary(text: str):
    summary = {"bbox": "", "point": "", "method": "", "used_fallback": ""}
    match = re.search(
        r"SUMMARY\|bbox=(?P<bbox>.*?)\|point=(?P<point>.*?)\|method=(?P<method>.*?)\|used_fallback=(?P<used_fallback>.*)",
        text,
    )
    if not match:
        return summary
    for key, value in match.groupdict().items():
        summary[key] = value.strip()
    return summary


def create_contact_sheet(output_dir: str, rows):
    image_items = []
    for row in rows:
        debug_path = os.path.join(row["result_dir"], "debug_overview.png")
        if os.path.exists(debug_path):
            image_items.append((row["image"], debug_path))
    if not image_items:
        return None
    cols = 2
    rows_count = int(math.ceil(len(image_items) / float(cols)))
    fig, axes = plt.subplots(rows_count, cols, figsize=(16, rows_count * 4.5), dpi=150)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for ax, (title, image_path) in zip(axes, image_items):
        img = plt.imread(image_path)
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    for ax in axes[len(image_items):]:
        ax.axis("off")
    plt.tight_layout()
    output_path = os.path.join(output_dir, "overview_contact_sheet.png")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_batch_summary_csv(output_dir: str, rows):
    csv_path = os.path.join(output_dir, "summary.csv")
    fieldnames = ["image", "returncode", "bbox", "point", "method", "used_fallback", "result_dir"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_batch_mode(input_dir: str):
    images = collect_images_from_directory(input_dir)
    if not images:
        raise FileNotFoundError(f"输入目录中没有可处理图片：{input_dir}")
    batch_output_dir = get_next_batch_output_dir()
    print("========== GCP 批量处理开始 ==========")
    print(f"输入目录：{input_dir}")
    print(f"输出目录：{batch_output_dir}")
    print(f"共发现 {len(images)} 张图片。")
    rows = []
    for index, image_path in enumerate(images, start=1):
        image_name = os.path.basename(image_path)
        image_stem = os.path.splitext(image_name)[0]
        image_output_dir = os.path.join(batch_output_dir, image_stem)
        os.makedirs(image_output_dir, exist_ok=True)
        shutil.copy2(image_path, os.path.join(image_output_dir, image_name))
        print(f"[{index}/{len(images)}] 处理中：{image_name}")
        env = os.environ.copy()
        env[ENV_SINGLE_IMAGE] = image_path
        env[ENV_DISABLE_WINDOWS] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        process = subprocess.run(
            [sys.executable, os.path.abspath(__file__)],
            cwd=image_output_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        combined_log = process.stdout or ""
        if process.stderr:
            combined_log = f"{combined_log}\n{process.stderr}" if combined_log else process.stderr
        with open(os.path.join(image_output_dir, "run.log"), "w", encoding="utf-8") as f:
            f.write(combined_log)
        parsed_summary = extract_machine_summary(combined_log)
        bbox = parsed_summary["bbox"] or extract_summary_value(combined_log, r"数字外接框坐标 \(x1, y1, x2, y2\): (.+)")
        point = parsed_summary["point"] or extract_summary_value(combined_log, r"沙漏交点最终坐标 \(x, y\): \(([^)]+)\)")
        method = parsed_summary["method"] or extract_summary_value(combined_log, r"交点定位方法: (.+)")
        used_fallback = parsed_summary["used_fallback"]
        rows.append(
            {
                "image": image_name,
                "returncode": process.returncode,
                "bbox": bbox,
                "point": point,
                "method": method,
                "used_fallback": used_fallback,
                "result_dir": image_output_dir,
            }
        )
        status = "OK" if process.returncode == 0 else f"FAIL({process.returncode})"
        print(
            f"  完成：{status} | "
            f"bbox={bbox if bbox else 'None'} | "
            f"point={point if point else 'None'} | "
            f"method={method if method else 'None'}"
        )
    write_batch_summary_csv(batch_output_dir, rows)
    contact_sheet_path = create_contact_sheet(batch_output_dir, rows)
    print("\n========== GCP 批量处理结束 ==========")
    print(f"结果目录：{batch_output_dir}")
    print(f"摘要文件：{os.path.join(batch_output_dir, 'summary.csv')}")
    if contact_sheet_path is not None:
        print(f"总览图：{contact_sheet_path}")


def main():
    print("========== GCP 自动定位提取开始 ==========")
    single_image_override = os.environ.get(ENV_SINGLE_IMAGE, "").strip()
    disable_windows = os.environ.get(ENV_DISABLE_WINDOWS, "").strip() == "1"
    if single_image_override:
        resolved_input_path = os.path.abspath(single_image_override)
    else:
        resolved_input_path, checked_paths = resolve_input_image_path(input_path)
        if resolved_input_path is None:
            raise FileNotFoundError(build_missing_input_message(input_path, checked_paths))
        if os.path.isdir(resolved_input_path):
            run_batch_mode(resolved_input_path)
            return

    image_bgr = cv2.imread(resolved_input_path)
    if image_bgr is None:
        raise RuntimeError(f"图像读取失败，请检查路径或文件完整性：{resolved_input_path}")

    warnings_list = []

    mask_red_raw = extract_red_mask(image_bgr)
    safe_imwrite("mask_red_raw.png", mask_red_raw)

    mask_hourglass_branch = build_hourglass_branch_mask(mask_red_raw)
    mask_digit_branch = build_digit_branch_mask(image_bgr, mask_red_raw)
    mask_digit_loose = build_digit_loose_mask(image_bgr)
    safe_imwrite("mask_hourglass_branch.png", mask_hourglass_branch)
    safe_imwrite("mask_digit_branch.png", mask_digit_branch)
    safe_imwrite("mask_digit_loose.png", mask_digit_loose)

    mask_red_clean = cv2.bitwise_or(mask_hourglass_branch, mask_digit_branch)
    safe_imwrite("mask_red_clean.png", mask_red_clean)

    hourglass_labels, hourglass_components = analyze_components(mask_hourglass_branch, "hourglass")
    hourglass_core = select_hourglass_core(hourglass_components, image_bgr.shape)
    if hourglass_core is None:
        warnings_list.append("未找到有效主靶标核心连通域。")
    hourglass_union_components = merge_hourglass_components(hourglass_core, hourglass_components, hourglass_labels, image_bgr.shape)
    hourglass_union_mask = build_mask_from_components(hourglass_labels, hourglass_union_components, image_bgr.shape)
    safe_imwrite("hourglass_union_mask.png", hourglass_union_mask)
    safe_imwrite("hourglass_mask.png", hourglass_union_mask)
    if not hourglass_union_components:
        warnings_list.append("主靶标 union 为空，后续数字 residual 与交点定位会退化。")

    mask_digit_residual = build_digit_residual_mask(mask_digit_branch, hourglass_union_mask)
    safe_imwrite("mask_digit_residual.png", mask_digit_residual)
    digit_branch_labels, digit_branch_components = analyze_components(mask_digit_branch, "digit_branch")
    digit_loose_labels, digit_loose_components = analyze_components(mask_digit_loose, "digit_loose")
    digit_labels, digit_components = analyze_components(mask_digit_residual, "digit_residual")
    digit_candidate_pool = attach_component_masks(digit_branch_components, digit_branch_labels)
    digit_candidate_pool.extend(attach_component_masks(digit_loose_components, digit_loose_labels))
    digit_selected_components = select_digit_components(
        digit_candidate_pool,
        digit_branch_labels,
        mask_digit_residual,
        hourglass_union_mask,
        hourglass_union_components,
        image_bgr.shape,
    )
    if not digit_selected_components:
        warnings_list.append("未从 residual mask 中找到可靠数字候选。")

    number_crop, number_bbox_xyxy = crop_number_region(image_bgr, digit_selected_components, NUMBER_PADDING)
    safe_imwrite("number_crop.jpg", number_crop)

    intersection_xy, used_fallback, locate_debug = locate_hourglass_intersection(hourglass_union_mask)
    if used_fallback:
        warnings_list.append("交点定位未成功使用主方法，触发了 fallback。")

    final_result = image_bgr.copy()
    if number_bbox_xyxy is not None:
        x1, y1, x2, y2 = number_bbox_xyxy
        cv2.rectangle(final_result, (x1, y1), (x2 - 1, y2 - 1), NUMBER_BOX_COLOR, NUMBER_BOX_THICKNESS)
    else:
        warnings_list.append("数字区域 bbox=None，final_result 中未绘制绿色框。")

    draw_crosshair(
        final_result,
        intersection_xy,
        size=CROSSHAIR_SIZE,
        color=CROSSHAIR_COLOR,
        thickness=CROSSHAIR_THICKNESS,
    )
    safe_imwrite("final_result.png", final_result)
    save_debug_overview(image_bgr, mask_red_clean, final_result, "debug_overview.png")

    print_component_report("Hourglass Branch Components", hourglass_components)
    print_component_report("Digit Branch Components", digit_branch_components)
    print_component_report("Digit Loose Components", digit_loose_components)
    print_component_report("Digit Residual Components", digit_components)

    print("\n========== 结果输出 ==========")
    if number_bbox_xyxy is not None:
        print(f"数字外接框坐标 (x1, y1, x2, y2): {number_bbox_xyxy}")
    else:
        print("数字外接框坐标 (x1, y1, x2, y2): None")
    print(f"沙漏交点最终坐标 (x, y): ({intersection_xy[0]:.3f}, {intersection_xy[1]:.3f})")
    print(f"是否使用了兜底质心: {used_fallback}")
    if locate_debug.get("bbox") is not None:
        print(f"主靶标 union bbox (x, y, w, h): {locate_debug['bbox']}")
    if locate_debug.get("roi_bbox") is not None:
        print(f"中心 ROI (x1, y1, x2, y2): {locate_debug['roi_bbox']}")
    if locate_debug.get("pca_angle_deg") is not None:
        print(f"PCA 主方向角度(度): {locate_debug['pca_angle_deg']:.3f}")
    if locate_debug.get("split_iter") is not None:
        print(f"split_midpoint 侵蚀轮次: {locate_debug['split_iter']}")
    if locate_debug.get("split_balance_ratio") is not None:
        print(f"split_midpoint 面积平衡比: {locate_debug['split_balance_ratio']:.3f}")
    if locate_debug.get("waist_point_rotated") is not None:
        wx, wy = locate_debug["waist_point_rotated"]
        print(f"旋转 ROI 中收腰点: ({wx:.3f}, {wy:.3f})")
    if locate_debug.get("method") is not None:
        print(f"交点定位方法: {locate_debug['method']}")
    print(build_machine_summary_line(number_bbox_xyxy, intersection_xy, locate_debug.get("method"), used_fallback))

    if warnings_list:
        print("\n========== 警告信息 ==========")
        for msg in warnings_list:
            print(f"- {msg}")

    print("\n已保存文件：")
    print("- number_crop.jpg")
    print("- mask_red_raw.png")
    print("- mask_red_clean.png")
    print("- mask_hourglass_branch.png")
    print("- mask_digit_branch.png")
    print("- mask_digit_loose.png")
    print("- mask_digit_residual.png")
    print("- hourglass_union_mask.png")
    print("- hourglass_mask.png")
    print("- final_result.png")
    print("- debug_overview.png")

    if SHOW_WINDOWS and not disable_windows:
        maybe_show_windows(
            {
                "input": image_bgr,
                "mask_red_raw": mask_red_raw,
                "mask_hourglass_branch": mask_hourglass_branch,
                "mask_digit_branch": mask_digit_branch,
                "mask_digit_loose": mask_digit_loose,
                "mask_digit_residual": mask_digit_residual,
                "hourglass_union_mask": hourglass_union_mask,
                "final_result": final_result,
            }
        )

    print("========== GCP 自动定位提取结束 ==========")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n[致命错误] 脚本执行失败：")
        print(str(exc))
        print("\n详细堆栈信息：")
        traceback.print_exc()
        sys.exit(1)
