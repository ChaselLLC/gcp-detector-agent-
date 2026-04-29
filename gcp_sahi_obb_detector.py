from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Sequence

import cv2
import numpy as np

from gcp_detection_models import DetectionRecord, RawSliceDetection, SliceInfo
from gcp_geometry import order_quad_clockwise, polygon_to_bbox_xyxy, quad_to_list


DEFAULT_SLICE_SIZE = 1024
DEFAULT_OVERLAP_RATIO = 0.2
DEFAULT_MATCH_THRESHOLD = 0.5
DEFAULT_CONFIDENCE_THRESHOLD = 0.85


@dataclass
class DetectionBackendInfo:
    backend_name: str
    used_sahi: bool


def _iter_slice_windows(
    image_bgr: np.ndarray,
    image_id: str,
    slice_size: int,
    overlap_ratio: float,
) -> Iterator[tuple[np.ndarray, SliceInfo]]:
    image_height, image_width = image_bgr.shape[:2]
    step = max(1, int(slice_size * (1.0 - overlap_ratio)))

    row_index = 0
    for offset_y in range(0, image_height, step):
        col_index = 0
        for offset_x in range(0, image_width, step):
            y_max = offset_y + slice_size
            x_max = offset_x + slice_size
            patch = image_bgr[offset_y:min(y_max, image_height), offset_x:min(x_max, image_width)]
            patch_height, patch_width = patch.shape[:2]
            if patch_height < slice_size or patch_width < slice_size:
                patch = cv2.copyMakeBorder(
                    patch,
                    0,
                    slice_size - patch_height,
                    0,
                    slice_size - patch_width,
                    cv2.BORDER_CONSTANT,
                    value=[0, 0, 0],
                )
            slice_info = SliceInfo(
                slice_row=row_index,
                slice_col=col_index,
                slice_x_offset=int(offset_x),
                slice_y_offset=int(offset_y),
                slice_width=int(min(slice_size, image_width - offset_x)),
                slice_height=int(min(slice_size, image_height - offset_y)),
                source_slice_name=f"{image_id}_row{row_index}_col{col_index}.jpg",
            )
            yield patch, slice_info
            col_index += 1
        row_index += 1


def _segmentation_to_quad(segmentation: Sequence[float] | Sequence[Sequence[float]]) -> list[list[float]]:
    if len(segmentation) == 1 and isinstance(segmentation[0], (list, tuple)):
        flat_values = [float(value) for value in segmentation[0]]
    else:
        flat_values = [float(value) for value in segmentation]

    points = np.asarray(flat_values, dtype=np.float32).reshape(-1, 2)
    if len(points) == 4:
        return quad_to_list(order_quad_clockwise(points))

    rect = cv2.minAreaRect(points.astype(np.float32))
    quad = cv2.boxPoints(rect)
    return quad_to_list(order_quad_clockwise(quad))


def _shift_polygon_xy(
    polygon_xy: Sequence[Sequence[float]],
    shift_x: float,
    shift_y: float,
) -> list[list[float]]:
    return [[float(point[0]) + shift_x, float(point[1]) + shift_y] for point in polygon_xy]


def _union_boxes(boxes: Sequence[Sequence[int]]) -> tuple[int, int, int, int]:
    x1 = min(int(box[0]) for box in boxes)
    y1 = min(int(box[1]) for box in boxes)
    x2 = max(int(box[2]) for box in boxes)
    y2 = max(int(box[3]) for box in boxes)
    return x1, y1, x2, y2


def _build_merged_detection_records(
    raw_detections: list[RawSliceDetection],
    match_threshold: float,
    class_agnostic: bool,
) -> list[DetectionRecord]:
    if not raw_detections:
        return []

    import torch

    from sahi.postprocess.combine import batched_greedy_nmm, greedy_nmm

    ordered_raw = sorted(raw_detections, key=lambda item: item.confidence, reverse=True)
    detection_tensor = torch.tensor(
        [
            [
                float(item.aabb_xyxy[0]),
                float(item.aabb_xyxy[1]),
                float(item.aabb_xyxy[2]),
                float(item.aabb_xyxy[3]),
                float(item.confidence),
                float(item.class_id),
            ]
            for item in ordered_raw
        ],
        dtype=torch.float32,
    )

    if class_agnostic:
        keep_to_merge = greedy_nmm(detection_tensor, match_metric="IOU", match_threshold=match_threshold)
    else:
        keep_to_merge = batched_greedy_nmm(detection_tensor, match_metric="IOU", match_threshold=match_threshold)

    merged_records: list[DetectionRecord] = []
    for keep_index in sorted(keep_to_merge.keys()):
        merge_indices = [keep_index, *keep_to_merge[keep_index]]
        members = [ordered_raw[index] for index in merge_indices]
        base_member = members[0]
        record = DetectionRecord(
            det_id="",
            class_id=base_member.class_id,
            class_name=base_member.class_name,
            confidence=float(max(member.confidence for member in members)),
            polygon_xy=base_member.polygon_xy,
            aabb_xyxy=_union_boxes([member.aabb_xyxy for member in members]),
            slice_info=base_member.slice_info.to_dict(),
            merged_from=[member.raw_id for member in members],
        )
        merged_records.append(record)

    return merged_records


def _run_sahi_slice_predictions(
    image_bgr: np.ndarray,
    image_id: str,
    weights_path: str,
    device: str,
    confidence_threshold: float,
    image_size: int,
    slice_size: int,
    overlap_ratio: float,
) -> list[RawSliceDetection]:
    from sahi.models.ultralytics import UltralyticsDetectionModel
    from sahi.predict import get_prediction

    model = UltralyticsDetectionModel(
        model_path=weights_path,
        confidence_threshold=confidence_threshold,
        device=device,
        image_size=image_size,
    )

    image_height, image_width = image_bgr.shape[:2]
    raw_detections: list[RawSliceDetection] = []
    for patch_bgr, slice_info in _iter_slice_windows(image_bgr, image_id, slice_size, overlap_ratio):
        patch_rgb = patch_bgr[:, :, ::-1]
        prediction = get_prediction(
            image=patch_rgb,
            detection_model=model,
            shift_amount=[slice_info.slice_x_offset, slice_info.slice_y_offset],
            full_shape=[image_height, image_width],
            verbose=0,
        )
        for index, object_prediction in enumerate(prediction.object_prediction_list):
            segmentation = object_prediction.mask.segmentation if object_prediction.mask is not None else None
            if segmentation:
                polygon_xy = _segmentation_to_quad(segmentation)
            else:
                bbox_xyxy = object_prediction.bbox.to_xyxy()
                polygon_xy = [
                    [bbox_xyxy[0], bbox_xyxy[1]],
                    [bbox_xyxy[2], bbox_xyxy[1]],
                    [bbox_xyxy[2], bbox_xyxy[3]],
                    [bbox_xyxy[0], bbox_xyxy[3]],
                ]
            polygon_xy = _shift_polygon_xy(
                quad_to_list(order_quad_clockwise(polygon_xy)),
                float(slice_info.slice_x_offset),
                float(slice_info.slice_y_offset),
            )
            bbox_xyxy = polygon_to_bbox_xyxy(polygon_xy)
            raw_detections.append(
                RawSliceDetection(
                    raw_id=f"{slice_info.source_slice_name}#raw{index:03d}",
                    class_id=int(object_prediction.category.id),
                    class_name=str(object_prediction.category.name),
                    confidence=float(object_prediction.score.value),
                    polygon_xy=polygon_xy,
                    aabb_xyxy=bbox_xyxy,
                    slice_info=slice_info,
                )
            )
    return raw_detections


def _run_ultralytics_slice_predictions(
    image_bgr: np.ndarray,
    image_id: str,
    weights_path: str,
    device: str,
    confidence_threshold: float,
    image_size: int,
    slice_size: int,
    overlap_ratio: float,
) -> list[RawSliceDetection]:
    from ultralytics import YOLO

    model = YOLO(weights_path)
    category_names = model.names
    raw_detections: list[RawSliceDetection] = []
    for patch_bgr, slice_info in _iter_slice_windows(image_bgr, image_id, slice_size, overlap_ratio):
        prediction_results = model(patch_bgr, imgsz=image_size, conf=confidence_threshold, device=device, verbose=False)
        result = prediction_results[0]
        if result.obb is None:
            continue

        for index in range(len(result.obb)):
            polygon_slice = result.obb.xyxyxyxy[index].detach().cpu().numpy().astype(np.float32)
            polygon_global = polygon_slice.copy()
            polygon_global[:, 0] += float(slice_info.slice_x_offset)
            polygon_global[:, 1] += float(slice_info.slice_y_offset)
            polygon_xy = quad_to_list(order_quad_clockwise(polygon_global))
            bbox_xyxy = polygon_to_bbox_xyxy(polygon_xy)
            class_id = int(result.obb.cls[index].item())
            class_name = str(category_names[class_id] if isinstance(category_names, dict) else category_names[class_id])
            raw_detections.append(
                RawSliceDetection(
                    raw_id=f"{slice_info.source_slice_name}#raw{index:03d}",
                    class_id=class_id,
                    class_name=class_name,
                    confidence=float(result.obb.conf[index].item()),
                    polygon_xy=polygon_xy,
                    aabb_xyxy=bbox_xyxy,
                    slice_info=slice_info,
                )
            )
    return raw_detections


def detect_large_image_obb(
    image_bgr: np.ndarray,
    image_id: str,
    weights_path: str,
    device: str = "0",
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    image_size: int = DEFAULT_SLICE_SIZE,
    slice_size: int = DEFAULT_SLICE_SIZE,
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    class_agnostic: bool = False,
    prefer_sahi: bool = True,
) -> tuple[list[DetectionRecord], DetectionBackendInfo]:
    raw_detections: list[RawSliceDetection]
    backend_info: DetectionBackendInfo
    effective_confidence_threshold = max(float(confidence_threshold), DEFAULT_CONFIDENCE_THRESHOLD)

    if prefer_sahi:
        try:
            raw_detections = _run_sahi_slice_predictions(
                image_bgr=image_bgr,
                image_id=image_id,
                weights_path=weights_path,
                device=device,
                confidence_threshold=effective_confidence_threshold,
                image_size=image_size,
                slice_size=slice_size,
                overlap_ratio=overlap_ratio,
            )
            backend_info = DetectionBackendInfo(backend_name="sahi_ultralytics_obb", used_sahi=True)
        except ImportError:
            raw_detections = _run_ultralytics_slice_predictions(
                image_bgr=image_bgr,
                image_id=image_id,
                weights_path=weights_path,
                device=device,
                confidence_threshold=effective_confidence_threshold,
                image_size=image_size,
                slice_size=slice_size,
                overlap_ratio=overlap_ratio,
            )
            backend_info = DetectionBackendInfo(backend_name="manual_ultralytics_obb", used_sahi=False)
    else:
        raw_detections = _run_ultralytics_slice_predictions(
            image_bgr=image_bgr,
            image_id=image_id,
            weights_path=weights_path,
            device=device,
            confidence_threshold=effective_confidence_threshold,
            image_size=image_size,
            slice_size=slice_size,
            overlap_ratio=overlap_ratio,
        )
        backend_info = DetectionBackendInfo(backend_name="manual_ultralytics_obb", used_sahi=False)

    merged_records = _build_merged_detection_records(
        raw_detections=raw_detections,
        match_threshold=match_threshold,
        class_agnostic=class_agnostic,
    )
    merged_records = [
        record
        for record in merged_records
        if float(record.confidence) >= effective_confidence_threshold
    ]
    merged_records.sort(key=lambda item: item.confidence, reverse=True)
    for index, record in enumerate(merged_records, start=1):
        record.det_id = f"det_{index:04d}"

    return merged_records, backend_info
