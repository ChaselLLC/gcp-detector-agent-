from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2

from gcp_crop_mapper import crop_roi_from_polygon, map_local_point_to_global
from gcp_fine_locator import DEFAULT_PYTHON_EXE, DEFAULT_SCRIPT_PATH, run_fine_locator
from gcp_geometry import homography_to_list, is_point_inside_image
from gcp_sahi_obb_detector import DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_OVERLAP_RATIO, DEFAULT_SLICE_SIZE, detect_large_image_obb
from gcp_visualization import annotate_image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CSV_FIELDNAMES = [
    "image_id",
    "det_id",
    "class_id",
    "class_name",
    "conf",
    "roi_x1",
    "roi_y1",
    "roi_x2",
    "roi_y2",
    "local_x",
    "local_y",
    "global_x",
    "global_y",
    "method",
    "used_fallback",
    "status",
    "error",
]


@dataclass
class PipelineConfig:
    input_path: str
    weights_path: str
    output_dir: str
    python_exe: str = DEFAULT_PYTHON_EXE
    detect_script_path: str = DEFAULT_SCRIPT_PATH
    device: str = "0"
    slice_size: int = DEFAULT_SLICE_SIZE
    overlap: float = DEFAULT_OVERLAP_RATIO
    conf: float = DEFAULT_CONFIDENCE_THRESHOLD
    roi_pad: int = 8
    max_images: int | None = None
    prefer_sahi: bool = True


def _collect_input_images(input_path: str, max_images: int | None = None) -> list[Path]:
    path = Path(input_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    if path.is_file():
        image_paths = [path]
    else:
        image_paths = sorted(
            child.resolve()
            for child in path.iterdir()
            if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
        )

    if max_images is not None:
        image_paths = image_paths[: max(0, int(max_images))]
    return image_paths


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_csv(csv_path: Path, rows: Sequence[dict[str, object]]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDNAMES})


def _write_json(json_path: Path, rows: Sequence[dict[str, object]]) -> None:
    with json_path.open("w", encoding="utf-8") as file_obj:
        json.dump(list(rows), file_obj, ensure_ascii=False, indent=2)


def _base_result_row(image_id: str, detection: dict[str, object] | None = None) -> dict[str, object]:
    row: dict[str, object] = {
        "image_id": image_id,
        "det_id": "",
        "class_id": "",
        "class_name": "",
        "conf": "",
        "roi_x1": "",
        "roi_y1": "",
        "roi_x2": "",
        "roi_y2": "",
        "local_x": "",
        "local_y": "",
        "global_x": "",
        "global_y": "",
        "method": "",
        "used_fallback": "",
        "status": "",
        "error": "",
        "polygon_xy": [],
        "aabb_xyxy": [],
        "slice_info": {},
        "crop_mode": "",
        "roi_bbox_xyxy": [],
        "local_point_xy": None,
        "global_point_xy": None,
        "global_to_roi_h": None,
        "roi_to_global_h": None,
        "merged_from": [],
        "fine_locator_artifacts": {},
        "backend_name": "",
        "used_sahi": None,
        "roi_image_path": "",
        "source_bbox_image_path": "",
    }
    if detection:
        row.update(detection)
    return row


def _build_no_detection_row(image_id: str, backend_name: str, used_sahi: bool, min_confidence: float) -> dict[str, object]:
    row = _base_result_row(image_id)
    row.update(
        {
            "status": "no_detection",
            "error": f"No detections after sliced OBB aggregation with conf >= {min_confidence:.2f}",
            "backend_name": backend_name,
            "used_sahi": used_sahi,
        }
    )
    return row


def process_single_image(image_path: Path, config: PipelineConfig) -> dict[str, object]:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    image_id = image_path.name
    image_stem = image_path.stem
    image_output_dir = _ensure_dir(Path(config.output_dir).resolve() / image_stem)
    rois_dir = _ensure_dir(image_output_dir / "rois")

    min_confidence = max(float(config.conf), DEFAULT_CONFIDENCE_THRESHOLD)

    detections, backend_info = detect_large_image_obb(
        image_bgr=image_bgr,
        image_id=image_stem,
        weights_path=config.weights_path,
        device=config.device,
        confidence_threshold=min_confidence,
        image_size=config.slice_size,
        slice_size=config.slice_size,
        overlap_ratio=config.overlap,
        prefer_sahi=config.prefer_sahi,
    )

    rows: list[dict[str, object]] = []
    if not detections:
        rows.append(_build_no_detection_row(image_id, backend_info.backend_name, backend_info.used_sahi, min_confidence))
    else:
        for detection in detections:
            detection_dir = _ensure_dir(rois_dir / detection.det_id)
            fine_locator_dir = _ensure_dir(detection_dir / "fine_locator")
            row = _base_result_row(
                image_id,
                detection={
                    "det_id": detection.det_id,
                    "class_id": detection.class_id,
                    "class_name": detection.class_name,
                    "conf": float(detection.confidence),
                    "polygon_xy": detection.polygon_xy,
                    "aabb_xyxy": list(detection.aabb_xyxy),
                    "slice_info": detection.slice_info,
                    "merged_from": list(detection.merged_from),
                    "backend_name": backend_info.backend_name,
                    "used_sahi": backend_info.used_sahi,
                },
            )

            crop_result = crop_roi_from_polygon(image_bgr, detection.polygon_xy, roi_pad=config.roi_pad)
            row["crop_mode"] = crop_result.crop_mode
            row["roi_bbox_xyxy"] = list(crop_result.roi_bbox_xyxy) if crop_result.roi_bbox_xyxy else []
            row["global_to_roi_h"] = homography_to_list(crop_result.global_to_roi_h)
            row["roi_to_global_h"] = homography_to_list(crop_result.roi_to_global_h)

            if crop_result.roi_bbox_xyxy is not None:
                x1, y1, x2, y2 = crop_result.roi_bbox_xyxy
                row["roi_x1"] = x1
                row["roi_y1"] = y1
                row["roi_x2"] = x2
                row["roi_y2"] = y2

            if not crop_result.success or crop_result.roi_image is None:
                row["status"] = "roi_error"
                row["error"] = crop_result.error or "ROI crop failed"
                rows.append(row)
                continue

            if crop_result.source_bbox_image is not None:
                source_bbox_path = detection_dir / "source_bbox_crop.png"
                cv2.imwrite(str(source_bbox_path), crop_result.source_bbox_image)
                row["source_bbox_image_path"] = str(source_bbox_path.resolve())

            roi_image_path = detection_dir / "crop.png"
            cv2.imwrite(str(roi_image_path), crop_result.roi_image)
            row["roi_image_path"] = str(roi_image_path.resolve())

            fine_result = run_fine_locator(
                crop_result.roi_image,
                work_dir=fine_locator_dir,
                python_exe=config.python_exe,
                script_path=config.detect_script_path,
            )
            row["fine_locator_artifacts"] = dict(fine_result.artifacts)
            row["method"] = fine_result.method or ""
            row["used_fallback"] = fine_result.used_fallback

            if not fine_result.success or fine_result.point_xy is None or crop_result.roi_to_global_h is None:
                row["status"] = "fine_locator_error"
                row["error"] = fine_result.error or "Fine locator failed"
                rows.append(row)
                continue

            local_point_xy = [float(fine_result.point_xy[0]), float(fine_result.point_xy[1])]
            row["local_point_xy"] = local_point_xy
            row["local_x"] = local_point_xy[0]
            row["local_y"] = local_point_xy[1]

            try:
                global_point_xy = map_local_point_to_global(local_point_xy, crop_result.roi_to_global_h)
            except Exception as exc:
                row["status"] = "fine_locator_error"
                row["error"] = f"Failed to map local point to global coordinates: {exc}"
                rows.append(row)
                continue

            row["global_point_xy"] = [float(global_point_xy[0]), float(global_point_xy[1])]
            row["global_x"] = float(global_point_xy[0])
            row["global_y"] = float(global_point_xy[1])

            if is_point_inside_image(global_point_xy, image_bgr.shape):
                row["status"] = "success"
            else:
                row["status"] = "point_out_of_bounds"
                row["error"] = "Mapped point lies outside the source image bounds"

            rows.append(row)

    annotated = annotate_image(image_bgr, rows)
    annotated_path = image_output_dir / "annotated.png"
    cv2.imwrite(str(annotated_path), annotated)

    csv_path = image_output_dir / "detections.csv"
    json_path = image_output_dir / "detections.json"
    _write_csv(csv_path, rows)
    _write_json(json_path, rows)

    return {
        "image_id": image_id,
        "image_path": str(image_path),
        "output_dir": str(image_output_dir),
        "annotated_path": str(annotated_path),
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "backend_name": backend_info.backend_name,
        "used_sahi": backend_info.used_sahi,
        "rows": rows,
    }


def run_pipeline(config: PipelineConfig) -> list[dict[str, object]]:
    image_paths = _collect_input_images(config.input_path, config.max_images)
    if not image_paths:
        raise FileNotFoundError(f"No supported input images found under: {config.input_path}")

    _ensure_dir(Path(config.output_dir).resolve())

    results: list[dict[str, object]] = []
    for index, image_path in enumerate(image_paths, start=1):
        print(f"[{index}/{len(image_paths)}] Processing {image_path.name}")
        result = process_single_image(image_path, config)
        success_count = sum(1 for row in result["rows"] if row.get("status") == "success")
        print(
            f"  backend={result['backend_name']} | "
            f"rows={len(result['rows'])} | "
            f"success={success_count} | "
            f"out={result['output_dir']}"
        )
        results.append(result)

    return results
