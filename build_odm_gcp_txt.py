from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_HEADER = "+proj=tmerc +lat_0=0 +lon_0=117 +k=1 +x_0=500000 +y_0=0 +ellps=GRS80 +units=m +no_defs"
DEFAULT_IMAGES_DIR = Path(r"D:\gcp\images")
DEFAULT_REFERENCE_GCP_LIST = Path(r"C:\Users\14768\Desktop\gcp_list.txt")
DEFAULT_REVIEW_REJECTIONS_NAME = "manual_review_rejections.json"
DEFAULT_MIN_CONFIDENCE = 0.85

METHOD_PRIORITY = {
    "core_inner_midpoint": 90,
    "inner_tip_midpoint": 80,
    "contour_inner_midpoint": 70,
    "split_midpoint_global": 60,
    "waist_subpixel": 50,
    "center_constrained_corner": 40,
    "roi_centroid": 30,
    "union_centroid": 20,
    "bbox_center": 10,
    "": 0,
    None: 0,
}


@dataclass(frozen=True)
class ControlPoint:
    name: str
    normalized_name: str
    x: float
    y: float
    z: float
    source_line: str


@dataclass(frozen=True)
class DetectionRow:
    image_name: str
    point_name: str
    normalized_point_name: str
    pixel_x: float
    pixel_y: float
    confidence: float
    method: str
    used_fallback: bool
    status: str
    raw_row: dict[str, str]


def normalize_point_name(name: str) -> str:
    cleaned = str(name).strip()
    if not cleaned:
        return ""
    try:
        numeric_value = float(cleaned)
        if math.isfinite(numeric_value) and numeric_value.is_integer():
            return str(int(numeric_value))
    except ValueError:
        pass
    return cleaned


def point_name_sort_key(name: str) -> tuple[int, float, str]:
    normalized = normalize_point_name(name)
    try:
        numeric_value = float(normalized)
        if math.isfinite(numeric_value):
            return 0, numeric_value, normalized
    except ValueError:
        pass
    return 1, math.inf, normalized.casefold()


def parse_control_point_line(line: str) -> ControlPoint | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    if "," in stripped:
        tokens = [token.strip() for token in stripped.split(",")]
    else:
        tokens = re.split(r"\s+", stripped)

    tokens = [token for token in tokens if token != ""]
    if len(tokens) < 4:
        raise ValueError(f"控制点行至少需要包含编号名和 XYZ: {line.rstrip()}")

    name = tokens[0]
    try:
        x_value = float(tokens[-3])
        y_value = float(tokens[-2])
        z_value = float(tokens[-1])
    except ValueError as exc:
        raise ValueError(f"无法从控制点行解析 XYZ: {line.rstrip()}") from exc

    return ControlPoint(
        name=name,
        normalized_name=normalize_point_name(name),
        x=x_value,
        y=y_value,
        z=z_value,
        source_line=line.rstrip(),
    )


def load_control_points(control_points_path: Path) -> list[ControlPoint]:
    control_points: list[ControlPoint] = []
    for line in control_points_path.read_text(encoding="utf-8").splitlines():
        parsed = parse_control_point_line(line)
        if parsed is not None:
            control_points.append(parsed)
    if not control_points:
        raise ValueError(f"控制点文件为空或无法解析: {control_points_path}")
    return control_points


def discover_control_points_txt(images_dir: Path) -> Path:
    txt_files = sorted(path for path in images_dir.glob("*.txt") if path.is_file())
    txt_files = [path for path in txt_files if not path.name.lower().startswith("odm_")]
    if len(txt_files) != 1:
        raise FileNotFoundError(
            f"期望在 {images_dir} 中找到唯一一个控制点 txt，实际找到 {len(txt_files)} 个: {[p.name for p in txt_files]}"
        )
    return txt_files[0]


def discover_results_root(repo_dir: Path) -> Path:
    candidates = list(repo_dir.glob("images_pipeline_output_warped_*"))
    if not candidates:
        candidates = list(repo_dir.glob("images_pipeline_output_*"))
    candidates = [path for path in candidates if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"在 {repo_dir} 下未找到 images_pipeline_output* 结果目录")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_header(header_file: Path | None) -> str:
    if header_file is not None and header_file.exists():
        first_line = header_file.read_text(encoding="utf-8").splitlines()[0].strip()
        if first_line:
            return first_line
    return DEFAULT_HEADER


def load_review_rejection_keys(review_rejections_path: Path | None) -> set[tuple[str, str]]:
    if review_rejections_path is None or not review_rejections_path.exists():
        return set()

    payload = json.loads(review_rejections_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        entries = payload.get("rejections", [])
    elif isinstance(payload, list):
        entries = payload
    else:
        raise ValueError(f"人工复检排除文件格式不支持: {review_rejections_path}")

    rejected: set[tuple[str, str]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        image_id = str(entry.get("image_id") or "").strip()
        det_id = str(entry.get("det_id") or "").strip()
        if image_id and det_id:
            rejected.add((image_id, det_id))
    return rejected


def load_success_detections(
    results_root: Path,
    rejected_detection_keys: set[tuple[str, str]] | None = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> list[DetectionRow]:
    rejected_detection_keys = rejected_detection_keys or set()
    min_confidence = max(float(min_confidence), DEFAULT_MIN_CONFIDENCE)
    detections: list[DetectionRow] = []
    for csv_path in sorted(results_root.glob("*/detections.csv")):
        with csv_path.open("r", encoding="utf-8-sig", newline="") as file_obj:
            for row in csv.DictReader(file_obj):
                if (row.get("status") or "").strip() != "success":
                    continue
                image_name = (row.get("image_id") or "").strip()
                point_name = (row.get("class_name") or "").strip()
                pixel_x = (row.get("global_x") or "").strip()
                pixel_y = (row.get("global_y") or "").strip()
                if not image_name or not point_name or not pixel_x or not pixel_y:
                    continue
                det_id = (row.get("det_id") or "").strip()
                if det_id and (image_name, det_id) in rejected_detection_keys:
                    continue
                try:
                    confidence = float((row.get("conf") or "0").strip() or "0")
                except ValueError:
                    continue
                if confidence < min_confidence:
                    continue
                detections.append(
                    DetectionRow(
                        image_name=image_name,
                        point_name=point_name,
                        normalized_point_name=normalize_point_name(point_name),
                        pixel_x=float(pixel_x),
                        pixel_y=float(pixel_y),
                        confidence=confidence,
                        method=(row.get("method") or "").strip(),
                        used_fallback=(row.get("used_fallback") or "").strip().lower() == "true",
                        status=(row.get("status") or "").strip(),
                        raw_row=row,
                    )
                )
    return detections


def rank_detection(detection: DetectionRow) -> tuple[int, int, float]:
    fallback_score = 0 if detection.used_fallback else 1
    method_score = METHOD_PRIORITY.get(detection.method, 1 if detection.method else 0)
    confidence_score = float(detection.confidence)
    return fallback_score, method_score, confidence_score


def select_best_detection_per_image_point(
    detections: list[DetectionRow],
    valid_point_names: set[str],
) -> list[DetectionRow]:
    best_by_image_point: dict[tuple[str, str], DetectionRow] = {}
    for detection in detections:
        if detection.normalized_point_name not in valid_point_names:
            continue
        key = (detection.image_name, detection.normalized_point_name)
        current_best = best_by_image_point.get(key)
        if current_best is None or rank_detection(detection) > rank_detection(current_best):
            best_by_image_point[key] = detection
    return sorted(
        best_by_image_point.values(),
        key=lambda detection: (
            point_name_sort_key(detection.normalized_point_name),
            detection.image_name.casefold(),
        ),
    )


def format_odm_line(control_point: ControlPoint, detection: DetectionRow) -> str:
    return (
        f"{control_point.x:.3f} "
        f"{control_point.y:.3f} "
        f"{control_point.z:.3f} "
        f"{detection.pixel_x:.2f} "
        f"{detection.pixel_y:.2f} "
        f"{detection.image_name} "
        f"{control_point.name}"
    )


def build_output_lines(control_points: list[ControlPoint], detections: list[DetectionRow]) -> tuple[list[str], list[str]]:
    control_points_by_name = {point.normalized_name: point for point in control_points}
    output_lines: list[str] = []
    warnings: list[str] = []
    used_pairs: set[tuple[str, str]] = set()
    used_point_names: set[str] = set()

    for detection in detections:
        control_point = control_points_by_name.get(detection.normalized_point_name)
        if control_point is None:
            warnings.append(f"检测结果中的编号 {detection.point_name} 在控制点文件中不存在，已跳过: {detection.image_name}")
            continue

        pair_key = (detection.image_name, control_point.normalized_name)
        if pair_key in used_pairs:
            continue

        output_lines.append(format_odm_line(control_point, detection))
        used_pairs.add(pair_key)
        used_point_names.add(control_point.normalized_name)

    missing_names = [point.name for point in control_points if point.normalized_name not in used_point_names]
    if missing_names:
        warnings.append(f"这些控制点在当前检测结果中没有匹配成功: {', '.join(missing_names)}")

    return output_lines, warnings


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="根据 images 中的控制点 txt 和流水线检测结果，生成 ODM 可用的 GCP 文本。")
    parser.add_argument("--images-dir", default=str(DEFAULT_IMAGES_DIR), help="包含图片和控制点 txt 的目录")
    parser.add_argument("--control-points", default=None, help="控制点 txt 路径；不传则自动读取 images 目录中唯一的 txt")
    parser.add_argument("--results-root", default=None, help="流水线结果根目录；不传则自动选择最新的 images_pipeline_output*")
    parser.add_argument("--output", default=None, help="输出 ODM GCP 文本路径；不传则输出到 images/odm_gcp_list.txt")
    parser.add_argument("--header-file", default=str(DEFAULT_REFERENCE_GCP_LIST), help="用于读取第一行坐标系头的参考文件")
    parser.add_argument(
        "--review-rejections",
        default=None,
        help=(
            "人工复检排除清单 JSON；不传时若 results-root 下存在 "
            f"{DEFAULT_REVIEW_REJECTIONS_NAME}，则自动使用它"
        ),
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
        help="最小导出置信度；低于 0.85 的值会按 0.85 处理",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parent
    images_dir = Path(args.images_dir).resolve()
    control_points_path = Path(args.control_points).resolve() if args.control_points else discover_control_points_txt(images_dir)
    results_root = Path(args.results_root).resolve() if args.results_root else discover_results_root(repo_dir)
    output_path = Path(args.output).resolve() if args.output else (images_dir / "odm_gcp_list.txt").resolve()
    header_file = Path(args.header_file).resolve() if args.header_file else None
    review_rejections_path = (
        Path(args.review_rejections).resolve()
        if args.review_rejections
        else (results_root / DEFAULT_REVIEW_REJECTIONS_NAME)
    )

    header = load_header(header_file)
    control_points = load_control_points(control_points_path)
    rejected_detection_keys = load_review_rejection_keys(review_rejections_path)
    min_confidence = max(float(args.min_conf), DEFAULT_MIN_CONFIDENCE)
    all_success_detections = load_success_detections(results_root, rejected_detection_keys, min_confidence=min_confidence)
    selected_detections = select_best_detection_per_image_point(
        detections=all_success_detections,
        valid_point_names={point.normalized_name for point in control_points},
    )
    output_lines, warnings = build_output_lines(control_points, selected_detections)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_text = "\n".join([header, *output_lines]) + "\n"
    output_path.write_text(output_text, encoding="utf-8")

    print(f"控制点文件: {control_points_path}")
    print(f"检测结果目录: {results_root}")
    print(f"人工复检排除文件: {review_rejections_path if review_rejections_path.exists() else '未使用'}")
    print(f"人工复检排除数量: {len(rejected_detection_keys)}")
    print(f"最小导出置信度: {min_confidence:.2f}")
    print(f"输出文件: {output_path}")
    print(f"控制点数量: {len(control_points)}")
    print(f"成功检测数量: {len(all_success_detections)}")
    print(f"每图每点保留数量: {len(selected_detections)}")
    print(f"输出行数: {len(output_lines)}")
    if warnings:
        print("警告:")
        for warning in warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
