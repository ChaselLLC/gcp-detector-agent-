from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import sys

from tqdm import tqdm

# Dataset root, for example:
# D:\gcp\yolov8 obb\dataset_gcp
base_dir = Path(r"D:\gcp\yolov8 obb\dataset_gcp")

# JSON source priority:
# 1. <base_dir>\jsons\<subset>
# 2. <base_dir>\labels\<subset>  (compatible with your current dataset)
JSON_DIR_CANDIDATES = ("jsons", "labels")
IMAGE_DIR_NAME = "images"
LABEL_DIR_NAME = "labels"
SUBSETS = ("train", "val")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASS_NAME_TO_ID = {
    "1": 0,
    "2": 1,
    "3": 2,
    "4": 3,
    "5": 4,
    "6": 5,
    "7": 6,
    "8": 7,
}


@dataclass
class SubsetStats:
    subset: str
    total_images: int = 0
    positive_images: int = 0
    negative_images: int = 0
    skipped_images: int = 0
    written_label_files: int = 0
    empty_label_files: int = 0
    total_objects: int = 0


def resolve_base_dir() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).expanduser().resolve()
    return base_dir.expanduser().resolve()


def find_json_root(dataset_dir: Path) -> Path:
    for candidate in JSON_DIR_CANDIDATES:
        candidate_root = dataset_dir / candidate
        if candidate_root.exists():
            return candidate_root
    raise FileNotFoundError(
        f"Could not find a JSON source directory under {dataset_dir}. "
        f"Tried: {', '.join(JSON_DIR_CANDIDATES)}"
    )


def list_image_files(images_dir: Path) -> list[Path]:
    return sorted(
        file_path
        for file_path in images_dir.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
    )


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def normalize_point(point: list[float], image_width: float, image_height: float) -> tuple[float, float]:
    x, y = point
    return clamp01(float(x) / image_width), clamp01(float(y) / image_height)


def parse_class_id(shape: dict) -> int | None:
    raw_label = shape.get("label")
    if raw_label is None:
        return None

    class_id = CLASS_NAME_TO_ID.get(str(raw_label).strip())
    return class_id


def shape_to_yolo_obb_line(shape: dict, image_width: int, image_height: int) -> str | None:
    points = shape.get("points")
    if not isinstance(points, list) or len(points) != 4:
        return None

    class_id = parse_class_id(shape)
    if class_id is None:
        return None

    normalized_values: list[str] = []
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            return None
        x_norm, y_norm = normalize_point(point, image_width, image_height)
        normalized_values.extend((f"{x_norm:.6f}", f"{y_norm:.6f}"))

    return f"{class_id} " + " ".join(normalized_values)


def convert_json_to_yolo_obb(json_path: Path) -> tuple[list[str], int]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    image_width = data.get("imageWidth")
    image_height = data.get("imageHeight")

    if not isinstance(image_width, (int, float)) or not isinstance(image_height, (int, float)):
        raise ValueError("Missing or invalid imageWidth/imageHeight")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("imageWidth/imageHeight must be positive")

    shapes = data.get("shapes")
    if not isinstance(shapes, list):
        raise ValueError("Missing or invalid shapes list")

    lines: list[str] = []
    skipped_shapes = 0

    for index, shape in enumerate(shapes, start=1):
        if not isinstance(shape, dict):
            skipped_shapes += 1
            print(f"[Warning] {json_path}: shape #{index} is not an object, skipped.")
            continue

        line = shape_to_yolo_obb_line(shape, int(image_width), int(image_height))
        if line is None:
            skipped_shapes += 1
            print(
                f"[Warning] {json_path}: shape #{index} has an invalid class label or "
                f"does not contain exactly 4 valid points, skipped."
            )
            continue

        lines.append(line)

    return lines, skipped_shapes


def process_subset(dataset_dir: Path, json_root: Path, subset: str) -> SubsetStats:
    images_dir = dataset_dir / IMAGE_DIR_NAME / subset
    json_dir = json_root / subset
    labels_dir = dataset_dir / LABEL_DIR_NAME / subset
    stats = SubsetStats(subset=subset)

    if not images_dir.exists():
        print(f"[Warning] Missing images directory, skipped: {images_dir}")
        return stats

    labels_dir.mkdir(parents=True, exist_ok=True)
    image_files = list_image_files(images_dir)

    progress_bar = tqdm(image_files, desc=f"{subset}", unit="img", ascii=True)
    for image_path in progress_bar:
        stats.total_images += 1
        stem = image_path.stem
        json_path = json_dir / f"{stem}.json"
        txt_path = labels_dir / f"{stem}.txt"

        if not json_path.exists():
            txt_path.touch()
            stats.negative_images += 1
            stats.empty_label_files += 1
            stats.written_label_files += 1
            continue

        try:
            lines, skipped_shapes = convert_json_to_yolo_obb(json_path)
        except Exception as exc:
            stats.skipped_images += 1
            print(f"[Warning] Failed to parse {json_path}: {exc}")
            continue

        if skipped_shapes > 0 and not lines:
            stats.skipped_images += 1
            print(f"[Warning] {json_path}: no valid OBB shapes found, skipped.")
            continue

        txt_content = "\n".join(lines)
        if txt_content:
            txt_content += "\n"
        txt_path.write_text(txt_content, encoding="utf-8")

        stats.positive_images += 1
        stats.written_label_files += 1
        stats.total_objects += len(lines)

    return stats


def print_subset_report(stats: SubsetStats) -> None:
    print(
        f"{stats.subset}: images={stats.total_images}, "
        f"positives={stats.positive_images}, "
        f"negatives={stats.negative_images}, "
        f"written_txt={stats.written_label_files}, "
        f"empty_txt={stats.empty_label_files}, "
        f"objects={stats.total_objects}, "
        f"skipped={stats.skipped_images}"
    )


def print_total_report(all_stats: list[SubsetStats]) -> None:
    total_images = sum(item.total_images for item in all_stats)
    total_positive = sum(item.positive_images for item in all_stats)
    total_negative = sum(item.negative_images for item in all_stats)
    total_written = sum(item.written_label_files for item in all_stats)
    total_empty = sum(item.empty_label_files for item in all_stats)
    total_objects = sum(item.total_objects for item in all_stats)
    total_skipped = sum(item.skipped_images for item in all_stats)

    print("-" * 72)
    print(
        f"Done: images={total_images}, positives={total_positive}, negatives={total_negative}, "
        f"written_txt={total_written}, empty_txt={total_empty}, objects={total_objects}, skipped={total_skipped}"
    )


def main() -> None:
    dataset_dir = resolve_base_dir()
    images_root = dataset_dir / IMAGE_DIR_NAME
    labels_root = dataset_dir / LABEL_DIR_NAME

    if not images_root.exists():
        raise FileNotFoundError(f"Missing images directory: {images_root}")

    labels_root.mkdir(parents=True, exist_ok=True)
    json_root = find_json_root(dataset_dir)

    print(f"Dataset root: {dataset_dir}")
    print(f"JSON source root: {json_root}")
    print(f"Output label root: {labels_root}")
    print("-" * 72)

    all_stats: list[SubsetStats] = []
    for subset in SUBSETS:
        stats = process_subset(dataset_dir, json_root, subset)
        all_stats.append(stats)
        print_subset_report(stats)

    print_total_report(all_stats)


if __name__ == "__main__":
    main()
