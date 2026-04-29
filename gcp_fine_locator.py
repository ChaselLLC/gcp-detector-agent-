from __future__ import annotations

import ast
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


DEFAULT_PYTHON_EXE = r"D:\Anaconda_envs\envs\gcp\python.exe"
DEFAULT_SCRIPT_PATH = str(Path(__file__).with_name("detect_gcp_hourglass.py"))
DEFAULT_INPUT_NAME = "crop_input.png"

SUMMARY_PATTERN = re.compile(
    r"SUMMARY\|bbox=(?P<bbox>.*?)\|point=(?P<point>.*?)\|method=(?P<method>.*?)\|used_fallback=(?P<used_fallback>.*)"
)
RESULT_BBOX_PATTERN = re.compile(r"数字外接框坐标 \(x1, y1, x2, y2\): (?P<bbox>.+)")
RESULT_POINT_PATTERN = re.compile(r"沙漏交点最终坐标 \(x, y\): \((?P<point>[^)]+)\)")
RESULT_METHOD_PATTERN = re.compile(r"交点定位方法: (?P<method>.+)")

ARTIFACT_FILENAMES = [
    "number_crop.jpg",
    "mask_red_raw.png",
    "mask_red_clean.png",
    "mask_hourglass_branch.png",
    "mask_digit_branch.png",
    "mask_digit_loose.png",
    "mask_digit_residual.png",
    "hourglass_union_mask.png",
    "hourglass_mask.png",
    "final_result.png",
    "debug_overview.png",
    "run.log",
]


@dataclass
class FineLocateResult:
    success: bool
    returncode: int
    point_xy: tuple[float, float] | None
    digit_bbox_xyxy: tuple[int, int, int, int] | None
    method: str | None
    used_fallback: bool | None
    warnings: list[str] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    artifacts: dict[str, str] = field(default_factory=dict)
    work_dir: str = ""
    error: str | None = None


def _parse_point_text(point_text: str | None) -> tuple[float, float] | None:
    if not point_text:
        return None
    cleaned = point_text.strip()
    if not cleaned or cleaned == "None":
        return None
    parts = [part.strip() for part in cleaned.split(",")]
    if len(parts) != 2:
        return None
    return float(parts[0]), float(parts[1])


def _parse_bbox_text(bbox_text: str | None) -> tuple[int, int, int, int] | None:
    if not bbox_text:
        return None
    cleaned = bbox_text.strip()
    if not cleaned or cleaned == "None":
        return None
    value = ast.literal_eval(cleaned)
    if not isinstance(value, tuple) or len(value) != 4:
        return None
    return tuple(int(item) for item in value)


def _extract_summary(stdout_text: str) -> dict[str, str]:
    match = SUMMARY_PATTERN.search(stdout_text)
    if match is None:
        return {"bbox": "", "point": "", "method": "", "used_fallback": ""}
    return {key: value.strip() for key, value in match.groupdict().items()}


def _extract_warnings(combined_text: str) -> list[str]:
    warnings: list[str] = []
    capture = False
    for raw_line in combined_text.splitlines():
        line = raw_line.strip()
        if "========== 警告信息 ==========" in line:
            capture = True
            continue
        if capture and line.startswith("=========="):
            break
        if capture and line.startswith("- "):
            warnings.append(line[2:].strip())
    return warnings


def _collect_artifacts(work_dir: Path) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    for filename in ARTIFACT_FILENAMES:
        path = work_dir / filename
        if path.exists():
            artifacts[filename] = str(path.resolve())
    return artifacts


def run_fine_locator(
    crop_bgr: np.ndarray,
    work_dir: str | os.PathLike[str],
    python_exe: str = DEFAULT_PYTHON_EXE,
    script_path: str = DEFAULT_SCRIPT_PATH,
) -> FineLocateResult:
    work_dir_path = Path(work_dir)
    work_dir_path.mkdir(parents=True, exist_ok=True)
    crop_path = work_dir_path / DEFAULT_INPUT_NAME
    ok = cv2.imwrite(str(crop_path), crop_bgr)
    if not ok:
        return FineLocateResult(
            success=False,
            returncode=-1,
            point_xy=None,
            digit_bbox_xyxy=None,
            method=None,
            used_fallback=None,
            work_dir=str(work_dir_path.resolve()),
            error="Failed to write ROI crop image",
        )
    return run_fine_locator_on_path(crop_path, work_dir_path, python_exe=python_exe, script_path=script_path)


def run_fine_locator_on_path(
    crop_path: str | os.PathLike[str],
    work_dir: str | os.PathLike[str],
    python_exe: str = DEFAULT_PYTHON_EXE,
    script_path: str = DEFAULT_SCRIPT_PATH,
) -> FineLocateResult:
    crop_path = Path(crop_path).resolve()
    work_dir_path = Path(work_dir).resolve()
    work_dir_path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["GCP_SINGLE_IMAGE"] = str(crop_path)
    env["GCP_DISABLE_WINDOWS"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    process = subprocess.run(
        [python_exe, script_path],
        cwd=str(work_dir_path),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    stdout_text = process.stdout or ""
    stderr_text = process.stderr or ""
    combined_text = stdout_text if not stderr_text else f"{stdout_text}\n{stderr_text}" if stdout_text else stderr_text

    run_log_path = work_dir_path / "run.log"
    run_log_path.write_text(combined_text, encoding="utf-8")

    summary = _extract_summary(stdout_text)
    bbox_text = summary["bbox"]
    if not bbox_text:
        bbox_match = RESULT_BBOX_PATTERN.search(stdout_text)
        bbox_text = bbox_match.group("bbox").strip() if bbox_match else ""

    point_text = summary["point"]
    if not point_text:
        point_match = RESULT_POINT_PATTERN.search(stdout_text)
        point_text = point_match.group("point").strip() if point_match else ""

    method = summary["method"] or None
    if method is None:
        method_match = RESULT_METHOD_PATTERN.search(stdout_text)
        method = method_match.group("method").strip() if method_match else None

    used_fallback: bool | None = None
    fallback_text = summary["used_fallback"]
    if fallback_text:
        used_fallback = fallback_text.strip().lower() == "true"

    point_xy = _parse_point_text(point_text)
    digit_bbox_xyxy = _parse_bbox_text(bbox_text)
    warnings = _extract_warnings(combined_text)
    artifacts = _collect_artifacts(work_dir_path)

    success = process.returncode == 0 and point_xy is not None
    error = None
    if process.returncode != 0:
        error = f"Fine locator exited with code {process.returncode}"
    elif point_xy is None:
        error = "Failed to parse point_xy from fine locator output"

    return FineLocateResult(
        success=success,
        returncode=process.returncode,
        point_xy=point_xy,
        digit_bbox_xyxy=digit_bbox_xyxy,
        method=method,
        used_fallback=used_fallback,
        warnings=warnings,
        stdout=stdout_text,
        stderr=stderr_text,
        artifacts=artifacts,
        work_dir=str(work_dir_path),
        error=error,
    )
