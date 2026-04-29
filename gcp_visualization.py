from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


SUCCESS_COLOR = (0, 255, 0)
WARNING_COLOR = (0, 165, 255)
ERROR_COLOR = (0, 0, 255)
POLYGON_COLOR = (0, 255, 255)
POINT_COLOR = (255, 0, 0)


def draw_polygon(image_bgr: np.ndarray, polygon_xy: Sequence[Sequence[float]], color: tuple[int, int, int], thickness: int = 2) -> None:
    if not polygon_xy:
        return
    points = np.asarray(polygon_xy, dtype=np.float32).reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(image_bgr, [points], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def draw_crosshair(
    image_bgr: np.ndarray,
    point_xy: Sequence[float],
    size: int = 12,
    color: tuple[int, int, int] = POINT_COLOR,
    thickness: int = 2,
) -> None:
    x_value = int(round(float(point_xy[0])))
    y_value = int(round(float(point_xy[1])))
    cv2.line(image_bgr, (x_value - size, y_value), (x_value + size, y_value), color, thickness, cv2.LINE_AA)
    cv2.line(image_bgr, (x_value, y_value - size), (x_value, y_value + size), color, thickness, cv2.LINE_AA)


def annotate_image(image_bgr: np.ndarray, detection_rows: Sequence[dict[str, object]]) -> np.ndarray:
    annotated = image_bgr.copy()
    if not detection_rows:
        return annotated

    for row in detection_rows:
        polygon_xy = row.get("polygon_xy") or []
        status = str(row.get("status") or "")
        if status == "success":
            color = SUCCESS_COLOR
        elif status == "point_out_of_bounds":
            color = WARNING_COLOR
        elif status == "no_detection":
            color = ERROR_COLOR
        else:
            color = POLYGON_COLOR if polygon_xy else ERROR_COLOR

        if polygon_xy:
            draw_polygon(annotated, polygon_xy, color=color, thickness=2)

        global_point_xy = row.get("global_point_xy")
        if isinstance(global_point_xy, (list, tuple)) and len(global_point_xy) == 2:
            draw_crosshair(annotated, global_point_xy, color=POINT_COLOR, thickness=2)

        label_parts = []
        det_id = row.get("det_id")
        if det_id:
            label_parts.append(str(det_id))
        class_name = row.get("class_name")
        if class_name:
            label_parts.append(str(class_name))
        confidence = row.get("conf")
        if confidence not in ("", None):
            label_parts.append(f"{float(confidence):.2f}")
        if status:
            label_parts.append(status)
        label = " ".join(label_parts).strip()

        anchor_x = 10
        anchor_y = 25
        if polygon_xy:
            points = np.asarray(polygon_xy, dtype=np.float32).reshape(-1, 2)
            anchor_x = int(np.min(points[:, 0]))
            anchor_y = max(20, int(np.min(points[:, 1])) - 8)

        if label:
            cv2.putText(
                annotated,
                label,
                (anchor_x, anchor_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    if all(str(row.get("status") or "") == "no_detection" for row in detection_rows):
        cv2.putText(
            annotated,
            "no_detection",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            ERROR_COLOR,
            2,
            cv2.LINE_AA,
        )

    return annotated
