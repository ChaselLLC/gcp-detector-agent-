from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SliceInfo:
    slice_row: int
    slice_col: int
    slice_x_offset: int
    slice_y_offset: int
    slice_width: int
    slice_height: int
    source_slice_name: str

    def to_dict(self) -> dict[str, int | str]:
        return {
            "slice_row": self.slice_row,
            "slice_col": self.slice_col,
            "slice_x_offset": self.slice_x_offset,
            "slice_y_offset": self.slice_y_offset,
            "slice_width": self.slice_width,
            "slice_height": self.slice_height,
            "source_slice_name": self.source_slice_name,
        }


@dataclass
class RawSliceDetection:
    raw_id: str
    class_id: int
    class_name: str
    confidence: float
    polygon_xy: list[list[float]]
    aabb_xyxy: tuple[int, int, int, int]
    slice_info: SliceInfo


@dataclass
class DetectionRecord:
    det_id: str
    class_id: int
    class_name: str
    confidence: float
    polygon_xy: list[list[float]]
    aabb_xyxy: tuple[int, int, int, int]
    slice_info: dict[str, int | str]
    merged_from: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "det_id": self.det_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": float(self.confidence),
            "polygon_xy": [[float(point[0]), float(point[1])] for point in self.polygon_xy],
            "aabb_xyxy": [int(value) for value in self.aabb_xyxy],
            "slice_info": dict(self.slice_info),
            "merged_from": list(self.merged_from),
        }
