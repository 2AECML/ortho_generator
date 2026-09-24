from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class PointCloudMetadata:
    source_path: str
    source_format: str
    point_count: int
    scan_count: int
    intensity_source: str
    scan_rows: int | None = None
    scan_columns: int | None = None
    structured_indices: bool = False

    @property
    def source_name(self) -> str:
        return Path(self.source_path).name


@dataclass(slots=True)
class PointBatch:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    z: NDArray[np.float64]
    intensity: NDArray[np.float64]
    row_index: NDArray[np.uint32] | None = None
    column_index: NDArray[np.uint32] | None = None

    def __post_init__(self) -> None:
        lengths = {len(self.x), len(self.y), len(self.z), len(self.intensity)}
        if len(lengths) != 1:
            raise ValueError("点批次中的坐标和强度数组必须等长。")
        for name in ("x", "y", "z", "intensity"):
            array = getattr(self, name)
            if array.dtype != np.float64:
                raise TypeError(f"{name} 必须为 float64。")
        for name in ("row_index", "column_index"):
            array = getattr(self, name)
            if array is not None:
                if len(array) != len(self.x):
                    raise ValueError(f"{name} 必须与坐标数组等长。")
                if array.dtype != np.uint32:
                    raise TypeError(f"{name} 必须为 uint32。")

    @property
    def point_count(self) -> int:
        return len(self.x)
