from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Callable

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree
import tifffile

from .cancellation import CancellationToken, ProcessingCancelledError
from .geometry import EllipseArcLookup, fit_reference_ellipse
from .sources import create_source


Progress = Callable[[str, int], None]


_SCANNER_RATE_PATTERN = re.compile(
    r"计算得到的每秒圈数\s*[：:]\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*圈\s*/\s*秒"
)
_HALF_SCAN_PATTERN = re.compile(
    r"半断面行数\s*=\s*(\d+)\s*[；;]\s*"
    r"扫描角度\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*度"
)


@dataclass(slots=True)
class GenerationSettings:
    pixel_size_m: float = 0.005
    image_length_m: float = 20.0
    adaptive_single_image: bool = True
    output_auxiliary_data: bool = False
    flat_image_output: bool = True
    expected_diameter_m: float = 5.4
    auto_estimate_diameter: bool = True
    vehicle_speed_mps: float = 0.4
    ring_width_m: float | None = None
    scanner_revolutions_per_second: float = 100.0
    structured_scan_span_degrees: float | None = None
    structured_expected_scan_rows: int | None = None
    longitudinal_min_m: float | None = None
    longitudinal_max_m: float | None = None
    display_min: float | None = None
    display_max: float | None = None
    maximum_display_interpolation_distance_m: float = 0.0075
    adaptive_display_interpolation: bool = True
    adaptive_display_interpolation_limit_m: float = 0.015
    maximum_display_hole_area_pixels: int = 25
    display_interpolation_method: str = "idw"
    suppress_display_impulses: bool = True
    repair_display_dark_speckles: bool = True
    las_depth_aware_display: bool = True
    maximum_las_display_relative_depth_m: float = 2.0
    las_display_depth_cluster_m: float = 0.03
    las_edge_antialias_strength: float = 0.5
    maximum_image_pixels: int = 80_000_000
    fit_sample_count: int = 100_000

    @property
    def scan_profile_spacing_m(self) -> float:
        return self.vehicle_speed_mps / self.scanner_revolutions_per_second

    def validate(self) -> None:
        if self.output_auxiliary_data and self.flat_image_output:
            raise ValueError("输出辅助数据时不能开启影像集中存放。")
        if not np.isfinite(self.vehicle_speed_mps) or self.vehicle_speed_mps <= 0:
            raise ValueError("小车速度必须为有限正数。")
        if self.ring_width_m is not None and (
            not np.isfinite(self.ring_width_m) or self.ring_width_m <= 0
        ):
            raise ValueError("环宽必须为有限正数。")
        if (
            not np.isfinite(self.scanner_revolutions_per_second)
            or self.scanner_revolutions_per_second <= 0
        ):
            raise ValueError("扫描仪每秒圈数必须为有限正数。")
        if self.structured_scan_span_degrees is not None and (
            not np.isfinite(self.structured_scan_span_degrees)
            or not 0 < self.structured_scan_span_degrees <= 360
        ):
            raise ValueError("结构化扫描角度必须在 0到360度之间。")
        if (
            self.structured_expected_scan_rows is not None
            and self.structured_expected_scan_rows <= 0
        ):
            raise ValueError("结构化扫描行数必须为正数。")
        if not np.isfinite(self.las_edge_antialias_strength) or not 0 <= self.las_edge_antialias_strength <= 1:
            raise ValueError("LAS 边缘抗锯齿强度必须在 0 到 1 之间。")
        for value in (self.maximum_las_display_relative_depth_m, self.las_display_depth_cluster_m):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("LAS 显示深度阈值必须为有限正数。")
        if not np.isfinite(self.pixel_size_m) or self.pixel_size_m <= 0:
            raise ValueError("像素尺寸必须为有限正数。")
        if not np.isfinite(self.image_length_m) or self.image_length_m <= 0:
            raise ValueError("每幅大图长度必须为有限正数。")
        if not np.isfinite(self.expected_diameter_m) or self.expected_diameter_m <= 0:
            raise ValueError("预期隧道直径必须为有限正数。")
        if (self.longitudinal_min_m is None) != (self.longitudinal_max_m is None):
            raise ValueError("纵向起止范围必须同时填写或同时留空。")
        if self.longitudinal_min_m is not None:
            if not np.isfinite((self.longitudinal_min_m, self.longitudinal_max_m)).all():
                raise ValueError("纵向起止范围必须是有限数值。")
            if self.longitudinal_max_m <= self.longitudinal_min_m:
                raise ValueError("纵向结束值必须大于开始值。")
        if (self.display_min is None) != (self.display_max is None):
            raise ValueError("显示上下限必须同时填写或同时留空。")
        if self.display_min is not None:
            if not np.isfinite((self.display_min, self.display_max)).all():
                raise ValueError("显示上下限必须是有限数值。")
            if self.display_max <= self.display_min:
                raise ValueError("显示上限必须大于下限。")
        if (
            not np.isfinite(self.maximum_display_interpolation_distance_m)
            or self.maximum_display_interpolation_distance_m < 0
        ):
            raise ValueError("最大显示插值距离必须是有限非负数。")
        if self.fit_sample_count < 1_000 or self.maximum_image_pixels < 1:
            raise ValueError("拟合样本数或单图像素上限无效。")
        if (
            not np.isfinite(self.adaptive_display_interpolation_limit_m)
            or self.adaptive_display_interpolation_limit_m < 0
        ):
            raise ValueError("自适应补孔上限必须是有限非负数。")
        if (
            self.adaptive_display_interpolation
            and self.maximum_display_interpolation_distance_m > 0
            and self.adaptive_display_interpolation_limit_m < self.maximum_display_interpolation_distance_m
        ):
            raise ValueError("自适应补孔上限不能小于基础插值距离。")
        if self.maximum_display_hole_area_pixels < 1:
            raise ValueError("小孔面积上限必须至少为 1 像素。")
        if self.display_interpolation_method not in ("nearest", "idw"):
            raise ValueError("补孔方式必须为 nearest 或 idw。")


@dataclass(frozen=True, slots=True)
class GenerationResult:
    output_directory: str
    manifest_path: str | None
    image_paths: tuple[str, ...]
    estimated_diameter_m: float
    actual_pixel_size_m: float
    resolution_adjusted: bool


def _available_image_path(directory: Path, filename: str) -> Path:
    """返回不覆盖已有文件的影像路径。"""
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    path = Path(filename)
    index = 2
    while True:
        candidate = directory / f"{path.stem}_{index}{path.suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def scanner_description_path(point_cloud_path: str | Path) -> Path:
    path = Path(point_cloud_path)
    return path.with_name(f"{path.stem}_扫描说明.txt")


@dataclass(frozen=True, slots=True)
class ScannerDescription:
    revolutions_per_second: float
    half_scan_rows: int | None
    half_scan_angle_degrees: float | None
    path: Path
    vehicle_speed_mps: float | None = None
    ring_width_m: float | None = None
    inner_diameter_m: float | None = None

    @property
    def total_scan_span_degrees(self) -> float | None:
        if self.half_scan_angle_degrees is None:
            return None
        return min(360.0, 2.0 * self.half_scan_angle_degrees)


def _optional_description_measurement(content: str, label: str, unit: str) -> float | None:
    """Ignore unavailable/invalid optional TLSD values in old or incomplete sidecars."""
    match = re.search(
        rf"^\s*{label}\s*[：:]\s*"
        rf"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*"
        rf"(?:{unit})\s*$",
        content,
        re.MULTILINE | re.IGNORECASE,
    )
    if match is None:
        return None
    value = float(match.group(1))
    return value if math.isfinite(value) and value > 0 else None


def read_scanner_description(point_cloud_path: str | Path) -> ScannerDescription:
    """Read scanner rate, coverage and optional TLSD parameters from its sidecar."""
    description = scanner_description_path(point_cloud_path)
    if not description.is_file():
        raise FileNotFoundError(f"未找到扫描说明文件：{description}")
    content = None
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            content = description.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    if content is None:
        raise ValueError(f"无法读取扫描说明文件编码：{description}")
    match = _SCANNER_RATE_PATTERN.search(content)
    if match is None:
        raise ValueError(
            f"扫描说明中未找到“计算得到的每秒圈数：{{数值}}圈/秒”：{description}"
        )
    value = Decimal(match.group(1))
    rounded = value.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    if rounded <= 0:
        raise ValueError(f"扫描仪每秒圈数必须为正数：{description}")
    geometry = _HALF_SCAN_PATTERN.search(content)
    half_rows = int(geometry.group(1)) if geometry else None
    half_angle = float(geometry.group(2)) if geometry else None
    if half_rows is not None and half_rows <= 0:
        raise ValueError(f"扫描说明中的半断面行数无效：{description}")
    if half_angle is not None and not 0 < half_angle <= 180:
        raise ValueError(f"扫描说明中的半断面角度无效：{description}")
    return ScannerDescription(
        float(rounded), half_rows, half_angle, description,
        vehicle_speed_mps=_optional_description_measurement(
            content, "小车速度", r"m\s*/\s*s|米\s*/\s*秒"
        ),
        ring_width_m=_optional_description_measurement(content, "环宽", "m|米"),
        inner_diameter_m=_optional_description_measurement(content, "内径", "m|米"),
    )


def settings_with_scanner_description(
    settings: GenerationSettings,
    description: ScannerDescription,
    automatic_parameters: bool,
) -> GenerationSettings:
    """Apply one input's sidecar to a fresh copy, preserving missing-field fallbacks."""
    changes = {
        "structured_scan_span_degrees": description.total_scan_span_degrees,
        "structured_expected_scan_rows": (
            None if description.half_scan_rows is None else 2 * description.half_scan_rows
        ),
    }
    if automatic_parameters:
        changes["scanner_revolutions_per_second"] = description.revolutions_per_second
        if description.vehicle_speed_mps is not None:
            changes["vehicle_speed_mps"] = description.vehicle_speed_mps
        if description.ring_width_m is not None:
            changes["ring_width_m"] = description.ring_width_m
        if description.inner_diameter_m is not None:
            changes["expected_diameter_m"] = description.inner_diameter_m
            changes["auto_estimate_diameter"] = False
    return replace(settings, **changes)


def scanner_parameter_summary(description: ScannerDescription) -> str:
    parts = [f"{description.revolutions_per_second:.3f} 圈/s"]
    if description.vehicle_speed_mps is not None:
        parts.append(f"小车速度 {description.vehicle_speed_mps:g} m/s")
    if description.ring_width_m is not None:
        parts.append(f"环宽 {description.ring_width_m:g} m（仅记录）")
    if description.inner_diameter_m is not None:
        parts.append(f"直径 {description.inner_diameter_m:g} m")
    return "；".join(parts)


def read_scanner_revolutions(point_cloud_path: str | Path) -> tuple[float, Path]:
    """Backward-compatible scanner-rate reader."""
    description = read_scanner_description(point_cloud_path)
    return description.revolutions_per_second, description.path


def _grid_shape(
    circumference_m: float,
    longitudinal_min_m: float,
    longitudinal_max_m: float,
    pixel_size_m: float,
    las_quantization: bool,
) -> tuple[int, int]:
    rows = int(np.ceil(circumference_m / pixel_size_m))
    columns = int(
        _longitudinal_columns(
            np.asarray([longitudinal_max_m]),
            longitudinal_min_m,
            pixel_size_m,
            las_quantization,
        )[0]
    ) + 1
    return rows, columns


def _adaptive_grid_shape(
    circumference_m: float,
    longitudinal_min_m: float,
    longitudinal_max_m: float,
    requested_pixel_size_m: float,
    las_quantization: bool,
    maximum_pixels: int,
) -> tuple[float, int, int]:
    """Use the finest uniform resolution whose complete image fits the limit."""
    requested_shape = _grid_shape(
        circumference_m,
        longitudinal_min_m,
        longitudinal_max_m,
        requested_pixel_size_m,
        las_quantization,
    )
    requested_pixels = requested_shape[0] * requested_shape[1]
    if requested_pixels <= maximum_pixels:
        return requested_pixel_size_m, *requested_shape

    lower = requested_pixel_size_m
    upper = requested_pixel_size_m * np.sqrt(requested_pixels / maximum_pixels)
    while True:
        upper_shape = _grid_shape(
            circumference_m,
            longitudinal_min_m,
            longitudinal_max_m,
            upper,
            las_quantization,
        )
        if upper_shape[0] * upper_shape[1] <= maximum_pixels:
            break
        upper *= 1.01
    for _ in range(64):
        middle = (lower + upper) / 2.0
        shape = _grid_shape(
            circumference_m,
            longitudinal_min_m,
            longitudinal_max_m,
            middle,
            las_quantization,
        )
        if shape[0] * shape[1] <= maximum_pixels:
            upper = middle
        else:
            lower = middle
    rows, columns = _grid_shape(
        circumference_m,
        longitudinal_min_m,
        longitudinal_max_m,
        upper,
        las_quantization,
    )
    return float(upper), rows, columns


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _close(array) -> None:
    array.flush()
    mmap = getattr(array, "_mmap", None)
    if mmap is not None:
        mmap.close()


def _selected(
    batch,
    settings: GenerationSettings,
    longitudinal_position: np.ndarray | None = None,
) -> np.ndarray:
    longitudinal = batch.y if longitudinal_position is None else longitudinal_position
    valid = (
        np.isfinite(batch.x)
        & np.isfinite(longitudinal)
        & np.isfinite(batch.z)
        & np.isfinite(batch.intensity)
    )
    if settings.longitudinal_min_m is not None:
        valid &= longitudinal >= settings.longitudinal_min_m
        valid &= longitudinal <= settings.longitudinal_max_m
    return valid


def _longitudinal_positions(batch, structured_indices: bool, settings: GenerationSettings):
    """Return metric tunnel positions for raster columns.

    Structured EasyTry E57 stores scan order in columnIndex. New conversions
    also mirror the effective global scan-column number in cartesianY so that
    segmented parts retain their global offset. Older conversions stored metric
    Y; those safely fall back to the local columnIndex sequence.
    """
    if not structured_indices:
        return batch.y.astype(np.float64, copy=False)
    if batch.column_index is None:
        raise ValueError("结构化 E57 缺少 columnIndex，无法按断面间距重建纵向坐标。")

    columns = batch.column_index.astype(np.float64)
    source_y = batch.y.astype(np.float64, copy=False)
    finite = np.isfinite(source_y)
    if np.any(finite):
        rounded_y = np.rint(source_y[finite])
        offsets = rounded_y - columns[finite]
        offset = float(np.median(offsets))
        if (
            np.max(np.abs(source_y[finite] - rounded_y)) <= 1e-6
            and np.max(np.abs(offsets - offset)) <= 1e-6
        ):
            columns = source_y
    return columns * settings.scan_profile_spacing_m


def _tiles(shape: tuple[int, int], tile_rows: int = 1024, tile_columns: int = 1024):
    for row0 in range(0, shape[0], tile_rows):
        row1 = min(shape[0], row0 + tile_rows)
        for column0 in range(0, shape[1], tile_columns):
            column1 = min(shape[1], column0 + tile_columns)
            yield np.s_[row0:row1, column0:column1]


def _read_with_halo(array: np.ndarray, destination, halo: int):
    row_slice, column_slice = destination
    row0 = max(0, row_slice.start - halo)
    row1 = min(array.shape[0], row_slice.stop + halo)
    column0 = max(0, column_slice.start - halo)
    column1 = min(array.shape[1], column_slice.stop + halo)
    block = np.asarray(array[row0:row1, column0:column1])
    core = np.s_[
        row_slice.start - row0 : row_slice.stop - row0,
        column_slice.start - column0 : column_slice.stop - column0,
    ]
    return block, core


def _bounded_internal_empty_groups(
    support: np.ndarray,
    radius: float,
):
    """Yield narrow empty runs bracketed by observations along either axis."""
    empty_columns = np.flatnonzero(~np.asarray(support, dtype=bool))
    for group in np.split(
        empty_columns, np.flatnonzero(np.diff(empty_columns) > 1) + 1
    ):
        if not len(group):
            continue
        left, right = int(group[0]) - 1, int(group[-1]) + 1
        if left < 0 or right >= len(support):
            continue
        if np.ceil(len(group) / 2) > radius + 1e-9:
            continue
        yield group, left, right


def _build_small_hole_mask(
    valid_mask: np.ndarray,
    output_path: Path,
    pixel_size_m: float,
    maximum_distance_m: float,
    token: CancellationToken,
    allow_scanline_gaps: bool = False,
    *,
    allow_scanrow_gaps: bool = False,
    adaptive: bool = False,
    adaptive_limit_m: float = 0.015,
    maximum_hole_area_pixels: int = 25,
) -> np.memmap:
    """Port of EasyTry's bounded display-only small-hole policy."""
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.uint8, shape=valid_mask.shape
    )
    output[:] = 0
    radius = maximum_distance_m / pixel_size_m
    if radius <= 0:
        output.flush()
        return output
    limit_radius = adaptive_limit_m / pixel_size_m if adaptive else radius
    row_gaps = tuple(_bounded_internal_empty_groups(
        np.any(valid_mask, axis=1), limit_radius,
    )) if allow_scanrow_gaps else ()
    # Include the density window and complete small-hole boundary across tiles.
    halo = int(np.ceil(limit_radius)) * 2 + 16 if adaptive else int(np.ceil(radius)) + 2
    for destination in _tiles(valid_mask.shape):
        token.throw_if_cancelled()
        block, core = _read_with_halo(valid_mask, destination, halo)
        valid = block.astype(bool)
        row_targets = np.zeros(valid.shape, dtype=bool)
        block_row0 = destination[0].start - core[0].start
        block_column0 = destination[1].start - core[1].start
        block_column1 = block_column0 + block.shape[1]
        for group, above, below in row_gaps:
            inside = group[(group >= block_row0) & (group < block_row0 + block.shape[0])]
            if not len(inside):
                continue
            bracketed = (
                (valid_mask[above, block_column0:block_column1] > 0)
                & (valid_mask[below, block_column0:block_column1] > 0)
            )
            row_targets[np.ix_(inside - block_row0, np.flatnonzero(bracketed))] = True
        # Close only bracketed rasterization rows for hole topology. Small
        # dropouts attached to those rows then become enclosed candidates too.
        # Distances and density still use real observations, never inferred data.
        topology_valid = valid | row_targets
        enclosed = ndimage.binary_fill_holes(topology_valid) & ~topology_valid
        distance = ndimage.distance_transform_edt(~valid)
        labels, label_count = ndimage.label(enclosed)
        if label_count:
            label_ids = np.arange(1, label_count + 1)
            maximum_distance = ndimage.maximum(
                distance, labels=labels, index=label_ids
            )
            accepted = label_ids[maximum_distance <= radius + 1e-9]
            if adaptive:
                # Occupied-cell fraction estimates spacing in the unfolded plane.
                # This is a display heuristic, not a measurement of XYZ accuracy.
                density = ndimage.uniform_filter(valid.astype(np.float32), size=31, mode="nearest")
                local_radius = np.minimum(
                    limit_radius,
                    np.maximum(radius, 2.0 / np.sqrt(np.maximum(density, 0.05))),
                )
                allowed = ndimage.minimum(local_radius, labels=labels, index=label_ids)
                areas = np.bincount(labels.ravel(), minlength=label_count + 1)[1:]
                accepted = label_ids[
                    (maximum_distance <= allowed + 1e-9)
                    & (areas <= maximum_hole_area_pixels)
                ]
                bounds = ndimage.find_objects(labels)
                accepted = np.asarray([
                    label_id for label_id in accepted
                    if max(s.stop - s.start for s in bounds[label_id - 1])
                    <= 2 * allowed[label_id - 1] + 1
                ], dtype=np.int32)
            fillable = np.isin(labels, accepted)
        else:
            fillable = np.zeros(valid.shape, dtype=bool)
        output[destination] = (fillable | row_targets)[core].astype(np.uint8)
    if allow_scanline_gaps:
        # A missing scan column touches the top/bottom image boundary, so it is
        # not an enclosed hole. Fill only narrow INTERNAL gaps bracketed by
        # real display support on both sides, never boundary/large gaps.
        column_support = np.any(valid_mask, axis=0)
        for group, left, right in _bounded_internal_empty_groups(
            column_support, limit_radius
        ):
            bracketed = (valid_mask[:, left] > 0) & (valid_mask[:, right] > 0)
            output[np.ix_(np.flatnonzero(bracketed), group)] = 1
    output.flush()
    return output


def _longitudinal_columns(y, origin: float, pixel_size: float, nearest: bool):
    position = (y - origin) / pixel_size
    return (
        np.floor(position + 0.5 + 1e-9).astype(np.int64)
        if nearest else np.floor(position + 1e-9).astype(np.int64)
    )


def _build_display_raster(
    field: np.ndarray,
    interpolation_mask: np.ndarray,
    limits: tuple[float, float],
    pixel_size_m: float,
    maximum_distance_m: float,
    token: CancellationToken,
    suppress_impulses: bool = True,
    foreground_depth: np.ndarray | None = None,
    *,
    allow_scanrow_gaps: bool = False,
    interpolation_method: str = "nearest",
    depth_tolerance_m: float = 0.03,
) -> tuple[np.ndarray, int]:
    if interpolation_method not in ("nearest", "idw"):
        raise ValueError("补孔方式必须为 nearest 或 idw。")
    display = np.zeros(field.shape, dtype=np.uint8)
    suppressed_count = 0
    halo = max(2, int(np.ceil(maximum_distance_m / pixel_size_m)) + 2)
    radius = maximum_distance_m / pixel_size_m
    column_support = np.zeros(field.shape[1], dtype=bool)
    row_support = np.zeros(field.shape[0], dtype=bool) if allow_scanrow_gaps else None
    for row0 in range(0, field.shape[0], 512):
        row1 = min(field.shape[0], row0 + 512)
        valid = np.isfinite(field[row0:row1])
        column_support |= np.any(valid, axis=0)
        if row_support is not None:
            row_support[row0:row1] = np.any(valid, axis=1)
    scanline_gaps = tuple(
        _bounded_internal_empty_groups(column_support, radius)
    )
    scanrow_gaps = tuple(_bounded_internal_empty_groups(row_support, radius)) if row_support is not None else ()
    lower, upper = limits
    for destination in _tiles(field.shape):
        token.throw_if_cancelled()
        values, core = _read_with_halo(field, destination, halo)
        interpolation, _ = _read_with_halo(
            interpolation_mask, destination, halo
        )
        source_valid = np.isfinite(values)
        depth_block = (
            _read_with_halo(foreground_depth, destination, halo)[0]
            if foreground_depth is not None else None
        )
        working = values.astype(np.float64, copy=True)
        fill_target = interpolation.astype(bool) & ~source_valid
        linear_filled = np.zeros(fill_target.shape, dtype=bool)
        scanline_targets = np.zeros(fill_target.shape, dtype=bool)
        row_slice, column_slice = destination
        block_row0 = max(0, row_slice.start - halo)
        block_row1 = min(field.shape[0], row_slice.stop + halo)
        block_column0 = max(0, column_slice.start - halo)
        block_column1 = min(field.shape[1], column_slice.stop + halo)
        for group, left, right in scanline_gaps:
            inside = group[(group >= block_column0) & (group < block_column1)]
            if not len(inside):
                continue
            left_values = np.asarray(field[block_row0:block_row1, left])
            right_values = np.asarray(field[block_row0:block_row1, right])
            bracketed = np.isfinite(left_values) & np.isfinite(right_values)
            for column in inside:
                local_column = int(column) - block_column0
                scanline_targets[:, local_column] = fill_target[:, local_column]
                take = bracketed & fill_target[:, local_column]
                if interpolation_method == "idw":
                    # Do not bridge strong radiometric or depth discontinuities.
                    take &= np.abs(left_values - right_values) <= (upper - lower) * (80.0 / 255.0)
                    if foreground_depth is not None:
                        ld = np.asarray(foreground_depth[block_row0:block_row1, left])
                        rd = np.asarray(foreground_depth[block_row0:block_row1, right])
                        take &= np.isfinite(ld) & np.isfinite(rd) & (np.abs(ld - rd) <= depth_tolerance_m)
                if not np.any(take):
                    continue
                right_weight = (int(column) - left) / (right - left)
                working[take, local_column] = (
                    left_values[take] * (1.0 - right_weight)
                    + right_values[take] * right_weight
                )
                linear_filled[take, local_column] = True
        for group, above, below in scanrow_gaps:
            inside = group[(group >= block_row0) & (group < block_row1)]
            if not len(inside):
                continue
            above_values = np.asarray(field[above, block_column0:block_column1])
            below_values = np.asarray(field[below, block_column0:block_column1])
            bracketed = np.isfinite(above_values) & np.isfinite(below_values)
            compatible = bracketed & (
                np.abs(above_values - below_values) <= (upper - lower) * (80.0 / 255.0)
            )
            if foreground_depth is not None:
                ad = np.asarray(foreground_depth[above, block_column0:block_column1])
                bd = np.asarray(foreground_depth[below, block_column0:block_column1])
                compatible &= np.isfinite(ad) & np.isfinite(bd) & (np.abs(ad - bd) <= depth_tolerance_m)
            for row in inside:
                local_row = int(row) - block_row0
                # A rejected pair must not fall back to one-sided interpolation.
                # Unpaired small holes can still use the bounded hole policy.
                scanline_targets[local_row] |= bracketed & fill_target[local_row]
                take = compatible & fill_target[local_row]
                if not np.any(take):
                    continue
                below_weight = (int(row) - above) / (below - above)
                working[local_row, take] = (
                    above_values[take] * (1.0 - below_weight)
                    + below_values[take] * below_weight
                )
                linear_filled[local_row, take] = True
        nearest_target = fill_target & ~linear_filled
        nearest_target &= ~scanline_targets
        filled = linear_filled.copy()
        if interpolation_method == "idw":
            if np.any(nearest_target) and np.any(source_valid):
                source_points = np.column_stack(np.nonzero(source_valid))
                target_points = np.column_stack(np.nonzero(nearest_target))
                tree = cKDTree(source_points)
                source_values = working[source_valid]
                source_depths = depth_block[source_valid] if depth_block is not None else None
                for start in range(0, len(target_points), 32768):
                    token.throw_if_cancelled()
                    points = target_points[start:start + 32768]
                    distances, indices = tree.query(
                        points, k=9, distance_upper_bound=radius + 1e-9,
                    )
                    # The eighth neighbour may tie with other samples. Resolve
                    # ties by row/column so changing tile boundaries cannot alter
                    # the chosen observations or the reconstructed grey value.
                    tied = np.isfinite(distances[:, 7]) & (
                        np.abs(distances[:, 7] - np.where(np.isfinite(distances[:, 8]), distances[:, 8], -1)) < 1e-9
                    )
                    distances = distances[:, :8].copy()
                    indices = indices[:, :8].copy()
                    for row in np.flatnonzero(tied):
                        candidates = np.asarray(tree.query_ball_point(
                            points[row], distances[row, 7] + 1e-9,
                        ), dtype=np.int64)
                        squared = np.sum((source_points[candidates] - points[row]) ** 2, axis=1)
                        order = np.lexsort((candidates, squared))[:8]
                        indices[row] = candidates[order]
                        distances[row] = np.sqrt(squared[order])
                    valid_neighbours = indices < len(source_values)
                    safe_indices = np.minimum(indices, len(source_values) - 1)
                    samples = source_values[safe_indices]
                    sufficient = valid_neighbours.sum(axis=1) >= 3
                    sample_min = np.min(np.where(valid_neighbours, samples, np.inf), axis=1)
                    sample_max = np.max(np.where(valid_neighbours, samples, -np.inf), axis=1)
                    sufficient &= sample_max - sample_min <= (upper - lower) * (80.0 / 255.0)
                    if source_depths is not None:
                        depths = source_depths[safe_indices]
                        valid_depths = valid_neighbours & np.isfinite(depths)
                        depth_min = np.min(np.where(valid_depths, depths, np.inf), axis=1)
                        depth_max = np.max(np.where(valid_depths, depths, -np.inf), axis=1)
                        sufficient &= (valid_depths.sum(axis=1) == valid_neighbours.sum(axis=1))
                        sufficient &= depth_max - depth_min <= depth_tolerance_m
                    weights = np.where(valid_neighbours, 1.0 / np.maximum(distances, 1e-9) ** 2, 0.0)
                    averaged = np.sum(samples * weights, axis=1) / np.maximum(weights.sum(axis=1), 1e-9)
                    rr, cc = points[sufficient].T
                    working[rr, cc] = averaged[sufficient]
                    filled[rr, cc] = True
        elif np.any(nearest_target) and np.any(source_valid):
            _, indices = ndimage.distance_transform_edt(
                ~source_valid, return_indices=True
            )
            working[nearest_target] = working[
                indices[0][nearest_target], indices[1][nearest_target]
            ]
            filled |= nearest_target
        fill_target = filled
        # Record actual successful fills, including row and edge rejections.
        interpolation_mask[destination] = filled[core].astype(np.uint8)
        display_valid = source_valid | fill_target
        if upper <= lower:
            mapped = np.where(display_valid, 127, 0).astype(np.uint8)
        else:
            normalized = np.clip((working - lower) / (upper - lower), 0.0, 1.0)
            normalized = np.where(display_valid, normalized, 0.0)
            mapped = np.rint(normalized * 255.0).astype(np.uint8)
        if suppress_impulses:
            median = ndimage.median_filter(mapped, size=3, mode="nearest")
            footprint = np.ones((3, 3), dtype=np.uint8)
            bright_support = ndimage.convolve(
                ((mapped >= 250) & display_valid).astype(np.uint8),
                footprint,
                mode="constant",
            )
            dark_support = ndimage.convolve(
                ((mapped <= 5) & display_valid).astype(np.uint8),
                footprint,
                mode="constant",
            )
            impulse = display_valid & (
                (
                    (mapped >= 250)
                    & (mapped.astype(np.int16) - median.astype(np.int16) >= 80)
                    & (bright_support == 1)
                )
                |
                (
                    (mapped <= 5)
                    & (median.astype(np.int16) - mapped.astype(np.int16) >= 80)
                    & (dark_support == 1)
                )
            )
            if foreground_depth is not None:
                depth_block, _ = _read_with_halo(foreground_depth, destination, halo)
                # Moderate grey impulses on foreground pipes are not necessarily
                # black/white saturated pixels. Require complete support and
                # no similarly coloured neighbours; short lines survive.
                similar_support = np.zeros(mapped.shape, dtype=np.uint8)
                signed = mapped.astype(np.int16)
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        neighbour = np.roll(signed, (dr, dc), axis=(0, 1))
                        neighbour_valid = np.roll(display_valid, (dr, dc), axis=(0, 1))
                        similar_support += ((np.abs(signed - neighbour) <= 24) & neighbour_valid).astype(np.uint8)
                complete_support = ndimage.convolve(display_valid.astype(np.uint8), footprint, mode="constant") == 9
                impulse |= (
                    (depth_block > 0.1) & complete_support
                    & (np.abs(signed - median.astype(np.int16)) >= 48)
                    & (similar_support == 1)
                )
            mapped[impulse] = median[impulse]
            suppressed_count += int(np.count_nonzero(impulse[core]))
        display[destination] = mapped[core]
    return display, suppressed_count


def _repair_display_dark_speckles(
    display: np.ndarray,
    token: CancellationToken,
    *,
    observed_mask: np.ndarray | None = None,
    maximum_distance_pixels: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Repair small dark dropouts in display only, preserving known observations.

    Without observation data the mask is inferred from appearance (PNG repair).
    Generation and auxiliary rebuild always supply the real display support.
    Component size and tile halos keep long lines and external gaps protected.
    """
    if display.ndim != 2 or display.dtype != np.uint8:
        raise ValueError("黑点修补需要 uint8 灰度图。")
    if observed_mask is not None and observed_mask.shape != display.shape:
        raise ValueError("观测掩膜与显示图尺寸不一致。")
    output = display.copy()
    changed = np.zeros(display.shape, dtype=np.uint8)
    if maximum_distance_pixels <= 0:
        return output, changed
    import cv2

    radius = min(3.0, maximum_distance_pixels)
    halo = 9 + 4 + int(np.ceil(radius)) + 2
    for destination in _tiles(display.shape):
        token.throw_if_cancelled()
        block, core = _read_with_halo(display, destination, halo)
        dark = (block <= 8).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
        if count <= 1:
            continue
        background = cv2.medianBlur(block, 9)
        area = stats[:, cv2.CC_STAT_AREA]
        contrast = background.astype(np.int16) - block.astype(np.int16)
        mean_contrast = np.bincount(
            labels.ravel(), weights=contrast.ravel(), minlength=count,
        ) / np.maximum(area, 1)
        mean_background = np.bincount(
            labels.ravel(), weights=background.ravel(), minlength=count,
        ) / np.maximum(area, 1)
        x, y, width, height, _ = stats.T
        selected = (
            (area <= 25) & (width <= 9) & (height <= 9)
            & (np.maximum(width, height) <= 4 * np.minimum(width, height))
            & (x > 4) & (y > 4)
            & (x + width < block.shape[1] - 4)
            & (y + height < block.shape[0] - 4)
            & (mean_contrast >= 50) & (mean_background >= 60)
        )
        selected[0] = False
        if not np.any(selected):
            continue
        distance = ndimage.distance_transform_edt(dark)
        maximum_distance = ndimage.maximum(distance, labels=labels, index=np.arange(count))
        selected &= maximum_distance <= maximum_distance_pixels + 1e-9
        candidate = selected[labels]
        if observed_mask is not None:
            support, _ = _read_with_halo(observed_mask, destination, halo)
            candidate &= support == 0
        if not np.any(candidate[core]):
            continue
        repaired = cv2.inpaint(block, candidate.astype(np.uint8) * 255, radius, cv2.INPAINT_TELEA)
        target = candidate[core]
        np.copyto(output[destination], repaired[core], where=target)
        changed[destination] = (target & (repaired[core] != block[core])).astype(np.uint8)
    return output, changed


def _display_coverage(field, hits, interpolation, pixel_size_m, token):
    """Summarize observed/filled/missing pixels without allocating full float copies."""
    observed = filled = missing = low_support = 0
    for destination in _tiles(field.shape):
        token.throw_if_cancelled()
        valid = np.isfinite(field[destination])
        observed += int(np.count_nonzero(valid))
        filled += int(np.count_nonzero(interpolation[destination]))
        missing += int(np.count_nonzero(~valid & ~interpolation[destination].astype(bool)))
        low_support += int(np.count_nonzero(valid & (hits[destination] == 1)))
    total = int(np.prod(field.shape))
    return {
        "observed_pixel_count": observed,
        "interpolated_pixel_count": filled,
        "remaining_missing_pixel_count": missing,
        "single_hit_pixel_count": low_support,
        "observed_fraction": observed / total,
        "remaining_missing_fraction": missing / total,
        "pixel_size_m": pixel_size_m,
    }


def _coverage_preview(field, interpolation, limits, token):
    """Palette diagnostic: red=missing, blue=filled, black=observed low intensity."""
    classes = np.zeros(field.shape, dtype=np.uint8)
    lower, upper = limits
    for destination in _tiles(field.shape):
        token.throw_if_cancelled()
        values = np.asarray(field[destination])
        valid = np.isfinite(values)
        block = valid.astype(np.uint8)
        if upper > lower:
            block[valid & (values <= lower + (upper - lower) * 5.0 / 255.0)] = 3
        block[interpolation[destination] > 0] = 2
        classes[destination] = block
    image = Image.fromarray(classes).convert("P")
    image.putpalette([255, 70, 70, 180, 180, 180, 0, 150, 255, 0, 0, 0] + [0] * (256 * 3 - 12))
    return image


def _antialias_foreground_edges(
    display: np.ndarray,
    foreground_depth: np.ndarray,
    valid_mask: np.ndarray,
    token: CancellationToken,
    strength: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Display-only tangent resampling at coherent foreground depth edges.

    This is contour antialiasing, not reconstruction of missing scan samples.
    Work from an immutable input with halos, so tile boundaries cannot change
    the result. Never sample across missing support or the image boundary.
    """
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("边缘抗锯齿强度必须在 0 到 1 之间。")
    if display.shape != foreground_depth.shape or display.shape != valid_mask.shape:
        raise ValueError("抗锯齿图像、深度和支持掩膜尺寸不一致。")
    output = display.copy()
    changed = np.zeros(display.shape, dtype=np.uint8)
    if strength == 0:
        return output, changed
    for destination in _tiles(display.shape):
        token.throw_if_cancelled()
        block, core = _read_with_halo(display, destination, 6)
        depth, _ = _read_with_halo(foreground_depth, destination, 6)
        support, _ = _read_with_halo(valid_mask, destination, 6)
        valid = (support > 0) & np.isfinite(depth)
        complete = ndimage.minimum_filter(valid.astype(np.uint8), size=5, mode="constant") > 0
        clean_depth = np.where(valid, depth, 0).astype(np.float32)
        depth_max = ndimage.maximum_filter(clean_depth, size=3)
        depth_min = ndimage.minimum_filter(clean_depth, size=3)
        values = block.astype(np.float32)
        gy = ndimage.sobel(values, axis=0, mode="nearest") / 8
        gx = ndimage.sobel(values, axis=1, mode="nearest") / 8
        # A local structure tensor stabilizes the tangent on pixel stairs.
        xx = ndimage.gaussian_filter(gx * gx, sigma=1, truncate=2)
        yy = ndimage.gaussian_filter(gy * gy, sigma=1, truncate=2)
        xy = ndimage.gaussian_filter(gx * gy, sigma=1, truncate=2)
        coherence = np.hypot(xx - yy, 2 * xy) / (xx + yy + 1e-6)
        normal_angle = 0.5 * np.arctan2(2 * xy, xx - yy)
        tangent_y = np.cos(normal_angle)
        tangent_x = -np.sin(normal_angle)
        contrast = ndimage.maximum_filter(values, size=3) - ndimage.minimum_filter(values, size=3)
        target = complete & (depth_max > 0.1) & (depth_max - depth_min > 0.03) & (contrast >= 40) & (coherence >= 0.6)
        rr, cc = np.nonzero(target)
        if len(rr):
            dy, dx = 0.75 * tangent_y[rr, cc], 0.75 * tangent_x[rr, cc]
            plus = ndimage.map_coordinates(values, [rr + dy, cc + dx], order=1, prefilter=False, mode="nearest")
            minus = ndimage.map_coordinates(values, [rr - dy, cc - dx], order=1, prefilter=False, mode="nearest")
            refined = values.copy()
            refined[rr, cc] = (1 - strength) * values[rr, cc] + strength * 0.5 * (plus + minus)
            refined = np.rint(np.clip(refined, 0, 255)).astype(np.uint8)
            output[destination] = refined[core]
            changed[destination] = (refined[core] != block[core]).astype(np.uint8)
    return output, changed


def _build_structured_row_arc_lookup(
    x: np.ndarray,
    z: np.ndarray,
    row_index: np.ndarray,
    scan_rows: int,
    scan_span_degrees: float,
    arc: EllipseArcLookup,
) -> tuple[np.ndarray, dict]:
    """Map observed scanner rows onto a smooth reference-section arc.

    FARO mobile scans commonly cover 300 degrees rather than a full revolution.
    The row angle comes from that actual span. Rays are intersected with the fitted
    reference ellipse, so rails, equipment and occlusion edges cannot stretch the
    output while the unobserved blind sector is not inserted into the image.
    """
    valid = (
        np.isfinite(x)
        & np.isfinite(z)
        & (row_index < scan_rows)
        & (np.hypot(x, z) > 0.1)
    )
    if np.count_nonzero(valid) < 50:
        raise ValueError("结构化扫描行的角度样本不足。")
    sample_rows = row_index[valid].astype(np.float64)
    observed_angle = np.arctan2(z[valid], x[valid])
    span_rad = float(np.deg2rad(scan_span_degrees))
    denominator = max(1, scan_rows - 1)
    nominal = span_rad * sample_rows / denominator
    candidates = []
    for direction in (1.0, -1.0):
        residual_phase = np.exp(1j * (observed_angle - direction * nominal))
        offset = float(np.angle(np.mean(residual_phase)))
        predicted = offset + direction * nominal
        residual = np.abs(np.angle(np.exp(1j * (observed_angle - predicted))))
        candidates.append((float(np.median(residual)), direction, offset, residual))
    _, direction, offset, residual = min(candidates, key=lambda item: item[0])

    source_rows = np.arange(scan_rows, dtype=np.float64)
    angles = offset + direction * span_rad * source_rows / denominator
    dx = np.cos(angles)
    dz = np.sin(angles)
    ellipse = arc.ellipse
    origin_x, origin_z = ellipse.world_to_local(
        np.asarray([0.0]), np.asarray([0.0])
    )
    cosine = np.cos(ellipse.rotation_rad)
    sine = np.sin(ellipse.rotation_rad)
    local_dx = cosine * dx + sine * dz
    local_dz = -sine * dx + cosine * dz
    a2 = ellipse.semi_axis_a**2
    b2 = ellipse.semi_axis_b**2
    quadratic_a = local_dx**2 / a2 + local_dz**2 / b2
    quadratic_b = 2.0 * (
        origin_x[0] * local_dx / a2 + origin_z[0] * local_dz / b2
    )
    quadratic_c = origin_x[0] ** 2 / a2 + origin_z[0] ** 2 / b2 - 1.0
    discriminant = quadratic_b**2 - 4.0 * quadratic_a * quadratic_c
    if np.any(discriminant < 0):
        raise ValueError("扫描原点的部分射线与拟合参考断面无交点。")
    sqrt_discriminant = np.sqrt(discriminant)
    root1 = (-quadratic_b + sqrt_discriminant) / (2.0 * quadratic_a)
    root2 = (-quadratic_b - sqrt_discriminant) / (2.0 * quadratic_a)
    distance = np.maximum(root1, root2)
    if not np.isfinite(distance).all() or np.any(distance <= 0):
        raise ValueError("扫描原点不在拟合参考断面内，无法建立稳定扫描行映射。")
    contour_x = distance * dx
    contour_z = distance * dz
    step = np.hypot(np.diff(contour_x), np.diff(contour_z))
    if not np.isfinite(step).all() or np.any(step <= 0):
        raise ValueError("结构化扫描轮廓弧长无效。")
    lookup = np.concatenate(([0.0], np.cumsum(step)))
    angular_p95 = float(np.percentile(residual, 95))
    if angular_p95 > np.deg2rad(1.0):
        raise ValueError(
            f"扫描行与扫描说明中的 {scan_span_degrees:.3f}° 范围不匹配："
            f"角度残差 P95={np.rad2deg(angular_p95):.3f}°。"
        )
    return lookup, {
        "method": "structured_row_actual_span_reference_section",
        "scan_rows": int(scan_rows),
        "scan_span_degrees": float(scan_span_degrees),
        "direction": "increasing" if direction > 0 else "decreasing",
        "angular_offset_rad": offset,
        "angular_residual_median_rad": float(np.median(residual)),
        "angular_residual_p95_rad": angular_p95,
        "reference_section_circumference_m": float(arc.circumference),
        "observed_arc_length_m": float(lookup[-1]),
        "unobserved_span_degrees": float(360.0 - scan_span_degrees),
    }


def generate_orthophotos(
    input_path: str | Path,
    output_parent: str | Path,
    settings: GenerationSettings | None = None,
    cancellation: CancellationToken | None = None,
    progress: Progress | None = None,
    source_factory=create_source,
) -> GenerationResult:
    cfg = settings or GenerationSettings()
    cfg.validate()
    token = cancellation or CancellationToken()
    emit = progress or (lambda _message, _percent: None)
    input_file = Path(input_path).resolve(strict=True)
    output_root = Path(output_parent).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = f"{input_file.stem}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    temporary_workspace: tempfile.TemporaryDirectory[str] | None = None
    if cfg.flat_image_output:
        temporary_workspace = tempfile.TemporaryDirectory(
            prefix=".easytry_ortho_", dir=output_root
        )
        work_directory = Path(temporary_workspace.name)
        result_directory = output_root
        orthophoto_directory = output_root
        auxiliary_root = work_directory / "work"
        manifest_path = work_directory / "manifest.json"
    else:
        result_directory = output_root / run_name
        result_directory.mkdir(parents=False, exist_ok=False)
        orthophoto_directory = result_directory / "orthophotos"
        if cfg.output_auxiliary_data:
            work_directory = result_directory
            auxiliary_root = result_directory / "auxiliary"
        else:
            temporary_workspace = tempfile.TemporaryDirectory(
                prefix=".easytry_ortho_", dir=result_directory
            )
            work_directory = Path(temporary_workspace.name)
            auxiliary_root = work_directory / "work"
        manifest_path = result_directory / "manifest.json"
    orthophoto_directory.mkdir(parents=True, exist_ok=True)
    auxiliary_root.mkdir(parents=True, exist_ok=True)
    state = {
        "schema_version": "tunnel-ortho-large-image-v1",
        "status": "RUNNING",
        "input_path": str(input_file),
        "settings": asdict(cfg),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "images": [],
    }
    _atomic_json(manifest_path, state)
    source = None
    try:
        emit("第一遍：检查输入并统计范围", 0)
        source = source_factory(input_file)
        metadata = source.open()
        is_las = metadata.source_format.upper() == "LAS"
        depth_display = is_las and cfg.las_depth_aware_display
        state["source"] = asdict(metadata)
        rng = np.random.default_rng(20260911)
        geometry_points = np.empty((0, 2), dtype=np.float64)
        geometry_rows = np.empty(0, dtype=np.uint32)
        geometry_keys = np.empty(0, dtype=np.float64)
        intensity_values = np.empty(0, dtype=np.float64)
        intensity_keys = np.empty(0, dtype=np.float64)
        y_min = math.inf
        y_max = -math.inf
        selected_points = 0
        read_points = 0
        maximum_radius = None if cfg.auto_estimate_diameter else cfg.expected_diameter_m * 2.0
        for batch in source.iter_batches(cancellation=token):
            token.throw_if_cancelled()
            read_points += batch.point_count
            longitudinal = _longitudinal_positions(
                batch, metadata.structured_indices, cfg
            )
            valid = _selected(batch, cfg, longitudinal)
            if np.any(valid):
                x = batch.x[valid]
                y = longitudinal[valid]
                z = batch.z[valid]
                intensity = batch.intensity[valid]
                selected_points += len(x)
                y_min = min(y_min, float(y.min()))
                y_max = max(y_max, float(y.max()))

                radius = np.hypot(x, z)
                fit = radius > 0.1
                if maximum_radius is not None:
                    fit &= radius <= maximum_radius
                candidate = np.column_stack((x[fit], z[fit]))
                candidate_rows = (
                    batch.row_index[valid][fit]
                    if metadata.structured_indices and batch.row_index is not None
                    else np.empty(0, dtype=np.uint32)
                )
                keys = rng.random(len(candidate))
                geometry_points = np.concatenate((geometry_points, candidate))
                if metadata.structured_indices:
                    geometry_rows = np.concatenate((geometry_rows, candidate_rows))
                geometry_keys = np.concatenate((geometry_keys, keys))
                if len(geometry_keys) > cfg.fit_sample_count:
                    keep = np.argpartition(geometry_keys, cfg.fit_sample_count - 1)[: cfg.fit_sample_count]
                    geometry_points = geometry_points[keep]
                    if metadata.structured_indices:
                        geometry_rows = geometry_rows[keep]
                    geometry_keys = geometry_keys[keep]

                intensity_limit = min(cfg.fit_sample_count, 200_000)
                value_keys = rng.random(len(intensity))
                intensity_values = np.concatenate((intensity_values, intensity))
                intensity_keys = np.concatenate((intensity_keys, value_keys))
                if len(intensity_keys) > intensity_limit:
                    keep = np.argpartition(intensity_keys, intensity_limit - 1)[:intensity_limit]
                    intensity_values = intensity_values[keep]
                    intensity_keys = intensity_keys[keep]
            percent = min(39, int(read_points / max(1, metadata.point_count) * 39))
            emit("第一遍：统计纵向、强度并抽样拟合", percent)
        source.close()
        source = None
        if selected_points == 0:
            raise ValueError("指定范围内没有包含有效强度的点。")
        if len(geometry_points) < 50:
            raise ValueError("有效断面样本不足，无法自动估算隧道直径。")

        if cfg.auto_estimate_diameter:
            radii = np.hypot(geometry_points[:, 0], geometry_points[:, 1])
            radial_limit = float(np.percentile(radii, 99.5))
            keep = np.isfinite(radii) & (radii <= radial_limit)
            geometry_points = geometry_points[keep]
            if metadata.structured_indices:
                geometry_rows = geometry_rows[keep]
        else:
            radial_limit = maximum_radius

        emit("拟合隧道参考椭圆", 40)
        expected = None if cfg.auto_estimate_diameter else cfg.expected_diameter_m
        ellipse, fit_report = fit_reference_ellipse(geometry_points, expected)
        fit_report["automatic_radial_sample_limit_m"] = radial_limit
        estimated_diameter = float(fit_report["estimated_equivalent_diameter_m"])
        state["estimated_diameter_m"] = estimated_diameter
        emit(
            f"隧道等效直径 {estimated_diameter:.3f} m（拟合长轴×短轴 "
            f"{fit_report['fitted_width_m']:.3f}×{fit_report['fitted_height_m']:.3f} m）",
            40,
        )
        if fit_report["warning"]:
            emit("警告：" + fit_report["warning"], 40)
        arc = EllipseArcLookup(ellipse)
        structured_row_lookup = None
        structured_mapping = None
        if metadata.structured_indices:
            if metadata.scan_rows is None or len(geometry_rows) != len(geometry_points):
                raise ValueError("结构化 E57 缺少有效的扫描行范围。")
            if (
                cfg.structured_expected_scan_rows is not None
                and metadata.scan_rows != cfg.structured_expected_scan_rows
            ):
                raise ValueError(
                    f"E57 扫描行数 {metadata.scan_rows} 与扫描说明中的 "
                    f"{cfg.structured_expected_scan_rows} 不一致。"
                )
            scan_span_degrees = cfg.structured_scan_span_degrees or 360.0
            if cfg.structured_scan_span_degrees is None:
                emit(
                    "警告：未从扫描说明获取扫描角度，结构化 E57 暂按 360° 处理。",
                    40,
                )
            structured_row_lookup, structured_mapping = _build_structured_row_arc_lookup(
                geometry_points[:, 0],
                geometry_points[:, 1],
                geometry_rows,
                metadata.scan_rows,
                scan_span_degrees,
                arc,
            )
            transverse_extent_m = float(structured_mapping["observed_arc_length_m"])
            emit(
                f"结构化断面按实际 {scan_span_degrees:.3f}° 扫描范围展开，"
                f"参考断面弧长 {transverse_extent_m:.3f} m。",
                40,
            )
        else:
            transverse_extent_m = arc.circumference
        requested_row_count, requested_column_count = _grid_shape(
            transverse_extent_m, y_min, y_max, cfg.pixel_size_m, is_las
        )
        if cfg.adaptive_single_image:
            effective_pixel_size_m, row_count, column_count = _adaptive_grid_shape(
                transverse_extent_m,
                y_min,
                y_max,
                cfg.pixel_size_m,
                is_las,
                cfg.maximum_image_pixels,
            )
            image_columns = column_count
        else:
            effective_pixel_size_m = cfg.pixel_size_m
            row_count, column_count = requested_row_count, requested_column_count
            image_columns = max(1, int(round(cfg.image_length_m / effective_pixel_size_m)))
            image_columns = min(image_columns, column_count)
            if row_count * image_columns > cfg.maximum_image_pixels:
                raise ValueError(
                    f"单幅图约 {row_count * image_columns:,} 像素，超过上限 "
                    f"{cfg.maximum_image_pixels:,}；请缩短大图长度或增大像素尺寸。"
                )
        resolution_adjusted = effective_pixel_size_m > cfg.pixel_size_m * (1.0 + 1e-12)
        if resolution_adjusted:
            emit(
                f"整图预计 {requested_row_count * requested_column_count:,} 像素，"
                f"超过上限 {cfg.maximum_image_pixels:,}；像素大小由 "
                f"{cfg.pixel_size_m * 1000:.3f} mm 自动调整为 "
                f"{effective_pixel_size_m * 1000:.3f} mm。",
                41,
            )
        total_pixels = row_count * column_count
        required_bytes = total_pixels * (48 if depth_display else 20) + 100 * 1024 * 1024
        storage_root = output_root if cfg.output_auxiliary_data else work_directory
        if shutil.disk_usage(storage_root).free < required_bytes:
            raise ValueError(f"磁盘空间不足，预计至少需要 {required_bytes / 1024**3:.2f} GB。")

        if cfg.display_min is None:
            display_min, display_max = np.percentile(intensity_values, (1.0, 99.0)).astype(float)
            if display_max <= display_min:
                display_min = float(np.min(intensity_values))
                display_max = float(np.max(intensity_values))
            display_method = "sample_percentile_1_99"
        else:
            display_min = float(cfg.display_min)
            display_max = float(cfg.display_max)
            display_method = "user_fixed"

        specs = []
        for index, column0 in enumerate(range(0, column_count, image_columns), start=1):
            column1 = min(column_count, column0 + image_columns)
            image_id = f"ORTHO_{index:05d}"
            directory = auxiliary_root / image_id
            directory.mkdir()
            shape = (row_count, column1 - column0)
            sums = np.lib.format.open_memmap(directory / "_sum.npy", mode="w+", dtype=np.float64, shape=shape)
            hits = np.lib.format.open_memmap(directory / "hit_count.npy", mode="w+", dtype=np.uint32, shape=shape)
            sums[:] = 0.0
            hits[:] = 0
            _close(sums)
            _close(hits)
            if depth_display:
                depth = np.lib.format.open_memmap(directory / "_depth_max.npy", mode="w+", dtype=np.float32, shape=shape)
                depth[:] = -np.inf
                _close(depth)
                for name, dtype in (("_display_sum.npy", np.float64), ("display_hit_count.npy", np.uint32)):
                    array = np.lib.format.open_memmap(directory / name, mode="w+", dtype=dtype, shape=shape)
                    array[:] = 0
                    _close(array)
            specs.append((directory, image_id, column0, column1, shape))

        state.update(
            status="RASTERIZING",
            selected_point_count=selected_points,
            ellipse=ellipse.to_dict(),
            ellipse_fit=fit_report,
            grid={
                "requested_pixel_size_m": cfg.pixel_size_m,
                "pixel_size_m": effective_pixel_size_m,
                "resolution_adjusted": resolution_adjusted,
                "resolution_scale": effective_pixel_size_m / cfg.pixel_size_m,
                "requested_row_count": requested_row_count,
                "requested_column_count": requested_column_count,
                "vehicle_speed_mps": cfg.vehicle_speed_mps,
                "scanner_revolutions_per_second": cfg.scanner_revolutions_per_second,
                "scan_profile_spacing_m": cfg.scan_profile_spacing_m,
                "row_count": row_count,
                "column_count": column_count,
                "circumference_m": arc.circumference,
                "transverse_extent_m": transverse_extent_m,
                "longitudinal_min_m": y_min,
                "longitudinal_max_m": y_max,
                "row_axis": (
                    "actual_span_structured_reference_section_arc"
                    if structured_row_lookup is not None
                    else "ellipse_arc_length_from_invert_counterclockwise"
                ),
                "column_axis": (
                    "scan_column_index_times_profile_spacing"
                    if metadata.structured_indices
                    else "tunnel_longitudinal_y"
                ),
                "column_quantization": "nearest_pixel_center" if is_las else "floor",
                "rasterization": (
                    "actual_span_structured_scan_rows"
                    if structured_row_lookup is not None
                    else "per_point_ellipse_projection"
                ),
            },
            display={"method": display_method, "minimum": display_min, "maximum": display_max},
            output_layout={
                "mode": "adaptive_single_image" if cfg.adaptive_single_image else "fixed_resolution_tiles",
                "orthophoto_directory": "." if cfg.flat_image_output else "orthophotos",
                "auxiliary_directory": "auxiliary" if cfg.output_auxiliary_data else None,
                "output_auxiliary_data": cfg.output_auxiliary_data,
                "flat_image_output": cfg.flat_image_output,
                "orthophoto_pattern": f"{input_file.stem}_ORTHO_{{sequence:05d}}.png",
            },
        )
        if structured_mapping is not None:
            state["structured_row_mapping"] = structured_mapping
            state["display_scanrow_resampling"] = {
                "enabled": cfg.maximum_display_interpolation_distance_m > 0,
                "method": "bounded_internal_rows_linear_with_edge_protection",
                "maximum_distance_m": (
                    cfg.adaptive_display_interpolation_limit_m
                    if cfg.adaptive_display_interpolation and cfg.maximum_display_interpolation_distance_m > 0
                    else cfg.maximum_display_interpolation_distance_m
                ),
                "scientific_data_modified": False,
            }
        _atomic_json(manifest_path, state)

        emit("第二遍：流式生成大幅正射栅格", 42)
        source = source_factory(input_file)
        source.open()
        read_points = 0
        projected_points = 0
        for batch in source.iter_batches(cancellation=token):
            token.throw_if_cancelled()
            read_points += batch.point_count
            longitudinal = _longitudinal_positions(
                batch, metadata.structured_indices, cfg
            )
            selected = _selected(batch, cfg, longitudinal)
            if np.any(selected):
                x = batch.x[selected]
                y = longitudinal[selected]
                z = batch.z[selected]
                values = batch.intensity[selected]
                if structured_row_lookup is not None:
                    assert batch.row_index is not None
                    source_rows = batch.row_index[selected]
                    valid_source_rows = source_rows < len(structured_row_lookup)
                    row_arc = np.full(len(source_rows), np.nan, dtype=np.float64)
                    row_arc[valid_source_rows] = structured_row_lookup[
                        source_rows[valid_source_rows]
                    ]
                    rows = np.floor(row_arc / effective_pixel_size_m).astype(np.int64)
                else:
                    rows = np.floor(arc.project(x, z) / effective_pixel_size_m).astype(np.int64)
                columns = _longitudinal_columns(y, y_min, effective_pixel_size_m, is_las)
                depths = ellipse.signed_relative_depth(x, z) if depth_display else None
                valid = (
                    (rows >= 0)
                    & (rows < row_count)
                    & (columns >= 0)
                    & (columns < column_count)
                    & np.isfinite(values)
                )
                rows = rows[valid]
                columns = columns[valid]
                values = values[valid]
                if depths is not None:
                    depths = depths[valid]
                projected_points += len(rows)
                destinations = columns // image_columns
                for destination in np.unique(destinations):
                    directory, _, column0, _, _ = specs[int(destination)]
                    take = destinations == destination
                    sums = np.load(directory / "_sum.npy", mmap_mode="r+")
                    hits = np.load(directory / "hit_count.npy", mmap_mode="r+")
                    np.add.at(sums, (rows[take], columns[take] - column0), values[take])
                    np.add.at(hits, (rows[take], columns[take] - column0), 1)
                    _close(sums)
                    _close(hits)
                    if depth_display:
                        depth = np.load(directory / "_depth_max.npy", mmap_mode="r+")
                        eligible = take & np.isfinite(depths) & (np.abs(depths) <= cfg.maximum_las_display_relative_depth_m)
                        np.maximum.at(depth, (rows[eligible], columns[eligible] - column0), depths[eligible])
                        _close(depth)
            percent = 42 + min(47, int(read_points / max(1, metadata.point_count) * 47))
            emit("第二遍：聚合正射像素", percent)
        source.close()
        source = None

        display_rejected_points = 0
        display_selected_points = 0
        if depth_display:
            emit("LAS 显示层：选择管线前景深度簇（科学强度不修改）", 89)
            source = source_factory(input_file)
            source.open()
            for batch in source.iter_batches(cancellation=token):
                longitudinal = _longitudinal_positions(
                    batch, metadata.structured_indices, cfg
                )
                selected = _selected(batch, cfg, longitudinal)
                x, z, values = (
                    getattr(batch, name)[selected]
                    for name in ("x", "z", "intensity")
                )
                y = longitudinal[selected]
                depths = ellipse.signed_relative_depth(x, z)
                rows = np.floor(arc.project(x, z) / effective_pixel_size_m).astype(np.int64)
                columns = _longitudinal_columns(y, y_min, effective_pixel_size_m, True)
                in_grid = (rows >= 0) & (rows < row_count) & (columns >= 0) & (columns < column_count)
                eligible = in_grid & np.isfinite(depths) & (np.abs(depths) <= cfg.maximum_las_display_relative_depth_m)
                display_rejected_points += int(np.count_nonzero(in_grid & ~eligible))
                for destination in np.unique(columns[eligible] // image_columns):
                    directory, _, column0, _, _ = specs[int(destination)]
                    take = eligible & (columns // image_columns == destination)
                    rr, cc = rows[take], columns[take] - column0
                    depth = np.load(directory / "_depth_max.npy", mmap_mode="r")
                    keep = depths[take] >= depth[rr, cc].astype(np.float64) - cfg.las_display_depth_cluster_m - 1e-6
                    display_selected_points += int(np.count_nonzero(keep))
                    sums = np.load(directory / "_display_sum.npy", mmap_mode="r+")
                    hits = np.load(directory / "display_hit_count.npy", mmap_mode="r+")
                    np.add.at(sums, (rr[keep], cc[keep]), values[take][keep])
                    np.add.at(hits, (rr[keep], cc[keep]), 1)
                    _close(depth)
                    _close(sums)
                    _close(hits)
            source.close()
            source = None
            state["las_display_aggregation"] = {
                "method": "frontmost_depth_cluster_mean",
                "maximum_absolute_relative_depth_m": cfg.maximum_las_display_relative_depth_m,
                "depth_cluster_m": cfg.las_display_depth_cluster_m,
                "gross_depth_rejected_point_count": display_rejected_points,
                "selected_point_count": display_selected_points,
                "scientific_data_modified": False,
            }

        image_paths: list[str] = []
        total_interpolated_pixels = 0
        for index, (directory, image_id, column0, column1, shape) in enumerate(specs, start=1):
            token.throw_if_cancelled()
            emit(f"写出大图 {index}/{len(specs)}", 89 + int(index / len(specs) * 10))
            sums = np.load(directory / "_sum.npy", mmap_mode="r")
            hits = np.load(directory / "hit_count.npy", mmap_mode="r")
            raw = np.lib.format.open_memmap(directory / "intensity.npy", mode="w+", dtype=np.float32, shape=shape)
            observed = np.zeros(shape, dtype=np.uint8)
            for row0 in range(0, row_count, 256):
                token.throw_if_cancelled()
                row1 = min(row_count, row0 + 256)
                count = np.asarray(hits[row0:row1])
                valid = count > 0
                block = np.full(count.shape, np.nan, dtype=np.float32)
                np.divide(sums[row0:row1], count, out=block, where=valid)
                raw[row0:row1] = block
                observed[row0:row1] = valid.astype(np.uint8)
            raw.flush()
            display_field = raw
            display_hits = hits
            if depth_display:
                display_hits = np.load(directory / "display_hit_count.npy", mmap_mode="r")
                display_sums = np.load(directory / "_display_sum.npy", mmap_mode="r")
                display_field = np.lib.format.open_memmap(directory / "display_intensity.npy", mode="w+", dtype=np.float32, shape=shape)
                for destination in _tiles(shape):
                    count = np.asarray(display_hits[destination])
                    block = np.full(count.shape, np.nan, dtype=np.float32)
                    np.divide(display_sums[destination], count, out=block, where=count > 0)
                    display_field[destination] = block
                display_field.flush()
                _close(display_sums)
            interpolation_path = directory / "_interpolated.npy"
            interpolation_distance_m = cfg.maximum_display_interpolation_distance_m
            if cfg.adaptive_display_interpolation and interpolation_distance_m > 0:
                interpolation_distance_m = cfg.adaptive_display_interpolation_limit_m
            interpolation = _build_small_hole_mask(
                display_hits,
                interpolation_path,
                effective_pixel_size_m,
                cfg.maximum_display_interpolation_distance_m,
                token,
                allow_scanline_gaps=is_las or metadata.structured_indices,
                allow_scanrow_gaps=metadata.structured_indices,
                adaptive=cfg.adaptive_display_interpolation,
                adaptive_limit_m=cfg.adaptive_display_interpolation_limit_m,
                maximum_hole_area_pixels=cfg.maximum_display_hole_area_pixels,
            )
            display_front_depth = np.load(directory / "_depth_max.npy", mmap_mode="r") if depth_display else None
            display, suppressed_pixels = _build_display_raster(
                display_field,
                interpolation,
                (display_min, display_max),
                effective_pixel_size_m,
                interpolation_distance_m,
                token,
                cfg.suppress_display_impulses,
                display_front_depth,
                allow_scanrow_gaps=metadata.structured_indices,
                interpolation_method=cfg.display_interpolation_method,
                depth_tolerance_m=cfg.las_display_depth_cluster_m,
            )
            speckle_enabled = cfg.repair_display_dark_speckles and interpolation_distance_m > 0
            repaired_speckles = 0
            if speckle_enabled:
                display, speckle_mask = _repair_display_dark_speckles(
                    display, token, observed_mask=display_hits,
                    maximum_distance_pixels=interpolation_distance_m / effective_pixel_size_m,
                )
                repaired_speckles = int(np.count_nonzero(speckle_mask))
                interpolation[speckle_mask > 0] = 1
                if cfg.output_auxiliary_data:
                    Image.fromarray(speckle_mask).save(directory / "display_dark_speckle_mask.png")
                del speckle_mask
            speckle_metadata = {
                "enabled": speckle_enabled,
                "method": "small_dark_components_v1_telea_missing_only",
                "modified_pixel_count": repaired_speckles,
                "scientific_data_modified": False,
            }
            state.setdefault("display_dark_speckle_repair", []).append({"image_id": image_id, **speckle_metadata})
            interpolated_pixels = int(np.count_nonzero(interpolation))
            total_interpolated_pixels += interpolated_pixels
            coverage = _display_coverage(display_field, display_hits, interpolation, effective_pixel_size_m, token)
            state.setdefault("display_coverage", []).append({"image_id": image_id, **coverage})
            emit(
                f"{image_id}：真实观测 {coverage['observed_fraction']:.1%}，"
                f"补孔 {interpolated_pixels:,} 像素（其中小黑孔修补 {repaired_speckles:,} 像素），"
                f"剩余缺测 {coverage['remaining_missing_fraction']:.1%}",
                89 + int(index / len(specs) * 10),
            )
            antialias_pixels = 0
            if display_front_depth is not None and cfg.las_edge_antialias_strength > 0:
                display, antialias_mask = _antialias_foreground_edges(
                    display, display_front_depth, display_hits, token,
                    cfg.las_edge_antialias_strength,
                )
                antialias_pixels = int(np.count_nonzero(antialias_mask))
                if cfg.output_auxiliary_data:
                    Image.fromarray(antialias_mask).save(directory / "display_antialias_mask.png")
            if display_front_depth is not None:
                _close(display_front_depth)
            tiff_path = directory / "intensity.tif"
            if cfg.output_auxiliary_data:
                tifffile.imwrite(
                    tiff_path,
                    raw,
                    bigtiff=raw.nbytes >= 3_500_000_000,
                    tile=(256, 256),
                    compression=None,
                    photometric="minisblack",
                    metadata={"axes": "YX", "unit": "source intensity", "pixel_size_m": effective_pixel_size_m},
                )
            annotation_name = f"{input_file.stem}_{image_id}.png"
            image_path = (
                _available_image_path(orthophoto_directory, annotation_name)
                if cfg.flat_image_output
                else orthophoto_directory / annotation_name
            )
            Image.fromarray(display).save(image_path)
            if cfg.output_auxiliary_data:
                Image.fromarray(observed).save(directory / "observed_mask.png")
                if depth_display:
                    Image.fromarray((display_hits > 0).astype(np.uint8)).save(directory / "display_support_mask.png")
                Image.fromarray(np.asarray(interpolation)).save(
                    directory / "interpolated_mask.png"
                )
                _coverage_preview(display_field, interpolation, (display_min, display_max), token).save(
                    directory / "coverage_preview.png"
                )
            image_metadata = {
                "schema_version": "tunnel-ortho-image-v1",
                "image_id": image_id,
                "sequence_index": index,
                "sequence_count": len(specs),
                "source_point_cloud": str(input_file),
                "shape": list(shape),
                "requested_pixel_size_m": cfg.pixel_size_m,
                "pixel_size_m": effective_pixel_size_m,
                "global_window": {"row0": 0, "row1": row_count, "column0": column0, "column1": column1},
                "longitudinal_bounds_m": [y_min + column0 * effective_pixel_size_m, y_min + (column1 - 1) * effective_pixel_size_m],
                "image": Path(os.path.relpath(image_path, directory)).as_posix(),
                "annotation_image": Path(os.path.relpath(image_path, directory)).as_posix(),
                "scientific_tiff": tiff_path.name,
                "scientific_array": "intensity.npy",
                "observed_mask": "observed_mask.png",
                "interpolated_mask": "interpolated_mask.png",
                "hit_count": "hit_count.npy",
                "display": state["display"],
                "display_interpolation": {
                    "method": (
                        "bounded_adaptive_idw_with_edge_protection"
                        if cfg.display_interpolation_method == "idw" else
                        "enclosed_small_holes_and_linear_bounded_internal_scanline_gaps"
                        if is_las or metadata.structured_indices
                        else "enclosed_small_holes_nearest_valid"
                    ),
                    "maximum_distance_m": cfg.maximum_display_interpolation_distance_m,
                    "adaptive": cfg.adaptive_display_interpolation,
                    "adaptive_limit_m": cfg.adaptive_display_interpolation_limit_m,
                    "maximum_hole_area_pixels": cfg.maximum_display_hole_area_pixels,
                    "interpolation_method": cfg.display_interpolation_method,
                    "allow_scanline_gaps": is_las or metadata.structured_indices,
                    "allow_scanrow_gaps": metadata.structured_indices,
                    "depth_tolerance_m": cfg.las_display_depth_cluster_m,
                    "maximum_neighbours": 8,
                    "minimum_neighbours": 3,
                    "maximum_neighbour_contrast_gray": 80,
                    "interpolated_pixel_count": interpolated_pixels,
                    "scientific_data_modified": False,
                },
                "display_impulse_suppression": {
                    "enabled": cfg.suppress_display_impulses,
                    "method": "isolated_3x3_extreme_and_foreground_grey_impulses" if depth_display else "isolated_3x3_bright_dark_median",
                    "suppressed_pixel_count": suppressed_pixels,
                    "scientific_data_modified": False,
                },
                "display_dark_speckle_repair": speckle_metadata,
                "ellipse": ellipse.to_dict(),
                "display_coverage": coverage,
                "coverage_preview": "coverage_preview.png",
            }
            if cfg.output_auxiliary_data and speckle_enabled:
                image_metadata["display_dark_speckle_mask"] = "display_dark_speckle_mask.png"
            if depth_display:
                image_metadata["display_edge_antialiasing"] = {
                    "method": "depth_gated_coherent_edge_tangent_resampling",
                    "strength": cfg.las_edge_antialias_strength,
                    "sample_offset_pixels": 0.75,
                    "modified_pixel_count": antialias_pixels,
                    "scientific_data_modified": False,
                }
                if cfg.las_edge_antialias_strength > 0:
                    image_metadata["display_antialias_mask"] = "display_antialias_mask.png"
                image_metadata["display_depth_aggregation"] = state["las_display_aggregation"]
                image_metadata["display_array"] = "display_intensity.npy"
                image_metadata["display_support_mask"] = "display_support_mask.png"
                image_metadata["display_hit_count"] = "display_hit_count.npy"
                image_metadata["display_front_depth"] = "display_front_depth.npy"
            if cfg.output_auxiliary_data:
                _atomic_json(directory / "image.json", image_metadata)
            _close(raw)
            _close(sums)
            _close(hits)
            _close(interpolation)
            if depth_display:
                _close(display_field)
                _close(display_hits)
                if cfg.output_auxiliary_data:
                    (directory / "_depth_max.npy").replace(directory / "display_front_depth.npy")
                    (directory / "_display_sum.npy").unlink()
            (directory / "_sum.npy").unlink()
            interpolation_path.unlink()
            relative_image = image_path.relative_to(result_directory).as_posix()
            state["images"].append(relative_image)
            image_paths.append(str(image_path.resolve()))
            _atomic_json(manifest_path, state)

        state.update(
            status="COMPLETED",
            projected_point_count=projected_points,
            display_interpolated_pixel_count=total_interpolated_pixels,
            completed_at=datetime.now().isoformat(timespec="seconds"),
        )
        result_manifest_path: Path | None
        if cfg.flat_image_output:
            result_manifest_path = None
            for image_path_text in image_paths:
                image_path = Path(image_path_text)
                image_manifest_path = image_path.with_name(
                    f"manifest_{image_path.stem}.json"
                )
                image_state = dict(state)
                image_state["primary_image"] = image_path.name
                _atomic_json(image_manifest_path, image_state)
                if result_manifest_path is None:
                    result_manifest_path = image_manifest_path
        else:
            _atomic_json(manifest_path, state)
            result_manifest_path = manifest_path
        emit(f"完成：生成 {len(image_paths)} 幅大图", 100)
        return GenerationResult(
            str(result_directory),
            str(result_manifest_path) if result_manifest_path is not None else None,
            tuple(image_paths),
            estimated_diameter,
            effective_pixel_size_m,
            resolution_adjusted,
        )
    except Exception as error:
        state.update(
            status="CANCELLED" if isinstance(error, ProcessingCancelledError) else "FAILED",
            error=str(error),
            finished_at=datetime.now().isoformat(timespec="seconds"),
        )
        _atomic_json(manifest_path, state)
        raise
    finally:
        if source is not None:
            source.close()
        if temporary_workspace is not None:
            try:
                temporary_workspace.cleanup()
            except OSError:
                pass
