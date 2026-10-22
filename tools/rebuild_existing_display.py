from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ortho_generator.cancellation import CancellationToken
from ortho_generator.service import (
    _build_display_raster,
    _antialias_foreground_edges,
    _build_small_hole_mask,
    _close,
    _coverage_preview,
    _display_coverage,
    _repair_display_dark_speckles,
    GenerationSettings,
)


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def rebuild(
    image_directory: Path, maximum_distance_m: float | None = None, antialias_strength: float | None = None,
    *, adaptive: bool | None = None, adaptive_limit_m: float | None = None,
    maximum_hole_area_pixels: int | None = None, interpolation_method: str | None = None,
    repair_dark_speckles: bool | None = None,
) -> int:
    if antialias_strength is not None and (not np.isfinite(antialias_strength) or not 0 <= antialias_strength <= 1):
        raise ValueError("抗锯齿强度必须在 0 到 1 之间。")
    image_directory = image_directory.resolve(strict=True)
    metadata_path = image_directory / "image.json"
    image_name = metadata_path.name
    existing_metadata = {}
    if metadata_path.is_file():
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        image_name = existing_metadata.get(
            "annotation_image", existing_metadata.get("image", "image.png")
        )
    image_path = (image_directory / image_name).resolve()
    raw_path = image_directory / existing_metadata.get("display_array", "intensity.npy")
    hit_path = image_directory / existing_metadata.get("display_hit_count", "display_hit_count.npy" if "display_array" in existing_metadata else "hit_count.npy")
    for required in (metadata_path, image_path, raw_path, hit_path):
        if not required.is_file():
            raise FileNotFoundError(f"缺少文件：{required}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    policy = metadata.get("display_interpolation", {})
    maximum_distance_m = float(policy.get("maximum_distance_m", 0.0075)) if maximum_distance_m is None else maximum_distance_m
    adaptive = bool(policy.get("adaptive", False)) if adaptive is None else adaptive
    adaptive_limit_m = float(policy.get("adaptive_limit_m", 0.015)) if adaptive_limit_m is None else adaptive_limit_m
    maximum_hole_area_pixels = int(policy.get("maximum_hole_area_pixels", 25)) if maximum_hole_area_pixels is None else maximum_hole_area_pixels
    interpolation_method = policy.get("interpolation_method", "nearest") if interpolation_method is None else interpolation_method
    if repair_dark_speckles is None:
        repair_dark_speckles = bool(metadata.get("display_dark_speckle_repair", {}).get("enabled", True))
    settings = GenerationSettings(
        maximum_display_interpolation_distance_m=maximum_distance_m,
        adaptive_display_interpolation=adaptive,
        adaptive_display_interpolation_limit_m=adaptive_limit_m,
        maximum_display_hole_area_pixels=maximum_hole_area_pixels,
        display_interpolation_method=interpolation_method,
        repair_display_dark_speckles=repair_dark_speckles,
    )
    settings.validate()
    search_distance_m = adaptive_limit_m if adaptive and maximum_distance_m > 0 else maximum_distance_m
    depth_tolerance_m = float(policy.get("depth_tolerance_m", metadata.get("display_depth_aggregation", {}).get("depth_cluster_m", 0.03)))
    raw = np.load(raw_path, mmap_mode="r")
    hits = np.load(hit_path, mmap_mode="r")
    if raw.shape != hits.shape or list(raw.shape) != metadata["shape"]:
        raise ValueError("科学强度、命中数和 image.json 的尺寸不一致。")

    backup_image = image_directory / f"{image_path.stem}_before_interpolation.png"
    backup_metadata = image_directory / "image_before_interpolation.json"
    if not backup_image.exists():
        shutil.copy2(image_path, backup_image)
    if not backup_metadata.exists():
        shutil.copy2(metadata_path, backup_metadata)

    work_path = image_directory / "_interpolated_rebuild.npy"
    source_path = str(metadata.get("source_point_cloud", "")).lower()
    run_manifest_path = image_directory.parents[1] / "manifest.json"
    run_manifest = (
        json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if run_manifest_path.is_file()
        else {}
    )
    allow_scanline_gaps = bool(policy.get("allow_scanline_gaps", False)) or source_path.endswith(".las") or bool(
        run_manifest.get("source", {}).get("structured_indices", False)
    )
    allow_scanrow_gaps = bool(policy.get(
        "allow_scanrow_gaps", run_manifest.get("source", {}).get("structured_indices", False),
    ))
    interpolation = _build_small_hole_mask(
        hits,
        work_path,
        float(metadata["pixel_size_m"]),
        maximum_distance_m,
        CancellationToken(),
        allow_scanline_gaps=allow_scanline_gaps,
        allow_scanrow_gaps=allow_scanrow_gaps,
        adaptive=adaptive,
        adaptive_limit_m=adaptive_limit_m,
        maximum_hole_area_pixels=maximum_hole_area_pixels,
    )
    try:
        depth_path = image_directory / metadata.get("display_front_depth", "display_front_depth.npy")
        front_depth = np.load(depth_path, mmap_mode="r") if depth_path.exists() else None
        display, suppressed_pixels = _build_display_raster(
            raw,
            interpolation,
            (
                float(metadata["display"]["minimum"]),
                float(metadata["display"]["maximum"]),
            ),
            float(metadata["pixel_size_m"]),
            search_distance_m,
            CancellationToken(),
            suppress_impulses=metadata.get("display_impulse_suppression", {}).get("enabled", True),
            foreground_depth=front_depth,
            interpolation_method=interpolation_method,
            depth_tolerance_m=depth_tolerance_m,
            allow_scanrow_gaps=allow_scanrow_gaps,
        )
        speckle_enabled = repair_dark_speckles and search_distance_m > 0
        repaired_speckles = 0
        if speckle_enabled:
            display, speckle_mask = _repair_display_dark_speckles(
                display, CancellationToken(), observed_mask=hits,
                maximum_distance_pixels=search_distance_m / float(metadata["pixel_size_m"]),
            )
            repaired_speckles = int(np.count_nonzero(speckle_mask))
            interpolation[speckle_mask > 0] = 1
            Image.fromarray(speckle_mask).save(image_directory / "display_dark_speckle_mask.png")
            del speckle_mask
        else:
            (image_directory / "display_dark_speckle_mask.png").unlink(missing_ok=True)
        strength = float(metadata.get("display_edge_antialiasing", {}).get("strength", 0)) if antialias_strength is None else antialias_strength
        antialias_pixels = 0
        if front_depth is not None and strength > 0:
            display, antialias_mask = _antialias_foreground_edges(
                display, front_depth, hits, CancellationToken(), strength,
            )
            antialias_pixels = int(np.count_nonzero(antialias_mask))
            Image.fromarray(antialias_mask).save(image_directory / "display_antialias_mask.png")
        if front_depth is not None:
            _close(front_depth)
        interpolated_pixels = int(np.count_nonzero(interpolation))
        coverage = _display_coverage(raw, hits, interpolation, float(metadata["pixel_size_m"]), CancellationToken())
        _coverage_preview(
            raw, interpolation,
            (float(metadata["display"]["minimum"]), float(metadata["display"]["maximum"])),
            CancellationToken(),
        ).save(image_directory / "coverage_preview.png")
        temporary_image = image_path.parent / f"{image_path.stem}.tmp.png"
        temporary_mask = image_directory / "interpolated_mask.tmp.png"
        Image.fromarray(display).save(temporary_image)
        Image.fromarray(np.asarray(interpolation)).save(temporary_mask)
        temporary_image.replace(image_path)
        temporary_mask.replace(image_directory / "interpolated_mask.png")
    finally:
        _close(interpolation)
        work_path.unlink(missing_ok=True)

    metadata["interpolated_mask"] = "interpolated_mask.png"
    metadata["coverage_preview"] = "coverage_preview.png"
    metadata["display_coverage"] = coverage
    metadata["display_dark_speckle_repair"] = {
        "enabled": speckle_enabled,
        "method": "small_dark_components_v1_telea_missing_only",
        "modified_pixel_count": repaired_speckles,
        "scientific_data_modified": False,
    }
    if speckle_enabled:
        metadata["display_dark_speckle_mask"] = "display_dark_speckle_mask.png"
    else:
        metadata.pop("display_dark_speckle_mask", None)
    if depth_path.exists():
        metadata["display_edge_antialiasing"] = {
            "method": "depth_gated_coherent_edge_tangent_resampling",
            "strength": strength,
            "sample_offset_pixels": 0.75,
            "modified_pixel_count": antialias_pixels,
            "scientific_data_modified": False,
        }
        if strength > 0:
            metadata["display_antialias_mask"] = "display_antialias_mask.png"
        else:
            metadata.pop("display_antialias_mask", None)
    metadata["display_interpolation"] = {
        "method": (
            "bounded_adaptive_idw_with_edge_protection"
            if interpolation_method == "idw" else
            "enclosed_small_holes_and_linear_bounded_internal_scanline_gaps"
            if allow_scanline_gaps
            else "enclosed_small_holes_nearest_valid"
        ),
        "maximum_distance_m": maximum_distance_m,
        "adaptive": adaptive,
        "adaptive_limit_m": adaptive_limit_m,
        "maximum_hole_area_pixels": maximum_hole_area_pixels,
        "interpolation_method": interpolation_method,
        "allow_scanline_gaps": allow_scanline_gaps,
        "allow_scanrow_gaps": allow_scanrow_gaps,
        "depth_tolerance_m": depth_tolerance_m,
        "maximum_neighbours": 8,
        "minimum_neighbours": 3,
        "maximum_neighbour_contrast_gray": 80,
        "interpolated_pixel_count": interpolated_pixels,
        "scientific_data_modified": False,
    }
    metadata["display_impulse_suppression"] = {
        "enabled": metadata.get("display_impulse_suppression", {}).get("enabled", True),
        "method": "isolated_3x3_extreme_and_foreground_grey_impulses" if depth_path.exists() else "isolated_3x3_bright_dark_median",
        "suppressed_pixel_count": suppressed_pixels,
        "scientific_data_modified": False,
    }
    _write_json_atomic(metadata_path, metadata)
    return interpolated_pixels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="对已有 ORTHO 目录重新生成受控小孔插值显示图。"
    )
    parser.add_argument("image_directory", type=Path)
    parser.add_argument("--maximum-distance-mm", type=float, default=None, help="基础补孔距离；默认沿用元数据，旧输出默认 7.5 mm")
    parser.add_argument("--antialias-strength", type=float, default=None, help="LAS 轮廓抗锯齿强度 0..1；0 关闭，默认沿用元数据")
    parser.add_argument("--adaptive", action=argparse.BooleanOptionalAction, default=None, help="启用/关闭局部密度自适应补孔；默认沿用元数据")
    parser.add_argument("--adaptive-limit-mm", type=float, default=None)
    parser.add_argument("--maximum-hole-area-pixels", type=int, default=None)
    parser.add_argument("--interpolation-method", choices=("nearest", "idw"), default=None)
    parser.add_argument("--repair-dark-speckles", action=argparse.BooleanOptionalAction, default=None,
                        help="启用/关闭缺测小黑孔修补；默认沿用元数据，旧输出默认开启")
    arguments = parser.parse_args()
    if arguments.maximum_distance_mm is not None and (not np.isfinite(arguments.maximum_distance_mm) or arguments.maximum_distance_mm < 0):
        parser.error("最大距离必须是有限非负数。")
    count = rebuild(
        arguments.image_directory,
        None if arguments.maximum_distance_mm is None else arguments.maximum_distance_mm / 1000.0,
        arguments.antialias_strength,
        adaptive=arguments.adaptive,
        adaptive_limit_m=None if arguments.adaptive_limit_mm is None else arguments.adaptive_limit_mm / 1000.0,
        maximum_hole_area_pixels=arguments.maximum_hole_area_pixels,
        interpolation_method=arguments.interpolation_method,
        repair_dark_speckles=arguments.repair_dark_speckles,
    )
    print(f"完成：填补 {count:,} 个显示像素；科学数据未修改。")


if __name__ == "__main__":
    main()
