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
)


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def rebuild(image_directory: Path, maximum_distance_m: float, antialias_strength: float | None = None) -> int:
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
    allow_scanline_gaps = source_path.endswith(".las") or bool(
        run_manifest.get("source", {}).get("structured_indices", False)
    )
    interpolation = _build_small_hole_mask(
        hits,
        work_path,
        float(metadata["pixel_size_m"]),
        maximum_distance_m,
        CancellationToken(),
        allow_scanline_gaps=allow_scanline_gaps,
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
            maximum_distance_m,
            CancellationToken(),
            suppress_impulses=metadata.get("display_impulse_suppression", {}).get("enabled", True),
            foreground_depth=front_depth,
        )
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
            "enclosed_small_holes_and_linear_bounded_internal_scanline_gaps"
            if allow_scanline_gaps
            else "enclosed_small_holes_nearest_valid"
        ),
        "maximum_distance_m": maximum_distance_m,
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
    parser.add_argument("--maximum-distance-mm", type=float, default=7.5)
    parser.add_argument("--antialias-strength", type=float, default=None, help="LAS 轮廓抗锯齿强度 0..1；0 关闭，默认沿用元数据")
    arguments = parser.parse_args()
    if not np.isfinite(arguments.maximum_distance_mm) or arguments.maximum_distance_mm < 0:
        parser.error("最大距离必须是有限非负数。")
    count = rebuild(
        arguments.image_directory,
        arguments.maximum_distance_mm / 1000.0,
        arguments.antialias_strength,
    )
    print(f"完成：填补 {count:,} 个显示像素；科学数据未修改。")


if __name__ == "__main__":
    main()
