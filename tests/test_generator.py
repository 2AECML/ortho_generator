from __future__ import annotations

from pathlib import Path
import struct

import numpy as np
from PIL import Image

from ortho_generator.cancellation import CancellationToken
from ortho_generator.geometry import EllipseArcLookup, EllipseParameters, fit_reference_ellipse
from ortho_generator.las_source import LasSource
from ortho_generator.models import PointBatch, PointCloudMetadata
from ortho_generator.service import (
    GenerationSettings,
    _adaptive_grid_shape,
    _antialias_foreground_edges,
    _build_display_raster,
    _build_small_hole_mask,
    _build_structured_row_arc_lookup,
    _close,
    _longitudinal_columns,
    _longitudinal_positions,
    generate_orthophotos,
    read_scanner_description,
    read_scanner_revolutions,
    scanner_description_path,
)


def _write_las_14(path: Path) -> None:
    header = bytearray(375)
    header[:4] = b"LASF"
    header[24:26] = bytes((1, 4))
    struct.pack_into("<H", header, 94, 375)
    struct.pack_into("<I", header, 96, 375)
    header[104] = 6
    struct.pack_into("<H", header, 105, 30)
    struct.pack_into("<I", header, 107, 0)
    struct.pack_into("<3d", header, 131, 0.01, 0.01, 0.01)
    struct.pack_into("<3d", header, 155, 100.0, 200.0, 300.0)
    struct.pack_into("<Q", header, 247, 3)
    records = bytearray(90)
    values = [(1, 2, 3, 10), (-2, 4, 6, 20), (5, -3, 7, 30)]
    for index, value in enumerate(values):
        struct.pack_into("<iiiH", records, index * 30, *value)
    path.write_bytes(header + records)


def test_las_14_point_format_6_streaming(tmp_path: Path) -> None:
    path = tmp_path / "sample.las"
    _write_las_14(path)
    source = LasSource(path)
    metadata = source.open()
    batches = list(source.iter_batches(2))
    assert metadata.point_count == 3
    assert [batch.point_count for batch in batches] == [2, 1]
    assert np.allclose(batches[0].x, [100.01, 99.98])
    assert np.allclose(batches[0].intensity, [10.0, 20.0])


def test_ellipse_arc_starts_at_invert() -> None:
    ellipse = EllipseParameters(0.0, 0.0, 2.0, 2.0, 0.0)
    arc = EllipseArcLookup(ellipse)
    value = arc.project(np.asarray([0.0]), np.asarray([-2.0]))
    assert value[0] < 1e-9


def test_automatic_diameter_uses_equivalent_ellipse_diameter() -> None:
    angle = np.linspace(0.0, 2.0 * np.pi, 2000, endpoint=False)
    points = np.column_stack((3.0 * np.cos(angle), 2.0 * np.sin(angle)))
    ellipse, report = fit_reference_ellipse(points, None)
    assert np.isclose(report["estimated_equivalent_diameter_m"], 2.0 * np.sqrt(6.0), atol=0.02)
    assert np.isclose(report["fitted_width_m"], 6.0, atol=0.02)
    assert np.isclose(report["fitted_height_m"], 4.0, atol=0.02)
    assert report["diameter_mode"] == "automatic"
    assert np.isclose(max(ellipse.semi_axis_a, ellipse.semi_axis_b), 3.0, atol=0.01)


def test_las_quantized_y_uses_nearest_pixel_centers() -> None:
    y = np.asarray([3380.0100, 3380.0149, 3380.0199, 3380.0250])
    assert _longitudinal_columns(y, 3375.0, 0.005, True).tolist() == [1002, 1003, 1004, 1005]
    assert _longitudinal_columns(y, 3375.0, 0.005, False).tolist() == [1002, 1002, 1003, 1005]


def test_adaptive_grid_uses_finest_complete_image_under_pixel_limit() -> None:
    pixel_size, rows, columns = _adaptive_grid_shape(
        circumference_m=40.0,
        longitudinal_min_m=0.0,
        longitudinal_max_m=100.0,
        requested_pixel_size_m=0.005,
        las_quantization=False,
        maximum_pixels=80_000_000,
    )
    assert rows * columns <= 80_000_000
    assert pixel_size > 0.005
    assert np.isclose(pixel_size, 0.005 * np.sqrt(2.0), rtol=0.01)


def test_scanner_revolutions_are_read_and_rounded_half_up(tmp_path: Path) -> None:
    point_cloud = tmp_path / "bdlsx1001.e57"
    point_cloud.touch()
    description = scanner_description_path(point_cloud)
    description.write_text(
        "其他说明\n计算得到的每秒圈数：95.3675圈/秒\n"
        "分辨率原始值：半断面行数=4267；扫描角度=150度\n",
        encoding="utf-8",
    )
    value, used_path = read_scanner_revolutions(point_cloud)
    assert value == 95.368
    assert used_path == description
    parsed = read_scanner_description(point_cloud)
    assert parsed.half_scan_rows == 4267
    assert parsed.half_scan_angle_degrees == 150.0
    assert parsed.total_scan_span_degrees == 300.0


def test_scanner_revolutions_support_gb18030_description(tmp_path: Path) -> None:
    point_cloud = tmp_path / "scan.las"
    point_cloud.touch()
    description = scanner_description_path(point_cloud)
    description.write_text(
        "计算得到的每秒圈数: 95.367432 圈/秒",
        encoding="gb18030",
    )
    value, _ = read_scanner_revolutions(point_cloud)
    assert value == 95.367


def test_batch_worker_continues_after_one_description_failure(tmp_path: Path) -> None:
    from PyQt5.QtCore import QCoreApplication
    import ortho_generator.app as app_module

    QCoreApplication.instance() or QCoreApplication([])
    first = tmp_path / "first.e57"
    second = tmp_path / "second.e57"
    first.touch()
    second.touch()
    scanner_description_path(first).write_text(
        "计算得到的每秒圈数：95.3675圈/秒", encoding="utf-8"
    )
    used_rates = []

    def fake_generate(source, output, settings, cancellation, progress):
        used_rates.append(settings.scanner_revolutions_per_second)
        return app_module.GenerationResult("run", "manifest", (), 10.0, 0.005, False)

    original = app_module.generate_orthophotos
    app_module.generate_orthophotos = fake_generate
    summaries = []
    try:
        worker = app_module.GenerationWorker(
            (str(first), str(second)),
            str(tmp_path / "out"),
            GenerationSettings(),
            auto_scanner_revolutions=True,
        )
        worker.succeeded.connect(summaries.append)
        worker.run()
    finally:
        app_module.generate_orthophotos = original

    assert used_rates == [95.368]
    assert len(summaries) == 1
    assert len(summaries[0].results) == 1
    assert len(summaries[0].failures) == 1
    assert Path(summaries[0].failures[0][0]).name == "second.e57"


def test_structured_e57_uses_speed_and_revolutions_for_profile_spacing() -> None:
    settings = GenerationSettings(
        vehicle_speed_mps=0.4,
        scanner_revolutions_per_second=100.0,
    )
    batch = PointBatch(
        np.zeros(3),
        np.asarray([100.0, 101.0, 102.0]),
        np.zeros(3),
        np.ones(3),
        np.zeros(3, dtype=np.uint32),
        np.asarray([0, 1, 2], dtype=np.uint32),
    )
    positions = _longitudinal_positions(batch, True, settings)
    assert settings.scan_profile_spacing_m == 0.004
    assert np.allclose(positions, [0.4, 0.404, 0.408])


def test_structured_e57_old_metric_y_falls_back_to_column_index() -> None:
    settings = GenerationSettings(
        vehicle_speed_mps=0.4,
        scanner_revolutions_per_second=100.0,
    )
    batch = PointBatch(
        np.zeros(3),
        np.asarray([0.0, 0.005, 0.010]),
        np.zeros(3),
        np.ones(3),
        np.zeros(3, dtype=np.uint32),
        np.asarray([0, 1, 2], dtype=np.uint32),
    )
    positions = _longitudinal_positions(batch, True, settings)
    assert np.allclose(positions, [0.0, 0.004, 0.008])


def test_internal_narrow_scanline_gap_is_display_only(tmp_path: Path) -> None:
    valid = np.ones((20, 20), dtype=np.uint8)
    valid[:, 0] = 0
    valid[:, 5] = 0
    valid[:, 10:14] = 0
    valid[3, 4] = 0
    result = _build_small_hole_mask(
        valid, tmp_path / "gap.npy", 0.005, 0.0075,
        CancellationToken(), allow_scanline_gaps=True,
    )
    assert np.all(result[:, 0] == 0)
    assert result[8, 5] == 1
    assert result[3, 5] == 0
    assert np.all(result[:, 10:14] == 0)
    assert valid[8, 5] == 0
    _close(result)


def test_internal_scanline_gap_uses_linear_longitudinal_resampling(tmp_path: Path) -> None:
    field = np.tile(np.arange(9, dtype=np.float32), (5, 1))
    field[:, 4] = np.nan
    valid = np.isfinite(field).astype(np.uint8)
    interpolation = _build_small_hole_mask(
        valid, tmp_path / "linear-gap.npy", 0.005, 0.0075,
        CancellationToken(), allow_scanline_gaps=True,
    )
    display, _ = _build_display_raster(
        field, interpolation, (0.0, 8.0), 0.005, 0.0075,
        CancellationToken(), suppress_impulses=False,
    )
    assert np.all(display[:, 4] == 128)
    assert np.all(np.isfinite(field[:, :4]))
    assert np.all(np.isnan(field[:, 4]))
    _close(interpolation)


def test_display_interpolation_only_selects_enclosed_small_holes(tmp_path: Path) -> None:
    valid = np.ones((30, 30), dtype=np.uint8)
    valid[15, 15] = 0
    valid[4:10, 4:10] = 0
    valid[0, 20] = 0
    interpolation = _build_small_hole_mask(
        valid,
        tmp_path / "interpolation.npy",
        pixel_size_m=0.005,
        maximum_distance_m=0.0075,
        token=CancellationToken(),
    )
    assert interpolation[15, 15] == 1
    assert interpolation[6, 6] == 0
    assert interpolation[0, 20] == 0
    _close(interpolation)


def test_display_suppresses_only_isolated_extreme_pixels() -> None:
    field = np.full((9, 9), 0.5, dtype=np.float32)
    field[4, 4] = 1.0
    field[2, 2] = 0.0
    field[6, 3:6] = 1.0
    interpolation = np.zeros(field.shape, dtype=np.uint8)
    display, count = _build_display_raster(
        field,
        interpolation,
        (0.0, 1.0),
        pixel_size_m=0.005,
        maximum_distance_m=0.0,
        token=CancellationToken(),
        suppress_impulses=True,
    )
    assert count == 2
    assert display[4, 4] == 128
    assert display[2, 2] == 128
    assert np.all(display[6, 3:6] == 255)


def test_structured_rows_get_one_stable_metric_arc_position() -> None:
    rows = np.arange(301, dtype=np.uint32)
    angle = -np.pi / 2 + np.deg2rad(300.0) * rows / (len(rows) - 1)
    # Range varies strongly, but scanner angle and source row stay exact.
    radius = np.full(len(rows), 5.0)
    x = (radius * np.cos(angle)).astype(np.float64)
    z = (radius * np.sin(angle)).astype(np.float64)
    lookup, report = _build_structured_row_arc_lookup(
        x,
        z,
        rows,
        len(rows),
        300.0,
        EllipseArcLookup(EllipseParameters(0.0, 0.0, 5.0, 5.0, 0.0)),
    )
    expected_length = 5.0 * np.deg2rad(300.0)
    assert np.all(np.diff(lookup) > 0)
    assert np.isclose(lookup[-1], expected_length, rtol=1e-4)
    assert report["scan_span_degrees"] == 300.0
    assert report["unobserved_span_degrees"] == 60.0
    assert report["angular_residual_p95_rad"] < 1e-12


def test_foreground_grey_impulse_filter_preserves_lines_and_wall() -> None:
    field = np.full((12, 12), 0.4, dtype=np.float32)
    field[3, 3] = 0.85
    field[3, 8] = 0.85
    field[8, 3:6] = 0.85
    depth = np.full(field.shape, 0.3, dtype=np.float32)
    depth[3, 8] = 0.0
    image, count = _build_display_raster(
        field, np.zeros(field.shape, dtype=np.uint8), (0, 1),
        0.005, 0, CancellationToken(), foreground_depth=depth,
    )
    assert count == 1
    assert image[3, 3] == 102
    assert image[3, 8] == 217
    assert np.all(image[8, 3:6] == 217)
    assert field[3, 3] == np.float32(0.85)


def test_antialias_softens_diagonal_stairs_without_changing_inputs() -> None:
    rr, cc = np.indices((64, 64))
    pipe = rr < (20 + cc / 3)
    image = np.where(pipe, 20, 220).astype(np.uint8)
    depth = np.where(pipe, 0.3, 0).astype(np.float32)
    before = image.copy()
    refined, mask = _antialias_foreground_edges(image, depth, np.ones(image.shape), CancellationToken())
    assert np.count_nonzero(mask) > 50
    assert np.any((refined > 20) & (refined < 220))
    assert np.array_equal(image, before)
    assert np.array_equal(mask > 0, refined != image)
    assert np.all(refined[:10] == image[:10])
    assert np.all(refined[50:] == image[50:])


def test_antialias_preserves_straight_thin_pipe_wall_and_missing_support() -> None:
    image = np.full((40, 40), 220, dtype=np.uint8)
    image[20, :] = 20
    depth = np.zeros(image.shape, dtype=np.float32)
    depth[20, :] = 0.3
    valid = np.ones(image.shape, dtype=np.uint8)
    refined, mask = _antialias_foreground_edges(image, depth, valid, CancellationToken())
    assert np.array_equal(refined, image)
    assert not np.any(mask)
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, size=image.shape, dtype=np.uint8)
    depth[:] = 0
    refined, mask = _antialias_foreground_edges(image, depth, valid, CancellationToken())
    assert np.array_equal(refined, image)
    depth[:] = rng.random(depth.shape)
    image[18:22, 18:22] = 0
    valid[18:22, 18:22] = 0
    refined, mask = _antialias_foreground_edges(image, depth, valid, CancellationToken())
    assert np.all(refined[18:22, 18:22] == 0)
    assert not np.any(mask[16:24, 16:24])


def test_antialias_disabled_and_invalid_strength() -> None:
    import pytest
    image = np.zeros((8, 8), dtype=np.uint8)
    refined, mask = _antialias_foreground_edges(image, image, image, CancellationToken(), 0)
    assert np.array_equal(refined, image)
    assert not np.any(mask)
    for strength in (-0.1, 1.1, np.nan):
        with pytest.raises(ValueError):
            GenerationSettings(las_edge_antialias_strength=strength).validate()


def test_antialias_preserves_one_pixel_diagonal_pipe_at_default_strength() -> None:
    rr, cc = np.indices((64, 64))
    pipe = rr == (20 + cc // 3)
    image = np.where(pipe, 20, 220).astype(np.uint8)
    depth = np.where(pipe, 0.3, 0).astype(np.float32)
    refined, _ = _antialias_foreground_edges(image, depth, np.ones(image.shape), CancellationToken())
    assert np.array_equal(refined < 120, pipe)


def test_rebuild_antialias_is_reproducible_and_reversible(tmp_path: Path) -> None:
    import json
    import runpy
    rr, cc = np.indices((64, 64))
    pipe = rr < (20 + cc / 3)
    field = np.where(pipe, 20, 220).astype(np.float32)
    depth = np.where(pipe, 0.3, 0).astype(np.float32)
    hits = np.ones(field.shape, dtype=np.uint32)
    np.save(tmp_path / "display_intensity.npy", field)
    np.save(tmp_path / "display_front_depth.npy", depth)
    np.save(tmp_path / "display_hit_count.npy", hits)
    metadata = {
        "image": "image.png", "shape": list(field.shape), "pixel_size_m": 0.005,
        "source_point_cloud": "test.las", "display_array": "display_intensity.npy",
        "display_hit_count": "display_hit_count.npy", "display_front_depth": "display_front_depth.npy",
        "display": {"minimum": 0, "maximum": 255},
        "display_impulse_suppression": {"enabled": False},
    }
    (tmp_path / "image.json").write_text(json.dumps(metadata), encoding="utf-8")
    Image.fromarray(field.astype(np.uint8)).save(tmp_path / "image.png")
    baseline = (tmp_path / "image.png").read_bytes()
    before = {p.name: p.read_bytes() for p in tmp_path.glob("*.npy")}
    module = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools" / "rebuild_existing_display.py"))
    module["rebuild"](tmp_path, 0, 0.5)
    refined = (tmp_path / "image.png").read_bytes()
    assert refined != baseline
    assert np.count_nonzero(np.asarray(Image.open(tmp_path / "display_antialias_mask.png"))) > 50
    module["rebuild"](tmp_path, 0)
    assert (tmp_path / "image.png").read_bytes() == refined
    module["rebuild"](tmp_path, 0, 0)
    assert (tmp_path / "image.png").read_bytes() == baseline
    assert all((tmp_path / name).read_bytes() == contents for name, contents in before.items())


def test_antialias_tile_halos_match_single_tile(monkeypatch) -> None:
    from ortho_generator import service
    rr, cc = np.indices((50, 60))
    pipe = rr < 15 + cc / 3
    image = np.where(pipe, 20, 220).astype(np.uint8)
    depth = np.where(pipe, 0.3, 0).astype(np.float32)
    valid = np.ones(image.shape, dtype=np.uint8)
    expected, expected_mask = _antialias_foreground_edges(image, depth, valid, CancellationToken())
    original_tiles = service._tiles
    monkeypatch.setattr(service, "_tiles", lambda shape: original_tiles(shape, 13, 17))
    actual, actual_mask = _antialias_foreground_edges(image, depth, valid, CancellationToken())
    assert np.array_equal(actual, expected)
    assert np.array_equal(actual_mask, expected_mask)


class SyntheticSource:
    def __init__(self, _path) -> None:
        angle = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        y = np.repeat(np.linspace(0.0, 1.0, 11), len(angle))
        self.x = np.tile(np.cos(angle), 11).astype(np.float64)
        self.y = y.astype(np.float64)
        self.z = np.tile(np.sin(angle), 11).astype(np.float64)
        self.intensity = (100.0 + 20.0 * np.tile(np.cos(angle), 11)).astype(np.float64)
        self.meta = PointCloudMetadata("synthetic.las", "LAS", len(self.x), 1, "intensity")

    def open(self):
        return self.meta

    def iter_batches(self, batch_size=1_000_000, cancellation=None):
        for start in range(0, len(self.x), 500):
            if cancellation:
                cancellation.throw_if_cancelled()
            end = min(len(self.x), start + 500)
            yield PointBatch(self.x[start:end], self.y[start:end], self.z[start:end], self.intensity[start:end])

    def close(self):
        pass


def test_generate_large_images_before_external_annotation(tmp_path: Path) -> None:
    input_path = tmp_path / "synthetic.las"
    input_path.touch()
    result = generate_orthophotos(
        input_path,
        tmp_path / "out",
        GenerationSettings(
            pixel_size_m=0.1,
            image_length_m=0.6,
            adaptive_single_image=False,
            output_auxiliary_data=True,
            flat_image_output=False,
            expected_diameter_m=2.0,
            maximum_image_pixels=100_000,
            fit_sample_count=2_000,
        ),
        source_factory=SyntheticSource,
    )
    assert len(result.image_paths) == 2
    first = Path(result.image_paths[0])
    assert first.name == "synthetic_ORTHO_00001.png"
    assert first.parent.name == "orthophotos"
    assert sorted(path.name for path in first.parent.glob("*.png")) == [
        "synthetic_ORTHO_00001.png", "synthetic_ORTHO_00002.png"
    ]
    with Image.open(first) as image:
        assert image.mode == "L"
        assert image.size[0] == 6
        assert image.size[1] in (63, 64)
    auxiliary = Path(result.output_directory) / "auxiliary" / "ORTHO_00001"
    assert (auxiliary / "intensity.tif").is_file()
    assert (auxiliary / "observed_mask.png").is_file()
    assert (auxiliary / "interpolated_mask.png").is_file()
    import json
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["status"] == "COMPLETED"
    assert np.isclose(result.estimated_diameter_m, 2.0, atol=0.02)
    assert np.isclose(manifest["estimated_diameter_m"], 2.0, atol=0.02)
    assert manifest["images"] == [
        "orthophotos/synthetic_ORTHO_00001.png",
        "orthophotos/synthetic_ORTHO_00002.png",
    ]
    assert manifest["output_layout"]["auxiliary_directory"] == "auxiliary"
    image_metadata = json.loads((auxiliary / "image.json").read_text(encoding="utf-8"))
    assert image_metadata["annotation_image"] == "../../orthophotos/synthetic_ORTHO_00001.png"
    assert (auxiliary / image_metadata["annotation_image"]).resolve() == first.resolve()


def test_adaptive_output_keeps_complete_cloud_in_one_image(tmp_path: Path) -> None:
    import json

    input_path = tmp_path / "adaptive.las"
    input_path.touch()
    result = generate_orthophotos(
        input_path,
        tmp_path / "out",
        GenerationSettings(
            pixel_size_m=0.1,
            expected_diameter_m=2.0,
            maximum_image_pixels=500,
            fit_sample_count=2_000,
        ),
        source_factory=SyntheticSource,
    )
    assert len(result.image_paths) == 1
    image_path = Path(result.image_paths[0])
    assert image_path.parent == (tmp_path / "out").resolve()
    assert not any(path.is_dir() for path in (tmp_path / "out").iterdir())
    assert Path(result.manifest_path).name == f"manifest_{image_path.stem}.json"
    assert result.resolution_adjusted
    assert result.actual_pixel_size_m > 0.1
    with Image.open(result.image_paths[0]) as image:
        assert image.width * image.height <= 500
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["primary_image"] == image_path.name
    assert manifest["output_layout"]["flat_image_output"] is True
    assert manifest["output_layout"]["auxiliary_directory"] is None
    assert manifest["output_layout"]["mode"] == "adaptive_single_image"
    assert manifest["grid"]["resolution_adjusted"] is True
    assert manifest["grid"]["pixel_size_m"] == result.actual_pixel_size_m


def test_nonflat_output_without_auxiliary_keeps_only_images_and_manifest(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "nonflat.las"
    input_path.touch()
    result = generate_orthophotos(
        input_path,
        tmp_path / "out",
        GenerationSettings(
            pixel_size_m=0.1,
            expected_diameter_m=2.0,
            output_auxiliary_data=False,
            flat_image_output=False,
            maximum_image_pixels=100_000,
            fit_sample_count=2_000,
        ),
        source_factory=SyntheticSource,
    )
    output_directory = Path(result.output_directory)
    assert Path(result.manifest_path) == output_directory / "manifest.json"
    assert (output_directory / "orthophotos").is_dir()
    assert not (output_directory / "auxiliary").exists()
    assert all(
        Path(image_path).parent == output_directory / "orthophotos"
        for image_path in result.image_paths
    )


class SourceWithGrossOutlier(SyntheticSource):
    def __init__(self, path):
        super().__init__(path)
        for name, value in (("x", 20.0), ("y", 0.0), ("z", 0.0), ("intensity", 1000.0)):
            setattr(self, name, np.append(getattr(self, name), value))
        self.meta = PointCloudMetadata("outlier.las", "LAS", len(self.x), 1, "intensity")


def test_depth_display_preserves_scientific_outlier_and_support(tmp_path: Path) -> None:
    import json
    input_path = tmp_path / "outlier.las"
    input_path.touch()
    result = generate_orthophotos(
        input_path, tmp_path / "out",
        GenerationSettings(pixel_size_m=0.1, image_length_m=2.0,
                           output_auxiliary_data=True, flat_image_output=False,
                           expected_diameter_m=2.0, fit_sample_count=2000),
        source_factory=SourceWithGrossOutlier,
    )
    directory = Path(result.output_directory) / "auxiliary" / "ORTHO_00001"
    scientific = np.load(directory / "intensity.npy")
    display = np.load(directory / "display_intensity.npy")
    hits = np.load(directory / "hit_count.npy")
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert np.nanmax(scientific) > 120
    assert np.nanmax(display) <= 120
    assert int(hits.sum()) == manifest["source"]["point_count"]
    assert manifest["las_display_aggregation"]["gross_depth_rejected_point_count"] == 1


class SourceWithPipe(SyntheticSource):
    def __init__(self, path):
        super().__init__(path)
        for name, value in (("x", 0.6), ("y", 0.0), ("z", 0.0), ("intensity", 50.0)):
            setattr(self, name, np.append(getattr(self, name), value))
        self.meta = PointCloudMetadata("pipe.las", "LAS", len(self.x), 1, "intensity")


def test_pipe_foreground_not_averaged_with_background(tmp_path: Path) -> None:
    path = tmp_path / "pipe.las"
    path.touch()
    result = generate_orthophotos(
        path, tmp_path / "out",
        GenerationSettings(pixel_size_m=0.1, image_length_m=2,
                           output_auxiliary_data=True, flat_image_output=False,
                           expected_diameter_m=2, fit_sample_count=2000),
        source_factory=SourceWithPipe,
    )
    directory = Path(result.output_directory) / "auxiliary" / "ORTHO_00001"
    scientific = np.load(directory / "intensity.npy")
    display = np.load(directory / "display_intensity.npy")
    pipe = np.isclose(display, 50)
    assert np.count_nonzero(pipe) == 1
    assert scientific[pipe][0] > 80


class StructuredSyntheticSource(SyntheticSource):
    def __init__(self, path):
        super().__init__(path)
        self.meta = PointCloudMetadata("structured.e57", "E57", len(self.x), 1,
                                       "intensity", 360, 11, True)
        self.rows = np.tile(np.arange(360, dtype=np.uint32), 11)
        self.columns = np.repeat(np.arange(11, dtype=np.uint32), 360)

    def iter_batches(self, batch_size=1000000, cancellation=None):
        for start in range(0, len(self.x), 500):
            end = min(len(self.x), start + 500)
            yield PointBatch(self.x[start:end], self.y[start:end], self.z[start:end],
                             self.intensity[start:end], self.rows[start:end], self.columns[start:end])


def test_structured_e57_does_not_enter_las_depth_path(tmp_path: Path) -> None:
    import json
    path = tmp_path / "structured.e57"
    path.touch()
    result = generate_orthophotos(
        path, tmp_path / "out",
        GenerationSettings(pixel_size_m=0.1, image_length_m=2,
                           expected_diameter_m=2, fit_sample_count=2000,
                           vehicle_speed_mps=1.0,
                           scanner_revolutions_per_second=10.0),
        source_factory=StructuredSyntheticSource,
    )
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["grid"]["rasterization"] == "actual_span_structured_scan_rows"
    assert manifest["grid"]["scan_profile_spacing_m"] == 0.1
    assert manifest["grid"]["column_axis"] == "scan_column_index_times_profile_spacing"
    assert "las_display_aggregation" not in manifest
    assert not (Path(result.output_directory) / "auxiliary" / "ORTHO_00001" / "display_intensity.npy").exists()


def test_rebuild_reproduces_las_display_sidecars_without_scientific_edits(tmp_path: Path) -> None:
    import runpy
    path = tmp_path / "pipe.las"
    path.touch()
    result = generate_orthophotos(
        path, tmp_path / "out",
        GenerationSettings(pixel_size_m=0.1, image_length_m=2,
                           output_auxiliary_data=True, flat_image_output=False,
                           expected_diameter_m=2, fit_sample_count=2000),
        source_factory=SourceWithPipe,
    )
    image_path = Path(result.image_paths[0])
    directory = Path(result.output_directory) / "auxiliary" / "ORTHO_00001"
    before_image = image_path.read_bytes()
    before_scientific = (directory / "intensity.npy").read_bytes()
    module = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools" / "rebuild_existing_display.py"))
    module["rebuild"](directory, 0.0075)
    assert image_path.read_bytes() == before_image
    assert (directory / "intensity.npy").read_bytes() == before_scientific
