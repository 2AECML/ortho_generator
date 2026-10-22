from dataclasses import replace
from pathlib import Path

import pytest

from ortho_generator.service import (
    GenerationSettings,
    read_scanner_description,
    scanner_description_path,
    settings_with_scanner_description,
)


def _write_description(source: Path, extra: str = "", encoding: str = "utf-8") -> None:
    source.touch()
    scanner_description_path(source).write_text(
        "计算得到的每秒圈数：95.3675圈/秒\n"
        "分辨率原始值：半断面行数=4267；扫描角度=150度\n" + extra,
        encoding=encoding,
    )


@pytest.fixture(scope="session")
def qt_application():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication

    application = QApplication.instance() or QApplication([])
    yield application


@pytest.mark.parametrize("encoding", ["utf-8-sig", "gb18030"])
def test_read_tlsd_measurements_from_matching_sidecar(tmp_path: Path, encoding: str) -> None:
    source = tmp_path / "scan.e57"
    _write_description(
        source, "小车速度：0.75 m/s\n环宽：1.2 m\n内径：5.8 m\n", encoding
    )
    # A neighbouring project's parameters must never be used.
    _write_description(tmp_path / "other.e57", "小车速度：9 m/s\n内径：20 m\n")

    description = read_scanner_description(source)

    assert description.vehicle_speed_mps == 0.75
    assert description.ring_width_m == 1.2
    assert description.inner_diameter_m == 5.8
    assert description.revolutions_per_second == 95.368
    assert description.total_scan_span_degrees == 300


@pytest.mark.parametrize("value", ["0", "-1", "NaN", "Infinity", "1e999", "未读取到"])
def test_unavailable_optional_values_keep_manual_fallback(tmp_path: Path, value: str) -> None:
    source = tmp_path / "scan.e57"
    _write_description(source, f"小车速度：{value} m/s\n环宽：{value} m\n内径：{value} m\n")
    base = GenerationSettings(vehicle_speed_mps=0.3, expected_diameter_m=6.2)

    parsed = read_scanner_description(source)
    actual = settings_with_scanner_description(base, parsed, True)

    assert parsed.vehicle_speed_mps is None
    assert parsed.ring_width_m is None
    assert parsed.inner_diameter_m is None
    assert actual.vehicle_speed_mps == 0.3
    assert actual.expected_diameter_m == 6.2
    assert actual.auto_estimate_diameter is True
    assert actual.scanner_revolutions_per_second == 95.368


def test_legacy_and_manual_modes_preserve_existing_parameters(tmp_path: Path) -> None:
    source = tmp_path / "old.e57"
    _write_description(source)
    baseline = GenerationSettings(vehicle_speed_mps=0.32, expected_diameter_m=6.4)
    legacy = settings_with_scanner_description(baseline, read_scanner_description(source), True)
    assert legacy.vehicle_speed_mps == baseline.vehicle_speed_mps
    assert legacy.expected_diameter_m == baseline.expected_diameter_m
    assert legacy.auto_estimate_diameter == baseline.auto_estimate_diameter
    assert legacy.ring_width_m is None

    _write_description(source, "小车速度：0.75 m/s\n环宽：1.2 m\n内径：5.8 m\n")
    manual = settings_with_scanner_description(baseline, read_scanner_description(source), False)
    assert manual.vehicle_speed_mps == baseline.vehicle_speed_mps
    assert manual.expected_diameter_m == baseline.expected_diameter_m
    assert manual.auto_estimate_diameter == baseline.auto_estimate_diameter
    assert manual.scanner_revolutions_per_second == baseline.scanner_revolutions_per_second
    assert manual.ring_width_m is None
    assert manual.structured_scan_span_degrees == 300


def test_applied_tlsd_values_are_saved_in_generation_manifest(tmp_path: Path) -> None:
    import json
    import numpy as np
    from ortho_generator.models import PointBatch, PointCloudMetadata
    from ortho_generator.service import generate_orthophotos

    source = tmp_path / "scan.las"
    _write_description(source, "小车速度：0.75 m/s\n环宽：1.2 m\n内径：2 m\n")

    class SyntheticSource:
        def __init__(self, path):
            angles = np.tile(np.linspace(0, 2 * np.pi, 360, endpoint=False), 11)
            self.batch = PointBatch(
                np.cos(angles), np.repeat(np.linspace(0, 1, 11), 360),
                np.sin(angles), np.full(angles.size, 100.0),
            )

        def open(self):
            return PointCloudMetadata(str(source), "LAS", self.batch.point_count, 1, "intensity")

        def iter_batches(self, batch_size=1_000_000, cancellation=None):
            yield self.batch

        def close(self):
            pass

    settings = settings_with_scanner_description(
        GenerationSettings(
            pixel_size_m=0.1, fit_sample_count=2_000,
            output_auxiliary_data=True, flat_image_output=False,
        ),
        read_scanner_description(source), True,
    )
    result = generate_orthophotos(source, tmp_path / "out", settings, source_factory=SyntheticSource)
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert len(result.image_paths) == 1
    assert Path(result.image_paths[0]).is_file()
    assert manifest["settings"]["vehicle_speed_mps"] == 0.75
    assert manifest["settings"]["ring_width_m"] == 1.2
    assert manifest["settings"]["expected_diameter_m"] == 2.0
    assert manifest["settings"]["auto_estimate_diameter"] is False
    assert manifest["ellipse_fit"]["diameter_mode"] == "manual_expected"


@pytest.mark.parametrize("task_mode", [False, True])
def test_each_batch_input_gets_its_own_metadata_and_missing_field_fallback(
    tmp_path: Path, monkeypatch, qt_application, task_mode: bool
) -> None:
    import ortho_generator.app as app
    from ortho_generator.task_list import OrthoTaskSpec

    first, second, legacy = (tmp_path / f"{name}.e57" for name in ("a", "b", "c"))
    _write_description(first, "小车速度：0.75 m/s\n环宽：1.2 m\n内径：5.8 m\n")
    _write_description(second, "小车速度：0.6 m/s\n环宽：1.5 m\n内径：6 m\n")
    _write_description(legacy)
    baseline = GenerationSettings(vehicle_speed_mps=0.31, expected_diameter_m=6.4)
    used = []

    def generate(source, output, settings, cancellation, progress):
        used.append((Path(source).name, replace(settings)))
        image = tmp_path / f"{Path(source).stem}.png"
        image.touch()
        return app.GenerationResult(str(tmp_path), None, (str(image),), 5.8, 0.005, False)

    monkeypatch.setattr(app, "generate_orthophotos", generate)
    monkeypatch.setattr(app, "has_safe_free_space", lambda _: (True, 1_000_000_000_000))
    if task_mode:
        worker = app.TaskListGenerationWorker(
            (OrthoTaskSpec(str(tmp_path / "*.e57"), str(tmp_path / "out")),),
            baseline, True, False,
        )
    else:
        worker = app.GenerationWorker(
            tuple(map(str, (first, second, legacy))), str(tmp_path / "out"), baseline, True
        )
    summaries = []
    worker.succeeded.connect(summaries.append)
    worker.run()

    assert len(summaries) == 1
    assert summaries[0].failures == ()
    assert [name for name, _ in used] == ["a.e57", "b.e57", "c.e57"]
    assert [settings.vehicle_speed_mps for _, settings in used] == [0.75, 0.6, 0.31]
    assert [settings.expected_diameter_m for _, settings in used] == [5.8, 6.0, 6.4]
    assert [settings.auto_estimate_diameter for _, settings in used] == [False, False, True]
    assert [settings.ring_width_m for _, settings in used] == [1.2, 1.5, None]
    assert baseline == GenerationSettings(vehicle_speed_mps=0.31, expected_diameter_m=6.4)


def test_ui_preview_restores_manual_baseline_on_input_and_mode_changes(
    tmp_path: Path, qt_application
) -> None:
    from ortho_generator.app import MainWindow

    first, legacy = tmp_path / "first.e57", tmp_path / "legacy.e57"
    _write_description(first, "小车速度：0.75 m/s\n环宽：1.2 m\n内径：5.8 m\n")
    _write_description(legacy)
    window = MainWindow()
    try:
        window.coordinate_confirmed.setChecked(True)
        window.vehicle_speed.setValue(0.31)
        window.diameter.setValue(6.4)
        window.auto_diameter.setChecked(True)
        window.source_edit.setPlainText(f"{first}\n{legacy}")
        assert window.vehicle_speed.value() == 0.75
        assert window.diameter.value() == 5.8
        assert not window.auto_diameter.isChecked()
        assert not window.vehicle_speed.isEnabled()
        assert not window.auto_diameter.isEnabled()
        # Workers receive the manual baseline; each sidecar is applied independently.
        settings = window._settings()
        assert settings.vehicle_speed_mps == 0.31
        assert settings.expected_diameter_m == 6.4
        assert settings.auto_estimate_diameter is True

        window.source_edit.setPlainText(str(legacy))
        assert window.vehicle_speed.value() == 0.31
        assert window.diameter.value() == 6.4
        assert window.auto_diameter.isChecked()
        assert window.vehicle_speed.isEnabled()
        assert window.auto_diameter.isEnabled()
        window.source_edit.setPlainText(str(first))
        window.auto_scanner_revolutions.setChecked(False)
        assert window.vehicle_speed.value() == 0.31
        assert window.diameter.value() == 6.4
        assert window.auto_diameter.isChecked()
        assert window.scanner_revolutions.value() == 100
        assert window.scanner_revolutions.isEnabled()
        window.auto_scanner_revolutions.setChecked(True)
        from ortho_generator.task_list import OrthoTaskSpec
        window.task_specs = (OrthoTaskSpec(str(tmp_path / "*.e57"), str(tmp_path / "out")),)
        window.source_edit.setPlainText(str(tmp_path / "*.e57"))
        assert window.vehicle_speed.value() == 0.31
        assert window.diameter.value() == 6.4
        assert window.auto_diameter.isChecked()
        assert window._scanner_preview_baseline == {}
    finally:
        window.close()
