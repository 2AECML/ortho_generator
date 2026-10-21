import json
import os
from pathlib import Path

import pytest
from PIL import Image

from ortho_generator.cancellation import ProcessingCancelledError
from ortho_generator.task_list import OrthoTaskSpec, TaskJournal, remove_generated_input


@pytest.fixture(scope="session")
def cleanup_qt_application():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication

    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture
def cleanup_case(tmp_path, monkeypatch, cleanup_qt_application):
    import ortho_generator.app as app

    calls = []
    output = tmp_path / "output"
    journal = TaskJournal(tmp_path / "journal.txt")

    def generate(source, destination, settings, token, progress):
        calls.append(source)
        image = Path(destination) / f"{Path(source).stem}.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", (2, 2)).save(image)
        return app.GenerationResult(str(image.parent), None, (str(image),), 5.4, 0.005, False)

    monkeypatch.setattr(app, "generate_orthophotos", generate)
    monkeypatch.setattr(app, "journal_path_for", lambda _: journal.path)
    monkeypatch.setattr(app, "has_safe_free_space", lambda _: (True, 1024**4))

    def source(extension=".e57"):
        cloud = tmp_path / ("input" + extension)
        cloud.write_bytes(b"tiny test input")
        description = cloud.with_name(f"{cloud.stem}_扫描说明.txt")
        description.write_text("test metadata", encoding="utf-8")
        return cloud, description

    def run(cloud, task_mode=True, remove=True, pattern=None):
        if task_mode:
            worker = app.TaskListGenerationWorker(
                (OrthoTaskSpec(pattern or str(cloud), str(output)),), str(tmp_path / "tasks.txt"),
                app.GenerationSettings(), False, remove,
            )
        else:
            worker = app.GenerationWorker(
                (str(cloud),), str(output), app.GenerationSettings(), False, remove,
            )
        summaries, errors = [], []
        worker.succeeded.connect(summaries.append)
        worker.failed.connect(errors.append)
        worker.run()
        return summaries, errors

    return app, source, run, output, journal, calls


@pytest.mark.parametrize("task_mode", [False, True])
@pytest.mark.parametrize("extension", [".e57", ".las"])
@pytest.mark.parametrize("enabled", [False, True])
def test_cleanup_option_controls_both_modes_and_formats(cleanup_case, task_mode, extension, enabled):
    _, make_source, run, _, journal, calls = cleanup_case
    source, description = make_source(extension)

    summaries, errors = run(source, task_mode, enabled)

    assert not errors and len(summaries) == 1
    summary = summaries[0]
    assert not summary.failures and len(calls) == 1
    assert source.exists() is (not enabled)
    assert description.exists() is (not enabled)
    assert summary.cleaned_sources == ((str(source),) if enabled else ())
    assert all(Path(path).is_file() for path in summary.results[0][1].image_paths)
    if task_mode:
        assert bool(journal.keys_with_status("REMOVED")) is enabled


def test_ordinary_ui_passes_checked_cleanup_option(cleanup_case, monkeypatch):
    app, make_source, _, output, _, _ = cleanup_case
    source, description = make_source()
    monkeypatch.setattr(app.GenerationWorker, "start", lambda self: self.run())
    messages = []
    monkeypatch.setattr(app.QMessageBox, "information", lambda *args: messages.append(args[-1]))
    monkeypatch.setattr(app.QMessageBox, "warning", lambda *args: pytest.fail(str(args[-1])))
    window = app.MainWindow()
    try:
        window.coordinate_confirmed.setChecked(True)
        window.auto_scanner_revolutions.setChecked(False)
        window.source_edit.setPlainText(str(source))
        window.output_edit.setText(str(output))
        window.remove_input_after_success.setChecked(True)
        window._generate()
        assert window.worker.remove_inputs is True
        assert not source.exists() and not description.exists()
        assert any("已清理 1 个" in message for message in messages)
    finally:
        window.close()


@pytest.mark.parametrize("task_mode", [False, True])
@pytest.mark.parametrize("missing_image", [False, True])
def test_missing_or_empty_output_never_deletes_input(cleanup_case, monkeypatch, task_mode, missing_image):
    app, make_source, run, output, _, _ = cleanup_case
    source, description = make_source()
    paths = (str(output / "missing.png"),) if missing_image else ()
    monkeypatch.setattr(app, "generate_orthophotos", lambda *args: app.GenerationResult(
        str(output), None, paths, 5.4, 0.005, False,
    ))

    summaries, errors = run(source, task_mode)

    assert not errors and len(summaries[0].failures) == 1
    assert "保留输入点云" in summaries[0].failures[0][1]
    assert not summaries[0].cleaned_sources
    assert source.exists() and description.exists()


@pytest.mark.parametrize("task_mode", [False, True])
@pytest.mark.parametrize("cancelled", [False, True])
def test_generation_failure_or_cancellation_preserves_input(cleanup_case, monkeypatch, task_mode, cancelled):
    app, make_source, run, _, _, _ = cleanup_case
    source, description = make_source()

    def fail(*args):
        raise ProcessingCancelledError("cancelled") if cancelled else ValueError("generation failed")

    monkeypatch.setattr(app, "generate_orthophotos", fail)
    summaries, errors = run(source, task_mode)
    assert source.exists() and description.exists()
    if cancelled:
        assert errors and not summaries
    else:
        assert not errors and len(summaries[0].failures) == 1


@pytest.mark.parametrize("task_mode", [False, True])
def test_cleanup_permission_failure_reports_error_and_preserves_outputs(cleanup_case, monkeypatch, task_mode):
    _, make_source, run, _, journal, _ = cleanup_case
    source, description = make_source()
    original_unlink = Path.unlink

    def deny_cloud(path, *args, **kwargs):
        if path == source:
            raise PermissionError("file is locked")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", deny_cloud)
    summaries, errors = run(source, task_mode)
    summary = summaries[0]
    assert not errors and "移除输入失败" in summary.failures[0][1]
    assert source.exists() and description.exists()
    assert all(Path(path).is_file() for path in summary.results[0][1].image_paths)
    assert not summary.cleaned_sources
    if task_mode:
        assert journal.keys_with_status("CLEANUP_FAILED")
        assert not summary.completed_specs


def test_completed_task_can_be_cleaned_later_without_regeneration(cleanup_case):
    _, make_source, run, _, _, calls = cleanup_case
    source, description = make_source()
    pattern = str(source.with_name("*.e57"))
    assert not run(source, remove=False, pattern=pattern)[0][0].failures

    summary = run(source, pattern=pattern)[0][0]

    assert len(calls) == 1 and not summary.results
    assert not summary.failures and summary.cleaned_sources == (str(source),)
    assert not source.exists() and not description.exists()
    assert len(summary.completed_specs) == 1


@pytest.mark.parametrize("layout", ["flat", "nested"])
def test_old_success_logs_recover_output_evidence_from_manifest(cleanup_case, layout):
    _, make_source, run, output, journal, calls = cleanup_case
    source, description = make_source()
    directory = output if layout == "flat" else output / "run"
    image = directory / "orthophotos" / "image.png"
    image.parent.mkdir(parents=True)
    Image.new("L", (2, 2)).save(image)
    manifest = directory / ("manifest_image.json" if layout == "flat" else "manifest.json")
    manifest.write_text(json.dumps(dict(
        status="COMPLETED", input_path=str(source), images=[image.relative_to(directory).as_posix()],
    )), encoding="utf-8")
    journal.append("GENERATED", str(source), str(output), "生成 1 幅影像")
    journal.append("SUCCESS", str(source), str(output), "旧版成功记录")

    summary = run(source)[0][0]

    assert not calls and not summary.failures
    assert not source.exists() and not description.exists() and image.exists()
    assert summary.cleaned_sources == (str(source),)


@pytest.mark.parametrize("invalid", ["no_manifest", "wrong_input", "running", "partial_images"])
def test_old_success_without_complete_matching_outputs_is_not_deleted(cleanup_case, invalid):
    _, make_source, run, output, journal, calls = cleanup_case
    source, description = make_source()
    output.mkdir()
    image = output / "image.png"
    Image.new("L", (2, 2)).save(image)
    if invalid != "no_manifest":
        (output / "manifest_image.json").write_text(json.dumps(dict(
            status="RUNNING" if invalid == "running" else "COMPLETED",
            input_path=str(source.with_name("other.e57")) if invalid == "wrong_input" else str(source),
            images=[image.name, "missing.png"] if invalid == "partial_images" else [image.name],
        )), encoding="utf-8")
    journal.append("GENERATED", str(source), str(output), "生成 1 幅影像")
    journal.append("SUCCESS", str(source), str(output), "旧版成功记录")

    summary = run(source)[0][0]

    assert not calls and summary.failures and not summary.completed_specs
    assert source.exists() and description.exists() and image.exists()
    assert journal.keys_with_status("CLEANUP_FAILED")


def test_partial_cleanup_retries_orphaned_description_without_regeneration(cleanup_case, monkeypatch):
    _, make_source, run, _, _, calls = cleanup_case
    source, description = make_source()
    original_unlink = Path.unlink
    denied = []

    def deny_description_once(path, *args, **kwargs):
        if path == description and not denied:
            denied.append(True)
            raise PermissionError("description is locked")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", deny_description_once)
    pattern = str(source.with_name("*.e57"))
    first = run(source, pattern=pattern)[0][0]
    assert first.failures and not source.exists() and description.exists()

    second = run(source, pattern=pattern)[0][0]

    assert len(calls) == 1 and not second.failures
    assert not description.exists() and second.cleaned_sources == (str(source),)
    assert len(second.completed_specs) == 1


def test_cleanup_disabled_still_skips_already_completed_tasks(cleanup_case):
    _, make_source, run, _, _, calls = cleanup_case
    source, description = make_source()
    run(source, remove=False)

    summary = run(source, remove=False)[0][0]

    assert len(calls) == 1 and not summary.results and not summary.failures
    assert source.exists() and description.exists()


def test_missing_task_does_not_match_another_task_in_same_output(cleanup_case):
    _, make_source, run, output, journal, calls = cleanup_case
    source, _ = make_source()
    other = source.with_name("other.e57")
    journal.append("SUCCESS", str(other), str(output))
    source.unlink()

    summary = run(source)[0][0]

    assert not calls and summary.failures and not summary.completed_specs


def test_cleanup_rejects_unrelated_file_types(tmp_path):
    unrelated = tmp_path / "notes.txt"
    unrelated.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="不支持清理"):
        remove_generated_input(str(unrelated))
    assert unrelated.read_text(encoding="utf-8") == "keep"
