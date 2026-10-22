import os
from pathlib import Path

import pytest
from PIL import Image

from ortho_generator.cancellation import ProcessingCancelledError
from ortho_generator.task_list import OrthoTaskSpec, remove_generated_input


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

    def generate(source, destination, settings, token, progress):
        calls.append(source)
        image = Path(destination) / f"{Path(source).stem}.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", (2, 2)).save(image)
        return app.GenerationResult(str(image.parent), None, (str(image),), 5.4, 0.005, False)

    monkeypatch.setattr(app, "generate_orthophotos", generate)
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
                (OrthoTaskSpec(pattern or str(cloud), str(output)),),
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

    return app, source, run, output, calls


@pytest.mark.parametrize("task_mode", [False, True])
@pytest.mark.parametrize("extension", [".e57", ".las"])
@pytest.mark.parametrize("enabled", [False, True])
def test_cleanup_option_controls_both_modes_and_formats(cleanup_case, task_mode, extension, enabled):
    _, make_source, run, _, calls = cleanup_case
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
        assert len(summary.completed_specs) == 1


def test_ordinary_ui_passes_checked_cleanup_option(cleanup_case, monkeypatch):
    app, make_source, _, output, _ = cleanup_case
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
    app, make_source, run, output, _ = cleanup_case
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
    app, make_source, run, _, _ = cleanup_case
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
    _, make_source, run, _, _ = cleanup_case
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
        assert not summary.completed_specs


def test_repeated_task_selection_runs_again_without_saved_history(cleanup_case):
    _, make_source, run, output, calls = cleanup_case
    source, description = make_source()
    first = run(source, remove=False)[0][0]
    second = run(source, remove=False)[0][0]

    assert len(calls) == 2
    assert first.results and second.results
    assert first.completed_specs == second.completed_specs
    assert source.exists() and description.exists()
    assert {path.name for path in output.iterdir()} == {"input.png"}


def test_completed_task_can_be_generated_and_cleaned_in_later_run(cleanup_case):
    _, make_source, run, _, calls = cleanup_case
    source, description = make_source()
    run(source, remove=False)
    summary = run(source)[0][0]

    assert len(calls) == 2 and summary.results and not summary.failures
    assert summary.cleaned_sources == (str(source),)
    assert not source.exists() and not description.exists()
    assert len(summary.completed_specs) == 1


def test_missing_task_is_not_marked_complete(cleanup_case):
    _, make_source, run, _, calls = cleanup_case
    source, description = make_source()
    source.unlink()

    summary = run(source)[0][0]

    assert not calls and summary.failures and not summary.completed_specs
    assert description.exists()


def test_task_list_removes_only_groups_completed_in_current_run(cleanup_case, monkeypatch, tmp_path):
    app, make_source, _, output, calls = cleanup_case
    source, _ = make_source()
    failing = source.with_name("failing.e57")
    failing.write_bytes(b"test")
    specs = (
        OrthoTaskSpec(str(source), str(output)),
        OrthoTaskSpec(str(failing), str(output)),
        OrthoTaskSpec(str(source.with_name("missing.e57")), str(output)),
    )
    task_file = tmp_path / "tasks.txt"
    task_file.write_text("".join(f"{spec.input_pattern}\t{spec.output_parent}\n" for spec in specs),
                         encoding="utf-8-sig")
    generate = app.generate_orthophotos

    def fail_one(path, *args):
        if path == str(failing):
            raise ValueError("invalid point cloud")
        return generate(path, *args)

    monkeypatch.setattr(app, "generate_orthophotos", fail_one)
    worker = app.TaskListGenerationWorker(specs, app.GenerationSettings(), False, False)
    summaries = []
    worker.succeeded.connect(summaries.append)
    worker.run()
    assert len(calls) == 1
    assert summaries[0].completed_specs == (specs[0],)
    assert len(summaries[0].failures) == 2

    messages = []
    monkeypatch.setattr(app.QMessageBox, "warning", lambda *args: messages.append(args[-1]))
    window = app.MainWindow()
    try:
        window.task_specs = specs
        window.task_list_path = str(task_file)
        window.remove_completed_task_items.setChecked(True)
        window._succeeded(summaries[0])
        assert app.read_task_list(task_file) == specs[1:]
        assert window.task_specs == specs[1:]
        assert "invalid point cloud" in messages[0] and "没有匹配到点云" in messages[0]
    finally:
        window.close()


def test_partial_group_is_not_removed_after_one_file_fails(cleanup_case, monkeypatch):
    app, make_source, run, _, _ = cleanup_case
    source, _ = make_source()
    failing = source.with_name("failing.e57")
    failing.touch()
    generate = app.generate_orthophotos

    def fail_one(path, *args):
        if path == str(failing):
            raise ValueError("generation failed")
        return generate(path, *args)

    monkeypatch.setattr(app, "generate_orthophotos", fail_one)
    summary = run(source, remove=False, pattern=str(source.with_name("*.e57")))[0][0]
    assert len(summary.results) == 1 and len(summary.failures) == 1
    assert not summary.completed_specs
    assert source.exists() and failing.exists()


def test_task_low_space_preserves_pending_input_and_completion_state(cleanup_case, monkeypatch):
    app, make_source, run, _, calls = cleanup_case
    source, description = make_source()
    monkeypatch.setattr(app, "has_safe_free_space", lambda _: (False, 9 * 1024**3))

    summary = run(source)[0][0]

    assert not calls and summary.stopped_reason
    assert not summary.results and not summary.completed_specs and not summary.cleaned_sources
    assert source.exists() and description.exists()


def test_ui_retains_log_history_without_file_access(cleanup_case, monkeypatch):
    app, _, _, _, _ = cleanup_case
    messages = []
    monkeypatch.setattr(app.QMessageBox, "warning", lambda *args: messages.append(args[-1]))
    window = app.MainWindow()
    try:
        assert set(window.findChildren(app.QPlainTextEdit)) == {window.source_edit, window.log}
        assert window.log.isReadOnly()

        def deny_file_access(*args, **kwargs):
            pytest.fail("UI logging must not read or write files")

        monkeypatch.setattr("builtins.open", deny_file_access)
        monkeypatch.setattr("io.open", deny_file_access)
        window._progressed("first input", 10)
        window._progressed("second input", 60)
        assert window.status_label.text() == "second input"
        assert window.progress.value() == 60
        assert "first input" in window.log.toPlainText()
        assert "second input" in window.log.toPlainText()
        window._failed("failed to read input")
        assert messages == ["failed to read input"]
        assert "failed to read input" in window.status_label.text()
        assert "failed to read input" in window.log.toPlainText()
        assert window.generate_button.isEnabled() and not window.cancel_button.isEnabled()
    finally:
        window.close()


def test_cleanup_rejects_unrelated_file_types(tmp_path):
    unrelated = tmp_path / "notes.txt"
    unrelated.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="不支持清理"):
        remove_generated_input(str(unrelated))
    assert unrelated.read_text(encoding="utf-8") == "keep"
