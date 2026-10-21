from __future__ import annotations

import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .cancellation import CancellationToken, ProcessingCancelledError
from .service import (
    GenerationResult,
    GenerationSettings,
    generate_orthophotos,
    read_scanner_description,
    scanner_parameter_summary,
    settings_with_scanner_description,
)
from .sources import create_source
from .task_list import (
    MINIMUM_FREE_BYTES,
    OrthoTaskSpec,
    TaskJournal,
    expand_task,
    has_safe_free_space,
    journal_path_for,
    read_task_list,
    remove_task_specs,
    remove_generated_input,
    require_generated_images,
    completed_image_outputs,
)


@dataclass(frozen=True, slots=True)
class BatchGenerationSummary:
    output_parent: str
    results: tuple[tuple[str, GenerationResult], ...]
    failures: tuple[tuple[str, str], ...]
    stopped_reason: str | None = None
    log_path: str | None = None
    completed_specs: tuple[OrthoTaskSpec, ...] = ()
    cleaned_sources: tuple[str, ...] = ()


class GenerationWorker(QThread):
    progressed = pyqtSignal(str, int)
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        sources: tuple[str, ...],
        output: str,
        settings: GenerationSettings,
        auto_scanner_revolutions: bool,
        remove_inputs: bool = False,
    ) -> None:
        super().__init__()
        self.sources = sources
        self.output = output
        self.settings = settings
        self.auto_scanner_revolutions = auto_scanner_revolutions
        self.remove_inputs = remove_inputs
        self.token = CancellationToken()

    def run(self) -> None:
        try:
            results: list[tuple[str, GenerationResult]] = []
            failures: list[tuple[str, str]] = []
            cleaned_sources: list[str] = []
            total = len(self.sources)
            for index, source in enumerate(self.sources, start=1):
                self.token.throw_if_cancelled()
                settings = replace(self.settings)
                try:
                    description = None
                    try:
                        description = read_scanner_description(source)
                    except (FileNotFoundError, ValueError):
                        if self.auto_scanner_revolutions:
                            raise
                    if description is not None:
                        settings = settings_with_scanner_description(
                            settings, description, self.auto_scanner_revolutions
                        )
                    if self.auto_scanner_revolutions and description is not None:
                        self.progressed.emit(
                            f"[{index}/{total}] {Path(source).name}：根据 "
                            f"{description.path.name} 设定 {scanner_parameter_summary(description)}",
                            int((index - 1) / total * 100),
                        )
                    if (
                        description is not None
                        and description.total_scan_span_degrees is not None
                    ):
                        self.progressed.emit(
                            f"[{index}/{total}] {Path(source).name}：实际断面扫描范围 "
                            f"{description.total_scan_span_degrees:.3f}°",
                            int((index - 1) / total * 100),
                        )
                    result = generate_orthophotos(
                        source,
                        self.output,
                        settings,
                        self.token,
                        lambda message, percent, i=index, path=source: self.progressed.emit(
                            f"[{i}/{total}] {Path(path).name}：{message}",
                            min(99, int(((i - 1) + percent / 100.0) / total * 100)),
                        ),
                    )
                    if self.remove_inputs:
                        require_generated_images(result.image_paths)
                    results.append((source, result))
                    if self.remove_inputs:
                        self.token.throw_if_cancelled()
                        try:
                            removed = remove_generated_input(source)
                            cleaned_sources.append(source)
                            self.progressed.emit(
                                f"[{index}/{total}] 已移除输入点云及扫描说明：{'；'.join(removed)}",
                                int(index / total * 100),
                            )
                        except Exception as error:
                            message = f"影像已生成，但移除输入失败：{error}"
                            failures.append((source, message))
                            self.progressed.emit(message, int(index / total * 100))
                except ProcessingCancelledError:
                    raise
                except Exception as error:
                    failures.append((source, str(error)))
                    self.progressed.emit(
                        f"[{index}/{total}] {Path(source).name} 失败：{error}",
                        int(index / total * 100),
                    )
            self.succeeded.emit(
                BatchGenerationSummary(
                    self.output, tuple(results), tuple(failures),
                    cleaned_sources=tuple(cleaned_sources),
                )
            )
        except Exception as error:
            self.failed.emit(str(error))


class TaskListGenerationWorker(QThread):
    progressed = pyqtSignal(str, int)
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        specs: tuple[OrthoTaskSpec, ...],
        task_list_path: str,
        settings: GenerationSettings,
        auto_scanner_revolutions: bool,
        remove_inputs: bool,
    ) -> None:
        super().__init__()
        self.specs = specs
        self.task_list_path = task_list_path
        self.settings = settings
        self.auto_scanner_revolutions = auto_scanner_revolutions
        self.remove_inputs = remove_inputs
        self.token = CancellationToken()

    def run(self) -> None:
        journal = TaskJournal(journal_path_for(self.task_list_path))
        try:
            expanded: list[tuple[str, str]] = []
            expanded_by_spec: dict[OrthoTaskSpec, tuple[str, ...]] = {}
            failures: list[tuple[str, str]] = []
            completed = journal.completed()
            generated = journal.keys_with_status("GENERATED")
            known_images = journal.generated_images()
            output_images: dict[str, dict[str, tuple[str, ...]]] = {}
            cleanup_failures: set[tuple[str, str]] = set()
            cleaned_sources: list[str] = []
            for spec in self.specs:
                sources = expand_task(spec)
                if self.remove_inputs:
                    orphaned = tuple(
                        source for source in journal.matching_sources(spec, ("GENERATED", "SUCCESS"))
                        if not Path(source).exists()
                        and Path(source).with_name(f"{Path(source).stem}_扫描说明.txt").is_file()
                    )
                    sources = tuple(dict.fromkeys((*sources, *orphaned)))
                expanded_by_spec[spec] = sources
                if not sources:
                    if self.remove_inputs and journal.matching_sources(spec, ("SUCCESS",)):
                        self.progressed.emit(
                            f"已完成并清理的任务组：{spec.input_pattern}", 0
                        )
                        continue
                    message = f"没有匹配到点云：{spec.input_pattern}"
                    failures.append((spec.input_pattern, message))
                    journal.append("NO_MATCH", spec.input_pattern, spec.output_parent, message)
                    continue
                expanded.extend((source, spec.output_parent) for source in sources)

            pending = [
                item for item in expanded
                if (str(Path(item[0]).resolve()).casefold(), str(Path(item[1]).resolve()).casefold())
                not in completed or self.remove_inputs
            ]
            skipped = len(expanded) - len(pending)
            if skipped:
                self.progressed.emit(f"根据执行日志跳过 {skipped} 个已成功点云。", 0)

            results: list[tuple[str, GenerationResult]] = []
            stopped_reason: str | None = None
            total = len(pending)
            for index, (source, output) in enumerate(pending, start=1):
                self.token.throw_if_cancelled()
                task_key = (
                    str(Path(source).resolve()).casefold(),
                    str(Path(output).resolve()).casefold(),
                )
                if self.remove_inputs and task_key in (generated | completed):
                    try:
                        image_paths = known_images.get(task_key)
                        if image_paths is None:
                            if task_key[1] not in output_images:
                                output_images[task_key[1]] = completed_image_outputs(output)
                            image_paths = output_images[task_key[1]].get(task_key[0], ())
                        require_generated_images(image_paths)
                        self.token.throw_if_cancelled()
                        removed = remove_generated_input(source)
                        cleaned_sources.append(source)
                        journal.append("REMOVED", source, output, "；".join(removed))
                        journal.append("SUCCESS", source, output, "恢复流程仅完成输入清理")
                        self.progressed.emit(
                            f"[{index}/{max(1, total)}] 已根据日志完成遗留输入清理。",
                            int(index / max(1, total) * 100),
                        )
                        continue
                    except ProcessingCancelledError:
                        raise
                    except Exception as error:
                        message = f"历史成图任务的输入清理失败：{error}"
                        failures.append((source, message))
                        cleanup_failures.add(task_key)
                        journal.append("CLEANUP_FAILED", source, output, message)
                        self.progressed.emit(message, int(index / max(1, total) * 100))
                        continue
                safe, free = has_safe_free_space(output)
                if not safe:
                    stopped_reason = (
                        f"输出盘剩余空间 {free / 1024**3:.2f} GB，低于安全余量 "
                        f"{MINIMUM_FREE_BYTES / 1024**3:.0f} GB，已停止。"
                    )
                    journal.append("STOP_NO_SPACE", source, output, stopped_reason)
                    break

                Path(output).mkdir(parents=True, exist_ok=True)
                journal.append("START", source, output)
                settings = replace(self.settings)
                low_space = False
                try:
                    description = None
                    try:
                        description = read_scanner_description(source)
                    except (FileNotFoundError, ValueError):
                        if self.auto_scanner_revolutions:
                            raise
                    if description is not None:
                        settings = settings_with_scanner_description(
                            settings, description, self.auto_scanner_revolutions
                        )
                    if self.auto_scanner_revolutions and description is not None:
                        self.progressed.emit(
                            f"[{index}/{max(1, total)}] {Path(source).name}：根据 "
                            f"{description.path.name} 设定 {scanner_parameter_summary(description)}",
                            int((index - 1) / max(1, total) * 100),
                        )

                    def report(message: str, percent: int) -> None:
                        nonlocal low_space
                        safe_now, _ = has_safe_free_space(output)
                        if not safe_now:
                            low_space = True
                            self.token.cancel()
                        self.progressed.emit(
                            f"[{index}/{max(1, total)}] {Path(source).name}：{message}",
                            min(99, int(((index - 1) + percent / 100.0) / max(1, total) * 100)),
                        )

                    result = generate_orthophotos(
                        source, output, settings, self.token, report
                    )
                    require_generated_images(result.image_paths)
                    results.append((source, result))
                    journal.record_generated(source, output, result.image_paths)
                    if self.remove_inputs:
                        self.token.throw_if_cancelled()
                        try:
                            removed = remove_generated_input(source)
                            cleaned_sources.append(source)
                            journal.append("REMOVED", source, output, "；".join(removed))
                            self.progressed.emit(
                                f"[{index}/{max(1, total)}] 已移除输入点云及扫描说明。",
                                int(index / max(1, total) * 100),
                            )
                        except Exception as error:
                            message = f"影像已生成，但移除输入失败：{error}"
                            failures.append((source, message))
                            cleanup_failures.add(task_key)
                            journal.append("CLEANUP_FAILED", source, output, message)
                            self.progressed.emit(message, int(index / max(1, total) * 100))
                            continue
                    journal.append("SUCCESS", source, output, f"生成 {len(result.image_paths)} 幅影像")
                except ProcessingCancelledError:
                    if low_space:
                        _, free = has_safe_free_space(output)
                        stopped_reason = f"输出盘剩余空间不足（{free / 1024**3:.2f} GB），已安全停止。"
                        journal.append("STOP_NO_SPACE", source, output, stopped_reason)
                        break
                    raise
                except Exception as error:
                    failures.append((source, str(error)))
                    journal.append("FAILED", source, output, str(error))
                    self.progressed.emit(
                        f"[{index}/{max(1, total)}] {Path(source).name} 失败：{error}",
                        int(index / max(1, total) * 100),
                    )

            output_parent = self.specs[-1].output_parent if self.specs else ""
            final_completed = journal.completed() - cleanup_failures
            completed_specs = tuple(
                spec
                for spec, sources in expanded_by_spec.items()
                if (
                    sources
                    and all(
                        (
                            str(Path(source).resolve()).casefold(),
                            str(Path(spec.output_parent).resolve()).casefold(),
                        ) in final_completed
                        for source in sources
                    )
                )
                or (
                    not sources
                    and self.remove_inputs
                    and journal.matching_sources(spec, ("SUCCESS",))
                )
            )
            self.succeeded.emit(BatchGenerationSummary(
                output_parent,
                tuple(results),
                tuple(failures),
                stopped_reason,
                str(journal.path),
                completed_specs,
                tuple(cleaned_sources),
            ))
        except Exception as error:
            self.failed.emit(str(error))


class TaskListSelectionDialog(QDialog):
    def __init__(self, specs: tuple[OrthoTaskSpec, ...], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("选择要执行的清单任务")
        self.resize(900, 560)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("请勾选本次需要执行的任务。默认不选择任何任务。"))
        self.list_widget = QListWidget()
        for spec in specs:
            item = QListWidgetItem(
                f"{spec.input_pattern}    →    {spec.output_parent}"
            )
            item.setData(Qt.UserRole, spec)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.list_widget.addItem(item)
        layout.addWidget(self.list_widget, 1)

        selection_actions = QHBoxLayout()
        select_all = QPushButton("全选")
        clear = QPushButton("清空")
        select_all.clicked.connect(lambda: self._set_all(Qt.Checked))
        clear.clicked.connect(lambda: self._set_all(Qt.Unchecked))
        selection_actions.addWidget(select_all)
        selection_actions.addWidget(clear)
        selection_actions.addStretch(1)
        layout.addLayout(selection_actions)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept_if_selected)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _set_all(self, state: Qt.CheckState) -> None:
        for index in range(self.list_widget.count()):
            self.list_widget.item(index).setCheckState(state)

    def _accept_if_selected(self) -> None:
        if self.selected_specs:
            self.accept()
        else:
            QMessageBox.warning(self, "尚未选择", "请至少选择一个任务。")

    @property
    def selected_specs(self) -> tuple[OrthoTaskSpec, ...]:
        return tuple(
            self.list_widget.item(index).data(Qt.UserRole)
            for index in range(self.list_widget.count())
            if self.list_widget.item(index).checkState() == Qt.Checked
        )


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.worker: GenerationWorker | TaskListGenerationWorker | None = None
        self.last_output: str | None = None
        self.task_specs: tuple[OrthoTaskSpec, ...] = ()
        self.task_list_path: str | None = None
        self._scanner_preview_baseline: dict[str, float | bool] = {}
        self.setWindowTitle("隧道点云大幅正射成图")
        self.setMinimumSize(680, 480)
        self.resize(920, 700)
        self._build_ui()
        self.source_edit.textChanged.connect(self._source_selection_changed)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        title = QLabel("LAS / E57 → 隧道大幅正射影像")
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        layout.addWidget(title)
        note = QLabel(
            "本程序只负责大图成图，不包含标注、复核或 640×640 切片。"
            "结构化 EasyTry E57 按 columnIndex 和断面间距重建纵向坐标；"
            "LAS/普通 E57 仍要求 Y=隧道纵向米制坐标。"
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        settings_scroll = QScrollArea()
        settings_scroll.setObjectName("settingsScroll")
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setFrameShape(QFrame.NoFrame)
        settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        settings_scroll.setMinimumHeight(220)
        settings_content = QWidget()
        settings_layout = QVBoxLayout(settings_content)
        settings_layout.setContentsMargins(0, 0, 0, 0)

        source_box = QGroupBox("输入与输出")
        source_form = QFormLayout(source_box)
        self.source_edit = QPlainTextEdit()
        self.source_edit.setMaximumHeight(64)
        self.source_edit.setPlaceholderText("每行一个 LAS/E57 文件")
        source_row = QHBoxLayout()
        source_row.addWidget(self.source_edit, 1)
        choose_source = QPushButton("选择文件…")
        choose_source.clicked.connect(self._choose_source)
        source_row.addWidget(choose_source)
        inspect_button = QPushButton("检查")
        inspect_button.clicked.connect(self._inspect_source)
        source_row.addWidget(inspect_button)
        source_form.addRow("点云文件", source_row)

        self.output_edit = QLineEdit()
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_edit, 1)
        choose_output = QPushButton("选择目录…")
        choose_output.clicked.connect(self._choose_output)
        output_row.addWidget(choose_output)
        source_form.addRow("输出父目录", output_row)
        self.choose_task_list = QPushButton("导入 Tab 分隔任务清单…")
        self.choose_task_list.clicked.connect(self._choose_task_list)
        source_form.addRow("自动任务", self.choose_task_list)
        self.output_auxiliary_data = QCheckBox("输出辅助数据")
        self.output_auxiliary_data.setChecked(False)
        self.output_auxiliary_data.setToolTip(
            "保留科学强度、掩膜、点击计数和每幅图的详细元数据。"
        )
        source_form.addRow("辅助输出", self.output_auxiliary_data)
        self.flat_image_output = QCheckBox("集中存放影像")
        self.flat_image_output.setChecked(True)
        self.flat_image_output.setToolTip(
            "将影像及其 manifest_"
            "{影像名}.json 直接存放到选定的输出目录。"
        )
        self.output_auxiliary_data.toggled.connect(
            self._output_auxiliary_data_toggled
        )
        source_form.addRow("影像存放", self.flat_image_output)
        self.remove_input_after_success = QCheckBox(
            "生成正射影像后移除输入点云文件"
        )
        self.remove_input_after_success.setChecked(False)
        self.remove_input_after_success.setToolTip(
            "普通文件与任务清单均生效；确认正射影像完整后移除 E57/LAS 与同名的扫描说明。"
            "清单中已成图但未清理的输入会核对输出后补清理。"
        )
        source_form.addRow("完成后清理", self.remove_input_after_success)
        self.remove_completed_task_items = QCheckBox("成功后从清单移除任务")
        self.remove_completed_task_items.setChecked(False)
        self.remove_completed_task_items.setToolTip(
            "一个清单任务组内的点云全部成功后，从原 TXT 中移除该行。"
        )
        source_form.addRow("清单维护", self.remove_completed_task_items)
        self.source_info = QLabel("尚未检查输入。")
        self.source_info.setWordWrap(True)
        source_form.addRow("输入信息", self.source_info)
        settings_layout.addWidget(source_box)

        parameters = QTabWidget()
        parameters.setObjectName("parameterTabs")
        basic_page = QWidget()
        form = QFormLayout(basic_page)
        self.pixel_size = QComboBox()
        self.pixel_size.setEditable(True)
        self.pixel_size.addItems(["5.0", "2.5", "1.5"])
        form.addRow("像素大小（mm）", self.pixel_size)
        self.vehicle_speed = QDoubleSpinBox()
        self.vehicle_speed.setRange(0.001, 100.0)
        self.vehicle_speed.setDecimals(4)
        self.vehicle_speed.setSingleStep(0.1)
        self.vehicle_speed.setValue(0.4)
        self.vehicle_speed.setSuffix(" m/s")
        self.vehicle_speed.setToolTip("仅用于带 rowIndex/columnIndex 的结构化 EasyTry E57。")
        form.addRow("小车速度", self.vehicle_speed)
        self.scanner_revolutions = QDoubleSpinBox()
        self.scanner_revolutions.setRange(0.001, 5000.0)
        self.scanner_revolutions.setDecimals(3)
        self.scanner_revolutions.setSingleStep(10.0)
        self.scanner_revolutions.setValue(100.0)
        self.scanner_revolutions.setSuffix(" 圈/s")
        self.scanner_revolutions.setToolTip("扫描仪每秒完成的断面扫描圈数。")
        form.addRow("扫描仪每秒圈数", self.scanner_revolutions)
        self.auto_scanner_revolutions = QCheckBox("根据扫描说明自动设定")
        self.auto_scanner_revolutions.setChecked(True)
        self.scanner_revolutions.setEnabled(False)
        self.auto_scanner_revolutions.setToolTip(
            "从每个点云同目录的“文件名_扫描说明.txt”读取圈数、小车速度和内径；"
            "环宽仅记录。缺少新增字段时沿用手工设置；批量处理按每个文件分别设定。"
        )
        self.auto_scanner_revolutions.toggled.connect(
            self._auto_scanner_revolutions_toggled
        )
        form.addRow("扫描参数来源", self.auto_scanner_revolutions)
        self.profile_spacing = QLabel()
        form.addRow("计算断面间距", self.profile_spacing)
        self.vehicle_speed.valueChanged.connect(self._update_profile_spacing)
        self.scanner_revolutions.valueChanged.connect(self._update_profile_spacing)
        self._update_profile_spacing()
        self.image_length = QDoubleSpinBox()
        self.image_length.setRange(0.1, 1000.0)
        self.image_length.setDecimals(1)
        self.image_length.setValue(20.0)
        self.image_length.setSuffix(" m")
        self.adaptive_single_image = QCheckBox("完整单图（超限自动降低分辨率）")
        self.adaptive_single_image.setChecked(True)
        self.adaptive_single_image.setToolTip(
            "保留完整隧道范围；若整图超过像素上限，自动使用不超限的最高分辨率。"
        )
        self.adaptive_single_image.toggled.connect(
            lambda checked: self.image_length.setEnabled(not checked)
        )
        self.image_length.setEnabled(False)
        form.addRow("输出模式", self.adaptive_single_image)
        form.addRow("每幅大图纵向长度", self.image_length)
        self.diameter = QDoubleSpinBox()
        self.diameter.setRange(0.5, 30.0)
        self.diameter.setDecimals(2)
        self.diameter.setValue(5.4)
        self.diameter.setSuffix(" m")
        self.auto_diameter = QCheckBox("根据点云断面自动拟合")
        self.auto_diameter.setChecked(True)
        self.auto_diameter.setToolTip("用稳健椭圆拟合估算等效直径；手工值仅在关闭自动估算时使用。")
        self.auto_diameter.toggled.connect(
            lambda checked: self.diameter.setEnabled(not checked)
        )
        self.diameter.setEnabled(False)
        form.addRow("直径模式", self.auto_diameter)
        form.addRow("手工预期直径", self.diameter)
        self.y_min = QLineEdit()
        self.y_max = QLineEdit()
        self.y_min.setPlaceholderText("留空使用全部点云")
        self.y_max.setPlaceholderText("留空使用全部点云")
        form.addRow("纵向 Y 起点（m）", self.y_min)
        form.addRow("纵向 Y 终点（m）", self.y_max)
        self.display_min = QLineEdit()
        self.display_max = QLineEdit()
        self.display_min.setPlaceholderText("留空自动使用 1% 分位")
        self.display_max.setPlaceholderText("留空自动使用 99% 分位")
        display_page = QWidget()
        display_form = QFormLayout(display_page)
        display_form.addRow("显示强度下限", self.display_min)
        display_form.addRow("显示强度上限", self.display_max)
        self.interpolation_distance = QDoubleSpinBox()
        self.interpolation_distance.setRange(0.0, 100.0)
        self.interpolation_distance.setDecimals(1)
        self.interpolation_distance.setValue(7.5)
        self.interpolation_distance.setSuffix(" mm")
        self.interpolation_distance.setToolTip(
            "只填补封闭小孔；0 表示关闭。科学强度与观测掩膜始终不修改。"
        )
        display_form.addRow("显示小孔最大插值距离", self.interpolation_distance)
        self.suppress_impulses = QCheckBox("抑制显示图中的孤立亮点/黑点")
        self.suppress_impulses.setChecked(True)
        self.suppress_impulses.setToolTip(
            "只修正显示 PNG 中孤立的单像素脉冲；科学强度、TIFF 与观测掩膜不修改。"
        )
        display_form.addRow("显示杂点处理", self.suppress_impulses)
        self.las_depth_display = QCheckBox("启用 LAS 前景深度簇聚合")
        self.las_depth_display.setChecked(True)
        self.las_depth_display.setToolTip(
            "仅调整显示 PNG 的前景深度聚合，科学数据保持不变。"
        )
        display_form.addRow("LAS 管线边缘处理", self.las_depth_display)
        self.las_max_depth = QDoubleSpinBox()
        self.las_max_depth.setRange(0.1, 20.0)
        self.las_max_depth.setValue(2.0)
        self.las_max_depth.setSuffix(" m")
        display_form.addRow("LAS 显示最大相对深度", self.las_max_depth)
        self.las_cluster = QDoubleSpinBox()
        self.las_cluster.setRange(1.0, 500.0)
        self.las_cluster.setValue(30.0)
        self.las_cluster.setSuffix(" mm")
        display_form.addRow("LAS 前景深度簇厚度", self.las_cluster)
        self.las_antialias = QDoubleSpinBox()
        self.las_antialias.setRange(0.0, 1.0)
        self.las_antialias.setSingleStep(0.1)
        self.las_antialias.setValue(0.5)
        self.las_antialias.setToolTip("仅在前景深度边缘沿轮廓做亚像素采样；0 关闭。不是缺测重建，科学数据不修改。需启用 LAS 深度簇聚合。")
        display_form.addRow("LAS 轮廓抗锯齿强度", self.las_antialias)
        self.coordinate_confirmed = QCheckBox("已确认点云坐标方向符合要求")
        self.coordinate_confirmed.setToolTip(
            "X/Z 为断面横向/竖向；LAS 或普通 E57 的 Y 为纵向米制坐标。"
        )
        form.addRow("坐标确认", self.coordinate_confirmed)
        parameters.addTab(basic_page, "基础与几何")
        parameters.addTab(display_page, "显示与 LAS 处理")
        settings_layout.addWidget(parameters)
        settings_layout.addStretch(1)
        settings_scroll.setWidget(settings_content)
        layout.addWidget(settings_scroll, 3)

        action_row = QHBoxLayout()
        self.generate_button = QPushButton("生成大幅正射影像")
        self.generate_button.clicked.connect(self._generate)
        action_row.addWidget(self.generate_button)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel)
        action_row.addWidget(self.cancel_button)
        self.open_button = QPushButton("打开最近输出")
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(self._open_output)
        action_row.addWidget(self.open_button)
        layout.addLayout(action_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(80)
        layout.addWidget(self.log, 1)

    def _choose_source(self) -> None:
        existing = self._source_paths()
        start = str(Path(existing[0]).parent) if existing else ""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择一个或多个点云", start, "点云 (*.las *.e57)"
        )
        if paths:
            was_task_mode = bool(self.task_specs)
            self.task_specs = ()
            self.task_list_path = None
            self.output_edit.setReadOnly(False)
            self.source_edit.setPlainText("\n".join(paths))
            if was_task_mode or not self.output_edit.text():
                self.output_edit.setText(str(Path(paths[0]).parent / "ortho_outputs"))
            if self.auto_scanner_revolutions.isChecked():
                self._sync_scanner_revolutions(show_errors=True)

    def _choose_task_list(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择正射影像任务清单",
            "",
            "文本任务清单 (*.txt);;所有文件 (*)",
        )
        if not path:
            return
        try:
            available_specs = read_task_list(path)
        except Exception as error:
            QMessageBox.warning(self, "任务清单无效", str(error))
            return
        selection = TaskListSelectionDialog(available_specs, self)
        if selection.exec_() != QDialog.Accepted:
            return
        specs = selection.selected_specs
        self.task_specs = specs
        self.task_list_path = str(Path(path).resolve())
        self.source_edit.setPlainText("\n".join(spec.input_pattern for spec in specs))
        self.output_edit.setText(f"由任务清单指定（{len(specs)} 组）")
        self.output_edit.setReadOnly(True)
        self.source_info.setText(
            f"已载入 {len(specs)} 组自动任务；再次运行将根据执行日志跳过已成功项。"
        )
        self.log.appendPlainText(f"已载入任务清单：{self.task_list_path}")
        self.log.appendPlainText(f"执行日志：{journal_path_for(self.task_list_path)}")

    def _choose_output(self) -> None:
        if self.task_specs:
            QMessageBox.information(self, "任务清单模式", "输出目录由任务清单第二列指定。")
            return
        path = QFileDialog.getExistingDirectory(
            self, "选择输出父目录", self.output_edit.text()
        )
        if path:
            self.output_edit.setText(path)

    def _inspect_source(self) -> None:
        try:
            if self.task_specs:
                counts = [len(expand_task(spec)) for spec in self.task_specs]
                missing = sum(count == 0 for count in counts)
                self.source_info.setText(
                    f"任务清单 {len(self.task_specs)} 组，当前匹配 {sum(counts)} 个点云，"
                    f"未匹配 {missing} 组。"
                )
                return
            paths = self._source_paths(require=True)
            details = []
            for path in paths:
                source = create_source(path)
                try:
                    metadata = source.open()
                    structured = (
                        f" | 结构化 {metadata.scan_rows}×{metadata.scan_columns}"
                        if metadata.structured_indices
                        else " | 通用几何投影"
                    )
                    details.append(
                        f"{Path(path).name}：{metadata.source_format} | "
                        f"{metadata.point_count:,} 点 | {metadata.scan_count} 个扫描对象"
                        f"{structured}"
                    )
                finally:
                    source.close()
            shown = details[:5]
            if len(details) > len(shown):
                shown.append(f"…其余 {len(details) - len(shown)} 个文件已检查")
            self.source_info.setText(
                f"已选择 {len(paths)} 个点云。\n" + "\n".join(shown)
            )
            if self.auto_scanner_revolutions.isChecked():
                self._sync_scanner_revolutions(show_errors=True)
        except Exception as error:
            QMessageBox.warning(self, "输入检查失败", str(error))

    def _source_paths(self, require: bool = False) -> tuple[str, ...]:
        paths = tuple(
            line.strip().strip('"')
            for line in self.source_edit.toPlainText().splitlines()
            if line.strip()
        )
        if require and not paths:
            raise ValueError("请选择至少一个点云文件。")
        return paths

    def _auto_scanner_revolutions_toggled(self, checked: bool) -> None:
        self._restore_scanner_parameter_preview()
        self.scanner_revolutions.setEnabled(not checked)
        if checked:
            self._sync_scanner_revolutions(show_errors=True)

    def _source_selection_changed(self) -> None:
        self._restore_scanner_parameter_preview()
        if self.auto_scanner_revolutions.isChecked():
            self._sync_scanner_revolutions(show_errors=False)

    def _restore_scanner_parameter_preview(self) -> None:
        baseline = self._scanner_preview_baseline
        self._scanner_preview_baseline = {}
        if "vehicle_speed_mps" in baseline:
            self.vehicle_speed.setValue(baseline["vehicle_speed_mps"])
        if "scanner_revolutions_per_second" in baseline:
            self.scanner_revolutions.setValue(baseline["scanner_revolutions_per_second"])
        if "expected_diameter_m" in baseline:
            self.diameter.setValue(baseline["expected_diameter_m"])
            self.auto_diameter.setChecked(baseline["auto_estimate_diameter"])
        self.vehicle_speed.setEnabled(True)
        self.auto_diameter.setEnabled(True)
        self.diameter.setEnabled(not self.auto_diameter.isChecked())

    def _output_auxiliary_data_toggled(self, checked: bool) -> None:
        if checked:
            self.flat_image_output.setChecked(False)
        self.flat_image_output.setEnabled(not checked)

    def _sync_scanner_revolutions(self, show_errors: bool) -> None:
        self._restore_scanner_parameter_preview()
        if self.task_specs:
            self.auto_scanner_revolutions.setToolTip("任务清单中的扫描参数将在成图时按文件分别读取。")
            return
        paths = self._source_paths()
        if not paths:
            self.auto_scanner_revolutions.setToolTip("选择点云后自动读取对应扫描说明中的参数。")
            return
        try:
            descriptions = [read_scanner_description(path) for path in paths]
            first = descriptions[0]
            self._scanner_preview_baseline["scanner_revolutions_per_second"] = (
                self.scanner_revolutions.value()
            )
            self.scanner_revolutions.setValue(first.revolutions_per_second)
            if first.vehicle_speed_mps is not None:
                self._scanner_preview_baseline["vehicle_speed_mps"] = self.vehicle_speed.value()
                self.vehicle_speed.setValue(first.vehicle_speed_mps)
                self.vehicle_speed.setEnabled(False)
            if first.inner_diameter_m is not None:
                self._scanner_preview_baseline.update(
                    expected_diameter_m=self.diameter.value(),
                    auto_estimate_diameter=self.auto_diameter.isChecked(),
                )
                self.auto_diameter.setChecked(False)
                self.auto_diameter.setEnabled(False)
                self.diameter.setValue(first.inner_diameter_m)
                self.diameter.setEnabled(False)
            detail = f"{Path(paths[0]).name}：{scanner_parameter_summary(first)}"
            if len(paths) > 1:
                detail += "。当前显示首个文件参数；成图时按每个文件分别读取，缺少新增字段使用原手工设置。"
            self.auto_scanner_revolutions.setToolTip(detail)
            self._update_profile_spacing()
        except Exception as error:
            self.auto_scanner_revolutions.setToolTip(f"扫描说明读取失败：{error}")
            if show_errors:
                QMessageBox.warning(self, "扫描说明读取失败", str(error))

    @staticmethod
    def _optional_number(field: QLineEdit) -> float | None:
        text = field.text().strip()
        return None if not text else float(text)

    def _update_profile_spacing(self, _value: float | None = None) -> None:
        spacing_m = self.vehicle_speed.value() / self.scanner_revolutions.value()
        self.profile_spacing.setText(
            f"{spacing_m:.9f} m（{spacing_m * 1000.0:.6f} mm）"
        )

    def _settings(self) -> GenerationSettings:
        if not self.coordinate_confirmed.isChecked():
            raise ValueError("请先确认点云坐标轴定义。")
        baseline = self._scanner_preview_baseline
        return GenerationSettings(
            pixel_size_m=float(self.pixel_size.currentText()) / 1000.0,
            image_length_m=self.image_length.value(),
            adaptive_single_image=self.adaptive_single_image.isChecked(),
            output_auxiliary_data=self.output_auxiliary_data.isChecked(),
            flat_image_output=self.flat_image_output.isChecked(),
            expected_diameter_m=baseline.get("expected_diameter_m", self.diameter.value()),
            auto_estimate_diameter=baseline.get("auto_estimate_diameter", self.auto_diameter.isChecked()),
            vehicle_speed_mps=baseline.get("vehicle_speed_mps", self.vehicle_speed.value()),
            scanner_revolutions_per_second=baseline.get(
                "scanner_revolutions_per_second", self.scanner_revolutions.value()
            ),
            longitudinal_min_m=self._optional_number(self.y_min),
            longitudinal_max_m=self._optional_number(self.y_max),
            display_min=self._optional_number(self.display_min),
            display_max=self._optional_number(self.display_max),
            maximum_display_interpolation_distance_m=(
                self.interpolation_distance.value() / 1000.0
            ),
            suppress_display_impulses=self.suppress_impulses.isChecked(),
            las_depth_aware_display=self.las_depth_display.isChecked(),
            maximum_las_display_relative_depth_m=self.las_max_depth.value(),
            las_display_depth_cluster_m=self.las_cluster.value() / 1000.0,
            las_edge_antialias_strength=self.las_antialias.value(),
        )

    def _generate(self) -> None:
        try:
            settings = self._settings()
            settings.validate()
            if self.task_specs:
                if not self.task_list_path:
                    raise ValueError("任务清单路径无效。")
                self.worker = TaskListGenerationWorker(
                    self.task_specs,
                    self.task_list_path,
                    settings,
                    self.auto_scanner_revolutions.isChecked(),
                    self.remove_input_after_success.isChecked(),
                )
                self.progress.setValue(0)
                self.log.appendPlainText(
                    f"自动任务：{len(self.task_specs)} 组；执行日志将用于断点续作。"
                )
                self.worker.progressed.connect(self._progressed)
                self.worker.succeeded.connect(self._succeeded)
                self.worker.failed.connect(self._failed)
                self.generate_button.setEnabled(False)
                self.cancel_button.setEnabled(True)
                self.remove_completed_task_items.setEnabled(False)
                self.worker.start()
                return
            sources = tuple(
                str(Path(path).resolve(strict=True))
                for path in self._source_paths(require=True)
            )
            invalid = [path for path in sources if Path(path).suffix.lower() not in (".las", ".e57")]
            if invalid:
                raise ValueError(f"不支持的点云文件：{invalid[0]}")
            output = self.output_edit.text().strip()
            if not output:
                raise ValueError("请选择输出父目录。")
            output = str(Path(output).resolve())
        except Exception as error:
            QMessageBox.warning(self, "参数错误", str(error))
            return
        self.progress.setValue(0)
        self.log.appendPlainText(f"批量输入：{len(sources)} 个点云")
        for source in sources:
            self.log.appendPlainText(f"  {source}")
        self.worker = GenerationWorker(
            sources,
            output,
            settings,
            self.auto_scanner_revolutions.isChecked(),
            self.remove_input_after_success.isChecked(),
        )
        self.worker.progressed.connect(self._progressed)
        self.worker.succeeded.connect(self._succeeded)
        self.worker.failed.connect(self._failed)
        self.generate_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.remove_completed_task_items.setEnabled(False)
        self.worker.start()

    def _progressed(self, message: str, percent: int) -> None:
        self.progress.setValue(percent)
        self.log.appendPlainText(f"{percent:3d}%  {message}")

    def _succeeded(self, summary: BatchGenerationSummary) -> None:
        self.progress.setValue(100)
        self.last_output = summary.output_parent
        total_images = 0
        for source, result in summary.results:
            total_images += len(result.image_paths)
            image_directory = (
                Path(result.image_paths[0]).parent
                if result.image_paths
                else Path(result.output_directory)
            )
            self.log.appendPlainText(f"完成 {Path(source).name}：{image_directory}")
            if result.manifest_path is not None:
                self.log.appendPlainText(f"  清单：{result.manifest_path}")
            self.log.appendPlainText(
                f"  等效直径 {result.estimated_diameter_m:.3f} m | "
                f"实际像素 {result.actual_pixel_size_m * 1000:.3f} mm"
            )
        if summary.results and "expected_diameter_m" not in self._scanner_preview_baseline:
            self.diameter.setValue(summary.results[-1][1].estimated_diameter_m)
        for source, error in summary.failures:
            self.log.appendPlainText(f"失败 {Path(source).name}：{error}")
        if summary.log_path:
            self.log.appendPlainText(f"执行日志：{summary.log_path}")
        if (
            self.remove_completed_task_items.isChecked()
            and self.task_list_path
            and summary.completed_specs
        ):
            try:
                removed = remove_task_specs(
                    self.task_list_path, summary.completed_specs
                )
                completed = set(summary.completed_specs)
                self.task_specs = tuple(
                    spec for spec in self.task_specs if spec not in completed
                )
                self.source_edit.setPlainText(
                    "\n".join(spec.input_pattern for spec in self.task_specs)
                )
                self.log.appendPlainText(f"已从任务清单移除 {removed} 个完成项。")
            except Exception as error:
                self.log.appendPlainText(f"从任务清单移除完成项失败：{error}")
        if summary.stopped_reason:
            self.log.appendPlainText("已停止：" + summary.stopped_reason)
        self.generate_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.remove_completed_task_items.setEnabled(True)
        self.open_button.setEnabled(bool(summary.results or summary.cleaned_sources))
        message = (
            f"批量处理完成：{len(summary.results)} 个成功，"
            f"{len(summary.failures)} 个失败，共生成 {total_images} 幅影像。"
        )
        if summary.cleaned_sources:
            message += f"\n已清理 {len(summary.cleaned_sources)} 个输入点云及对应扫描说明。"
        if summary.stopped_reason:
            QMessageBox.warning(
                self,
                "空间不足，流程已停止",
                message + "\n" + summary.stopped_reason + "\n释放空间后重新载入同一清单即可继续。",
            )
        elif summary.failures:
            QMessageBox.warning(self, "批量成图完成", message + "\n请查看日志中的失败原因。")
        else:
            QMessageBox.information(self, "批量成图完成", message)

    def _failed(self, message: str) -> None:
        self.log.appendPlainText("失败：" + message)
        self.generate_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.remove_completed_task_items.setEnabled(True)
        QMessageBox.warning(self, "成图未完成", message)

    def _cancel(self) -> None:
        if self.worker is not None:
            self.worker.token.cancel()
            self.log.appendPlainText("正在安全取消…")

    def _open_output(self) -> None:
        if self.last_output and Path(self.last_output).is_dir():
            os.startfile(self.last_output)

    def closeEvent(self, event) -> None:
        if self.worker is not None and self.worker.isRunning():
            answer = QMessageBox.question(self, "任务正在运行", "是否取消任务并退出？")
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self.worker.token.cancel()
            self.worker.wait(5000)
        event.accept()


def main() -> int:
    application = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return application.exec_()
