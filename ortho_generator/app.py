from __future__ import annotations

import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .cancellation import CancellationToken, ProcessingCancelledError
from .service import (
    GenerationResult,
    GenerationSettings,
    generate_orthophotos,
    read_scanner_revolutions,
)
from .sources import create_source


@dataclass(frozen=True, slots=True)
class BatchGenerationSummary:
    output_parent: str
    results: tuple[tuple[str, GenerationResult], ...]
    failures: tuple[tuple[str, str], ...]


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
    ) -> None:
        super().__init__()
        self.sources = sources
        self.output = output
        self.settings = settings
        self.auto_scanner_revolutions = auto_scanner_revolutions
        self.token = CancellationToken()

    def run(self) -> None:
        try:
            results: list[tuple[str, GenerationResult]] = []
            failures: list[tuple[str, str]] = []
            total = len(self.sources)
            for index, source in enumerate(self.sources, start=1):
                self.token.throw_if_cancelled()
                settings = replace(self.settings)
                try:
                    if self.auto_scanner_revolutions:
                        revolutions, description = read_scanner_revolutions(source)
                        settings.scanner_revolutions_per_second = float(revolutions)
                        self.progressed.emit(
                            f"[{index}/{total}] {Path(source).name}：根据 "
                            f"{description.name} 设定为 {revolutions:.3f} 圈/s",
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
                    results.append((source, result))
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
                    self.output, tuple(results), tuple(failures)
                )
            )
        except Exception as error:
            self.failed.emit(str(error))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.worker: GenerationWorker | None = None
        self.last_output: str | None = None
        self.setWindowTitle("隧道点云大幅正射成图")
        self.resize(900, 680)
        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        title = QLabel("LAS / E57 → 隧道大幅正射影像")
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        layout.addWidget(title)
        note = QLabel(
            "本程序只负责大图成图，不包含标注、复核或 640×640 切片。"
            "结构化 EasyTry E57 按 columnIndex 和断面间距重建纵向坐标；"
            "LAS/普通 E57 仍要求 Y=隧道纵向米制坐标。"
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        source_box = QGroupBox("输入与输出")
        source_form = QFormLayout(source_box)
        self.source_edit = QPlainTextEdit()
        self.source_edit.setMaximumHeight(82)
        self.source_edit.setPlaceholderText("每行一个 LAS/E57 文件")
        source_row = QHBoxLayout()
        source_row.addWidget(self.source_edit, 1)
        choose_source = QPushButton("选择 LAS/E57（可多选）…")
        choose_source.clicked.connect(self._choose_source)
        source_row.addWidget(choose_source)
        inspect_button = QPushButton("检查输入")
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
        self.source_info = QLabel("尚未检查输入。")
        self.source_info.setWordWrap(True)
        source_form.addRow("输入信息", self.source_info)
        layout.addWidget(source_box)

        parameters = QGroupBox("成图参数")
        form = QFormLayout(parameters)
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
        self.auto_scanner_revolutions.setChecked(False)
        self.auto_scanner_revolutions.setToolTip(
            "从每个点云同目录的“文件名_扫描说明.txt”读取并四舍五入；"
            "批量处理时每个文件分别设定。"
        )
        self.auto_scanner_revolutions.toggled.connect(
            self._auto_scanner_revolutions_toggled
        )
        form.addRow("圈数来源", self.auto_scanner_revolutions)
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
        self.adaptive_single_image = QCheckBox("完整单图（超限时自动调整像素大小）")
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
        form.addRow("显示强度下限", self.display_min)
        form.addRow("显示强度上限", self.display_max)
        self.interpolation_distance = QDoubleSpinBox()
        self.interpolation_distance.setRange(0.0, 100.0)
        self.interpolation_distance.setDecimals(1)
        self.interpolation_distance.setValue(7.5)
        self.interpolation_distance.setSuffix(" mm")
        self.interpolation_distance.setToolTip(
            "只填补封闭小孔；0 表示关闭。科学强度与观测掩膜始终不修改。"
        )
        form.addRow("显示小孔最大插值距离", self.interpolation_distance)
        self.suppress_impulses = QCheckBox("抑制显示图中的孤立亮点和黑点")
        self.suppress_impulses.setChecked(True)
        self.suppress_impulses.setToolTip(
            "只修正显示 PNG 中孤立的单像素脉冲；科学强度、TIFF 与观测掩膜不修改。"
        )
        form.addRow("显示杂点处理", self.suppress_impulses)
        self.las_depth_display = QCheckBox("LAS 显示图启用前景深度簇聚合（科学数据保留）")
        self.las_depth_display.setChecked(True)
        form.addRow("LAS 管线边缘处理", self.las_depth_display)
        self.las_max_depth = QDoubleSpinBox()
        self.las_max_depth.setRange(0.1, 20.0)
        self.las_max_depth.setValue(2.0)
        self.las_max_depth.setSuffix(" m")
        form.addRow("LAS 显示最大相对深度", self.las_max_depth)
        self.las_cluster = QDoubleSpinBox()
        self.las_cluster.setRange(1.0, 500.0)
        self.las_cluster.setValue(30.0)
        self.las_cluster.setSuffix(" mm")
        form.addRow("LAS 前景深度簇厚度", self.las_cluster)
        self.las_antialias = QDoubleSpinBox()
        self.las_antialias.setRange(0.0, 1.0)
        self.las_antialias.setSingleStep(0.1)
        self.las_antialias.setValue(0.5)
        self.las_antialias.setToolTip("仅在前景深度边缘沿轮廓做亚像素采样；0 关闭。不是缺测重建，科学数据不修改。需启用 LAS 深度簇聚合。")
        form.addRow("LAS 轮廓抗锯齿强度", self.las_antialias)
        self.coordinate_confirmed = QCheckBox(
            "已确认 X/Z 为断面横向/竖向；LAS 或普通 E57 的 Y 为纵向米制坐标"
        )
        form.addRow("坐标确认", self.coordinate_confirmed)
        layout.addWidget(parameters)

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
        layout.addWidget(self.log, 1)

    def _choose_source(self) -> None:
        existing = self._source_paths()
        start = str(Path(existing[0]).parent) if existing else ""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择一个或多个点云", start, "点云 (*.las *.e57)"
        )
        if paths:
            self.source_edit.setPlainText("\n".join(paths))
            if not self.output_edit.text():
                self.output_edit.setText(str(Path(paths[0]).parent / "ortho_outputs"))
            if self.auto_scanner_revolutions.isChecked():
                self._sync_scanner_revolutions(show_errors=True)

    def _choose_output(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "选择输出父目录", self.output_edit.text()
        )
        if path:
            self.output_edit.setText(path)

    def _inspect_source(self) -> None:
        try:
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
        self.scanner_revolutions.setEnabled(not checked)
        if checked:
            self._sync_scanner_revolutions(show_errors=True)

    def _sync_scanner_revolutions(self, show_errors: bool) -> None:
        paths = self._source_paths()
        if not paths:
            return
        try:
            rates = [read_scanner_revolutions(path)[0] for path in paths]
            self.scanner_revolutions.setValue(float(rates[0]))
            if len(set(rates)) == 1:
                detail = f"已从扫描说明读取：{rates[0]:.3f} 圈/s"
            else:
                detail = (
                    f"批次内圈数不同（{min(rates):.3f}–{max(rates):.3f} 圈/s）；"
                    "成图时将按文件分别使用。"
                )
            self.auto_scanner_revolutions.setToolTip(detail)
            self._update_profile_spacing()
        except Exception as error:
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
        return GenerationSettings(
            pixel_size_m=float(self.pixel_size.currentText()) / 1000.0,
            image_length_m=self.image_length.value(),
            adaptive_single_image=self.adaptive_single_image.isChecked(),
            expected_diameter_m=self.diameter.value(),
            auto_estimate_diameter=self.auto_diameter.isChecked(),
            vehicle_speed_mps=self.vehicle_speed.value(),
            scanner_revolutions_per_second=self.scanner_revolutions.value(),
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
        )
        self.worker.progressed.connect(self._progressed)
        self.worker.succeeded.connect(self._succeeded)
        self.worker.failed.connect(self._failed)
        self.generate_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.worker.start()

    def _progressed(self, message: str, percent: int) -> None:
        self.progress.setValue(percent)
        self.log.appendPlainText(f"{percent:3d}%  {message}")

    def _succeeded(self, summary: BatchGenerationSummary) -> None:
        self.progress.setValue(100)
        self.last_output = summary.output_parent
        total_images = 0
        for source, result in summary.results:
            orthophotos = Path(result.output_directory) / "orthophotos"
            total_images += len(result.image_paths)
            self.log.appendPlainText(f"完成 {Path(source).name}：{orthophotos}")
            self.log.appendPlainText(
                f"  等效直径 {result.estimated_diameter_m:.3f} m | "
                f"实际像素 {result.actual_pixel_size_m * 1000:.3f} mm"
            )
        if summary.results:
            self.diameter.setValue(summary.results[-1][1].estimated_diameter_m)
        for source, error in summary.failures:
            self.log.appendPlainText(f"失败 {Path(source).name}：{error}")
        self.generate_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.open_button.setEnabled(bool(summary.results))
        message = (
            f"批量处理完成：{len(summary.results)} 个成功，"
            f"{len(summary.failures)} 个失败，共生成 {total_images} 幅影像。"
        )
        if summary.failures:
            QMessageBox.warning(self, "批量成图完成", message + "\n请查看日志中的失败原因。")
        else:
            QMessageBox.information(self, "批量成图完成", message)

    def _failed(self, message: str) -> None:
        self.log.appendPlainText("失败：" + message)
        self.generate_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
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
