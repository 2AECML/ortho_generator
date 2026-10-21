from __future__ import annotations

import glob
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


MINIMUM_FREE_BYTES = 10 * 1024**3


@dataclass(frozen=True, slots=True)
class OrthoTaskSpec:
    input_pattern: str
    output_parent: str


def read_task_list(path: str | Path) -> tuple[OrthoTaskSpec, ...]:
    task_path = Path(path).resolve(strict=True)
    tasks: list[OrthoTaskSpec] = []
    for line_number, raw_line in enumerate(
        task_path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        columns = raw_line.split("\t")
        if len(columns) != 2 or not all(column.strip() for column in columns):
            raise ValueError(f"任务清单第 {line_number} 行必须用一个 Tab 分隔两列。")
        tasks.append(OrthoTaskSpec(columns[0].strip(), columns[1].strip()))
    if not tasks:
        raise ValueError("任务清单中没有有效任务。")
    return tuple(tasks)


def expand_task(spec: OrthoTaskSpec) -> tuple[str, ...]:
    paths = {
        str(Path(path).resolve())
        for path in glob.glob(spec.input_pattern)
        if Path(path).is_file() and Path(path).suffix.lower() in (".las", ".e57")
    }
    return tuple(sorted(paths, key=str.casefold))


def journal_path_for(task_list_path: str | Path) -> Path:
    path = Path(task_list_path).resolve()
    log_directory = Path(__file__).resolve().parent.parent / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    return log_directory / f"{path.stem}_执行日志.txt"


def remove_task_specs(
    task_list_path: str | Path, specs: tuple[OrthoTaskSpec, ...]
) -> int:
    if not specs:
        return 0
    path = Path(task_list_path).resolve(strict=True)
    keys = {
        (spec.input_pattern.strip().casefold(), str(Path(spec.output_parent).resolve()).casefold())
        for spec in specs
    }
    retained: list[str] = []
    removed = 0
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        columns = line.split("\t")
        key = None
        if len(columns) == 2:
            key = (
                columns[0].strip().casefold(),
                str(Path(columns[1].strip()).resolve()).casefold(),
            )
        if key in keys:
            removed += 1
        else:
            retained.append(line)
    content = "\n".join(retained)
    if retained:
        content += "\n"
    path.write_text(content, encoding="utf-8-sig")
    return removed


class TaskJournal:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, status: str, source: str, output: str, message: str = "") -> None:
        clean = message.replace("\t", " ").replace("\r", " ").replace("\n", " ")
        line = (
            f"{datetime.now():%Y-%m-%d %H:%M:%S}\t{status}\t"
            f"{source}\t{output}\t{clean}\n"
        )
        with self.path.open("a", encoding="utf-8-sig", newline="") as stream:
            stream.write(line)

    def completed(self) -> set[tuple[str, str]]:
        return self.keys_with_status("SUCCESS")

    def keys_with_status(self, status: str) -> set[tuple[str, str]]:
        result: set[tuple[str, str]] = set()
        for columns in self._rows():
            if columns[1] == status:
                result.add((_normal(columns[2]), _normal(columns[3])))
        return result

    def record_generated(self, source: str, output: str, image_paths: tuple[str, ...]) -> None:
        self.append("GENERATED", source, output, json.dumps(
            {"image_paths": list(image_paths)}, ensure_ascii=False,
        ))

    def generated_images(self) -> dict[tuple[str, str], tuple[str, ...]]:
        result: dict[tuple[str, str], tuple[str, ...]] = {}
        for columns in self._rows():
            if columns[1] != "GENERATED" or len(columns) < 5:
                continue
            key = (_normal(columns[2]), _normal(columns[3]))
            result.pop(key, None)
            try:
                payload = json.loads(columns[4])
            except (ValueError, TypeError):
                continue
            paths = payload.get("image_paths") if isinstance(payload, dict) else None
            if isinstance(paths, list) and paths and all(isinstance(p, str) for p in paths):
                result[key] = tuple(paths)
        return result

    def matching_sources(self, spec: OrthoTaskSpec, statuses: tuple[str, ...]) -> tuple[str, ...]:
        output = _normal(spec.output_parent)
        pattern = str(Path(spec.input_pattern).resolve())
        matches = {}
        output_names = {}
        for columns in self._rows():
            if columns[1] not in statuses:
                continue
            if columns[3] not in output_names:
                output_names[columns[3]] = _normal(columns[3])
            if output_names[columns[3]] != output:
                continue
            source = Path(columns[2]).resolve()
            if source.match(pattern) and source.suffix.lower() in (".e57", ".las"):
                matches[_normal(str(source))] = str(source)
        return tuple(matches.values())

    def _rows(self) -> list[list[str]]:
        if not self.path.exists():
            return []
        return [columns for line in self.path.read_text(encoding="utf-8-sig").splitlines()
                if len(columns := line.split("\t", 4)) >= 4]


def require_generated_images(image_paths: tuple[str, ...]) -> None:
    if not image_paths or not all(Path(path).is_file() for path in image_paths):
        raise RuntimeError("没有确认到完整的正射影像输出，保留输入点云。")


def completed_image_outputs(output_parent: str) -> dict[str, tuple[str, ...]]:
    """Recover output evidence from manifests written before image paths were journaled."""
    root = Path(output_parent).resolve()
    result: dict[str, tuple[str, ...]] = {}
    manifests = list(root.glob("manifest_*.json")) + list(root.glob("*/manifest.json"))
    for manifest in manifests:
        try:
            data = json.loads(manifest.read_text(encoding="utf-8-sig"))
            if not isinstance(data, dict) or data.get("status") != "COMPLETED":
                continue
            source, images = data.get("input_path"), data.get("images")
            if (not isinstance(source, str) or not isinstance(images, list) or not images
                    or not all(isinstance(path, str) for path in images)):
                continue
            paths = tuple((manifest.parent / path).resolve() for path in images)
            if all(path.is_relative_to(root) and path.suffix.lower() == ".png"
                   and path.is_file() for path in paths):
                result[_normal(source)] = tuple(map(str, paths))
        except (OSError, ValueError):
            continue
    return result


def has_safe_free_space(output_parent: str, minimum: int = MINIMUM_FREE_BYTES) -> tuple[bool, int]:
    path = Path(output_parent).resolve()
    existing = path
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    free = shutil.disk_usage(existing).free
    return free >= minimum, free


def remove_generated_input(source: str) -> tuple[str, ...]:
    point_cloud = Path(source).resolve()
    if point_cloud.suffix.lower() not in (".e57", ".las"):
        raise ValueError(f"不支持清理的输入点云格式：{point_cloud}")
    removed: list[str] = []
    try:
        point_cloud.unlink()
    except FileNotFoundError:
        pass
    else:
        removed.append(str(point_cloud))
    description = point_cloud.with_name(f"{point_cloud.stem}_扫描说明.txt")
    if description.is_file():
        description.unlink()
        removed.append(str(description))
    return tuple(removed)


def _normal(path: str) -> str:
    return str(Path(path).resolve()).casefold()
