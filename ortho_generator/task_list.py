from __future__ import annotations

import glob
import shutil
from dataclasses import dataclass
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


def require_generated_images(image_paths: tuple[str, ...]) -> None:
    if not image_paths or not all(Path(path).is_file() for path in image_paths):
        raise RuntimeError("没有确认到完整的正射影像输出，保留输入点云。")


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
