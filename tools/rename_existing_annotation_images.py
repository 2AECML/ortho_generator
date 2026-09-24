from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def rename_run(run_directory: Path) -> list[Path]:
    run_directory = run_directory.resolve(strict=True)
    manifest_path = run_directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"缺少总清单：{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_path = Path(manifest["input_path"])
    source_stem = source_path.stem
    image_entries = list(manifest.get("images", []))
    if not image_entries:
        raise ValueError("总清单中没有正射影像。")

    backup_manifest = run_directory / "manifest_before_annotation_naming.json"
    if not backup_manifest.exists():
        shutil.copy2(manifest_path, backup_manifest)

    renamed: list[Path] = []
    updated_entries: list[str] = []
    for sequence_index, entry in enumerate(image_entries, start=1):
        old_path = run_directory / entry
        image_id = f"ORTHO_{sequence_index:05d}"
        modern_auxiliary = run_directory / "auxiliary" / image_id
        image_directory = modern_auxiliary if (modern_auxiliary / "image.json").is_file() else old_path.parent
        metadata_path = image_directory / "image.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        image_id = metadata.get("image_id", image_id)
        current_name = metadata.get("annotation_image", metadata.get("image", old_path.name))
        current_path = (image_directory / current_name).resolve()
        new_path = old_path.parent / f"{source_stem}_{image_id}.png"

        backup_metadata = image_directory / "image_before_annotation_naming.json"
        if not backup_metadata.exists():
            shutil.copy2(metadata_path, backup_metadata)
        if current_path != new_path:
            if new_path.exists():
                raise FileExistsError(f"目标标注图已存在：{new_path}")
            current_path.replace(new_path)

        metadata.update(
            image=Path(os.path.relpath(new_path, image_directory)).as_posix(),
            annotation_image=Path(os.path.relpath(new_path, image_directory)).as_posix(),
            source_point_cloud=str(source_path),
            sequence_index=sequence_index,
            sequence_count=len(image_entries),
        )
        _write_json_atomic(metadata_path, metadata)
        renamed.append(new_path)
        updated_entries.append(new_path.relative_to(run_directory).as_posix())

    manifest["images"] = updated_entries
    manifest["annotation_naming"] = {
        "pattern": "{source_stem}_ORTHO_{sequence:05d}.png",
        "source_point_cloud": str(source_path),
        "sequence_count": len(updated_entries),
    }
    _write_json_atomic(manifest_path, manifest)
    return renamed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将旧运行中的 image.png 改为点云名加连续序号。"
    )
    parser.add_argument("run_directory", type=Path)
    arguments = parser.parse_args()
    for path in rename_run(arguments.run_directory):
        print(path)


if __name__ == "__main__":
    main()
