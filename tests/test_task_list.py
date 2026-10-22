from pathlib import Path

from ortho_generator.task_list import (
    OrthoTaskSpec,
    expand_task,
    read_task_list,
    remove_task_specs,
    remove_generated_input,
)


def test_read_and_expand_task_list(tmp_path: Path) -> None:
    source = tmp_path / "输入 数据"
    source.mkdir()
    (source / "b.e57").touch()
    (source / "a.e57").touch()
    task_list = tmp_path / "tasks.txt"
    task_list.write_text(
        f"{source.as_posix()}/*.e57\t{tmp_path / 'ortho_outputs'}\n",
        encoding="utf-8-sig",
    )

    specs = read_task_list(task_list)

    assert len(specs) == 1
    assert [Path(path).name for path in expand_task(specs[0])] == ["a.e57", "b.e57"]


def test_remove_generated_input_removes_matching_description(tmp_path: Path) -> None:
    source = tmp_path / "scan.e57"
    description = tmp_path / "scan_扫描说明.txt"
    source.touch()
    description.touch()

    removed = remove_generated_input(str(source))

    assert set(map(Path, removed)) == {source, description}
    assert not source.exists()
    assert not description.exists()


def test_task_list_requires_tab_separator(tmp_path: Path) -> None:
    task_list = tmp_path / "tasks.txt"
    task_list.write_text("input output\n", encoding="utf-8")

    try:
        read_task_list(task_list)
    except ValueError as error:
        assert "Tab" in str(error)
    else:
        raise AssertionError("missing separator must fail")


def test_remove_selected_task_specs(tmp_path: Path) -> None:
    task_list = tmp_path / "tasks.txt"
    first = OrthoTaskSpec("E:/Data/a/*.e57", str(tmp_path / "a_out"))
    second = OrthoTaskSpec("E:/Data/b/*.e57", str(tmp_path / "b_out"))
    task_list.write_text(
        f"{first.input_pattern}\t{first.output_parent}\n"
        f"{second.input_pattern}\t{second.output_parent}\n",
        encoding="utf-8-sig",
    )

    removed = remove_task_specs(task_list, (first,))

    assert removed == 1
    assert read_task_list(task_list) == (second,)
