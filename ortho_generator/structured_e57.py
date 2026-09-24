"""Reader for EasyTry E57 files that retain scanner row/column indices."""

from __future__ import annotations

from pathlib import Path
import struct
import subprocess

import numpy as np

from .cancellation import CancellationToken
from .models import PointBatch, PointCloudMetadata


_HEADER = struct.Struct("<8sIIQI")
_FRAME = struct.Struct("<Q")
_MAGIC = b"ETRDR001"
_FIELD_MASK = 0x7F


def reader_path() -> Path:
    return Path(__file__).resolve().parent / "native" / "x64" / "EasyTryE57Reader.exe"


def reader_available() -> bool:
    executable = reader_path()
    return executable.is_file() and (executable.parent / "xerces-c_3_2.dll").is_file()


class StructuredE57Source:
    """Streams XYZ, intensity, and exact scanner indices from an EasyTry E57."""

    def __init__(self, source_path: str | Path) -> None:
        self.path = Path(source_path)
        self._metadata: PointCloudMetadata | None = None

    def _start(self, capacity: int) -> subprocess.Popen:
        executable = reader_path()
        if not reader_available():
            raise RuntimeError(f"找不到结构化 E57 读取器或其运行库：{executable}")
        return subprocess.Popen(
            [str(executable), str(self.path), str(capacity)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            bufsize=0,
        )

    @staticmethod
    def _read_exact(stream, count: int) -> bytes:
        result = bytearray()
        while len(result) < count:
            chunk = stream.read(count - len(result))
            if not chunk:
                raise RuntimeError("结构化 E57 读取器输出意外中断。")
            result.extend(chunk)
        return bytes(result)

    @staticmethod
    def _process_error(process: subprocess.Popen) -> str:
        assert process.stderr is not None
        detail = process.stderr.read().decode("utf-8", errors="replace").strip()
        return detail or f"结构化 E57 读取器退出代码：{process.returncode}"

    def _read_header(self, process: subprocess.Popen) -> tuple[int, int, int]:
        assert process.stdout is not None
        try:
            magic, rows, columns, point_count, field_mask = _HEADER.unpack(
                self._read_exact(process.stdout, _HEADER.size)
            )
        except Exception as error:
            if process.poll() is None:
                process.wait(timeout=10)
            detail = self._process_error(process)
            raise RuntimeError(detail) from error
        if magic != _MAGIC or field_mask != _FIELD_MASK:
            raise RuntimeError("E57 不包含 EasyTry 所需的结构化扫描字段。")
        if rows <= 0 or columns <= 0 or point_count <= 0:
            raise RuntimeError("E57 的结构化扫描范围无效。")
        return int(rows), int(columns), int(point_count)

    def open(self) -> PointCloudMetadata:
        if not self.path.is_file():
            raise ValueError(f"E57 文件不存在：{self.path}")
        process = self._start(1)
        try:
            rows, columns, point_count = self._read_header(process)
            self._metadata = PointCloudMetadata(
                source_path=str(self.path.resolve()),
                source_format="E57",
                point_count=point_count,
                scan_count=1,
                intensity_source="intensity",
                scan_rows=rows,
                scan_columns=columns,
                structured_indices=True,
            )
            return self._metadata
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate()

    @property
    def metadata(self) -> PointCloudMetadata:
        return self._metadata or self.open()

    def iter_batches(
        self,
        batch_size: int = 262_144,
        cancellation: CancellationToken | None = None,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0。")
        token = cancellation or CancellationToken()
        process = self._start(batch_size)
        assert process.stdout is not None
        try:
            rows, columns, point_count = self._read_header(process)
            if self._metadata is None:
                self._metadata = PointCloudMetadata(
                    str(self.path.resolve()), "E57", point_count, 1, "intensity",
                    rows, columns, True,
                )
            while True:
                token.throw_if_cancelled()
                count = _FRAME.unpack(
                    self._read_exact(process.stdout, _FRAME.size)
                )[0]
                if count == 0:
                    break
                size = int(count)
                arrays = [
                    np.frombuffer(
                        self._read_exact(process.stdout, size * 8), dtype="<f8"
                    ).copy()
                    for _ in range(3)
                ]
                processed = np.frombuffer(
                    self._read_exact(process.stdout, size * 4), dtype="<f4"
                ).astype(np.float64)
                # rawIntensity is retained in the E57, but orthophoto generation uses
                # the converter's processed intensity field.
                self._read_exact(process.stdout, size * 8)
                row_index = np.frombuffer(
                    self._read_exact(process.stdout, size * 4), dtype="<u4"
                ).copy()
                column_index = np.frombuffer(
                    self._read_exact(process.stdout, size * 4), dtype="<u4"
                ).copy()
                yield PointBatch(
                    x=arrays[0],
                    y=arrays[1],
                    z=arrays[2],
                    intensity=processed,
                    row_index=row_index,
                    column_index=column_index,
                )
            return_code = process.wait(timeout=30)
            if return_code != 0:
                raise RuntimeError(self._process_error(process))
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate()

    def close(self) -> None:
        self._metadata = None
