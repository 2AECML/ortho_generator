"""Binary protocol wrapper for the bundled generic libE57Format reader."""

from __future__ import annotations

from pathlib import Path
import struct
import subprocess

import numpy as np

from .cancellation import CancellationToken
from .models import PointBatch, PointCloudMetadata


_HEADER = struct.Struct("<8sIIQ")
_FRAME = struct.Struct("<Q")
_MAGIC = b"ORTH57V1"


def reader_path() -> Path:
    return Path(__file__).resolve().parent / "native" / "x64" / "OrthoE57Reader.exe"


class E57Source:
    def __init__(self, source_path: str | Path) -> None:
        self.path = Path(source_path)
        self._metadata: PointCloudMetadata | None = None

    def _start(self, capacity: int) -> subprocess.Popen:
        executable = reader_path()
        if not executable.is_file():
            raise RuntimeError(f"找不到通用 E57 读取器：{executable}")
        return subprocess.Popen(
            [str(executable), str(self.path), str(capacity)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    @staticmethod
    def _read_exact(stream, count: int) -> bytes:
        data = stream.read(count)
        if len(data) != count:
            raise RuntimeError("E57 读取器输出意外中断。")
        return data

    def open(self) -> PointCloudMetadata:
        if not self.path.is_file():
            raise ValueError(f"E57 文件不存在：{self.path}")
        process = self._start(1)
        assert process.stdout is not None
        try:
            magic, scan_count, flags, point_count = _HEADER.unpack(
                self._read_exact(process.stdout, _HEADER.size)
            )
            if magic != _MAGIC:
                raise RuntimeError("E57 读取器协议不匹配。")
            intensity_source = "intensity" if flags & 1 else "rgb_luminance"
            self._metadata = PointCloudMetadata(
                source_path=str(self.path.resolve()),
                source_format="E57",
                point_count=int(point_count),
                scan_count=int(scan_count),
                intensity_source=intensity_source,
            )
            return self._metadata
        finally:
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
        assert process.stdout is not None and process.stderr is not None
        try:
            magic, scan_count, flags, point_count = _HEADER.unpack(
                self._read_exact(process.stdout, _HEADER.size)
            )
            if magic != _MAGIC:
                raise RuntimeError("E57 读取器协议不匹配。")
            if self._metadata is None:
                self._metadata = PointCloudMetadata(
                    str(self.path.resolve()), "E57", int(point_count), int(scan_count),
                    "intensity" if flags & 1 else "rgb_luminance",
                )
            while True:
                token.throw_if_cancelled()
                count = _FRAME.unpack(self._read_exact(process.stdout, _FRAME.size))[0]
                if count == 0:
                    break
                byte_count = count * 8
                arrays = [
                    np.frombuffer(self._read_exact(process.stdout, byte_count), dtype="<f8").copy()
                    for _ in range(4)
                ]
                yield PointBatch(*arrays)
            return_code = process.wait(timeout=10)
            if return_code != 0:
                detail = process.stderr.read().decode("utf-8", errors="replace").strip()
                raise RuntimeError(detail or f"E57 读取器退出代码：{return_code}")
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate()

    def close(self) -> None:
        self._metadata = None
