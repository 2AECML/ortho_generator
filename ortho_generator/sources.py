from __future__ import annotations

from pathlib import Path

from .las_source import LasSource
from .native_e57 import E57Source as GenericE57Source
from .structured_e57 import StructuredE57Source, reader_available


class E57Source:
    """Prefer exact scanner indices and fall back to the generic E57 reader."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._delegate = None

    def open(self):
        if reader_available():
            structured = StructuredE57Source(self.path)
            try:
                metadata = structured.open()
                self._delegate = structured
                return metadata
            except (RuntimeError, ValueError):
                structured.close()
        generic = GenericE57Source(self.path)
        metadata = generic.open()
        self._delegate = generic
        return metadata

    @property
    def metadata(self):
        return self._delegate.metadata if self._delegate is not None else self.open()

    def iter_batches(self, batch_size=262_144, cancellation=None):
        if self._delegate is None:
            self.open()
        yield from self._delegate.iter_batches(batch_size, cancellation)

    def close(self) -> None:
        if self._delegate is not None:
            self._delegate.close()
            self._delegate = None


def create_source(path: str | Path):
    suffix = Path(path).suffix.lower()
    if suffix == ".las":
        return LasSource(path)
    if suffix == ".e57":
        return E57Source(path)
    if suffix == ".laz":
        raise ValueError("当前版本支持未压缩 LAS，不支持 LAZ。请先转换为 .las。")
    raise ValueError("只支持 .las 或 .e57 点云。")
