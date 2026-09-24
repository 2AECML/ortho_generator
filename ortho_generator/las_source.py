"""Bounded-memory reader for uncompressed LAS 1.0-1.4 point formats 0-10."""

from __future__ import annotations

from pathlib import Path
import struct

import numpy as np

from .cancellation import CancellationToken
from .models import PointBatch, PointCloudMetadata


class LasSource:
    def __init__(self, source_path: str | Path) -> None:
        self.path = Path(source_path)
        self._metadata: PointCloudMetadata | None = None
        self._offset = 0
        self._record_length = 0
        self._point_format = 0
        self._point_count = 0
        self._scale = np.ones(3, dtype=np.float64)
        self._xyz_offset = np.zeros(3, dtype=np.float64)

    def open(self) -> PointCloudMetadata:
        if not self.path.is_file():
            raise ValueError(f"LAS 文件不存在：{self.path}")
        with self.path.open("rb") as stream:
            header = stream.read(375)
        if len(header) < 227 or header[:4] != b"LASF":
            raise ValueError("文件不是有效的 LAS。")
        header_size = struct.unpack_from("<H", header, 94)[0]
        self._offset = struct.unpack_from("<I", header, 96)[0]
        raw_format = header[104]
        if raw_format & 0x80:
            raise ValueError("检测到压缩 LAS/LAZ 数据；当前版本只读取未压缩 .las。")
        self._point_format = raw_format & 0x3F
        if self._point_format > 10:
            raise ValueError(f"不支持 LAS 点格式 {self._point_format}。")
        self._record_length = struct.unpack_from("<H", header, 105)[0]
        legacy_count = struct.unpack_from("<I", header, 107)[0]
        self._point_count = legacy_count
        if header_size >= 375 and len(header) >= 255:
            extended_count = struct.unpack_from("<Q", header, 247)[0]
            if extended_count:
                self._point_count = extended_count
        if self._point_count <= 0 or self._record_length < 14:
            raise ValueError("LAS 点记录或点数无效。")
        self._scale = np.asarray(struct.unpack_from("<3d", header, 131), dtype=np.float64)
        self._xyz_offset = np.asarray(struct.unpack_from("<3d", header, 155), dtype=np.float64)
        expected = self._offset + self._point_count * self._record_length
        if self.path.stat().st_size < expected:
            raise ValueError("LAS 文件长度小于文件头声明。")
        self._metadata = PointCloudMetadata(
            source_path=str(self.path.resolve()),
            source_format="LAS",
            point_count=int(self._point_count),
            scan_count=1,
            intensity_source="intensity",
        )
        return self._metadata

    @property
    def metadata(self) -> PointCloudMetadata:
        return self._metadata or self.open()

    def iter_batches(
        self,
        batch_size: int = 1_000_000,
        cancellation: CancellationToken | None = None,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0。")
        if self._metadata is None:
            self.open()
        token = cancellation or CancellationToken()
        dtype = np.dtype(
            {
                "names": ["X", "Y", "Z", "intensity", "tail"],
                "formats": ["<i4", "<i4", "<i4", "<u2", ("u1", self._record_length - 14)],
                "offsets": [0, 4, 8, 12, 14],
                "itemsize": self._record_length,
            }
        )
        points = np.memmap(
            self.path,
            dtype=dtype,
            mode="r",
            offset=self._offset,
            shape=(self._point_count,),
        )
        try:
            for start in range(0, self._point_count, batch_size):
                token.throw_if_cancelled()
                part = points[start : min(self._point_count, start + batch_size)]
                xyz = np.column_stack((part["X"], part["Y"], part["Z"])).astype(np.float64)
                xyz *= self._scale
                xyz += self._xyz_offset
                yield PointBatch(
                    x=np.ascontiguousarray(xyz[:, 0]),
                    y=np.ascontiguousarray(xyz[:, 1]),
                    z=np.ascontiguousarray(xyz[:, 2]),
                    intensity=part["intensity"].astype(np.float64),
                )
        finally:
            del points

    def close(self) -> None:
        self._metadata = None
