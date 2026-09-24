from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class EllipseParameters:
    center_x: float
    center_z: float
    semi_axis_a: float
    semi_axis_b: float
    rotation_rad: float

    def __post_init__(self) -> None:
        values = (
            self.center_x,
            self.center_z,
            self.semi_axis_a,
            self.semi_axis_b,
            self.rotation_rad,
        )
        if not np.isfinite(values).all() or self.semi_axis_a <= 0 or self.semi_axis_b <= 0:
            raise ValueError("椭圆参数无效。")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    def world_to_local(self, x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dx = x - self.center_x
        dz = z - self.center_z
        cosine = np.cos(self.rotation_rad)
        sine = np.sin(self.rotation_rad)
        return cosine * dx + sine * dz, -sine * dx + cosine * dz

    def radial_parameter(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        local_x, local_z = self.world_to_local(x, z)
        return np.mod(
            np.arctan2(local_z / self.semi_axis_b, local_x / self.semi_axis_a),
            2.0 * np.pi,
        )

    def signed_relative_depth(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        local_x, local_z = self.world_to_local(x, z)
        radius = np.hypot(local_x, local_z)
        safe = np.maximum(radius, np.finfo(np.float64).eps)
        denominator = np.sqrt(
            (local_x / safe / self.semi_axis_a) ** 2
            + (local_z / safe / self.semi_axis_b) ** 2
        )
        reference = np.where(radius > 0, 1.0 / denominator, np.nan)
        return reference - radius


class EllipseArcLookup:
    def __init__(self, ellipse: EllipseParameters, sample_count: int = 65536) -> None:
        self.ellipse = ellipse
        self.parameter = np.linspace(0.0, 2.0 * np.pi, sample_count + 1)
        speed = np.sqrt(
            (ellipse.semi_axis_a * np.sin(self.parameter)) ** 2
            + (ellipse.semi_axis_b * np.cos(self.parameter)) ** 2
        )
        step = 2.0 * np.pi / sample_count
        self.cumulative = np.concatenate(
            ([0.0], np.cumsum(0.5 * (speed[:-1] + speed[1:]) * step))
        )
        self.circumference = float(self.cumulative[-1])
        # The image starts at the tunnel invert (world -Z) and runs counter-clockwise.
        seam_x = np.asarray([ellipse.center_x], dtype=np.float64)
        seam_z = np.asarray([ellipse.center_z - 1.0], dtype=np.float64)
        seam_parameter = float(ellipse.radial_parameter(seam_x, seam_z)[0])
        self.seam_arc = float(np.interp(seam_parameter, self.parameter, self.cumulative))

    def project(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        parameter = self.ellipse.radial_parameter(x, z)
        from_zero = np.interp(parameter, self.parameter, self.cumulative)
        return np.mod(from_zero - self.seam_arc, self.circumference)


def fit_reference_ellipse(
    points: np.ndarray,
    expected_diameter_m: float | None,
) -> tuple[EllipseParameters, dict]:
    if len(points) < 50:
        raise ValueError("有效断面样本不足，无法拟合参考椭圆。")
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("参考椭圆拟合需要 opencv-python。") from error

    rng = np.random.default_rng(20260911)
    score_points = points[
        np.linspace(0, len(points) - 1, min(len(points), 20_000), dtype=np.int64)
    ]
    best: EllipseParameters | None = None
    best_score = np.inf
    sample_size = min(12, len(points))
    for _ in range(256):
        indices = rng.choice(len(points), size=sample_size, replace=False)
        try:
            (cx, cz), (da, db), angle = cv2.fitEllipse(
                points[indices].astype(np.float32).reshape(-1, 1, 2)
            )
            candidate = EllipseParameters(float(cx), float(cz), float(da) / 2, float(db) / 2, float(np.deg2rad(angle)))
        except (ValueError, cv2.error):
            continue
        smaller = min(candidate.semi_axis_a, candidate.semi_axis_b)
        larger = max(candidate.semi_axis_a, candidate.semi_axis_b)
        if larger / smaller > 3.0:
            continue
        residual = np.abs(candidate.signed_relative_depth(score_points[:, 0], score_points[:, 1]))
        score = float(np.nanmedian(residual))
        if np.isfinite(score) and score < best_score:
            best, best_score = candidate, score
    if best is None:
        raise RuntimeError("参考椭圆初始拟合失败。")

    ellipse = best
    active = np.ones(len(points), dtype=bool)
    for _ in range(4):
        residual = np.abs(ellipse.signed_relative_depth(points[:, 0], points[:, 1]))
        finite = residual[np.isfinite(residual)]
        median = float(np.median(finite))
        mad = float(np.median(np.abs(finite - median)))
        limit = min(median + 3.5 * 1.4826 * max(mad, 1e-6), float(np.percentile(finite, 70)))
        active = np.isfinite(residual) & (residual <= limit)
        if active.sum() < 50:
            break
        (cx, cz), (da, db), angle = cv2.fitEllipse(
            points[active].astype(np.float32).reshape(-1, 1, 2)
        )
        ellipse = EllipseParameters(float(cx), float(cz), float(da) / 2, float(db) / 2, float(np.deg2rad(angle)))

    smaller = min(ellipse.semi_axis_a, ellipse.semi_axis_b)
    larger = max(ellipse.semi_axis_a, ellipse.semi_axis_b)
    axis_ratio = larger / smaller
    estimated_diameter = 2.0 * float(np.sqrt(smaller * larger))
    residual = np.abs(ellipse.signed_relative_depth(points[:, 0], points[:, 1]))
    finite_mask = np.isfinite(residual)
    finite_residual = residual[finite_mask]
    median = float(np.median(finite_residual))
    mad = float(np.median(np.abs(finite_residual - median)))
    limit = min(
        median + 3.5 * 1.4826 * max(mad, 1e-6),
        float(np.percentile(finite_residual, 70)),
    )
    # Tunnel scans also contain rails, cable trays and equipment.  They are useful
    # image content, but are not observations of the reference tunnel envelope.
    # Validate the ellipse against the final wall inliers only.
    active = finite_mask & (residual <= limit)
    if active.sum() < 50:
        raise RuntimeError("参考椭圆有效墙面内点不足。")
    p95 = float(np.percentile(residual[active], 95))
    if expected_diameter_m is not None:
        expected_radius = expected_diameter_m / 2.0
        if axis_ratio > 1.5:
            raise RuntimeError("参考椭圆长短半轴比超过 1.5，请检查坐标方向或点云范围。")
        if smaller < 0.65 * expected_radius or larger > 1.40 * expected_radius:
            raise RuntimeError(
                f"拟合半轴 {smaller:.3f}/{larger:.3f} m 与预期直径 {expected_diameter_m:.3f} m 不符。"
            )
    if not np.isfinite(p95):
        raise RuntimeError("参考椭圆拟合残差无效。")
    if expected_diameter_m is not None and p95 > 0.25:
        raise RuntimeError(f"参考椭圆拟合残差过大：P95={p95:.3f} m。")
    if expected_diameter_m is None and p95 > max(0.75, 0.15 * estimated_diameter):
        raise RuntimeError(
            f"无法可靠地自动估算直径：参考椭圆残差 P95={p95:.3f} m。"
        )
    warnings = []
    if expected_diameter_m is None and axis_ratio > 1.5:
        warnings.append("自动拟合的长短半轴比超过 1.5，建议检查坐标方向和点云范围。")
    if expected_diameter_m is None and p95 > 0.25:
        warnings.append(f"拟合残差 P95={p95:.3f} m，建议在成图后抽查展开几何。")
    return ellipse, {
        "method": "ransac_iterative_fit",
        "diameter_mode": "automatic" if expected_diameter_m is None else "manual_expected",
        "sample_count": int(len(points)),
        "inlier_count": int(active.sum()),
        "residual_p95_m": p95,
        "estimated_equivalent_diameter_m": estimated_diameter,
        "fitted_width_m": 2.0 * larger,
        "fitted_height_m": 2.0 * smaller,
        "axis_ratio": axis_ratio,
        "warning": " ".join(warnings) or None,
    }
