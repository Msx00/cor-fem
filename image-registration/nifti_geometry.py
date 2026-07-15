"""Explicit NIfTI geometry conversions for physical-space FEM registration.

SimpleITK image indices are ``(x, y, z)`` and physical points are LPS
millimetres.  Arrays returned by ``GetArrayFromImage`` are ``(z, y, x)``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import SimpleITK as sitk


@dataclass(frozen=True)
class NiftiGeometry:
    size_xyz: tuple[int, int, int]
    spacing_xyz: tuple[float, float, float]
    origin_lps_mm: tuple[float, float, float]
    direction_lps: tuple[float, ...]

    @property
    def direction_matrix(self) -> np.ndarray:
        return np.asarray(self.direction_lps, dtype=np.float64).reshape(3, 3)

    @property
    def index_to_physical_matrix(self) -> np.ndarray:
        return self.direction_matrix @ np.diag(self.spacing_xyz)

    @property
    def physical_to_index_matrix(self) -> np.ndarray:
        return np.linalg.inv(self.index_to_physical_matrix)

    def to_dict(self) -> dict:
        result = asdict(self)
        result.update(
            coordinate_system="LPS",
            physical_unit="mm",
            image_index_axis_order="xyz",
            numpy_array_axis_order="zyx",
        )
        return result


def geometry_from_image(image: sitk.Image) -> NiftiGeometry:
    if image.GetDimension() != 3:
        raise ValueError(f"Expected a 3D image, got dimension={image.GetDimension()}")
    return NiftiGeometry(
        size_xyz=tuple(int(value) for value in image.GetSize()),
        spacing_xyz=tuple(float(value) for value in image.GetSpacing()),
        origin_lps_mm=tuple(float(value) for value in image.GetOrigin()),
        direction_lps=tuple(float(value) for value in image.GetDirection()),
    )


def read_nifti(path: str | Path) -> tuple[sitk.Image, NiftiGeometry]:
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"NIfTI file not found: {image_path}")
    image = sitk.ReadImage(str(image_path))
    return image, geometry_from_image(image)


def array_zyx_to_index_xyz(values: np.ndarray | Sequence[float]) -> np.ndarray:
    points = np.asarray(values)
    if points.shape[-1] != 3:
        raise ValueError(f"Expected (..., 3) coordinates, got {points.shape}")
    return points[..., ::-1].copy()


def index_xyz_to_array_zyx(values: np.ndarray | Sequence[float]) -> np.ndarray:
    points = np.asarray(values)
    if points.shape[-1] != 3:
        raise ValueError(f"Expected (..., 3) coordinates, got {points.shape}")
    return points[..., ::-1].copy()


def index_xyz_to_physical(
    indices_xyz: np.ndarray | Sequence[float],
    geometry: NiftiGeometry,
) -> np.ndarray:
    indices = np.asarray(indices_xyz, dtype=np.float64)
    if indices.shape[-1] != 3:
        raise ValueError(f"Expected (..., 3) indices, got {indices.shape}")
    origin = np.asarray(geometry.origin_lps_mm, dtype=np.float64)
    return indices @ geometry.index_to_physical_matrix.T + origin


def physical_to_continuous_index_xyz(
    physical_lps_mm: np.ndarray | Sequence[float],
    geometry: NiftiGeometry,
) -> np.ndarray:
    points = np.asarray(physical_lps_mm, dtype=np.float64)
    if points.shape[-1] != 3:
        raise ValueError(f"Expected (..., 3) points, got {points.shape}")
    origin = np.asarray(geometry.origin_lps_mm, dtype=np.float64)
    return (points - origin) @ geometry.physical_to_index_matrix.T


def physical_bounds_lps_mm(geometry: NiftiGeometry) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounds including half-voxel physical extent."""
    size = np.asarray(geometry.size_xyz, dtype=np.float64)
    low = np.full(3, -0.5, dtype=np.float64)
    high = size - 0.5
    corners = np.asarray(
        [
            (x, y, z)
            for x in (low[0], high[0])
            for y in (low[1], high[1])
            for z in (low[2], high[2])
        ],
        dtype=np.float64,
    )
    physical = index_xyz_to_physical(corners, geometry)
    return physical.min(axis=0), physical.max(axis=0)


def geometries_equal(
    first: NiftiGeometry,
    second: NiftiGeometry,
    atol: float = 1e-6,
) -> bool:
    return bool(
        first.size_xyz == second.size_xyz
        and np.allclose(first.spacing_xyz, second.spacing_xyz, atol=atol, rtol=0.0)
        and np.allclose(first.origin_lps_mm, second.origin_lps_mm, atol=atol, rtol=0.0)
        and np.allclose(first.direction_lps, second.direction_lps, atol=atol, rtol=0.0)
    )


def image_geometry_summary(path: str | Path) -> dict:
    image, geometry = read_nifti(path)
    bounds_min, bounds_max = physical_bounds_lps_mm(geometry)
    return {
        "path": str(Path(path).resolve()),
        "pixel_type": image.GetPixelIDTypeAsString(),
        **geometry.to_dict(),
        "physical_bounds_lps_mm": {
            "min": bounds_min.tolist(),
            "max": bounds_max.tolist(),
        },
    }
