"""Inverse tetrahedral mapping and pull resampling for deformed FEM meshes.

All point coordinates are ``(x, y, z)`` physical LPS millimetres.  Numpy image
arrays remain ``(z, y, x)`` and are converted explicitly through
``nifti_geometry``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pyvista as pv
from scipy.ndimage import map_coordinates
import SimpleITK as sitk

from nifti_geometry import (
    geometry_from_image,
    index_xyz_to_array_zyx,
    index_xyz_to_physical,
    physical_to_continuous_index_xyz,
)


OutsideMode = Literal["identity", "zero", "nearest"]
InterpolationMode = Literal["linear", "nearest"]


@dataclass
class TetrahedralLocator:
    reference_nodes: np.ndarray
    deformed_nodes: np.ndarray
    tetrahedra: np.ndarray
    deformed_grid: pv.UnstructuredGrid


def _make_tetrahedral_grid(points: np.ndarray, tetrahedra: np.ndarray) -> pv.UnstructuredGrid:
    count = tetrahedra.shape[0]
    cells = np.hstack(
        [np.full((count, 1), 4, dtype=np.int64), tetrahedra.astype(np.int64)]
    ).ravel()
    cell_types = np.full(count, pv.CellType.TETRA, dtype=np.uint8)
    return pv.UnstructuredGrid(cells, cell_types, points.astype(np.float64))


def validate_tetrahedral_deformation(
    reference_nodes: np.ndarray,
    deformed_nodes: np.ndarray,
    tetrahedra: np.ndarray,
) -> None:
    reference = np.asarray(reference_nodes)
    deformed = np.asarray(deformed_nodes)
    tets = np.asarray(tetrahedra)
    if reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError(f"reference_nodes must have shape (N, 3), got {reference.shape}")
    if deformed.shape != reference.shape:
        raise ValueError(
            "reference_nodes and deformed_nodes must have identical shapes, "
            f"got {reference.shape} and {deformed.shape}"
        )
    if tets.ndim != 2 or tets.shape[1] != 4:
        raise ValueError(f"tetrahedra must have shape (T, 4), got {tets.shape}")
    if reference.shape[0] == 0 or tets.shape[0] == 0:
        raise ValueError("The tetrahedral deformation must not be empty")
    if not np.issubdtype(tets.dtype, np.integer):
        raise TypeError("tetrahedra must contain integer node indices")
    if int(tets.min()) < 0 or int(tets.max()) >= reference.shape[0]:
        raise IndexError(
            f"tetrahedra indices must be in [0, {reference.shape[0] - 1}], "
            f"got [{int(tets.min())}, {int(tets.max())}]"
        )
    if not np.isfinite(reference).all() or not np.isfinite(deformed).all():
        raise ValueError("reference_nodes/deformed_nodes contain NaN or Inf")


def prepare_tetrahedral_locator(
    reference_nodes: np.ndarray,
    deformed_nodes: np.ndarray,
    tetrahedra: np.ndarray,
) -> TetrahedralLocator:
    """Build a VTK-backed spatial locator over the deformed tetrahedral mesh."""
    reference = np.asarray(reference_nodes, dtype=np.float64)
    deformed = np.asarray(deformed_nodes, dtype=np.float64)
    tets = np.asarray(tetrahedra, dtype=np.int64)
    validate_tetrahedral_deformation(reference, deformed, tets)
    return TetrahedralLocator(
        reference_nodes=reference,
        deformed_nodes=deformed,
        tetrahedra=tets,
        deformed_grid=_make_tetrahedral_grid(deformed, tets),
    )


def locate_points_in_tetrahedra(
    locator: TetrahedralLocator,
    physical_points_lps_mm: np.ndarray,
    outside_mode: OutsideMode = "identity",
) -> tuple[np.ndarray, np.ndarray]:
    """Return cell ids and the original inside-mesh mask for physical points."""
    if outside_mode not in {"identity", "zero", "nearest"}:
        raise ValueError(f"Unsupported outside_mode: {outside_mode}")
    points = np.asarray(physical_points_lps_mm, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=bool)

    cell_ids = np.asarray(
        locator.deformed_grid.find_containing_cell(points), dtype=np.int64
    ).reshape(-1)
    inside = cell_ids >= 0
    if outside_mode == "nearest" and np.any(~inside):
        nearest = locator.deformed_grid.find_closest_cell(points[~inside])
        cell_ids[~inside] = np.asarray(nearest, dtype=np.int64).reshape(-1)
    return cell_ids, inside


def barycentric_coordinates(
    points_xyz: np.ndarray,
    tetrahedron_vertices_xyz: np.ndarray,
    degeneracy_epsilon: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute vectorized tetrahedral barycentric weights.

    Returns ``(weights, valid)`` where weights have shape ``(N, 4)``.  Invalid
    degenerate tetrahedra receive NaN weights.
    """
    points = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    vertices = np.asarray(tetrahedron_vertices_xyz, dtype=np.float64)
    if vertices.shape != (points.shape[0], 4, 3):
        raise ValueError(
            f"tetrahedron vertices must have shape {(points.shape[0], 4, 3)}, "
            f"got {vertices.shape}"
        )
    matrix = np.stack(
        (
            vertices[:, 1] - vertices[:, 0],
            vertices[:, 2] - vertices[:, 0],
            vertices[:, 3] - vertices[:, 0],
        ),
        axis=2,
    )
    determinant = np.linalg.det(matrix)
    valid = np.isfinite(determinant) & (np.abs(determinant) > degeneracy_epsilon)
    weights = np.full((points.shape[0], 4), np.nan, dtype=np.float64)
    if np.any(valid):
        # NumPy 2.x treats a batched right-hand side shaped (N, 3) as a
        # matrix RHS rather than N independent 3-vectors.  Keep the final
        # singleton column dimension explicit: (N, 3, 3) @ (N, 3, 1).
        rhs = (points[valid] - vertices[valid, 0])[..., np.newaxis]
        solved = np.linalg.solve(matrix[valid], rhs)[..., 0]
        weights[valid, 1:] = solved
        weights[valid, 0] = 1.0 - solved.sum(axis=1)
    return weights, valid


def map_deformed_to_reference(
    locator: TetrahedralLocator,
    physical_points_lps_mm: np.ndarray,
    outside_mode: OutsideMode = "identity",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate ``phi_inverse`` from deformed to reference physical space.

    The returned tuple is ``(reference_points, cell_ids, weights, inside_mask)``.
    With ``identity``, outside points keep their input physical coordinates. With
    ``zero``, outside points are marked by NaN coordinates. With ``nearest``, the
    closest tetrahedron is extrapolated using its barycentric coordinates.
    """
    points = np.asarray(physical_points_lps_mm, dtype=np.float64).reshape(-1, 3)
    cell_ids, inside = locate_points_in_tetrahedra(locator, points, outside_mode)
    mapped = points.copy() if outside_mode == "identity" else np.full_like(points, np.nan)
    weights = np.full((points.shape[0], 4), np.nan, dtype=np.float64)
    candidates = cell_ids >= 0
    if np.any(candidates):
        node_ids = locator.tetrahedra[cell_ids[candidates]]
        candidate_weights, valid = barycentric_coordinates(
            points[candidates], locator.deformed_nodes[node_ids]
        )
        candidate_indices = np.flatnonzero(candidates)
        weights[candidate_indices] = candidate_weights
        if np.any(valid):
            valid_indices = candidate_indices[valid]
            reference_vertices = locator.reference_nodes[node_ids[valid]]
            mapped[valid_indices] = np.einsum(
                "ni,nij->nj", candidate_weights[valid], reference_vertices
            )
        invalid_indices = candidate_indices[~valid]
        if outside_mode == "identity" and invalid_indices.size:
            mapped[invalid_indices] = points[invalid_indices]
    if outside_mode == "zero":
        mapped[~inside] = np.nan
    return mapped, cell_ids, weights, inside


def _as_image(value: sitk.Image | str | Path) -> sitk.Image:
    return value if isinstance(value, sitk.Image) else sitk.ReadImage(str(value))


def resample_volume_with_tetrahedral_deformation(
    source_image: sitk.Image | str | Path,
    target_reference_image: sitk.Image | str | Path,
    locator: TetrahedralLocator,
    interpolation: InterpolationMode = "linear",
    outside_mode: OutsideMode = "identity",
    point_chunk: int = 250_000,
    binary_label: bool = False,
) -> tuple[sitk.Image, dict]:
    """Pull-resample an MR volume onto the target US geometry."""
    if interpolation not in {"linear", "nearest"}:
        raise ValueError(f"Unsupported interpolation: {interpolation}")
    source = _as_image(source_image)
    target = _as_image(target_reference_image)
    source_geometry = geometry_from_image(source)
    target_geometry = geometry_from_image(target)
    source_array = sitk.GetArrayFromImage(source)
    if source_array.ndim != 3:
        raise ValueError(f"Expected scalar 3D source image, got {source_array.shape}")
    # scipy.ndimage.map_coordinates otherwise defaults to the input dtype.  An
    # integer MR volume would therefore truncate linear-interpolation results.
    sampling_array = (
        source_array.astype(np.float32, copy=False)
        if interpolation == "linear"
        else source_array
    )

    size_x, size_y, size_z = target_geometry.size_xyz
    output_dtype = np.uint8 if binary_label else (
        np.float32 if interpolation == "linear" else source_array.dtype
    )
    output = np.zeros((size_z, size_y, size_x), dtype=output_dtype)
    plane_size = max(1, size_x * size_y)
    slices_per_block = max(1, int(point_chunk) // plane_size)
    inside_count = 0
    mapped_count = 0

    for z_start in range(0, size_z, slices_per_block):
        z_end = min(size_z, z_start + slices_per_block)
        zz, yy, xx = np.meshgrid(
            np.arange(z_start, z_end, dtype=np.float64),
            np.arange(size_y, dtype=np.float64),
            np.arange(size_x, dtype=np.float64),
            indexing="ij",
        )
        target_indices_xyz = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
        deformed_points = index_xyz_to_physical(target_indices_xyz, target_geometry)
        reference_points, _, _, inside = map_deformed_to_reference(
            locator, deformed_points, outside_mode=outside_mode
        )
        inside_count += int(inside.sum())

        mapping_valid = np.isfinite(reference_points).all(axis=1)
        reference_indices_xyz = np.zeros_like(reference_points)
        if np.any(mapping_valid):
            reference_indices_xyz[mapping_valid] = physical_to_continuous_index_xyz(
                reference_points[mapping_valid], source_geometry
            )
        reference_indices_zyx = index_xyz_to_array_zyx(reference_indices_xyz)
        sampled = map_coordinates(
            sampling_array,
            reference_indices_zyx.T,
            order=1 if interpolation == "linear" else 0,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        sampled[~mapping_valid] = 0
        mapped_count += int(mapping_valid.sum())
        if binary_label:
            sampled = (sampled >= 0.5).astype(np.uint8, copy=False)
        else:
            sampled = sampled.astype(output_dtype, copy=False)
        output[z_start:z_end] = sampled.reshape(z_end - z_start, size_y, size_x)

    result = sitk.GetImageFromArray(output)
    result.CopyInformation(target)
    total = int(output.size)
    statistics = {
        "interpolation": interpolation,
        "outside_mode": outside_mode,
        "total_output_voxels": total,
        "fem_inside_voxels": inside_count,
        "fem_inside_fraction": float(inside_count / total) if total else 0.0,
        "mapped_voxels": mapped_count,
        "mapped_fraction": float(mapped_count / total) if total else 0.0,
        "source_array_axis_order": "zyx",
        "mesh_coordinate_system": "LPS mm",
    }
    return result, statistics


def signed_tetrahedron_volumes(nodes: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    points = np.asarray(nodes, dtype=np.float64)
    tets = np.asarray(tetrahedra, dtype=np.int64)
    vertices = points[tets]
    matrices = np.stack(
        (
            vertices[:, 1] - vertices[:, 0],
            vertices[:, 2] - vertices[:, 0],
            vertices[:, 3] - vertices[:, 0],
        ),
        axis=2,
    )
    return np.linalg.det(matrices) / 6.0


def tetrahedral_quality_statistics(
    reference_nodes: np.ndarray,
    deformed_nodes: np.ndarray,
    tetrahedra: np.ndarray,
    near_zero_epsilon_mm3: float = 1e-9,
) -> dict:
    validate_tetrahedral_deformation(reference_nodes, deformed_nodes, tetrahedra)
    reference_volume = signed_tetrahedron_volumes(reference_nodes, tetrahedra)
    deformed_volume = signed_tetrahedron_volumes(deformed_nodes, tetrahedra)
    reference_near_zero = np.abs(reference_volume) <= near_zero_epsilon_mm3
    deformed_near_zero = np.abs(deformed_volume) <= near_zero_epsilon_mm3
    usable = ~reference_near_zero
    ratio = np.full(reference_volume.shape, np.nan, dtype=np.float64)
    ratio[usable] = np.abs(deformed_volume[usable]) / np.abs(reference_volume[usable])
    finite_ratio = ratio[np.isfinite(ratio)]
    percentiles = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    ratio_percentiles = (
        {str(value): float(np.percentile(finite_ratio, value)) for value in percentiles}
        if finite_ratio.size
        else {}
    )
    flipped = usable & (np.signbit(reference_volume) != np.signbit(deformed_volume))
    return {
        "tetrahedron_count": int(tetrahedra.shape[0]),
        "reference_negative_tetra_count": int((reference_volume < 0).sum()),
        "deformed_negative_tetra_count": int((deformed_volume < 0).sum()),
        "flipped_tetra_count": int(flipped.sum()),
        "near_zero_reference_tetra_count": int(reference_near_zero.sum()),
        "near_zero_deformed_tetra_count": int(deformed_near_zero.sum()),
        "near_zero_epsilon_mm3": float(near_zero_epsilon_mm3),
        "minimum_volume_ratio": float(finite_ratio.min()) if finite_ratio.size else None,
        "volume_ratio_percentiles": ratio_percentiles,
        "reference_signed_volume_mm3": {
            "min": float(reference_volume.min()),
            "max": float(reference_volume.max()),
            "sum": float(reference_volume.sum()),
        },
        "deformed_signed_volume_mm3": {
            "min": float(deformed_volume.min()),
            "max": float(deformed_volume.max()),
            "sum": float(deformed_volume.sum()),
        },
    }
