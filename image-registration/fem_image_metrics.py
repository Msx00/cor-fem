"""Quality-control helpers for FEM image registration outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.ndimage import binary_erosion, distance_transform_edt
import SimpleITK as sitk

from nifti_geometry import (
    array_zyx_to_index_xyz,
    geometry_from_image,
    index_xyz_to_physical,
    physical_bounds_lps_mm,
)


def _bounds_dict(bounds_min: np.ndarray, bounds_max: np.ndarray) -> dict:
    return {
        "min": np.asarray(bounds_min, dtype=np.float64).tolist(),
        "max": np.asarray(bounds_max, dtype=np.float64).tolist(),
        "size": (np.asarray(bounds_max) - np.asarray(bounds_min)).astype(float).tolist(),
        "center": (0.5 * (np.asarray(bounds_max) + np.asarray(bounds_min))).astype(float).tolist(),
    }


def image_physical_bounds(image: sitk.Image | str | Path) -> dict:
    value = image if isinstance(image, sitk.Image) else sitk.ReadImage(str(image))
    minimum, maximum = physical_bounds_lps_mm(geometry_from_image(value))
    return _bounds_dict(minimum, maximum)


def binary_label_physical_bounds(image: sitk.Image | str | Path) -> dict | None:
    value = image if isinstance(image, sitk.Image) else sitk.ReadImage(str(image))
    array = sitk.GetArrayFromImage(value) != 0
    if not array.any():
        return None
    occupied_z = np.flatnonzero(array.any(axis=(1, 2)))
    occupied_y = np.flatnonzero(array.any(axis=(0, 2)))
    occupied_x = np.flatnonzero(array.any(axis=(0, 1)))
    minimum_zyx = np.asarray(
        [occupied_z[0], occupied_y[0], occupied_x[0]], dtype=np.float64
    ) - 0.5
    maximum_zyx = np.asarray(
        [occupied_z[-1], occupied_y[-1], occupied_x[-1]], dtype=np.float64
    ) + 0.5
    minimum_xyz = array_zyx_to_index_xyz(minimum_zyx)
    maximum_xyz = array_zyx_to_index_xyz(maximum_zyx)
    corners = np.asarray(
        [
            [x, y, z]
            for x in (minimum_xyz[0], maximum_xyz[0])
            for y in (minimum_xyz[1], maximum_xyz[1])
            for z in (minimum_xyz[2], maximum_xyz[2])
        ]
    )
    physical = index_xyz_to_physical(corners, geometry_from_image(value))
    return _bounds_dict(physical.min(axis=0), physical.max(axis=0))


def mesh_physical_bounds(mesh: pv.DataSet | str | Path) -> dict:
    value = mesh if isinstance(mesh, pv.DataSet) else pv.read(str(mesh))
    bounds = np.asarray(value.bounds, dtype=np.float64).reshape(3, 2)
    return _bounds_dict(bounds[:, 0], bounds[:, 1])


def compare_bounds(first: dict, second: dict) -> dict:
    first_min = np.asarray(first["min"], dtype=np.float64)
    first_max = np.asarray(first["max"], dtype=np.float64)
    second_min = np.asarray(second["min"], dtype=np.float64)
    second_max = np.asarray(second["max"], dtype=np.float64)
    intersection_size = np.maximum(0.0, np.minimum(first_max, second_max) - np.maximum(first_min, second_min))
    first_size = np.maximum(0.0, first_max - first_min)
    second_size = np.maximum(0.0, second_max - second_min)
    first_volume = float(np.prod(first_size))
    second_volume = float(np.prod(second_size))
    intersection_volume = float(np.prod(intersection_size))
    union_volume = first_volume + second_volume - intersection_volume
    return {
        "center_distance_mm": float(
            np.linalg.norm(0.5 * (first_min + first_max - second_min - second_max))
        ),
        "intersection_size_mm": intersection_size.tolist(),
        "intersection_over_union": (
            float(intersection_volume / union_volume) if union_volume > 0.0 else 0.0
        ),
        "first_coverage": (
            float(intersection_volume / first_volume) if first_volume > 0.0 else 0.0
        ),
        "second_coverage": (
            float(intersection_volume / second_volume) if second_volume > 0.0 else 0.0
        ),
    }


def mesh_label_bounds_check(
    mesh_path: str | Path,
    label_path: str | Path,
) -> dict:
    mesh_bounds = mesh_physical_bounds(mesh_path)
    label_bounds = binary_label_physical_bounds(label_path)
    return {
        "mesh_bounds_lps_mm": mesh_bounds,
        "label_bounds_lps_mm": label_bounds,
        "comparison": compare_bounds(mesh_bounds, label_bounds) if label_bounds else None,
    }


def mesh_geometry_change(before_path: str | Path, after_path: str | Path) -> dict:
    before = mesh_physical_bounds(before_path)
    after = mesh_physical_bounds(after_path)
    before_size = np.asarray(before["size"], dtype=np.float64)
    after_size = np.asarray(after["size"], dtype=np.float64)
    relative_size_change = np.divide(
        after_size - before_size,
        before_size,
        out=np.zeros(3, dtype=np.float64),
        where=before_size > 0,
    )
    return {
        "before_bounds_lps_mm": before,
        "after_bounds_lps_mm": after,
        "center_shift_mm": float(
            np.linalg.norm(np.asarray(after["center"]) - np.asarray(before["center"]))
        ),
        "relative_axis_size_change": relative_size_change.tolist(),
        "note": "Smoothing is performed in-place in LPS mm, but may change shape, bounds and volume.",
    }


def dice_hd95(
    prediction: sitk.Image | str | Path,
    target: sitk.Image | str | Path,
) -> dict:
    prediction_image = (
        prediction if isinstance(prediction, sitk.Image) else sitk.ReadImage(str(prediction))
    )
    target_image = target if isinstance(target, sitk.Image) else sitk.ReadImage(str(target))
    prediction_array = sitk.GetArrayFromImage(prediction_image) != 0
    target_array = sitk.GetArrayFromImage(target_image) != 0
    if prediction_array.shape != target_array.shape:
        raise ValueError(
            f"Dice/HD95 require matching array shapes, got {prediction_array.shape} and {target_array.shape}"
        )
    intersection = int(np.logical_and(prediction_array, target_array).sum())
    denominator = int(prediction_array.sum() + target_array.sum())
    dice = float(2.0 * intersection / denominator) if denominator else 1.0
    if not prediction_array.any() or not target_array.any():
        return {
            "dice": dice,
            "hd95_mm": None,
            "warning": "HD95 is undefined when either mask is empty",
        }

    structure = np.ones((3, 3, 3), dtype=bool)
    prediction_surface = prediction_array & ~binary_erosion(
        prediction_array, structure=structure, border_value=0
    )
    target_surface = target_array & ~binary_erosion(
        target_array, structure=structure, border_value=0
    )
    spacing_zyx = tuple(reversed(target_image.GetSpacing()))
    distance_to_target = distance_transform_edt(~target_surface, sampling=spacing_zyx)
    distance_to_prediction = distance_transform_edt(~prediction_surface, sampling=spacing_zyx)
    distances = np.concatenate(
        (distance_to_target[prediction_surface], distance_to_prediction[target_surface])
    )
    return {
        "dice": dice,
        "hd95_mm": float(np.percentile(distances, 95.0)),
        "prediction_voxels": int(prediction_array.sum()),
        "target_voxels": int(target_array.sum()),
    }
