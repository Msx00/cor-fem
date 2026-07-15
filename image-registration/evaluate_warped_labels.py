#!/usr/bin/env python3
"""Evaluate MR-to-US warped binary NIfTI labels produced by COR-FEM.

For every ``outputs/case_*/warped/warped_labelN.nii.gz``, this script compares
the warped MR label with ``test/case_*/us/labelN.nii.gz`` in the US grid.

Metric definitions follow the supplied MONAI evaluation scripts:

* TRE: Euclidean distance between physical-space binary-mask centroids (mm).
* Dice: binary Dice coefficient.
* HD95: ``medpy.metric.binary.hd95`` (mm).
* ASD: directed ``medpy.metric.binary.asd(warped, fixed)`` (mm).
* CD: symmetric ``medpy.metric.binary.assd(warped, fixed)`` (mm).
* MI: discrete mutual information of the two binary label masks (nats).

The input files must share the same voxel grid for Dice and voxel-wise MI to be
valid. A geometry mismatch is recorded as a row with ``status=geometry_mismatch``
instead of silently resampling either label.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np
from medpy import metric


METRIC_COLUMNS = ("tre_mm", "dice", "mi", "hd95_mm", "cd_mm", "asd_mm")
ROW_COLUMNS = (
    "case",
    "label",
    "status",
    "warped_label_path",
    "fixed_label_path",
    "geometry_equal",
    "warped_foreground_voxels",
    "fixed_foreground_voxels",
    *METRIC_COLUMNS,
    "message",
)


@dataclass(frozen=True)
class LabelVolume:
    """A 3D binary NIfTI mask with its voxel-to-physical affine."""

    mask: np.ndarray
    affine: np.ndarray
    spacing_mm: tuple[float, float, float]


def natural_case_key(path: Path) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    )


def parse_labels(value: str) -> list[int]:
    labels = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not labels or any(label < 0 for label in labels):
        raise argparse.ArgumentTypeError("--labels must contain non-negative integers")
    return labels


def load_binary_label(path: Path, threshold: float) -> LabelVolume:
    image = nib.load(str(path))
    array = np.asarray(image.get_fdata(dtype=np.float32))
    array = np.squeeze(array)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI after squeeze, got {array.shape}: {path}")
    affine = np.asarray(image.affine, dtype=np.float64)
    spacing = tuple(float(value) for value in np.linalg.norm(affine[:3, :3], axis=0))
    return LabelVolume(mask=array > threshold, affine=affine, spacing_mm=spacing)


def geometry_equal(first: LabelVolume, second: LabelVolume, affine_atol: float) -> bool:
    return bool(
        first.mask.shape == second.mask.shape
        and np.allclose(first.affine, second.affine, rtol=0.0, atol=affine_atol)
    )


def centroid_physical_mm(mask: np.ndarray, affine: np.ndarray) -> np.ndarray | None:
    """Return the foreground centroid transformed with the NIfTI affine."""
    indices_xyz = np.argwhere(mask)
    if indices_xyz.size == 0:
        return None
    # NIfTI voxel coordinates identify voxel centres, so no half-voxel offset
    # is needed before applying its affine.
    centroid_xyz = indices_xyz.mean(axis=0, dtype=np.float64)
    return affine[:3, :3] @ centroid_xyz + affine[:3, 3]


def tre_centroid_mm(warped: LabelVolume, fixed: LabelVolume) -> float:
    warped_centroid = centroid_physical_mm(warped.mask, warped.affine)
    fixed_centroid = centroid_physical_mm(fixed.mask, fixed.affine)
    if warped_centroid is None or fixed_centroid is None:
        return math.nan
    return float(np.linalg.norm(warped_centroid - fixed_centroid))


def dice_binary(warped_mask: np.ndarray, fixed_mask: np.ndarray) -> float:
    warped_voxels = int(np.count_nonzero(warped_mask))
    fixed_voxels = int(np.count_nonzero(fixed_mask))
    # Match the supplied MONAI evaluator: an all-background pair is excluded
    # from mean Dice rather than counted as a perfect registration.
    if warped_voxels == 0 and fixed_voxels == 0:
        return math.nan
    if warped_voxels == 0 or fixed_voxels == 0:
        return 0.0
    intersection = int(np.count_nonzero(warped_mask & fixed_mask))
    return float(2.0 * intersection / (warped_voxels + fixed_voxels))


def binary_mutual_information(warped_mask: np.ndarray, fixed_mask: np.ndarray) -> float:
    """Discrete binary-label MI in nats, calculated from the joint histogram."""
    warped_voxels = int(np.count_nonzero(warped_mask))
    fixed_voxels = int(np.count_nonzero(fixed_mask))
    if warped_voxels == 0 or fixed_voxels == 0:
        return math.nan
    encoded = warped_mask.astype(np.uint8, copy=False).ravel() * 2
    encoded += fixed_mask.astype(np.uint8, copy=False).ravel()
    joint = np.bincount(encoded, minlength=4).reshape(2, 2).astype(np.float64)
    joint /= joint.sum()
    marginal_warped = joint.sum(axis=1, keepdims=True)
    marginal_fixed = joint.sum(axis=0, keepdims=True)
    valid = joint > 0.0
    return float(np.sum(joint[valid] * np.log((joint / (marginal_warped * marginal_fixed))[valid])))


def surface_metrics_mm(
    warped_mask: np.ndarray,
    fixed_mask: np.ndarray,
    spacing_mm: tuple[float, float, float],
) -> dict[str, float]:
    """Compute MedPy HD95, directed ASD, and symmetric ASD/CD in millimetres."""
    if not np.any(warped_mask) or not np.any(fixed_mask):
        return {"hd95_mm": math.nan, "cd_mm": math.nan, "asd_mm": math.nan}
    spacing = tuple(float(value) for value in spacing_mm)
    return {
        "hd95_mm": float(metric.binary.hd95(warped_mask, fixed_mask, voxelspacing=spacing)),
        "cd_mm": float(metric.binary.assd(warped_mask, fixed_mask, voxelspacing=spacing)),
        "asd_mm": float(metric.binary.asd(warped_mask, fixed_mask, voxelspacing=spacing)),
    }


def empty_row(case_name: str, label: int, warped_path: Path, fixed_path: Path) -> dict[str, Any]:
    return {
        "case": case_name,
        "label": label,
        "status": "unknown",
        "warped_label_path": str(warped_path),
        "fixed_label_path": str(fixed_path),
        "geometry_equal": "",
        "warped_foreground_voxels": "",
        "fixed_foreground_voxels": "",
        **{column: math.nan for column in METRIC_COLUMNS},
        "message": "",
    }


def evaluate_pair(
    case_name: str,
    label: int,
    warped_path: Path,
    fixed_path: Path,
    threshold: float,
    affine_atol: float,
) -> dict[str, Any]:
    row = empty_row(case_name, label, warped_path, fixed_path)
    if not warped_path.is_file():
        row.update(status="missing_warped_label", message="Warped label file not found")
        return row
    if not fixed_path.is_file():
        row.update(status="missing_fixed_label", message="US reference label file not found")
        return row
    try:
        warped = load_binary_label(warped_path, threshold)
        fixed = load_binary_label(fixed_path, threshold)
        same_geometry = geometry_equal(warped, fixed, affine_atol)
        row.update(
            geometry_equal=same_geometry,
            warped_foreground_voxels=int(np.count_nonzero(warped.mask)),
            fixed_foreground_voxels=int(np.count_nonzero(fixed.mask)),
        )
        if not same_geometry:
            row.update(
                status="geometry_mismatch",
                message="Shape or NIfTI affine differs; metrics were not computed",
            )
            return row

        row.update(
            tre_mm=tre_centroid_mm(warped, fixed),
            dice=dice_binary(warped.mask, fixed.mask),
            mi=binary_mutual_information(warped.mask, fixed.mask),
            **surface_metrics_mm(warped.mask, fixed.mask, fixed.spacing_mm),
        )
        if not np.any(warped.mask) and not np.any(fixed.mask):
            row.update(status="both_empty", message="Both masks are empty; metrics are undefined")
        elif not np.any(warped.mask) or not np.any(fixed.mask):
            row.update(status="one_empty", message="One mask is empty; distance metrics are undefined")
        else:
            row.update(status="ok")
    except Exception as error:  # Keep one bad pair from aborting all cases.
        row.update(status="error", message=repr(error))
    return row


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def metric_summary(rows: Iterable[dict[str, Any]], label: int | str) -> dict[str, Any]:
    rows = list(rows)
    summary: dict[str, Any] = {
        "label": label,
        "n_rows": len(rows),
        "n_cases": len({str(row["case"]) for row in rows}),
        "n_ok": sum(row["status"] == "ok" for row in rows),
    }
    for metric_name in METRIC_COLUMNS:
        values = np.asarray([safe_float(row[metric_name]) for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        summary[f"{metric_name}_mean"] = float(values.mean()) if values.size else math.nan
        summary[f"{metric_name}_std"] = float(values.std()) if values.size else math.nan
        summary[f"{metric_name}_n"] = int(values.size)
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--test-root", type=Path, default=Path("test"))
    parser.add_argument("--output-dir", type=Path, default=None)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--case", help="Evaluate one case, for example case_0000")
    selection.add_argument("--all-cases", action="store_true", help="Evaluate all case_* output directories")
    parser.add_argument("--labels", type=parse_labels, default=parse_labels("0,1,2,3,4,5"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--affine-atol", type=float, default=1e-5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outputs_root = args.outputs_root.resolve()
    test_root = args.test_root.resolve()
    output_dir = (args.output_dir or outputs_root / "label_evaluation").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.case:
        case_dirs = [outputs_root / args.case]
    else:
        case_dirs = sorted(
            (path for path in outputs_root.glob("case_*") if path.is_dir()),
            key=natural_case_key,
        )
    if not case_dirs:
        raise FileNotFoundError(f"No case_* directories found in {outputs_root}")

    rows: list[dict[str, Any]] = []
    for case_dir in case_dirs:
        case_name = case_dir.name
        for label in args.labels:
            rows.append(
                evaluate_pair(
                    case_name=case_name,
                    label=label,
                    warped_path=case_dir / "warped" / f"warped_label{label}.nii.gz",
                    fixed_path=test_root / case_name / "us" / f"label{label}.nii.gz",
                    threshold=args.threshold,
                    affine_atol=args.affine_atol,
                )
            )

    summary_rows = [
        metric_summary([row for row in rows if row["label"] == label], label)
        for label in args.labels
    ]
    summary_rows.append(metric_summary(rows, "ALL"))
    summary_columns = tuple(summary_rows[0].keys())
    write_csv(output_dir / "warped_label_metrics.csv", rows, ROW_COLUMNS)
    write_csv(output_dir / "warped_label_summary.csv", summary_rows, summary_columns)
    with (output_dir / "warped_label_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary_rows, stream, indent=2, ensure_ascii=False, allow_nan=True)

    print(f"[DONE] per-label metrics: {output_dir / 'warped_label_metrics.csv'}")
    print(f"[DONE] summary: {output_dir / 'warped_label_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
