#!/usr/bin/env python3
"""End-to-end NIfTI surface-driven COR-FEM image registration.

The MR ``label0.nii.gz`` surface drives COR-FEM registration to the US
``label0.nii.gz`` surface.  The resulting tetrahedral deformation is then used
as an inverse physical-space map to pull-resample MR image/labels onto the US
image grid.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
from scipy.ndimage import binary_erosion, distance_transform_edt
import SimpleITK as sitk

from cor_fem_registration import (
    copy_registration_artifacts,
    load_registration_result,
    run_cor_fem_registration,
)
from nifti_geometry import geometries_equal, geometry_from_image, image_geometry_summary
from stl_smooth import smooth_stl_vtk
from tetrahedral_warp import (
    prepare_tetrahedral_locator,
    resample_volume_with_tetrahedral_deformation,
    tetrahedral_quality_statistics,
)
from volume2mesh import nii_label_to_stl


SCRIPT_DIR = Path(__file__).resolve().parent


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, default=_json_default)


def natural_case_key(path: Path) -> tuple:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    )


def discover_cases(data_root: Path) -> list[Path]:
    return sorted(
        (path for path in data_root.glob("case_*") if path.is_dir()),
        key=natural_case_key,
    )


def case_input_paths(args: argparse.Namespace, case_dir: Path) -> dict[str, Path]:
    source_dir = case_dir / args.source_modality
    target_dir = case_dir / args.target_modality
    return {
        "source_image": source_dir / "image.nii.gz",
        "source_label": source_dir / args.source_label,
        "target_image": target_dir / "image.nii.gz",
        "target_label": target_dir / args.target_label,
        "source_dir": source_dir,
        "target_dir": target_dir,
    }


def validate_case_inputs(paths: dict[str, Path]) -> None:
    required = ("source_image", "source_label", "target_image", "target_label")
    missing = [str(paths[name]) for name in required if not paths[name].is_file()]
    if missing:
        raise FileNotFoundError("Missing required case files: " + ", ".join(missing))


def _surface_needs_update(output: Path, source: Path, force: bool) -> bool:
    return bool(
        force
        or not output.is_file()
        or source.stat().st_mtime_ns > output.stat().st_mtime_ns
    )


def prepare_registration_surfaces(
    args: argparse.Namespace,
    paths: dict[str, Path],
    meshes_dir: Path,
) -> tuple[Path, Path]:
    """Create physical LPS-mm surfaces and smooth only the MR source."""
    source_raw = meshes_dir / "mr_label0_raw.stl"
    source_smooth = meshes_dir / "mr_label0_smooth.stl"
    target_raw = meshes_dir / "us_label0_raw.stl"

    if _surface_needs_update(source_raw, paths["source_label"], args.force):
        ok = nii_label_to_stl(
            label_path=paths["source_label"],
            out_path=source_raw,
            image_path=paths["source_image"],
            threshold=args.threshold,
            coord_system="LPS",
            pad_boundary=True,
            smooth=False,
            process_mesh=False,
            half_voxel_shift=False,
            save_meta=True,
        )
        if not ok:
            raise RuntimeError(f"Source label is empty: {paths['source_label']}")

    if args.smooth_source:
        if _surface_needs_update(source_smooth, source_raw, args.force):
            ok = smooth_stl_vtk(
                in_stl=source_raw,
                out_stl=source_smooth,
                method=args.smooth_method,
                iterations=args.smooth_iterations,
                pass_band=args.smooth_pass_band,
                relaxation_factor=args.smooth_relaxation_factor,
                feature_angle=180.0,
                boundary_smoothing=True,
                feature_edge_smoothing=True,
                compute_normals=True,
                subdivision=args.smooth_subdivision,
                subdivision_method="loop",
            )
            if not ok:
                raise RuntimeError(f"Source smoothing failed: {source_raw}")
        source_surface = source_smooth
    else:
        source_surface = source_raw

    if _surface_needs_update(target_raw, paths["target_label"], args.force):
        ok = nii_label_to_stl(
            label_path=paths["target_label"],
            out_path=target_raw,
            image_path=paths["target_image"],
            threshold=args.threshold,
            coord_system="LPS",
            pad_boundary=True,
            smooth=False,
            process_mesh=False,
            half_voxel_shift=False,
            save_meta=True,
        )
        if not ok:
            raise RuntimeError(f"Target label is empty: {paths['target_label']}")
    return source_surface, target_raw


def save_surface_stl(vertices: np.ndarray, faces: np.ndarray, path: Path) -> None:
    triangles = np.asarray(faces, dtype=np.int64)
    vtk_faces = np.hstack(
        [np.full((triangles.shape[0], 1), 3, dtype=np.int64), triangles]
    ).ravel()
    mesh = pv.PolyData(np.asarray(vertices, dtype=np.float64), vtk_faces)
    mesh.save(path)


def binary_label_metrics(warped_path: Path, target_path: Path) -> dict:
    warped_image = sitk.ReadImage(str(warped_path))
    target_image = sitk.ReadImage(str(target_path))
    warped_geometry = geometry_from_image(warped_image)
    target_geometry = geometry_from_image(target_image)
    if not geometries_equal(warped_geometry, target_geometry):
        raise ValueError(
            f"Cannot compare labels with different geometry: {warped_path}, {target_path}"
        )
    warped = sitk.GetArrayFromImage(warped_image) > 0
    target = sitk.GetArrayFromImage(target_image) > 0
    intersection = int(np.count_nonzero(warped & target))
    denominator = int(np.count_nonzero(warped) + np.count_nonzero(target))
    dice = 1.0 if denominator == 0 else (2.0 * intersection / denominator)

    if not np.any(warped) and not np.any(target):
        hd95 = 0.0
    elif not np.any(warped) or not np.any(target):
        hd95 = None
    else:
        warped_surface = warped & ~binary_erosion(warped)
        target_surface = target & ~binary_erosion(target)
        spacing_zyx = tuple(reversed(target_geometry.spacing_xyz))
        distance_to_target = distance_transform_edt(~target_surface, sampling=spacing_zyx)
        distance_to_warped = distance_transform_edt(~warped_surface, sampling=spacing_zyx)
        distances = np.concatenate(
            (distance_to_target[warped_surface], distance_to_warped[target_surface])
        )
        hd95 = float(np.percentile(distances, 95.0)) if distances.size else None
    return {
        "dice": float(dice),
        "hd95_mm": hd95,
        "warped_foreground_voxels": int(np.count_nonzero(warped)),
        "target_foreground_voxels": int(np.count_nonzero(target)),
    }


def dry_run_case(args: argparse.Namespace, case_dir: Path) -> dict:
    paths = case_input_paths(args, case_dir)
    validate_case_inputs(paths)
    output_dir = args.output_root / case_dir.name
    return {
        "case": case_dir.name,
        "inputs": {name: str(path.resolve()) for name, path in paths.items()},
        "source_geometry": image_geometry_summary(paths["source_image"]),
        "target_geometry": image_geometry_summary(paths["target_image"]),
        "planned_output": str(output_dir.resolve()),
        "image_outside_mode": args.outside_mode,
        "label_outside_mode": args.label_outside_mode,
    }


def run_case(args: argparse.Namespace, case_dir: Path) -> dict:
    paths = case_input_paths(args, case_dir)
    validate_case_inputs(paths)
    case_output = (args.output_root / case_dir.name).resolve()
    meshes_dir = case_output / "meshes"
    deformation_dir = case_output / "deformation"
    warped_dir = case_output / "warped"
    metrics_dir = case_output / "metrics"
    registration_dir = case_output / "registration_work"
    logs_dir = case_output / "logs"
    for directory in (
        meshes_dir,
        deformation_dir,
        warped_dir,
        metrics_dir,
        registration_dir,
        logs_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    source_surface, target_surface = prepare_registration_surfaces(
        args, paths, meshes_dir
    )
    shutil.copy2(source_surface, meshes_dir / "source_surface_before_registration.stl")
    shutil.copy2(target_surface, meshes_dir / "target_surface.stl")

    deformation_path = registration_dir / "deformation_data.npz"
    registration_current = bool(
        args.reuse_registration
        and not args.force
        and deformation_path.is_file()
        and deformation_path.stat().st_mtime_ns >= source_surface.stat().st_mtime_ns
        and deformation_path.stat().st_mtime_ns >= target_surface.stat().st_mtime_ns
    )
    if registration_current:
        registration = load_registration_result(registration_dir)
    else:
        tet_cache = deformation_dir / "source_tet_cache.npz"
        if tet_cache.is_file() and source_surface.stat().st_mtime_ns > tet_cache.stat().st_mtime_ns:
            tet_cache.unlink()
        registration = run_cor_fem_registration(
            source_surface=source_surface,
            target_surface=target_surface,
            output_dir=registration_dir,
            cor_fem_script=args.cor_fem_script,
            python_executable=args.python_bin,
            device=args.device,
            dtype="float32",
            tet_cache=tet_cache,
            max_iterations=args.max_iters,
            target_sample_count=args.target_sample_count,
            extra_arguments=args.cor_fem_extra_arg,
        )

    artifact_paths = copy_registration_artifacts(
        registration, meshes_dir, deformation_dir
    )
    save_surface_stl(
        registration["surface_vertices_final"],
        registration["surface_faces"],
        meshes_dir / "source_surface_after_registration.stl",
    )

    quality = tetrahedral_quality_statistics(
        registration["reference_nodes"],
        registration["deformed_nodes"],
        registration["tetrahedra"],
    )
    write_json(metrics_dir / "jacobian_metrics.json", quality)
    if args.fail_on_inverted_tet and quality["flipped_tetra_count"] > 0:
        raise RuntimeError(
            f"Deformed mesh contains {quality['flipped_tetra_count']} flipped tetrahedra"
        )

    locator = prepare_tetrahedral_locator(
        registration["reference_nodes"],
        registration["deformed_nodes"],
        registration["tetrahedra"],
    )
    warped_image, image_stats = resample_volume_with_tetrahedral_deformation(
        source_image=paths["source_image"],
        target_reference_image=paths["target_image"],
        locator=locator,
        interpolation="linear",
        outside_mode=args.outside_mode,
        point_chunk=args.point_chunk,
        binary_label=False,
    )
    warped_image = sitk.Cast(warped_image, sitk.sitkFloat32)
    warped_image_path = warped_dir / "warped_mr_image.nii.gz"
    sitk.WriteImage(warped_image, str(warped_image_path), True)
    # Short aliases allow the whole warped MR set to be loaded as one familiar
    # image/label group, while the explicit names preserve mapping direction.
    shutil.copy2(warped_image_path, warped_dir / "image.nii.gz")
    fixed_us_image_path = warped_dir / "fixed_us_image.nii.gz"
    shutil.copy2(paths["target_image"], fixed_us_image_path)

    label_resampling: dict[str, Any] = {}
    label_metrics: dict[str, Any] = {}
    warped_label_outputs: dict[str, Any] = {}
    warnings: list[str] = []
    for label_index in range(args.label_count):
        name = f"label{label_index}.nii.gz"
        source_label = paths["source_dir"] / name
        if not source_label.is_file():
            warnings.append(f"Missing source label, skipped: {source_label}")
            continue
        warped_label, statistics = resample_volume_with_tetrahedral_deformation(
            source_image=source_label,
            target_reference_image=paths["target_image"],
            locator=locator,
            interpolation="nearest",
            # A label must not retain source foreground outside the FEM domain.
            # With the image-wide ``identity`` fallback, any raw MR label voxel
            # outside the tetrahedra is copied unchanged into US space.  That
            # produces a mixture of deformed and undeformed anatomy.  Keep the
            # background at zero outside the deformation instead.
            outside_mode=args.label_outside_mode,
            point_chunk=args.point_chunk,
            binary_label=True,
        )
        output_path = warped_dir / f"warped_label{label_index}.nii.gz"
        sitk.WriteImage(sitk.Cast(warped_label, sitk.sitkUInt8), str(output_path), True)
        short_nii_path = warped_dir / f"label{label_index}.nii.gz"
        shutil.copy2(output_path, short_nii_path)

        warped_stl_path = warped_dir / f"warped_label{label_index}.stl"
        stl_created = nii_label_to_stl(
            label_path=output_path,
            out_path=warped_stl_path,
            image_path=fixed_us_image_path,
            threshold=0.5,
            coord_system="LPS",
            pad_boundary=True,
            smooth=False,
            process_mesh=False,
            half_voxel_shift=False,
            save_meta=True,
        )
        short_stl_path: Path | None = None
        if stl_created:
            short_stl_path = warped_dir / f"label{label_index}.stl"
            shutil.copy2(warped_stl_path, short_stl_path)
        else:
            warnings.append(
                f"Warped label{label_index} is empty; NIfTI was saved but STL was skipped"
            )

        label_resampling[f"label{label_index}"] = statistics
        warped_label_outputs[f"label{label_index}"] = {
            "explicit_nifti": output_path,
            "short_nifti": short_nii_path,
            "explicit_stl": warped_stl_path if stl_created else None,
            "short_stl": short_stl_path,
        }
        target_label = paths["target_dir"] / name
        if target_label.is_file():
            try:
                label_metrics[f"label{label_index}"] = binary_label_metrics(
                    output_path, target_label
                )
            except Exception as error:
                warnings.append(
                    f"Could not evaluate warped label{label_index} against target: {error}"
                )

    source_label_map = paths["source_dir"] / "label_map.nii.gz"
    if source_label_map.is_file():
        warped_map, map_stats = resample_volume_with_tetrahedral_deformation(
            source_image=source_label_map,
            target_reference_image=paths["target_image"],
            locator=locator,
            interpolation="nearest",
            # label_map is categorical, so it follows the same zero-background
            # policy as the individual binary labels.
            outside_mode=args.label_outside_mode,
            point_chunk=args.point_chunk,
            binary_label=False,
        )
        warped_map_path = warped_dir / "warped_label_map.nii.gz"
        sitk.WriteImage(warped_map, str(warped_map_path), True)
        shutil.copy2(warped_map_path, warped_dir / "label_map.nii.gz")
        label_resampling["label_map"] = map_stats

    registration_metrics = {
        "case": case_dir.name,
        "coordinate_system": "LPS mm",
        "mapping": "US output point -> deformed tet -> barycentric weights -> reference tet -> MR sample",
        "outside_policy": {
            "image": args.outside_mode,
            "labels_and_label_map": args.label_outside_mode,
            "reason": (
                "Image background may use identity sampling outside the FEM domain; "
                "labels use zero outside the FEM domain to prevent undeformed "
                "source foreground from leaking into the warped label."
            ),
        },
        "image_resampling": image_stats,
        "label_resampling": label_resampling,
        "label_metrics": label_metrics,
        "warped_label_outputs": warped_label_outputs,
        "warnings": warnings,
        "source_image_geometry": image_geometry_summary(paths["source_image"]),
        "target_image_geometry": image_geometry_summary(paths["target_image"]),
        "artifacts": artifact_paths,
    }
    write_json(metrics_dir / "registration_metrics.json", registration_metrics)
    write_json(
        case_output / "config.json",
        {
            "case": case_dir.name,
            "inputs": paths,
            "arguments": vars(args),
            "outputs": {
                "warped_mr_image": warped_image_path,
                "warped_mr_image_short_name": warped_dir / "image.nii.gz",
                "fixed_us_image": fixed_us_image_path,
                "warped_labels": warped_label_outputs,
            },
        },
    )
    (logs_dir / "pipeline.log").write_text(
        "\n".join(["Pipeline completed successfully", *warnings]) + "\n",
        encoding="utf-8",
    )
    return registration_metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NIfTI-to-NIfTI COR-FEM non-rigid image registration"
    )
    parser.add_argument("--data-root", type=Path, default=Path("test"))
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--case")
    selection.add_argument("--all-cases", default="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--source-modality", default="mr")
    parser.add_argument("--target-modality", default="us")
    parser.add_argument("--source-label", default="label0.nii.gz")
    parser.add_argument("--target-label", default="label0.nii.gz")
    parser.add_argument("--label-count", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--outside-mode",
        choices=["identity", "zero", "nearest"],
        default="identity",
        help=(
            "Outside-FEM policy for the warped MR intensity image. "
            "Default: identity (sample MR at the same physical point)."
        ),
    )
    parser.add_argument(
        "--label-outside-mode",
        choices=["zero", "identity", "nearest"],
        default="zero",
        help=(
            "Outside-FEM policy for label*.nii.gz and label_map.nii.gz. "
            "Default: zero, preventing undeformed MR labels from leaking into "
            "the US-space warped labels."
        ),
    )
    parser.add_argument("--point-chunk", type=int, default=250_000)
    parser.add_argument(
        "--smooth-source", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--smooth-method", choices=["windowed_sinc", "laplacian"], default="windowed_sinc"
    )
    parser.add_argument("--smooth-iterations", type=int, default=250)
    parser.add_argument("--smooth-pass-band", type=float, default=0.005)
    parser.add_argument("--smooth-relaxation-factor", type=float, default=0.15)
    parser.add_argument("--smooth-subdivision", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--cor-fem-script", type=Path, default=SCRIPT_DIR / "cor-fem-single.py"
    )
    parser.add_argument("--max-iters", type=int, default=200)
    parser.add_argument("--target-sample-count", type=int, default=6000)
    parser.add_argument("--cor-fem-extra-arg", action="append", default=[])
    parser.add_argument(
        "--reuse-registration", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--fail-on-inverted-tet", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.data_root = args.data_root.resolve()
    args.output_root = args.output_root.resolve()
    args.cor_fem_script = args.cor_fem_script.resolve()
    args.python_bin = args.python_bin.resolve()
    if not args.data_root.is_dir():
        raise NotADirectoryError(f"Data root not found: {args.data_root}")
    cases = discover_cases(args.data_root) if args.all_cases else [args.data_root / args.case]
    if not cases:
        raise FileNotFoundError(f"No case_* directories found below {args.data_root}")

    failed = 0
    for case_dir in cases:
        try:
            if args.dry_run:
                print(json.dumps(dry_run_case(args, case_dir), indent=2, ensure_ascii=False))
            else:
                print(f"[CASE] {case_dir.name}", flush=True)
                result = run_case(args, case_dir)
                print(
                    f"[DONE] {case_dir.name}: FEM inside fraction="
                    f"{result['image_resampling']['fem_inside_fraction']:.6f}",
                    flush=True,
                )
        except Exception as error:
            failed += 1
            print(f"[FAILED] {case_dir.name}: {error}", file=sys.stderr, flush=True)
            if not args.continue_on_error:
                raise
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
