#!/usr/bin/env python3
"""Windows-compatible NIfTI-to-surface batch launcher for COR-FEM.

For every case, the MR and US registration surfaces are extracted from
``label0.nii.gz`` in physical millimetres.  Only the MR source surface is
smoothed.  The existing single-case COR-FEM program is then invoked with an
argument list (``shell=False``), so Windows paths containing spaces are handled
correctly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from stl_smooth import smooth_stl_vtk
from volume2mesh import nii_label_to_stl


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_COR_FEM_SCRIPT = SCRIPT_DIR / "cor-fem-single.py"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "prostate" / "mureg_hybrid"

# Keep these values synchronized with the former run_prostate_hybrid.sh
# command.  They are written explicitly here so a Windows run has exactly the
# same registration configuration without needing Bash environment variables.
COR_FEM_ARGUMENTS: tuple[str, ...] = (
    "--dtype", "float32",
    "--young", "500",
    "--poisson", "0.45",
    "--w-t2s", "1.0",
    "--w-s2t", "1.0",
    "--tangent-weight-start", "0.70",
    "--tangent-weight", "0.22",
    "--tangent-anneal-iters", "40",
    "--surface-smooth-k", "1.5",
    "--cover-dist-start", "4.0",
    "--cover-dist-end", "12.0",
    "--vol-preserve-k", "10",
    "--vol-barrier-k", "40",
    "--vol-j-min", "0.35",
    "--min-accepted-j", "0.10",
    "--max-delta-u", "1.0",
    "--trim-normal-cos", "0.25",
    "--trim-quantile", "0.99",
    "--trim-max-dist", "50",
    "--penalty-k", "100",
    "--data-area-scale", "1.0",
    "--robust-sigma", "20",
    "--target-sample-count", "12000",
    "--pcg-tol", "1e-5",
    "--pcg-max-iter", "500",
    "--max-iters", "100",
    "--min-iters", "10",
    "--grad-tol", "1e-6",
    "--tol", "1e-5",
    "--stop_mean_err", "1e-3",
    "--line-search-init", "1.0",
    "--armijo-c1", "1e-4",
    "--backtrack-factor", "0.5",
    "--min-step", "1e-5",
    "--max-backtracks", "14",
    "--max-reject-streak", "3",
    "--outer-merit-rel-tol", "1e-8",
    "--boundary-mode", "none",
    "--seed", "0",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hybrid COR-FEM cases on Windows without Bash.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Dataset root containing the split folder, or cases directly when --split is empty.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split below --dataset-root. Pass an empty string if case_* is directly below it.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Defaults to <output-root>/_mesh_cache.",
    )
    parser.add_argument("--start-case", type=int, default=0)
    parser.add_argument("--end-case", type=int, default=32)
    parser.add_argument("--source-modality", default="mr")
    parser.add_argument("--target-modality", default="us")
    parser.add_argument("--source-label", default="label0.nii.gz")
    parser.add_argument("--target-label", default="label0.nii.gz")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--coord-system",
        choices=["LPS", "RAS"],
        default="LPS",
        help="Physical coordinate convention for both generated STL files.",
    )
    parser.add_argument(
        "--smooth-source",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Smooth only the MR/source surface before registration (default: true).",
    )
    parser.add_argument(
        "--smooth-method",
        choices=["windowed_sinc", "laplacian"],
        default="windowed_sinc",
    )
    parser.add_argument("--smooth-iterations", type=int, default=250)
    parser.add_argument("--smooth-pass-band", type=float, default=0.005)
    parser.add_argument("--smooth-relaxation-factor", type=float, default=0.15)
    parser.add_argument("--smooth-feature-angle", type=float, default=180.0)
    parser.add_argument("--smooth-subdivision", type=int, default=2)
    parser.add_argument(
        "--smooth-subdivision-method",
        choices=["loop", "butterfly", "linear"],
        default="loop",
    )
    parser.add_argument(
        "--force-mesh",
        action="store_true",
        help="Rebuild cached STL surfaces even when they are newer than the NIfTI labels.",
    )
    parser.add_argument("--gpu-id", default="0", help="Value assigned to CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--cor-fem-script", type=Path, default=DEFAULT_COR_FEM_SCRIPT)
    parser.add_argument("--eval-label-start-index", type=int, default=-1)
    parser.add_argument("--skip-done", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue with later cases after a failed registration (default: true).",
    )
    parser.add_argument(
        "--extra-cor-fem-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="Append one raw argument to cor-fem-single.py; repeat for multiple arguments.",
    )
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, TypeError):
        return None


def _finite_metric(mapping: dict[str, Any], name: str) -> bool:
    try:
        return math.isfinite(float(mapping[name]))
    except (KeyError, TypeError, ValueError):
        return False


def valid_completed_result(case_outdir: Path) -> tuple[bool, dict[str, Any] | None]:
    """Validate the current nested run_meta.json schema.

    The shell launcher looked for ``chamfer_distance`` directly below
    ``surface_metrics``.  The current COR-FEM output stores it below
    ``surface_metrics.after_registration``, so that old check rejected valid
    results.
    """
    meta_path = case_outdir / "run_meta.json"
    if not meta_path.is_file() or not (case_outdir / "surface_metrics.json").is_file():
        return False, None

    meta = _load_json(meta_path)
    if meta is None:
        return False, None
    surface_metrics = meta.get("surface_metrics", {})
    if not isinstance(surface_metrics, dict):
        return False, meta
    after = surface_metrics.get("after_registration", surface_metrics)
    if not isinstance(after, dict):
        return False, meta

    valid = bool(
        meta.get("process_completed")
        and meta.get("registration_improved")
        and meta.get("jacobian_valid")
        and _finite_metric(after, "chamfer_distance")
        and _finite_metric(after, "hd95")
    )
    return valid, meta


def _metric_text(meta: dict[str, Any] | None) -> str:
    if not meta:
        return "metrics unavailable"
    surface_metrics = meta.get("surface_metrics", {})
    after = surface_metrics.get("after_registration", surface_metrics)
    try:
        return f"CD={float(after['chamfer_distance']):.6f} mm, HD95={float(after['hd95']):.6f} mm"
    except (KeyError, TypeError, ValueError):
        return "metrics unavailable"


def _needs_rebuild(output_path: Path, *input_paths: Path) -> bool:
    if not output_path.is_file():
        return True
    output_time = output_path.stat().st_mtime_ns
    return any(path.stat().st_mtime_ns > output_time for path in input_paths)


def prepare_case_surfaces(
    args: argparse.Namespace,
    case_dir: Path,
    case_cache: Path,
) -> tuple[Path, Path]:
    """Extract label0 surfaces and smooth only the source surface.

    ``volume2mesh`` reads nibabel arrays in ``(x,y,z)`` order, applies the full
    NIfTI affine and, for the default LPS convention, flips RAS x/y after that
    affine.  Consequently the returned mesh points are physical coordinates in
    millimetres rather than numpy array indices.
    """
    source_dir = case_dir / args.source_modality
    target_dir = case_dir / args.target_modality
    source_label = source_dir / args.source_label
    target_label = target_dir / args.target_label
    source_image = source_dir / "image.nii.gz"
    target_image = target_dir / "image.nii.gz"

    if not source_label.is_file() or not target_label.is_file():
        missing = [str(path) for path in (source_label, target_label) if not path.is_file()]
        raise FileNotFoundError("Missing registration label(s): " + ", ".join(missing))

    source_mesh_dir = case_cache / args.source_modality
    target_mesh_dir = case_cache / args.target_modality
    source_raw = source_mesh_dir / "label0_raw.stl"
    source_smooth = source_mesh_dir / "label0.stl"
    target_raw = target_mesh_dir / "label0.stl"

    rebuild_source_raw = args.force_mesh or _needs_rebuild(source_raw, source_label)
    if rebuild_source_raw:
        ok = nii_label_to_stl(
            label_path=source_label,
            out_path=source_raw,
            image_path=source_image if source_image.is_file() else None,
            threshold=args.threshold,
            coord_system=args.coord_system,
            pad_boundary=True,
            smooth=False,
            process_mesh=False,
            half_voxel_shift=False,
            save_meta=True,
        )
        if not ok:
            raise RuntimeError(f"Could not extract a non-empty source surface: {source_label}")

    if args.smooth_source:
        if args.force_mesh or rebuild_source_raw or _needs_rebuild(source_smooth, source_raw):
            ok = smooth_stl_vtk(
                in_stl=source_raw,
                out_stl=source_smooth,
                method=args.smooth_method,
                iterations=args.smooth_iterations,
                pass_band=args.smooth_pass_band,
                relaxation_factor=args.smooth_relaxation_factor,
                feature_angle=args.smooth_feature_angle,
                boundary_smoothing=True,
                feature_edge_smoothing=True,
                compute_normals=True,
                subdivision=args.smooth_subdivision,
                subdivision_method=args.smooth_subdivision_method,
            )
            if not ok:
                raise RuntimeError(f"Could not smooth source surface: {source_raw}")
        source_surface = source_smooth
    else:
        source_surface = source_raw

    if args.force_mesh or _needs_rebuild(target_raw, target_label):
        ok = nii_label_to_stl(
            label_path=target_label,
            out_path=target_raw,
            image_path=target_image if target_image.is_file() else None,
            threshold=args.threshold,
            coord_system=args.coord_system,
            pad_boundary=True,
            smooth=False,
            process_mesh=False,
            half_voxel_shift=False,
            save_meta=True,
        )
        if not ok:
            raise RuntimeError(f"Could not extract a non-empty target surface: {target_label}")

    return source_surface, target_raw


def _result_is_newer_than_inputs(case_outdir: Path, *inputs: Path) -> bool:
    meta_path = case_outdir / "run_meta.json"
    if not meta_path.is_file():
        return False
    meta_time = meta_path.stat().st_mtime_ns
    return all(path.is_file() and path.stat().st_mtime_ns <= meta_time for path in inputs)


def build_case_command(
    args: argparse.Namespace,
    case_dir: Path,
    case_outdir: Path,
    cache_root: Path,
    source_mesh: Path,
    target_mesh: Path,
) -> list[str]:
    case_cache = cache_root / case_dir.name
    command = [
        str(args.python_bin),
        str(args.cor_fem_script),
        "--source", str(source_mesh),
        "--target", str(target_mesh),
        "--outdir", str(case_outdir),
        "--preprocessed-source", str(case_cache / "source_repaired.ply"),
        "--tet-cache", str(case_cache / "source_tet_cache.npz"),
        "--eval-label-source-dir", str(source_mesh.parent),
        "--eval-label-target-dir", str(target_mesh.parent),
        "--eval-label-start-index", str(args.eval_label_start_index),
        "--device", args.device,
        *COR_FEM_ARGUMENTS,
        *args.extra_cor_fem_arg,
    ]
    return command


def append_summary(summary_path: Path, message: str) -> None:
    print(message, flush=True)
    with summary_path.open("a", encoding="utf-8") as stream:
        stream.write(message + "\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.start_case < 0 or args.end_case < args.start_case:
        raise ValueError("Require 0 <= --start-case <= --end-case")

    dataset_root = args.dataset_root.resolve()
    data_root = dataset_root / args.split if args.split else dataset_root
    output_root = args.output_root.resolve()
    cache_root = (args.cache_root or (output_root / "_mesh_cache")).resolve()
    cor_fem_script = args.cor_fem_script.resolve()
    python_bin = args.python_bin.resolve()
    args.cor_fem_script = cor_fem_script
    args.python_bin = python_bin

    if not cor_fem_script.is_file():
        raise FileNotFoundError(f"COR-FEM script not found: {cor_fem_script}")
    if not python_bin.is_file():
        raise FileNotFoundError(f"Python executable not found: {python_bin}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"Dataset directory not found: {data_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "batch_summary.txt"
    summary_path.write_text("", encoding="utf-8")

    print("=" * 60)
    print("Hybrid strict-energy COR-FEM prostate registration")
    print(f"Cases        : {args.start_case} ~ {args.end_case}")
    print(f"Dataset      : {data_root}")
    print(f"Output       : {output_root}")
    print(f"Mesh cache   : {cache_root}")
    print(f"Physical GPU : {args.gpu_id}")
    print(f"Device       : {args.device}")
    print(f"Python       : {python_bin}")
    print(f"Python file  : {cor_fem_script}")
    print("=" * 60)

    counts = {"success": 0, "failed": 0, "skipped": 0}
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    for case_id in range(args.start_case, args.end_case + 1):
        case_name = f"case_{case_id:04d}"
        case_dir = data_root / case_name
        source_label = case_dir / args.source_modality / args.source_label
        target_label = case_dir / args.target_modality / args.target_label
        case_outdir = output_root / case_name
        case_cache = cache_root / case_name

        print()
        print("-" * 60)
        print(f"[CASE] {case_name}")
        print(f"[SRC NIfTI] {source_label}")
        print(f"[TGT NIfTI] {target_label}")
        print(f"[OUT ] {case_outdir}")
        print("-" * 60)

        if not source_label.is_file() or not target_label.is_file():
            append_summary(summary_path, f"[SKIP] {case_name}: missing label0.nii.gz pair")
            counts["skipped"] += 1
            continue

        result_valid, result_meta = valid_completed_result(case_outdir)
        if (
            args.skip_done
            and result_valid
            and _result_is_newer_than_inputs(case_outdir, source_label, target_label)
        ):
            append_summary(
                summary_path,
                f"[SKIP] {case_name}: valid result exists, {_metric_text(result_meta)}",
            )
            counts["skipped"] += 1
            continue

        case_outdir.mkdir(parents=True, exist_ok=True)
        case_cache.mkdir(parents=True, exist_ok=True)
        try:
            source_mesh, target_mesh = prepare_case_surfaces(args, case_dir, case_cache)
        except Exception as error:
            append_summary(summary_path, f"[FAILED] {case_name}: surface preparation: {error}")
            counts["failed"] += 1
            if not args.continue_on_error:
                break
            continue

        tet_cache = case_cache / "source_tet_cache.npz"
        if tet_cache.is_file() and source_mesh.stat().st_mtime_ns > tet_cache.stat().st_mtime_ns:
            tet_cache.unlink()

        print(f"[SRC STL] {source_mesh}")
        print(f"[TGT STL] {target_mesh}")
        command = build_case_command(
            args,
            case_dir,
            case_outdir,
            cache_root,
            source_mesh,
            target_mesh,
        )
        completed = subprocess.run(command, env=environment, check=False, shell=False)

        result_valid, result_meta = valid_completed_result(case_outdir)
        if completed.returncode == 0 and result_valid:
            append_summary(
                summary_path,
                f"[SUCCESS] {case_name}: {_metric_text(result_meta)}",
            )
            counts["success"] += 1
        else:
            append_summary(
                summary_path,
                f"[FAILED] {case_name}: exit={completed.returncode}; missing/invalid/non-improved metadata",
            )
            counts["failed"] += 1
            if not args.continue_on_error:
                break

    append_summary(summary_path, "")
    append_summary(summary_path, "=" * 60)
    append_summary(summary_path, f"Success : {counts['success']}")
    append_summary(summary_path, f"Failed  : {counts['failed']}")
    append_summary(summary_path, f"Skipped : {counts['skipped']}")
    append_summary(summary_path, f"Summary : {summary_path}")
    append_summary(summary_path, "=" * 60)
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
