#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STL mesh smoothing with optional subdivision.

Compared with pure smoothing, subdivision increases mesh density first, so the final
surface can become much rounder/smoother instead of only moving existing vertices.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import vtk


def subdivide_polydata(poly: vtk.vtkPolyData, subdivisions: int = 0, method: str = "loop") -> vtk.vtkPolyData:
    """Subdivide triangle mesh to create more vertices/faces before smoothing."""
    subdivisions = int(subdivisions)
    if subdivisions <= 0:
        return poly

    method = method.lower()
    if method == "loop":
        sub = vtk.vtkLoopSubdivisionFilter()
    elif method == "butterfly":
        sub = vtk.vtkButterflySubdivisionFilter()
    elif method == "linear":
        sub = vtk.vtkLinearSubdivisionFilter()
    else:
        raise ValueError("subdivision_method must be 'loop', 'butterfly', or 'linear'")

    sub.SetInputData(poly)
    sub.SetNumberOfSubdivisions(subdivisions)
    sub.Update()
    return sub.GetOutput()


def make_smoother(
    poly: vtk.vtkPolyData,
    method: str,
    iterations: int,
    pass_band: float,
    relaxation_factor: float,
    feature_angle: float,
    boundary_smoothing: bool,
    feature_edge_smoothing: bool,
):
    """Create and configure VTK smoother."""
    method = method.lower()

    if method == "windowed_sinc":
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputData(poly)
        smoother.SetNumberOfIterations(iterations)
        smoother.SetPassBand(pass_band)
        smoother.SetFeatureAngle(feature_angle)
        smoother.NormalizeCoordinatesOn()

        if boundary_smoothing:
            smoother.BoundarySmoothingOn()
        else:
            smoother.BoundarySmoothingOff()

        if feature_edge_smoothing:
            smoother.FeatureEdgeSmoothingOn()
        else:
            smoother.FeatureEdgeSmoothingOff()

        print(
            f"[Smooth] method=windowed_sinc, iterations={iterations}, "
            f"pass_band={pass_band}, feature_angle={feature_angle}, "
            f"feature_edge_smoothing={feature_edge_smoothing}"
        )
        return smoother

    if method == "laplacian":
        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputData(poly)
        smoother.SetNumberOfIterations(iterations)
        smoother.SetRelaxationFactor(relaxation_factor)
        smoother.SetFeatureAngle(feature_angle)

        if boundary_smoothing:
            smoother.BoundarySmoothingOn()
        else:
            smoother.BoundarySmoothingOff()

        if feature_edge_smoothing:
            smoother.FeatureEdgeSmoothingOn()
        else:
            smoother.FeatureEdgeSmoothingOff()

        print(
            f"[Smooth] method=laplacian, iterations={iterations}, "
            f"relaxation_factor={relaxation_factor}, feature_angle={feature_angle}, "
            f"feature_edge_smoothing={feature_edge_smoothing}"
        )
        return smoother

    raise ValueError("method must be 'windowed_sinc' or 'laplacian'")


def smooth_stl_vtk(
    in_stl: Path,
    out_stl: Path,
    method: str = "windowed_sinc",
    iterations: int = 120,
    pass_band: float = 0.02,
    relaxation_factor: float = 0.15,
    feature_angle: float = 180.0,
    boundary_smoothing: bool = True,
    feature_edge_smoothing: bool = True,
    compute_normals: bool = True,
    subdivision: int = 2,
    subdivision_method: str = "loop",
):
    """
    Smooth one STL file using VTK.

    Recommended for very smooth organ-like surfaces:
        method='windowed_sinc', subdivision=1~2, iterations=100~200,
        pass_band=0.01~0.03, feature_angle=180, feature_edge_smoothing=True.

    Notes:
        - subdivision increases triangle count, giving the smoother more geometry to work with.
        - very aggressive smoothing may erase anatomical details and change volume/shape.
    """
    in_stl = Path(in_stl)
    out_stl = Path(out_stl)

    if not in_stl.exists():
        print(f"[Missing] {in_stl}")
        return False

    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(in_stl))
    reader.Update()
    poly = reader.GetOutput()

    if poly is None or poly.GetNumberOfPoints() == 0:
        print(f"[Skip Empty] {in_stl}")
        return False

    n_points_before = poly.GetNumberOfPoints()
    n_cells_before = poly.GetNumberOfCells()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(poly)
    cleaner.Update()

    triangle = vtk.vtkTriangleFilter()
    triangle.SetInputConnection(cleaner.GetOutputPort())
    triangle.Update()

    current_poly = triangle.GetOutput()

    if subdivision > 0:
        current_poly = subdivide_polydata(
            current_poly,
            subdivisions=subdivision,
            method=subdivision_method,
        )
        print(
            f"[Subdivision] method={subdivision_method}, levels={subdivision}, "
            f"points={current_poly.GetNumberOfPoints()}, cells={current_poly.GetNumberOfCells()}"
        )

    smoother = make_smoother(
        poly=current_poly,
        method=method,
        iterations=iterations,
        pass_band=pass_band,
        relaxation_factor=relaxation_factor,
        feature_angle=feature_angle,
        boundary_smoothing=boundary_smoothing,
        feature_edge_smoothing=feature_edge_smoothing,
    )
    smoother.Update()
    smoothed_poly = smoother.GetOutput()

    if compute_normals:
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(smoothed_poly)
        normals.SetFeatureAngle(feature_angle)
        normals.ConsistencyOn()
        normals.AutoOrientNormalsOn()
        normals.SplittingOff()
        normals.Update()
        final_poly = normals.GetOutput()
    else:
        final_poly = smoothed_poly

    out_stl.parent.mkdir(parents=True, exist_ok=True)

    writer = vtk.vtkSTLWriter()
    writer.SetFileName(str(out_stl))
    writer.SetInputData(final_poly)
    writer.SetFileTypeToBinary()
    writer.Write()

    n_points_after = final_poly.GetNumberOfPoints()
    n_cells_after = final_poly.GetNumberOfCells()

    print(
        f"[OK] {in_stl} -> {out_stl} | "
        f"points {n_points_before}->{n_points_after}, "
        f"cells {n_cells_before}->{n_cells_after}"
    )

    return True


def smooth_flat_modalities(
    in_root: Path,
    out_root: Path,
    modalities,
    labels,
    method: str,
    iterations: int,
    pass_band: float,
    relaxation_factor: float,
    feature_angle: float,
    boundary_smoothing: bool,
    feature_edge_smoothing: bool,
    compute_normals: bool,
    subdivision: int,
    subdivision_method: str,
):
    total = 0
    success = 0
    skipped = 0

    print("\n" + "=" * 80)
    print(f"[Flat Layout] in_root={in_root}")
    print("=" * 80)

    for modality in modalities:
        modality_dir = in_root / modality

        if not modality_dir.exists():
            print(f"[Skip] modality folder not found: {modality_dir}")
            continue

        for label_id in labels:
            total += 1
            in_stl = modality_dir / f"label{label_id}.stl"
            out_stl = out_root / modality / f"label{label_id}.stl"

            if not in_stl.exists():
                print(f"[Missing] {in_stl}")
                skipped += 1
                continue

            try:
                ok = smooth_stl_vtk(
                    in_stl=in_stl,
                    out_stl=out_stl,
                    method=method,
                    iterations=iterations,
                    pass_band=pass_band,
                    relaxation_factor=relaxation_factor,
                    feature_angle=feature_angle,
                    boundary_smoothing=boundary_smoothing,
                    feature_edge_smoothing=feature_edge_smoothing,
                    compute_normals=compute_normals,
                    subdivision=subdivision,
                    subdivision_method=subdivision_method,
                )

                if ok:
                    success += 1
                else:
                    skipped += 1

            except Exception as e:
                skipped += 1
                print(f"[FAIL] {in_stl} | {repr(e)}")

    print("\n" + "=" * 80)
    print("[Done]")
    print(f"Total STL:   {total}")
    print(f"Smoothed:    {success}")
    print(f"Skipped:     {skipped}")
    print(f"Output root: {out_root}")
    print("=" * 80)


def smooth_dataset(
    in_root: Path,
    out_root: Path,
    splits,
    modalities,
    labels,
    method: str,
    iterations: int,
    pass_band: float,
    relaxation_factor: float,
    feature_angle: float,
    boundary_smoothing: bool,
    feature_edge_smoothing: bool,
    compute_normals: bool,
    subdivision: int,
    subdivision_method: str,
):
    total = 0
    success = 0
    skipped = 0

    for split in splits:
        split_dir = in_root / split

        if not split_dir.exists():
            print(f"[Skip] split not found: {split_dir}")
            continue

        case_dirs = sorted([
            p for p in split_dir.iterdir()
            if p.is_dir() and p.name.startswith("case_")
        ])

        print("\n" + "=" * 80)
        print(f"[Split] {split}, cases={len(case_dirs)}")
        print("=" * 80)

        for case_dir in case_dirs:
            for modality in modalities:
                for label_id in labels:
                    total += 1
                    in_stl = case_dir / modality / f"label{label_id}.stl"
                    out_stl = out_root / in_stl.relative_to(in_root)

                    if not in_stl.exists():
                        print(f"[Missing] {in_stl}")
                        skipped += 1
                        continue

                    try:
                        ok = smooth_stl_vtk(
                            in_stl=in_stl,
                            out_stl=out_stl,
                            method=method,
                            iterations=iterations,
                            pass_band=pass_band,
                            relaxation_factor=relaxation_factor,
                            feature_angle=feature_angle,
                            boundary_smoothing=boundary_smoothing,
                            feature_edge_smoothing=feature_edge_smoothing,
                            compute_normals=compute_normals,
                            subdivision=subdivision,
                            subdivision_method=subdivision_method,
                        )

                        if ok:
                            success += 1
                        else:
                            skipped += 1

                    except Exception as e:
                        skipped += 1
                        print(f"[FAIL] {in_stl} | {repr(e)}")

    print("\n" + "=" * 80)
    print("[Done]")
    print(f"Total STL:   {total}")
    print(f"Smoothed:    {success}")
    print(f"Skipped:     {skipped}")
    print(f"Output root: {out_root}")
    print("=" * 80)


def detect_layout(in_root: Path, modalities):
    for modality in modalities:
        modality_dir = in_root / modality
        if modality_dir.exists() and any(modality_dir.glob("label*.stl")):
            return "flat"
    return "dataset"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--in_root", type=str, default="mesh", help="Input STL root folder.")
    parser.add_argument("--out_root", type=str, default="mesh_smooth", help="Output smoothed STL root folder.")
    parser.add_argument(
        "--layout",
        type=str,
        default="auto",
        choices=["auto", "flat", "dataset"],
        help=(
            "Input layout. flat means in_root/mr/label1.stl and "
            "in_root/us/label1.stl. dataset keeps the old split/case layout."
        ),
    )
    parser.add_argument("--splits", type=str, default="test", help="Comma-separated splits, e.g. train,val,test or test.")
    parser.add_argument("--modalities", type=str, default="mr,us", help="Comma-separated modalities, e.g. mr,us.")
    parser.add_argument("--labels", type=str, default="1,2,3", help="Comma-separated label ids.")

    parser.add_argument(
        "--method",
        type=str,
        default="windowed_sinc",
        choices=["windowed_sinc", "laplacian"],
        help="Smoothing method.",
    )
    parser.add_argument("--iterations", type=int, default=250, help="Number of smoothing iterations.")
    parser.add_argument(
        "--pass_band",
        type=float,
        default=0.005,
        help="Pass band for windowed sinc smoothing. Smaller means smoother.",
    )
    parser.add_argument("--relaxation_factor", type=float, default=0.15, help="Relaxation factor for Laplacian smoothing.")
    parser.add_argument("--feature_angle", type=float, default=180.0, help="Feature angle for smoothing and normals.")

    parser.add_argument(
        "--subdivision",
        type=int,
        default=2,
        help="Subdivision levels before smoothing. 0 disables. 1-2 recommended, 3 may be heavy.",
    )
    parser.add_argument(
        "--subdivision_method",
        type=str,
        default="loop",
        choices=["loop", "butterfly", "linear"],
        help="Subdivision method. loop is usually roundest; linear only adds triangles without changing shape much.",
    )

    parser.add_argument("--no_boundary_smoothing", action="store_true", help="Disable boundary smoothing.")
    parser.add_argument("--no_feature_edge_smoothing", action="store_true", help="Disable feature edge smoothing.")
    parser.add_argument("--no_normals", action="store_true", help="Do not recompute normals after smoothing.")

    parser.add_argument(
        "--ultra_smooth",
        action="store_true",
        help="Shortcut: very aggressive smoothing. May erase shape details.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.ultra_smooth:
        args.method = "windowed_sinc"
        args.iterations = max(args.iterations, 220)
        args.pass_band = min(args.pass_band, 0.008)
        args.feature_angle = max(args.feature_angle, 180.0)
        args.subdivision = max(args.subdivision, 2)
        args.subdivision_method = "loop"

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)

    splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    modalities = [x.strip() for x in args.modalities.split(",") if x.strip()]
    labels = [int(x.strip()) for x in args.labels.split(",") if x.strip()]

    layout = args.layout
    if layout == "auto":
        layout = detect_layout(in_root, modalities)

    common_kwargs = dict(
        in_root=in_root,
        out_root=out_root,
        modalities=modalities,
        labels=labels,
        method=args.method,
        iterations=args.iterations,
        pass_band=args.pass_band,
        relaxation_factor=args.relaxation_factor,
        feature_angle=args.feature_angle,
        boundary_smoothing=not args.no_boundary_smoothing,
        feature_edge_smoothing=not args.no_feature_edge_smoothing,
        compute_normals=not args.no_normals,
        subdivision=args.subdivision,
        subdivision_method=args.subdivision_method,
    )

    if layout == "flat":
        smooth_flat_modalities(**common_kwargs)
    else:
        smooth_dataset(splits=splits, **common_kwargs)


if __name__ == "__main__":
    main()
