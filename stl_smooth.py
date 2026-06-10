#!/usr/bin/env python3
# -*- coding: utf-8 -*
import argparse
from pathlib import Path

import vtk


def smooth_stl_vtk(
    in_stl: Path,
    out_stl: Path,
    method: str = "windowed_sinc",
    iterations: int = 30,
    pass_band: float = 0.1,
    relaxation_factor: float = 0.15,
    feature_angle: float = 120.0,
    boundary_smoothing: bool = True,
    feature_edge_smoothing: bool = False,
    compute_normals: bool = True,
):
    """
    Smooth one STL file using VTK.

    method:
        "windowed_sinc": 推荐，形状收缩较小
        "laplacian": 普通 Laplacian smoothing，可能会收缩
    """

    in_stl = Path(in_stl)
    out_stl = Path(out_stl)

    if not in_stl.exists():
        print(f"[Missing] {in_stl}")
        return False

    # ------------------------------------------------------------
    # 1. Read STL
    # ------------------------------------------------------------
    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(in_stl))
    reader.Update()

    poly = reader.GetOutput()

    if poly is None or poly.GetNumberOfPoints() == 0:
        print(f"[Skip Empty] {in_stl}")
        return False

    n_points_before = poly.GetNumberOfPoints()
    n_cells_before = poly.GetNumberOfCells()

    # ------------------------------------------------------------
    # 2. Clean mesh
    # ------------------------------------------------------------
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(poly)
    cleaner.Update()

    # ------------------------------------------------------------
    # 3. Make sure mesh is triangular
    # ------------------------------------------------------------
    triangle = vtk.vtkTriangleFilter()
    triangle.SetInputConnection(cleaner.GetOutputPort())
    triangle.Update()

    # ------------------------------------------------------------
    # 4. Smooth
    # ------------------------------------------------------------
    method = method.lower()

    if method == "windowed_sinc":
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputConnection(triangle.GetOutputPort())
        smoother.SetNumberOfIterations(iterations)
        smoother.SetPassBand(pass_band)

        if boundary_smoothing:
            smoother.BoundarySmoothingOn()
        else:
            smoother.BoundarySmoothingOff()

        if feature_edge_smoothing:
            smoother.FeatureEdgeSmoothingOn()
        else:
            smoother.FeatureEdgeSmoothingOff()

        smoother.SetFeatureAngle(feature_angle)

        # Normalization helps numerical stability
        smoother.NormalizeCoordinatesOn()

        print(
            f"[Smooth] method=windowed_sinc, iterations={iterations}, "
            f"pass_band={pass_band}, feature_angle={feature_angle}"
        )

    elif method == "laplacian":
        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputConnection(triangle.GetOutputPort())
        smoother.SetNumberOfIterations(iterations)
        smoother.SetRelaxationFactor(relaxation_factor)

        if boundary_smoothing:
            smoother.BoundarySmoothingOn()
        else:
            smoother.BoundarySmoothingOff()

        if feature_edge_smoothing:
            smoother.FeatureEdgeSmoothingOn()
        else:
            smoother.FeatureEdgeSmoothingOff()

        smoother.SetFeatureAngle(feature_angle)

        print(
            f"[Smooth] method=laplacian, iterations={iterations}, "
            f"relaxation_factor={relaxation_factor}, feature_angle={feature_angle}"
        )

    else:
        raise ValueError("method must be 'windowed_sinc' or 'laplacian'")

    smoother.Update()

    smoothed_poly = smoother.GetOutput()

    # ------------------------------------------------------------
    # 5. Recompute normals
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # 6. Write STL
    # ------------------------------------------------------------
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

                    rel_path = in_stl.relative_to(in_root)
                    out_stl = out_root / rel_path

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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--in_root",
        type=str,
        default="E:/CODE/DataSet/Prostate/muregdataset/mureg_data_shixing_rigid_us2mr_quality_gated/mureg_stl_lps",
        help="Input STL root folder.",
    )

    parser.add_argument(
        "--out_root",
        type=str,
        default="E:/CODE/DataSet/Prostate/muregdataset/mureg_data_shixing_rigid_us2mr_quality_gated/mureg_stl_lps_smooth",
        help="Output smoothed STL root folder.",
    )

    parser.add_argument(
        "--splits",
        type=str,
        default="test",
        help="Comma-separated splits, e.g. train,val,test or test.",
    )

    parser.add_argument(
        "--modalities",
        type=str,
        default="mr,us",
        help="Comma-separated modalities, e.g. mr,us.",
    )

    parser.add_argument(
        "--labels",
        type=str,
        default="0,1,2,3,4,5",
        help="Comma-separated label ids.",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="windowed_sinc",
        choices=["windowed_sinc", "laplacian"],
        help="Smoothing method.",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=30,
        help="Number of smoothing iterations.",
    )

    parser.add_argument(
        "--pass_band",
        type=float,
        default=0.1,
        help="Pass band for windowed sinc smoothing. Smaller means smoother.",
    )

    parser.add_argument(
        "--relaxation_factor",
        type=float,
        default=0.15,
        help="Relaxation factor for Laplacian smoothing.",
    )

    parser.add_argument(
        "--feature_angle",
        type=float,
        default=120.0,
        help="Feature angle for smoothing and normals.",
    )

    parser.add_argument(
        "--no_boundary_smoothing",
        action="store_true",
        help="Disable boundary smoothing.",
    )

    parser.add_argument(
        "--feature_edge_smoothing",
        action="store_true",
        help="Enable feature edge smoothing.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)

    splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    modalities = [x.strip() for x in args.modalities.split(",") if x.strip()]
    labels = [int(x.strip()) for x in args.labels.split(",") if x.strip()]

    smooth_dataset(
        in_root=in_root,
        out_root=out_root,
        splits=splits,
        modalities=modalities,
        labels=labels,
        method=args.method,
        iterations=args.iterations,
        pass_band=args.pass_band,
        relaxation_factor=args.relaxation_factor,
        feature_angle=args.feature_angle,
        boundary_smoothing=not args.no_boundary_smoothing,
        feature_edge_smoothing=args.feature_edge_smoothing,
    )


if __name__ == "__main__":
    main()
