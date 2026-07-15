import argparse
import json
from pathlib import Path

import numpy as np
import nibabel as nib
from skimage import measure
import trimesh


def check_image_label_affine(image_path: Path, label_path: Path, atol: float = 1e-5):
    """
    Check whether image.nii.gz and label{i}.nii.gz have the same affine.
    """
    result = {
        "image_path": str(image_path),
        "label_path": str(label_path),
        "image_exists": image_path.exists(),
        "label_exists": label_path.exists(),
        "same_affine": None,
        "message": "",
    }

    if not image_path.exists():
        result["message"] = "image.nii.gz not found"
        return result

    if not label_path.exists():
        result["message"] = "label file not found"
        return result

    img = nib.load(str(image_path))
    lab = nib.load(str(label_path))

    img_aff = np.asarray(img.affine, dtype=np.float64)
    lab_aff = np.asarray(lab.affine, dtype=np.float64)

    result["image_shape"] = tuple(np.squeeze(np.asarray(img.dataobj)).shape)
    result["label_shape"] = tuple(np.squeeze(np.asarray(lab.dataobj)).shape)
    result["image_affine"] = img_aff.tolist()
    result["label_affine"] = lab_aff.tolist()
    result["same_affine"] = bool(np.allclose(img_aff, lab_aff, atol=atol))

    if result["same_affine"]:
        result["message"] = "image affine and label affine are consistent"
    else:
        result["message"] = "WARNING: image affine and label affine are different"

    return result


def nii_label_to_stl(
    label_path: Path,
    out_path: Path,
    image_path: Path = None,
    threshold: float = 0.5,
    coord_system: str = "LPS",
    pad_boundary: bool = True,
    smooth: bool = False,
    process_mesh: bool = False,
    half_voxel_shift: bool = False,
    save_meta: bool = True,
):
    """
    Convert one NIfTI label to STL.

    Parameters
    ----------
    coord_system:
        "RAS": keep nibabel affine world coordinates.
        "LPS": flip x and y after affine transform. Recommended for MITK / DKFZ.
    """

    label_path = Path(label_path)
    out_path = Path(out_path)

    if not label_path.exists():
        print(f"[Missing] {label_path}")
        return False

    nii = nib.load(str(label_path))
    data = np.asarray(nii.get_fdata())
    data = np.squeeze(data)

    if data.ndim != 3:
        raise ValueError(f"Expected 3D label, got shape={data.shape}, path={label_path}")

    mask = data > threshold

    if np.count_nonzero(mask) == 0:
        print(f"[Skip Empty] {label_path}")
        return False

    affine = np.asarray(nii.affine, dtype=np.float64)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)

    affine_check = None
    if image_path is not None:
        image_path = Path(image_path)
        try:
            affine_check = check_image_label_affine(image_path, label_path)
        except Exception as e:
            affine_check = {
                "image_path": str(image_path),
                "label_path": str(label_path),
                "image_exists": image_path.exists(),
                "label_exists": label_path.exists(),
                "same_affine": None,
                "message": f"image affine check skipped: {repr(e)}",
            }
            print(f"[WARN] {label_path}")
            print(f"       image affine check skipped: {repr(e)}")
            print("       STL will be generated using LABEL affine.")

        if affine_check["same_affine"] is False:
            print(f"[WARN] {label_path}")
            print("       image affine and label affine are different.")
            print("       STL will be generated using LABEL affine.")

    # ------------------------------------------------------------------
    # 1. Marching cubes on binary label
    # ------------------------------------------------------------------
    if pad_boundary:
        # Pad one voxel to avoid clipped/open surface if mask touches image boundary.
        mask_for_mc = np.pad(
            mask.astype(np.float32),
            pad_width=1,
            mode="constant",
            constant_values=0,
        )

        verts, faces, normals, values = measure.marching_cubes(
            mask_for_mc,
            level=0.5,
            spacing=(1.0, 1.0, 1.0),
            allow_degenerate=False,
        )

        # Remove padding offset.
        verts = verts - 1.0

    else:
        verts, faces, normals, values = measure.marching_cubes(
            mask.astype(np.float32),
            level=0.5,
            spacing=(1.0, 1.0, 1.0),
            allow_degenerate=False,
        )

    # Optional test only.
    # Normally keep False. Use it only if you observe a uniform half-voxel offset.
    if half_voxel_shift:
        verts = verts + 0.5

    # ------------------------------------------------------------------
    # 2. Voxel -> NIfTI world coordinate
    # ------------------------------------------------------------------
    verts_world = verts @ affine[:3, :3].T + affine[:3, 3]

    # ------------------------------------------------------------------
    # 3. Coordinate system conversion
    # ------------------------------------------------------------------
    coord_system = coord_system.upper()

    if coord_system == "RAS":
        # Keep as NIfTI/nibabel world coordinate.
        pass

    elif coord_system == "LPS":
        # RAS -> LPS:
        # x: Right  -> Left
        # y: Anterior -> Posterior
        # z: Superior unchanged
        verts_world[:, 0] *= -1.0
        verts_world[:, 1] *= -1.0

    else:
        raise ValueError("coord_system must be 'RAS' or 'LPS'")

    # ------------------------------------------------------------------
    # 4. Save STL
    # ------------------------------------------------------------------
    mesh = trimesh.Trimesh(
        vertices=verts_world,
        faces=faces,
        process=process_mesh,
    )

    if smooth:
        print("[WARN] smooth=True will move the STL surface away from original label boundary.")
        trimesh.smoothing.filter_laplacian(mesh)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_path))

    print(
        f"[OK] {label_path} -> {out_path} | "
        f"shape={data.shape}, spacing={spacing}, "
        f"verts={len(mesh.vertices)}, faces={len(mesh.faces)}, "
        f"coord={coord_system}, process={process_mesh}, smooth={smooth}"
    )

    if save_meta:
        meta = {
            "label_path": str(label_path),
            "image_path": str(image_path) if image_path is not None else "",
            "out_path": str(out_path),
            "threshold": float(threshold),
            "coord_system": coord_system,
            "ras_to_lps_flip_xy": bool(coord_system == "LPS"),
            "pad_boundary": bool(pad_boundary),
            "smooth": bool(smooth),
            "process_mesh": bool(process_mesh),
            "half_voxel_shift": bool(half_voxel_shift),
            "label_shape": tuple(int(x) for x in data.shape),
            "spacing_from_affine": spacing.tolist(),
            "label_affine": affine.tolist(),
            "num_label_voxels": int(np.count_nonzero(mask)),
            "num_mesh_vertices": int(len(mesh.vertices)),
            "num_mesh_faces": int(len(mesh.faces)),
            "mesh_bounds": mesh.bounds.tolist(),
            "affine_check": affine_check,
            "note": (
                "For MITK/DKFZ, coord_system=LPS is recommended. "
                "It flips x and y after voxel-to-world affine transform."
            ),
        }

        meta_path = out_path.with_suffix(".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    return True


def export_dataset(
    data_root: Path,
    out_root: Path,
    subsets,
    modalities,
    labels,
    threshold: float,
    coord_system: str,
    pad_boundary: bool,
    smooth: bool,
    process_mesh: bool,
    half_voxel_shift: bool,
    save_meta: bool,
):
    total = 0
    success = 0
    skipped = 0
    failed = 0

    for subset in subsets:
        subset_dir = data_root / subset

        if not subset_dir.exists():
            print(f"[Skip] subset not found: {subset_dir}")
            continue

        case_dirs = sorted([
            p for p in subset_dir.iterdir()
            if p.is_dir() and p.name.startswith("case_")
        ])

        print("\n" + "=" * 80)
        print(f"[Subset] {subset}, cases={len(case_dirs)}")
        print("=" * 80)

        for case_dir in case_dirs:
            for modality in modalities:
                image_path = case_dir / modality / "image.nii.gz"

                for label_id in labels:
                    total += 1

                    label_path = case_dir / modality / f"label{label_id}.nii.gz"
                    out_path = (
                        out_root
                        / subset
                        / case_dir.name
                        / modality
                        / f"label{label_id}.stl"
                    )

                    if not label_path.exists():
                        print(f"[Missing] {label_path}")
                        skipped += 1
                        continue

                    try:
                        ok = nii_label_to_stl(
                            label_path=label_path,
                            out_path=out_path,
                            image_path=image_path,
                            threshold=threshold,
                            coord_system=coord_system,
                            pad_boundary=pad_boundary,
                            smooth=smooth,
                            process_mesh=process_mesh,
                            half_voxel_shift=half_voxel_shift,
                            save_meta=save_meta,
                        )

                        if ok:
                            success += 1
                        else:
                            skipped += 1

                    except Exception as e:
                        failed += 1
                        print(f"[FAIL] {label_path} | {repr(e)}")

    print("\n" + "=" * 80)
    print("[Done]")
    print(f"Total labels: {total}")
    print(f"Exported STL: {success}")
    print(f"Skipped:      {skipped}")
    print(f"Failed:       {failed}")
    print(f"Output root:  {out_root}")
    print("=" * 80)



def find_image_path(modality_dir: Path):
    """
    Find the image volume next to labels when possible.
    """
    for name in ("image.nii.gz", "image.nii"):
        image_path = modality_dir / name
        if image_path.exists():
            return image_path
    return None


def export_flat_modalities(
    data_root: Path,
    out_root: Path,
    modalities,
    labels,
    threshold: float,
    coord_system: str,
    pad_boundary: bool,
    smooth: bool,
    process_mesh: bool,
    half_voxel_shift: bool,
    save_meta: bool,
):
    total = 0
    success = 0
    skipped = 0
    failed = 0

    print("\n" + "=" * 80)
    print(f"[Flat Layout] data_root={data_root}")
    print("=" * 80)

    for modality in modalities:
        modality_dir = data_root / modality

        if not modality_dir.exists():
            print(f"[Skip] modality folder not found: {modality_dir}")
            continue

        image_path = find_image_path(modality_dir)
        if image_path is None:
            print(f"[WARN] image file not found in {modality_dir}; affine check skipped.")

        for label_id in labels:
            total += 1
            label_path = modality_dir / f"label{label_id}.nii.gz"
            if not label_path.exists():
                label_path = modality_dir / f"label{label_id}.nii"

            out_path = out_root / modality / f"label{label_id}.stl"

            if not label_path.exists():
                print(f"[Missing] {modality_dir / f'label{label_id}.nii.gz'}")
                skipped += 1
                continue

            try:
                ok = nii_label_to_stl(
                    label_path=label_path,
                    out_path=out_path,
                    image_path=image_path,
                    threshold=threshold,
                    coord_system=coord_system,
                    pad_boundary=pad_boundary,
                    smooth=smooth,
                    process_mesh=process_mesh,
                    half_voxel_shift=half_voxel_shift,
                    save_meta=save_meta,
                )

                if ok:
                    success += 1
                else:
                    skipped += 1

            except Exception as e:
                failed += 1
                print(f"[FAIL] {label_path} | {repr(e)}")

    print("\n" + "=" * 80)
    print("[Done]")
    print(f"Total labels: {total}")
    print(f"Exported STL: {success}")
    print(f"Skipped:      {skipped}")
    print(f"Failed:       {failed}")
    print(f"Output root:  {out_root}")
    print("=" * 80)


def detect_layout(data_root: Path, modalities):
    """
    Detect whether labels are directly under data_root/mr and data_root/us.
    """
    for modality in modalities:
        modality_dir = data_root / modality
        if not modality_dir.exists():
            continue

        if any(modality_dir.glob("label*.nii.gz")) or any(modality_dir.glob("label*.nii")):
            return "flat"

    return "dataset"

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root",
        type=str,
        default=".",
        help="Root folder. For this workspace, use current folder containing mr/ and us/.",
    )

    parser.add_argument(
        "--out_root",
        type=str,
        default="mesh",
        help="Output STL folder.",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="auto",
        choices=["auto", "flat", "dataset"],
        help=(
            "Input layout. flat means data_root/mr/label1.nii.gz and "
            "data_root/us/label1.nii.gz. dataset keeps the old subset/case layout."
        ),
    )

    parser.add_argument(
        "--subsets",
        type=str,
        default="test",
        help="Comma-separated subsets, e.g. train,val,test or test.",
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
        default="1,2,3",
        help="Comma-separated label ids.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binary threshold for label.",
    )

    parser.add_argument(
        "--coord_system",
        type=str,
        default="LPS",
        choices=["RAS", "LPS"],
        help=(
            "Coordinate system for exported STL. "
            "Use LPS for MITK/DKFZ. Use RAS for RAS-based viewers."
        ),
    )

    parser.add_argument(
        "--no_pad_boundary",
        action="store_true",
        help="Disable one-voxel zero padding before marching cubes.",
    )

    parser.add_argument(
        "--smooth",
        action="store_true",
        help="Apply Laplacian smoothing. Not recommended for exact label alignment.",
    )

    parser.add_argument(
        "--process_mesh",
        action="store_true",
        help="Let trimesh process mesh. Not recommended for exact alignment.",
    )

    parser.add_argument(
        "--half_voxel_shift",
        action="store_true",
        help=(
            "Add +0.5 voxel shift before affine. "
            "Normally do not use this."
        ),
    )

    parser.add_argument(
        "--no_meta",
        action="store_true",
        help="Do not save per-STL JSON metadata.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)

    subsets = [x.strip() for x in args.subsets.split(",") if x.strip()]
    modalities = [x.strip() for x in args.modalities.split(",") if x.strip()]
    labels = [int(x.strip()) for x in args.labels.split(",") if x.strip()]

    layout = args.layout
    if layout == "auto":
        layout = detect_layout(data_root, modalities)

    if layout == "flat":
        export_flat_modalities(
            data_root=data_root,
            out_root=out_root,
            modalities=modalities,
            labels=labels,
            threshold=args.threshold,
            coord_system=args.coord_system,
            pad_boundary=not args.no_pad_boundary,
            smooth=args.smooth,
            process_mesh=args.process_mesh,
            half_voxel_shift=args.half_voxel_shift,
            save_meta=not args.no_meta,
        )
    else:
        export_dataset(
            data_root=data_root,
            out_root=out_root,
            subsets=subsets,
            modalities=modalities,
            labels=labels,
            threshold=args.threshold,
            coord_system=args.coord_system,
            pad_boundary=not args.no_pad_boundary,
            smooth=args.smooth,
            process_mesh=args.process_mesh,
            half_voxel_shift=args.half_voxel_shift,
            save_meta=not args.no_meta,
        )


if __name__ == "__main__":
    main()


