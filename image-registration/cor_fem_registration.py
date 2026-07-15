"""Stable Python wrapper around the existing ``cor-fem-single.py`` CLI."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


REQUIRED_DEFORMATION_KEYS = (
    "reference_nodes",
    "deformed_nodes",
    "tetrahedra",
    "surface_node_ids",
    "surface_faces",
    "surface_vertices_initial",
    "surface_vertices_final",
)


def load_registration_result(output_dir: str | Path) -> dict:
    """Load a previously exported COR-FEM tetrahedral deformation."""
    output_dir = Path(output_dir)
    data_path = output_dir / "deformation_data.npz"
    if not data_path.is_file():
        raise FileNotFoundError(
            "COR-FEM completed without deformation_data.npz. Ensure the bundled "
            "cor-fem-single.py result-export patch is present."
        )
    with np.load(data_path, allow_pickle=False) as data:
        missing = [key for key in REQUIRED_DEFORMATION_KEYS if key not in data]
        if missing:
            raise KeyError(f"Missing deformation arrays in {data_path}: {missing}")
        result = {key: np.asarray(data[key]) for key in REQUIRED_DEFORMATION_KEYS}
    result["deformation_data_path"] = data_path
    result["tetra_reference_path"] = output_dir / "reference_volume.vtu"
    result["tetra_deformed_path"] = output_dir / "deformed_volume.vtu"
    result["deformed_surface_path"] = output_dir / "deformed_surface.ply"
    result["run_meta_path"] = output_dir / "run_meta.json"
    return result


def run_cor_fem_registration(
    source_surface: str | Path,
    target_surface: str | Path,
    output_dir: str | Path,
    *,
    cor_fem_script: str | Path | None = None,
    python_executable: str | Path | None = None,
    device: str = "cuda:0",
    dtype: str = "float32",
    tet_cache: str | Path | None = None,
    max_iterations: int = 100,
    target_sample_count: int = 12_000,
    extra_arguments: Iterable[str] = (),
    quiet: bool = False,
) -> dict:
    """Run the current COR-FEM program without replacing its registration model.

    The subprocess is invoked with an argument list (never ``shell=True``), so
    Windows paths containing spaces are handled without manual quoting.
    """
    source = Path(source_surface).resolve()
    target = Path(target_surface).resolve()
    out = Path(output_dir).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source surface not found: {source}")
    if not target.is_file():
        raise FileNotFoundError(f"Target surface not found: {target}")
    script = Path(cor_fem_script or Path(__file__).with_name("cor-fem-single.py")).resolve()
    if not script.is_file():
        raise FileNotFoundError(f"COR-FEM script not found: {script}")
    executable = str(Path(python_executable).resolve()) if python_executable else sys.executable
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(tet_cache).resolve() if tet_cache else out / "source_tet_cache.npz"

    command = [
        executable,
        str(script),
        "--source",
        str(source),
        "--target",
        str(target),
        "--outdir",
        str(out),
        "--tet-cache",
        str(cache),
        "--device",
        device,
        "--dtype",
        dtype,
        "--max-iters",
        str(int(max_iterations)),
        "--target-sample-count",
        str(int(target_sample_count)),
        "--no-blender-repair-source",
        "--boundary-mode",
        "none",
    ]
    if quiet:
        command.append("--quiet")
    command.extend(str(argument) for argument in extra_arguments)
    subprocess.run(command, check=True, cwd=str(script.parent))
    result = load_registration_result(out)
    result["command"] = command
    return result


def copy_registration_artifacts(
    result: dict,
    meshes_dir: str | Path,
    deformation_dir: str | Path,
) -> dict:
    """Copy existing COR-FEM outputs into the public case output layout."""
    mesh_output = Path(meshes_dir)
    deformation_output = Path(deformation_dir)
    mesh_output.mkdir(parents=True, exist_ok=True)
    deformation_output.mkdir(parents=True, exist_ok=True)
    paths = {
        "tetra_reference": mesh_output / "tetra_reference.vtu",
        "tetra_deformed": mesh_output / "tetra_deformed.vtu",
        "deformation_data": deformation_output / "deformation_data.npz",
        "displacement_nodes": deformation_output / "displacement_nodes.npy",
    }
    shutil.copy2(result["tetra_reference_path"], paths["tetra_reference"])
    shutil.copy2(result["tetra_deformed_path"], paths["tetra_deformed"])
    shutil.copy2(result["deformation_data_path"], paths["deformation_data"])
    displacement = result["deformed_nodes"] - result["reference_nodes"]
    np.save(paths["displacement_nodes"], displacement.astype(np.float64))
    return paths
