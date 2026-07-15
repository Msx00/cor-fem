#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The command-line entry point used by run.sh. No local project imports are required.
"""
import os
import json
import time
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pyvista as pv
import tetgen
import torch
from scipy.spatial import cKDTree


# ---------------- Torch speed knobs ----------------
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass


# =============================================================================
# Utils
# =============================================================================

def log(msg: str, quiet: bool = False):
    if not quiet:
        print(msg, flush=True)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def axis_to_index(axis: str) -> int:
    axis = axis.lower()
    if axis == "x":
        return 0
    if axis == "y":
        return 1
    if axis == "z":
        return 2
    raise ValueError(f"Unsupported axis: {axis}")


def polydata_to_faces_tri(poly: pv.PolyData) -> np.ndarray:
    poly = poly.triangulate()
    faces = np.asarray(poly.faces, dtype=np.int64).reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        raise RuntimeError("Non-triangle faces exist. Use triangulate().")
    return faces[:, 1:4].astype(np.int64)


def make_tet_ugrid(points: np.ndarray, tets: np.ndarray) -> pv.UnstructuredGrid:
    points = np.asarray(points, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64)
    nT = tets.shape[0]
    cells = np.hstack([np.full((nT, 1), 4, dtype=np.int64), tets]).ravel()
    celltypes = np.full((nT,), pv.CellType.TETRA, dtype=np.uint8)
    return pv.UnstructuredGrid(cells, celltypes, points)



# =============================================================================
# Blender mesh repair (embedded)
# =============================================================================

BLENDER_REPAIR_PY = r"""
import bpy
import os
import sys
import argparse
import bmesh


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []

    p = argparse.ArgumentParser()
    p.add_argument("--inp", required=True)
    p.add_argument("--out", required=True)

    p.add_argument("--merge_dist", type=float, default=0.0)
    p.add_argument("--voxel_size", type=float, default=0.0)
    p.add_argument("--smooth_iters", type=int, default=0)
    p.add_argument("--smooth_factor", type=float, default=0.2)
    p.add_argument("--decimate_ratio", type=float, default=1.0)

    p.add_argument("--no_apply_transform", action="store_true", default=False)
    p.add_argument("--triangulate", action="store_true", default=False)
    return p.parse_args(argv)


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_mesh(path: str):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=path)
        else:
            bpy.ops.import_scene.obj(filepath=path)
    elif ext == ".ply":
        bpy.ops.import_mesh.ply(filepath=path)
    elif ext == ".stl":
        bpy.ops.import_mesh.stl(filepath=path)
    else:
        raise RuntimeError(f"Unsupported input extension: {ext}")

    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("No mesh object imported.")

    obj = meshes[0]
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj


def export_mesh(path: str, obj):
    ext = os.path.splitext(path)[1].lower()

    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    if ext == ".obj":
        if hasattr(bpy.ops.wm, "obj_export"):
            bpy.ops.wm.obj_export(
                filepath=path,
                export_selected_objects=True,
                export_triangulated_mesh=True,
            )
        else:
            bpy.ops.export_scene.obj(filepath=path, use_selection=True)
    elif ext == ".ply":
        bpy.ops.export_mesh.ply(filepath=path, use_selection=True)
    elif ext == ".stl":
        bpy.ops.export_mesh.stl(filepath=path, use_selection=True)
    else:
        raise RuntimeError(f"Unsupported output extension: {ext}")


def apply_transforms(obj):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)


def merge_by_distance_bmesh(obj, merge_dist: float):
    if merge_dist <= 0:
        return

    bpy.context.view_layer.objects.active = obj
    if obj.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()

    bmesh.ops.remove_doubles(
        bm,
        verts=bm.verts,
        dist=float(merge_dist),
    )

    bm.to_mesh(me)
    bm.free()
    me.update()


def recalc_normals(obj):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')


def remesh_voxel_apply(obj, voxel_size: float):
    if voxel_size <= 0:
        return

    mod = obj.modifiers.new(name="VoxelRemesh", type='REMESH')
    mod.mode = 'VOXEL'
    mod.voxel_size = float(voxel_size)
    if hasattr(mod, "adaptivity"):
        mod.adaptivity = 0.0

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=mod.name)


def smooth_apply(obj, factor: float, iterations: int):
    if iterations <= 0 or factor <= 0:
        return

    mod = obj.modifiers.new(name="SmoothFix", type='SMOOTH')
    mod.factor = float(factor)
    mod.iterations = int(iterations)

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=mod.name)


def decimate_apply(obj, ratio: float):
    if ratio >= 0.999999:
        return
    if ratio <= 0.0:
        raise ValueError("--decimate_ratio must be in (0,1].")

    mod = obj.modifiers.new(name="DecimateFix", type='DECIMATE')
    mod.decimate_type = 'COLLAPSE'
    mod.ratio = float(ratio)

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=mod.name)


def triangulate_apply(obj):
    mod = obj.modifiers.new(name="TriangulateFix", type='TRIANGULATE')
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=mod.name)


def main():
    args = parse_args()

    inp = os.path.abspath(args.inp)
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    clear_scene()
    obj = import_mesh(inp)

    if not args.no_apply_transform:
        apply_transforms(obj)

    merge_by_distance_bmesh(obj, float(args.merge_dist))
    remesh_voxel_apply(obj, float(args.voxel_size))
    smooth_apply(obj, float(args.smooth_factor), int(args.smooth_iters))
    decimate_apply(obj, float(args.decimate_ratio))

    if args.triangulate:
        triangulate_apply(obj)

    recalc_normals(obj)
    export_mesh(out, obj)

    print(f"[OK] repaired mesh exported: {out}")
    print(f"[STAT] verts={len(obj.data.vertices)}, faces={len(obj.data.polygons)}")


if __name__ == "__main__":
    main()
"""


def _which_blender(blender_bin: str) -> str:
    if os.path.isabs(blender_bin) and os.path.exists(blender_bin):
        return blender_bin
    found = shutil.which(blender_bin)
    if found is None:
        raise RuntimeError(f"[ERROR] Cannot find blender executable: {blender_bin}")
    return found


def run_blender_repair(
    inp_mesh: str,
    out_mesh: str,
    blender_bin: str = "blender",
    merge_dist: float = 0.0,
    voxel_size: float = 0.0,
    smooth_iters: int = 0,
    smooth_factor: float = 0.2,
    decimate_ratio: float = 1.0,
    no_apply_transform: bool = False,
    triangulate: bool = True,
    script_path: Optional[str] = None,
    use_cache: bool = True,
) -> str:
    inp_mesh = os.path.abspath(inp_mesh)
    out_mesh = os.path.abspath(out_mesh)
    os.makedirs(os.path.dirname(out_mesh), exist_ok=True)

    if use_cache and os.path.exists(out_mesh):
        try:
            if os.path.getmtime(out_mesh) >= os.path.getmtime(inp_mesh):
                print(f"[BLENDER] cache hit, skip repair: {out_mesh}")
                return out_mesh
        except Exception:
            pass

    blender_exe = _which_blender(blender_bin)

    if script_path is None:
        script_path = os.path.join(os.path.dirname(out_mesh), "_blender_repair_embedded.py")
    script_path = os.path.abspath(script_path)

    with open(script_path, "w", encoding="utf-8") as f:
        f.write(BLENDER_REPAIR_PY)

    cmd = [
        blender_exe,
        "-b",
        "-P", script_path,
        "--",
        "--inp", inp_mesh,
        "--out", out_mesh,
        "--merge_dist", str(float(merge_dist)),
        "--voxel_size", str(float(voxel_size)),
        "--smooth_iters", str(int(smooth_iters)),
        "--smooth_factor", str(float(smooth_factor)),
        "--decimate_ratio", str(float(decimate_ratio)),
    ]

    if no_apply_transform:
        cmd.append("--no_apply_transform")
    if triangulate:
        cmd.append("--triangulate")

    print("[BLENDER CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)

    if not os.path.exists(out_mesh):
        raise RuntimeError(f"[ERROR] Blender repaired output not found: {out_mesh}")

    return out_mesh


# =============================================================================
# IO
# =============================================================================

def _safe_compute_point_normals(mesh: pv.PolyData) -> np.ndarray:
    mesh = mesh.clean().triangulate()
    try:
        mesh_n = mesh.compute_normals(
            point_normals=True,
            cell_normals=False,
            consistent_normals=True,
            auto_orient_normals=True,
            split_vertices=False,
            inplace=False,
        )
        normals = np.asarray(mesh_n.point_data["Normals"], dtype=np.float64)
    except Exception:
        normals = np.zeros((mesh.n_points, 3), dtype=np.float64)

    if normals.shape[0] != mesh.n_points:
        normals = np.zeros((mesh.n_points, 3), dtype=np.float64)

    nrm = np.linalg.norm(normals, axis=1, keepdims=True)
    good = nrm[:, 0] > 1e-12
    normals[good] = normals[good] / nrm[good]
    return normals


def load_surface_mesh(path: str, quiet: bool = False) -> pv.PolyData:
    mesh = pv.read(path)
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    mesh = mesh.clean().triangulate()

    faces_tri = polydata_to_faces_tri(mesh)
    if faces_tri.shape[0] == 0:
        raise ValueError(f"Source must be a triangle surface mesh: {path}")

    log(f"[Source] V={mesh.n_points}, F={faces_tri.shape[0]}", quiet)
    return mesh


def load_target_points_with_normals(path: str, max_points: int, quiet: bool = False):
    geom = pv.read(path)
    if not isinstance(geom, pv.PolyData):
        geom = geom.extract_surface()
    geom = geom.clean().triangulate()

    pts = np.asarray(geom.points, dtype=np.float64)
    if pts.shape[0] == 0:
        raise ValueError(f"Target has no points: {path}")

    normals = _safe_compute_point_normals(geom)

    if pts.shape[0] > max_points > 0:
        idx = np.random.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]
        normals = normals[idx]

    log(f"[Target] points={pts.shape[0]}", quiet)
    return pts, normals


# =============================================================================
# TetGen
# =============================================================================

def tetrahedralize_with_tetgen(
    surf: pv.PolyData,
    mindihedral: float,
    minratio: float,
    maxvolume: float,
    quiet: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    if not surf.is_all_triangles:
        surf = surf.triangulate()

    tg = tetgen.TetGen(surf)
    try:
        kw = dict(mindihedral=float(mindihedral), minratio=float(minratio))
        if maxvolume > 0:
            kw["maxvolume"] = float(maxvolume)
        tg.tetrahedralize(verbose=0 if quiet else 1, **kw)
    except RuntimeError:
        log("[WARN] TetGen failed with quality constraints, fallback to quality=True", quiet)
        kw = dict(quality=True)
        if maxvolume > 0:
            kw["maxvolume"] = float(maxvolume)
        tg.tetrahedralize(verbose=0 if quiet else 1, **kw)

    grid = tg.grid
    points = np.asarray(grid.points, dtype=np.float64)
    cells = np.asarray(grid.cells, dtype=np.int64).reshape(-1, 5)
    tets = cells[:, 1:5].astype(np.int64)

    return points, tets


def match_surface_points_to_volume(
    surface_points: np.ndarray,
    volume_points: np.ndarray,
    tol: float = 1e-6
) -> np.ndarray:
    tree = cKDTree(volume_points)
    d, idx = tree.query(surface_points, k=1)
    bad = np.where(d > tol)[0]
    if bad.size > 0:
        raise RuntimeError(
            f"surface->volume mapping failed: {bad.size} points exceed tol={tol}, max={d.max():.6e}"
        )
    return idx.astype(np.int64)


# =============================================================================
# Boundary
# =============================================================================

def auto_fixed_vertex_ids(vertices: np.ndarray, mode: str, percentile: float, axis: str) -> np.ndarray:
    if mode == "none":
        return np.empty((0,), dtype=np.int64)

    ax = axis_to_index(axis)
    coord = vertices[:, ax]

    if mode in ("auto_posterior", "percentile_min"):
        thr = np.percentile(coord, percentile)
        ids = np.where(coord <= thr)[0]
        return ids.astype(np.int64)

    if mode == "percentile_max":
        thr = np.percentile(coord, 100.0 - percentile)
        ids = np.where(coord >= thr)[0]
        return ids.astype(np.int64)

    raise ValueError(f"Unsupported boundary mode: {mode}")


@torch.no_grad()
def build_surface_unique_edges(faces_tri: np.ndarray) -> np.ndarray:
    faces = np.asarray(faces_tri, dtype=np.int64)
    if faces.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    e01 = faces[:, [0, 1]]
    e12 = faces[:, [1, 2]]
    e20 = faces[:, [2, 0]]
    edges = np.vstack([e01, e12, e20])
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    return edges.astype(np.int64)


# =============================================================================

# =============================================================================
# Strict energy-consistent corotational FEM registration
# =============================================================================

def compute_lame(E: float, nu: float):
    lam = nu * E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return float(lam), float(mu)


@torch.no_grad()
def precompute_tet_ref(
    X: torch.Tensor,
    tets: torch.Tensor,
    dtype: torch.dtype,
    det_eps: float = 1e-12,
    chunk: int = 131072,
):
    device = X.device
    T = tets.shape[0]
    grads_list = []
    vol_list = []
    invDm_list = []
    I = torch.eye(3, device=device, dtype=dtype).view(1, 3, 3)

    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        tet = tets[s:e]
        x0 = X[tet[:, 0]]
        x1 = X[tet[:, 1]]
        x2 = X[tet[:, 2]]
        x3 = X[tet[:, 3]]
        Dm = torch.stack([x1 - x0, x2 - x0, x3 - x0], dim=2)
        detDm = torch.linalg.det(Dm)
        vol = detDm.abs() / 6.0

        valid = vol > det_eps
        if not valid.all():
            vol = torch.where(valid, vol, torch.full_like(vol, det_eps))
            Dm = torch.where(valid.view(-1, 1, 1), Dm, I.expand(e - s, 3, 3))

        invDm = torch.linalg.inv(Dm)
        invDmT = invDm.transpose(1, 2)
        g1 = invDmT[:, :, 0]
        g2 = invDmT[:, :, 1]
        g3 = invDmT[:, :, 2]
        g0 = -(g1 + g2 + g3)
        grads = torch.stack([g0, g1, g2, g3], dim=1)

        grads_list.append(grads)
        vol_list.append(vol)
        invDm_list.append(invDm)

    return torch.cat(grads_list), torch.cat(vol_list), torch.cat(invDm_list)


@torch.no_grad()
def knn1_gpu(P: torch.Tensor, Q: torch.Tensor, p_chunk: int = 4096, q_chunk: int = 32768):
    device = P.device
    dtype = P.dtype
    Np, Nq = P.shape[0], Q.shape[0]
    out_idx = torch.empty((Np,), device=device, dtype=torch.long)
    out_d2 = torch.empty((Np,), device=device, dtype=dtype)

    for ps in range(0, Np, p_chunk):
        pe = min(ps + p_chunk, Np)
        Pblk = P[ps:pe]
        P2 = (Pblk * Pblk).sum(dim=1, keepdim=True)
        best_d2 = torch.full((pe - ps,), float("inf"), device=device, dtype=dtype)
        best_j = torch.zeros((pe - ps,), device=device, dtype=torch.long)

        for qs in range(0, Nq, q_chunk):
            qe = min(qs + q_chunk, Nq)
            Qblk = Q[qs:qe]
            Q2 = (Qblk * Qblk).sum(dim=1).view(1, -1)
            d2 = (P2 + Q2 - 2.0 * (Pblk @ Qblk.transpose(0, 1))).clamp_min(0.0)
            d2_min, j = torch.min(d2, dim=1)
            better = d2_min < best_d2
            if better.any():
                best_d2[better] = d2_min[better]
                best_j[better] = j[better] + qs

        out_idx[ps:pe] = best_j
        out_d2[ps:pe] = best_d2

    return out_idx, out_d2


@torch.no_grad()
def polar_rotation_from_F(F: torch.Tensor) -> torch.Tensor:
    U, _, Vh = torch.linalg.svd(F, full_matrices=False)
    R = U @ Vh
    neg = torch.linalg.det(R) < 0
    if neg.any():
        U_fix = U.clone()
        U_fix[neg, :, 2] *= -1.0
        R = U_fix @ Vh
    return R


class StrictProstateCORFEMReg:
    """
    Energy-consistent nonlinear corotational FEM registration.

    The objective is explicit:
      E = E_cor + E_vol + E_surface + E_match.

    - E_cor uses the rotation-invariant corotational strain S-I, where F=RS.
    - E_vol contains a quadratic volume penalty and a correctly signed
      shifted logarithmic compression barrier.
    - E_surface is an exact edge-length energy for lagged coverage weights.
    - E_match is an exact Welsch robust energy for lagged correspondences
      and blended normals.

    Correspondences, trimming masks, blended normals, and coverage weights are
    updated in the outer ICP loop and held fixed during each nonlinear inner
    solve. The inner problem is minimized by limited-memory BFGS with Armijo
    backtracking. No approximate corotational tangent or PCG SPD assumption is
    used.
    """

    def __init__(
        self,
        vol_points: np.ndarray,
        vol_tets: np.ndarray,
        surf_map: np.ndarray,
        src_faces_tri: np.ndarray,
        target_points: np.ndarray,
        target_normals: np.ndarray,
        w_t2s: float,
        w_s2t: float,
        fixed_vertex_ids: np.ndarray,
        young: float,
        poisson: float,
        penalty_k: float,
        robust_sigma: float,
        tangent_weight: float,
        tangent_weight_start: float,
        tangent_anneal_iters: int,
        trim_quantile: float,
        trim_max_dist: float,
        trim_normal_cos: float,
        max_delta_u: float,
        vol_preserve_k: float,
        vol_barrier_k: float,
        vol_j_min: float,
        min_accepted_j: float,
        surface_smooth_k: float,
        cover_dist_start: float,
        cover_dist_end: float,
        inner_iters: int,
        lbfgs_history: int,
        grad_tol: float,
        armijo_c1: float,
        line_search_init: float,
        backtrack_factor: float,
        min_step: float,
        max_backtracks: int,
        max_reject_streak: int,
        outer_energy_rel_tol: float,
        outer_error_rel_tol: float,
        device: str,
        dtype: str,
        tet_chunk: int,
        knn_p_chunk: int,
        knn_q_chunk: int,
        quiet: bool = False,
    ):
        dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        tdtype = torch.float32 if dtype == "float32" else torch.float64
        self.device = dev
        self.dtype = tdtype
        self.quiet = quiet

        self.X0 = torch.tensor(np.asarray(vol_points, dtype=np.float64), device=dev, dtype=tdtype)
        self.tets = torch.tensor(np.asarray(vol_tets, dtype=np.int64), device=dev, dtype=torch.long)
        self.surf_map = torch.tensor(np.asarray(surf_map, dtype=np.int64), device=dev, dtype=torch.long)
        self.src_faces_tri = np.asarray(src_faces_tri, dtype=np.int64)
        self.surf_faces = torch.tensor(self.src_faces_tri, device=dev, dtype=torch.long)
        self.surface_edges_np = build_surface_unique_edges(self.src_faces_tri)
        self.surface_edges = torch.tensor(self.surface_edges_np, device=dev, dtype=torch.long)
        self.target = torch.tensor(np.asarray(target_points, dtype=np.float64), device=dev, dtype=tdtype)
        self.target_normals = torch.tensor(np.asarray(target_normals, dtype=np.float64), device=dev, dtype=tdtype)

        self.V = int(self.X0.shape[0])
        self.ndof = self.V * 3
        self.w_t2s = float(w_t2s)
        self.w_s2t = float(w_s2t)
        self.young = float(young)
        self.poisson = float(poisson)
        self.penalty_k = float(penalty_k)
        self.robust_sigma = float(robust_sigma)
        self.tangent_weight = float(tangent_weight)
        self.tangent_weight_start = float(tangent_weight_start)
        self.tangent_anneal_iters = int(tangent_anneal_iters)
        self.trim_quantile = float(trim_quantile)
        self.trim_max_dist = float(trim_max_dist)
        self.trim_normal_cos = float(trim_normal_cos)
        self.max_delta_u = float(max_delta_u)
        self.vol_preserve_k = float(vol_preserve_k)
        self.vol_barrier_k = float(vol_barrier_k)
        self.vol_j_min = float(vol_j_min)
        self.min_accepted_j = float(min_accepted_j)
        if not (0.0 < self.min_accepted_j < self.vol_j_min):
            raise ValueError(
                f"Require 0 < min_accepted_j < vol_j_min, got "
                f"{self.min_accepted_j} and {self.vol_j_min}."
            )
        self.surface_smooth_k = float(surface_smooth_k)
        self.cover_dist_start = float(cover_dist_start)
        self.cover_dist_end = float(cover_dist_end)
        self.inner_iters = max(1, int(inner_iters))
        self.lbfgs_history = max(1, int(lbfgs_history))
        self.grad_tol = float(grad_tol)
        self.armijo_c1 = float(armijo_c1)
        self.line_search_init = float(line_search_init)
        self.backtrack_factor = float(backtrack_factor)
        self.min_step = float(min_step)
        self.max_backtracks = int(max_backtracks)
        self.max_reject_streak = int(max_reject_streak)
        self.outer_energy_rel_tol = max(0.0, float(outer_energy_rel_tol))
        self.outer_error_rel_tol = max(0.0, float(outer_error_rel_tol))
        self.tet_chunk = int(tet_chunk)
        self.knn_p_chunk = int(knn_p_chunk)
        self.knn_q_chunk = int(knn_q_chunk)
        self.iter_id = 0

        self.lam, self.mu = compute_lame(self.young, self.poisson)
        self.grads, self.vol, self.invDm = precompute_tet_ref(
            self.X0, self.tets, dtype=tdtype, chunk=self.tet_chunk
        )
        # Normalize volumetric energies by the reference organ volume.  Without
        # this normalization, the matching term scales with the number of
        # sampled surface points while the FEM term scales with mesh volume,
        # making the relative weights change whenever either mesh is resampled.
        self.total_ref_volume = self.vol.sum().clamp_min(1e-12)
        self.I3 = torch.eye(3, device=dev, dtype=tdtype).view(1, 3, 3)
        self.Xsurf0 = self.X0[self.surf_map]

        fixed_vertex_ids = np.asarray(fixed_vertex_ids, dtype=np.int64)
        fixed_dofs = []
        for vid in fixed_vertex_ids:
            fixed_dofs.extend([3 * int(vid), 3 * int(vid) + 1, 3 * int(vid) + 2])
        self.fixed_dofs = torch.tensor(fixed_dofs, device=dev, dtype=torch.long)

        self.u = torch.zeros((self.ndof,), device=dev, dtype=tdtype)
        log(
            f"[INFO] strict prostate COR-FEM: device={dev}, dtype={tdtype}, "
            f"V={self.V}, T={self.tets.shape[0]}, bidirectional=True, "
            f"solver=nonlinear-LBFGS+Armijo",
            quiet,
        )

    @torch.no_grad()
    def current_tangent_weight(self) -> float:
        if self.tangent_anneal_iters <= 0:
            return float(self.tangent_weight)
        a = min(max(float(self.iter_id) / float(self.tangent_anneal_iters), 0.0), 1.0)
        return float((1.0 - a) * self.tangent_weight_start + a * self.tangent_weight)

    @torch.no_grad()
    def volume_vertices_from_u(self, u_vec: torch.Tensor) -> torch.Tensor:
        return self.X0 + u_vec.view(self.V, 3)

    @torch.no_grad()
    def surface_vertices_from_u(self, u_vec: torch.Tensor) -> torch.Tensor:
        return self.volume_vertices_from_u(u_vec)[self.surf_map]

    @torch.no_grad()
    def _surface_normals_from_u(self, u_vec: torch.Tensor) -> torch.Tensor:
        verts = self.surface_vertices_from_u(u_vec)
        faces = self.surf_faces
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        fn = torch.cross(v1 - v0, v2 - v0, dim=1)
        vn = torch.zeros_like(verts)
        vn.index_add_(0, faces[:, 0], fn)
        vn.index_add_(0, faces[:, 1], fn)
        vn.index_add_(0, faces[:, 2], fn)
        return self._normalize_rows(vn)

    @torch.no_grad()
    def _normalize_rows(self, x: torch.Tensor) -> torch.Tensor:
        return x / torch.norm(x, dim=1, keepdim=True).clamp_min(1e-12)

    @torch.no_grad()
    def _blend_normals(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = self._normalize_rows(a)
        b = self._normalize_rows(b)
        dot = (a * b).sum(dim=1, keepdim=True)
        sign = torch.where(dot < 0.0, -torch.ones_like(dot), torch.ones_like(dot))
        n = a + sign * b
        n_norm = torch.norm(n, dim=1, keepdim=True)
        fallback = torch.where(
            (torch.norm(a, dim=1, keepdim=True) > 1e-8), a, b
        )
        return torch.where(n_norm > 1e-8, n / n_norm.clamp_min(1e-12), fallback)

    @torch.no_grad()
    def _trim_mask(self, dist, src_normals, tgt_normals):
        d = dist.reshape(-1)
        mask = torch.ones_like(d, dtype=torch.bool)
        if 0.0 < self.trim_quantile < 1.0 and d.numel() > 8:
            mask &= d <= torch.quantile(d, self.trim_quantile)
        if self.trim_max_dist > 0.0:
            mask &= d <= self.trim_max_dist
        if self.trim_normal_cos > 0.0:
            sn = self._normalize_rows(src_normals)
            tn = self._normalize_rows(tgt_normals)
            mask &= torch.abs((sn * tn).sum(dim=1)) >= self.trim_normal_cos
        if not mask.any():
            mask = torch.ones_like(mask)
        return mask

    @torch.no_grad()
    def compute_current_J(self, u_vec: Optional[torch.Tensor] = None) -> torch.Tensor:
        if u_vec is None:
            u_vec = self.u
        x = self.volume_vertices_from_u(u_vec)
        xe = x[self.tets]
        Ds = torch.stack([xe[:, 1] - xe[:, 0], xe[:, 2] - xe[:, 0], xe[:, 3] - xe[:, 0]], dim=2)
        return torch.linalg.det(Ds @ self.invDm)

    @torch.no_grad()
    def _make_match_block(self, ids, targets, src_normals, tgt_normals, dist, direction_weight):
        mask = self._trim_mask(dist, src_normals, tgt_normals)
        return {
            "ids": ids[mask],
            "targets": targets[mask],
            "normals": self._blend_normals(src_normals[mask], tgt_normals[mask]),
            "direction_weight": float(direction_weight),
            "active_ratio": float(mask.float().mean().item()),
        }

    @torch.no_grad()
    def build_outer_state(self, u_vec: torch.Tensor):
        surf = self.surface_vertices_from_u(u_vec)
        sn = self._surface_normals_from_u(u_vec)

        # target -> source
        idx_t2s, d2_t2s = knn1_gpu(self.target, surf, self.knn_p_chunk, self.knn_q_chunk)
        t2s = self._make_match_block(
            ids=self.surf_map[idx_t2s],
            targets=self.target,
            src_normals=sn[idx_t2s],
            tgt_normals=self.target_normals,
            dist=torch.sqrt(d2_t2s.clamp_min(0.0)),
            direction_weight=self.w_t2s,
        )

        # source -> target
        idx_s2t, d2_s2t = knn1_gpu(surf, self.target, self.knn_p_chunk, self.knn_q_chunk)
        s2t = self._make_match_block(
            ids=self.surf_map,
            targets=self.target[idx_s2t],
            src_normals=sn,
            tgt_normals=self.target_normals[idx_s2t],
            dist=torch.sqrt(d2_s2t.clamp_min(0.0)),
            direction_weight=self.w_s2t,
        )

        # Smooth lagged coverage weights: hit vertices are observed; otherwise
        # the regularization weight ramps continuously with source-target distance.
        hit = torch.zeros((surf.shape[0],), device=self.device, dtype=self.dtype)
        hit.index_add_(0, idx_t2s, torch.ones_like(idx_t2s, dtype=self.dtype))
        hit = hit > 0
        dist_s2t = torch.sqrt(d2_s2t.clamp_min(0.0))
        if self.cover_dist_end > self.cover_dist_start:
            cover_w = torch.clamp(
                (dist_s2t - self.cover_dist_start)
                / max(self.cover_dist_end - self.cover_dist_start, 1e-12),
                0.0,
                1.0,
            )
        else:
            cover_w = (dist_s2t >= self.cover_dist_start).to(self.dtype)
        cover_w = torch.where(hit, torch.zeros_like(cover_w), cover_w)

        if self.surface_edges.numel() > 0:
            e = self.surface_edges
            edge_w = 0.5 * (cover_w[e[:, 0]] + cover_w[e[:, 1]])
        else:
            edge_w = torch.empty((0,), device=self.device, dtype=self.dtype)

        return {
            "matches": [t2s, s2t],
            "edge_w": edge_w,
            "beta": float(self.current_tangent_weight()),
            "t2s_active_ratio": t2s["active_ratio"],
            "s2t_active_ratio": s2t["active_ratio"],
        }

    @torch.no_grad()
    def _corotational_volume_energy_gradient(self, u_vec: torch.Tensor):
        x_nodes = self.volume_vertices_from_u(u_vec)
        grad_nodes = torch.zeros_like(x_nodes)
        E_cor = torch.zeros((), device=self.device, dtype=self.dtype)
        E_vol = torch.zeros((), device=self.device, dtype=self.dtype)
        min_J = float("inf")
        T = int(self.tets.shape[0])

        for s in range(0, T, self.tet_chunk):
            e = min(s + self.tet_chunk, T)
            tet = self.tets[s:e]
            xe = x_nodes[tet]
            Ds = torch.stack([xe[:, 1] - xe[:, 0], xe[:, 2] - xe[:, 0], xe[:, 3] - xe[:, 0]], dim=2)
            F = Ds @ self.invDm[s:e]
            J = torch.linalg.det(F)
            min_J = min(min_J, float(J.min().item()))

            R = polar_rotation_from_F(F)
            S = R.transpose(1, 2) @ F
            S = 0.5 * (S + S.transpose(1, 2))
            A = S - self.I3
            trA = A.diagonal(dim1=1, dim2=2).sum(dim=1)
            vv = self.vol[s:e]
            wv = vv / self.total_ref_volume

            density_cor = self.mu * (A * A).sum(dim=(1, 2)) + 0.5 * self.lam * trA * trA
            E_cor = E_cor + (wv * density_cor).sum()
            stress_rot = 2.0 * self.mu * A + self.lam * trA.view(-1, 1, 1) * self.I3
            P_cor = R @ stress_rot

            # True shifted logarithmic compression barrier. It is zero with
            # zero first derivative at J_min and diverges as J approaches the
            # strictly positive floor J_floor=min_accepted_j:
            #   phi(z) = -log(z) + z - 1,
            #   z = (J-J_floor)/(J_min-J_floor),  J < J_min.
            # Hence d phi / dJ = (1 - 1/z)/(J_min-J_floor) < 0.
            j_range = self.vol_j_min - self.min_accepted_j
            active = J < self.vol_j_min
            z = ((J - self.min_accepted_j) / j_range).clamp_min(1e-12)
            barrier_density = torch.where(
                active, -torch.log(z) + z - 1.0, torch.zeros_like(J)
            )
            density_vol = (
                0.5 * self.vol_preserve_k * (J - 1.0) ** 2
                + self.vol_barrier_k * barrier_density
            )
            E_vol = E_vol + (wv * density_vol).sum()

            dbarrier_dJ = torch.where(
                active,
                self.vol_barrier_k * (1.0 - 1.0 / z) / j_range,
                torch.zeros_like(J),
            )
            coeff = self.vol_preserve_k * (J - 1.0) + dbarrier_dJ
            FinvT = torch.linalg.inv(F).transpose(1, 2)
            P_vol = (coeff * J).view(-1, 1, 1) * FinvT
            P = P_cor + P_vol

            g = self.grads[s:e]
            for a in range(4):
                fa = torch.bmm(P, g[:, a, :].unsqueeze(2)).squeeze(2) * wv.view(-1, 1)
                grad_nodes.index_add_(0, tet[:, a], fa)

        return E_cor, E_vol, grad_nodes, min_J

    @torch.no_grad()
    def energy_gradient(self, u_vec: torch.Tensor, state):
        E_cor, E_vol, grad_nodes, min_J = self._corotational_volume_energy_gradient(u_vec)
        x_nodes = self.volume_vertices_from_u(u_vec)

        # Exact gradient of edge-length energy with lagged edge weights.
        E_surface = torch.zeros((), device=self.device, dtype=self.dtype)
        if self.surface_smooth_k > 0.0 and self.surface_edges.numel() > 0:
            xsurf = x_nodes[self.surf_map]
            e = self.surface_edges
            d = xsurf[e[:, 0]] - xsurf[e[:, 1]]
            d0 = self.Xsurf0[e[:, 0]] - self.Xsurf0[e[:, 1]]
            l = torch.norm(d, dim=1).clamp_min(1e-12)
            l0 = torch.norm(d0, dim=1)
            diff = l - l0
            ew = state["edge_w"]
            # A weighted mean makes the regularizer independent of the number
            # of remeshed surface edges.
            surface_norm = ew.sum().clamp_min(1.0)
            E_surface = 0.5 * self.surface_smooth_k * (ew * diff * diff).sum() / surface_norm
            ge = (
                self.surface_smooth_k
                * ew.view(-1, 1)
                * (diff / l).view(-1, 1)
                * d
                / surface_norm
            )
            gsurf = torch.zeros_like(xsurf)
            gsurf.index_add_(0, e[:, 0], ge)
            gsurf.index_add_(0, e[:, 1], -ge)
            grad_nodes.index_add_(0, self.surf_map, gsurf)

        # Exact Welsch robust matching energy for fixed correspondences/normals.
        E_match = torch.zeros((), device=self.device, dtype=self.dtype)
        beta = float(state["beta"])
        for block in state["matches"]:
            if block["ids"].numel() == 0 or block["direction_weight"] <= 0.0:
                continue
            ids = block["ids"]
            delta = block["targets"] - x_nodes[ids]
            n = block["normals"]
            dn_scalar = (delta * n).sum(dim=1, keepdim=True)
            dn = dn_scalar * n
            dt = delta - dn
            q = (dn * dn).sum(dim=1) + beta * (dt * dt).sum(dim=1)
            # Normalize t2s and s2t independently.  This prevents the target
            # sampling count (e.g. 12,000 points) from overwhelming the FEM
            # energy or changing the result when the same surface is resampled.
            match_norm = float(max(int(ids.numel()), 1))
            scale = self.penalty_k * block["direction_weight"] / match_norm

            if self.robust_sigma > 0.0:
                sigma2 = self.robust_sigma * self.robust_sigma
                rw = torch.exp(-q / (2.0 * sigma2))
                E_match = E_match + scale * sigma2 * (1.0 - rw).sum()
            else:
                rw = torch.ones_like(q)
                E_match = E_match + 0.5 * scale * q.sum()

            # grad(E_match) = - matching force
            f = scale * rw.view(-1, 1) * (dn + beta * dt)
            grad_nodes.index_add_(0, ids, -f)

        grad = grad_nodes.reshape(-1)
        if self.fixed_dofs.numel() > 0:
            grad = grad.clone()
            grad[self.fixed_dofs] = 0.0

        total = E_cor + E_vol + E_surface + E_match
        terms = {
            "total": float(total.item()),
            "cor": float(E_cor.item()),
            "vol": float(E_vol.item()),
            "surface": float(E_surface.item()),
            "match": float(E_match.item()),
            "min_J": float(min_J),
        }
        return total, grad, terms

    @torch.no_grad()
    def _lbfgs_direction(self, grad, s_hist, y_hist, rho_hist):
        q = grad.clone()
        alphas = []
        for s, y, rho in zip(reversed(s_hist), reversed(y_hist), reversed(rho_hist)):
            a = rho * torch.dot(s, q)
            alphas.append(a)
            q = q - a * y

        if s_hist:
            sy = torch.dot(s_hist[-1], y_hist[-1])
            yy = torch.dot(y_hist[-1], y_hist[-1]).clamp_min(1e-20)
            gamma = sy / yy
        else:
            gamma = torch.tensor(1.0, device=self.device, dtype=self.dtype)
        r = gamma * q

        for s, y, rho, a in zip(s_hist, y_hist, rho_hist, reversed(alphas)):
            b = rho * torch.dot(y, r)
            r = r + s * (a - b)
        return -r

    @torch.no_grad()
    def _apply_constraints(self, u_vec):
        if self.fixed_dofs.numel() > 0:
            u_vec = u_vec.clone()
            u_vec[self.fixed_dofs] = 0.0
        return u_vec

    @torch.no_grad()
    def optimize_fixed_state(self, u_start: torch.Tensor, state):
        u = self._apply_constraints(u_start.clone())
        E, g, terms = self.energy_gradient(u, state)
        s_hist, y_hist, rho_hist = [], [], []
        accepted_steps = 0
        last_step = 0.0

        for _ in range(self.inner_iters):
            if float(torch.norm(g).item()) <= self.grad_tol:
                break

            p = self._lbfgs_direction(g, s_hist, y_hist, rho_hist)
            p = p.clone()
            if self.fixed_dofs.numel() > 0:
                p[self.fixed_dofs] = 0.0
            gtp = torch.dot(g, p)
            if (not torch.isfinite(gtp)) or gtp >= -1e-12:
                s_hist, y_hist, rho_hist = [], [], []
                p = -g
                if self.fixed_dofs.numel() > 0:
                    p[self.fixed_dofs] = 0.0
                gtp = torch.dot(g, p)

            if self.max_delta_u > 0.0:
                max_norm = torch.norm(p.view(self.V, 3), dim=1).max().clamp_min(1e-12)
                scale = min(1.0, self.max_delta_u / float(max_norm.item()))
                p = p * scale
                gtp = torch.dot(g, p)

            alpha = self.line_search_init
            accepted = False
            E_new = None
            g_new = None
            terms_new = None
            u_new = None

            for _bt in range(self.max_backtracks + 1):
                candidate = self._apply_constraints(u + alpha * p)
                min_j = float(self.compute_current_J(candidate).min().item())
                if min_j <= self.min_accepted_j:
                    alpha *= self.backtrack_factor
                    if alpha < self.min_step:
                        break
                    continue

                E_cand, g_cand, terms_cand = self.energy_gradient(candidate, state)
                if torch.isfinite(E_cand) and E_cand <= E + self.armijo_c1 * alpha * gtp:
                    accepted = True
                    u_new, E_new, g_new, terms_new = candidate, E_cand, g_cand, terms_cand
                    break

                alpha *= self.backtrack_factor
                if alpha < self.min_step:
                    break

            if not accepted:
                break

            s_vec = u_new - u
            y_vec = g_new - g
            sy = torch.dot(s_vec, y_vec)
            if torch.isfinite(sy) and sy > 1e-10 * torch.norm(s_vec) * torch.norm(y_vec):
                if len(s_hist) >= self.lbfgs_history:
                    s_hist.pop(0); y_hist.pop(0); rho_hist.pop(0)
                s_hist.append(s_vec.clone())
                y_hist.append(y_vec.clone())
                rho_hist.append(1.0 / sy)
            else:
                s_hist, y_hist, rho_hist = [], [], []

            u, E, g, terms = u_new, E_new, g_new, terms_new
            accepted_steps += 1
            last_step = float(alpha)

        return u, terms, accepted_steps, last_step

    @torch.no_grad()
    def compute_error_on_u(self, u_vec: torch.Tensor):
        surf = self.surface_vertices_from_u(u_vec)
        idx_t2s, d2_t2s = knn1_gpu(self.target, surf, self.knn_p_chunk, self.knn_q_chunk)
        idx_s2t, d2_s2t = knn1_gpu(surf, self.target, self.knn_p_chunk, self.knn_q_chunk)
        dt = torch.sqrt(d2_t2s.clamp_min(0.0))
        ds = torch.sqrt(d2_s2t.clamp_min(0.0))

        if self.robust_sigma > 0.0:
            sigma2 = self.robust_sigma ** 2
            wt = torch.exp(-d2_t2s / (2.0 * sigma2))
            ws = torch.exp(-d2_s2t / (2.0 * sigma2))
        else:
            wt = torch.ones_like(dt)
            ws = torch.ones_like(ds)

        t_rob = float((wt * dt).sum().item() / (wt.sum().item() + 1e-12))
        s_rob = float((ws * ds).sum().item() / (ws.sum().item() + 1e-12))
        wsum = max(self.w_t2s + self.w_s2t, 1e-12)
        return {
            "t2s_mean": float(dt.mean().item()),
            "t2s_max": float(dt.max().item()),
            "s2t_mean": float(ds.mean().item()),
            "s2t_max": float(ds.max().item()),
            "t2s_mean_robust": t_rob,
            "s2t_mean_robust": s_rob,
            "mean_sym": float((self.w_t2s * dt.mean().item() + self.w_s2t * ds.mean().item()) / wsum),
            "max_sym": float(max(dt.max().item(), ds.max().item())),
            "mean_sym_robust": float((self.w_t2s * t_rob + self.w_s2t * s_rob) / wsum),
        }

    @torch.no_grad()
    def displacement_stats(self, u_vec=None):
        if u_vec is None:
            u_vec = self.u
        d = torch.norm(u_vec.view(self.V, 3), dim=1)
        return float(d.mean().item()), float(d.max().item())

    @torch.no_grad()
    def fit(self, max_iters: int, tol: float, min_iters: int = 20, stop_mean_err: float = 3.0):
        hist = []
        accepted_hist = []
        stop_reason = "max_iters"
        reject_streak = 0
        t0 = time.time()

        for it in range(max_iters):
            self.iter_id = it
            err_before = self.compute_error_on_u(self.u)
            if err_before["mean_sym"] < stop_mean_err:
                stop_reason = f"mean_err<{stop_mean_err}"
                hist.append(err_before["mean_sym_robust"])
                break

            state = self.build_outer_state(self.u)
            u_old = self.u.clone()
            E_before, _, terms_before = self.energy_gradient(u_old, state)
            E_before_value = float(E_before.item())

            candidate, terms, inner_accepts, inner_step = self.optimize_fixed_state(u_old, state)

            # Evaluate candidate states with correspondences, normals, trimming,
            # and coverage weights rebuilt at that candidate.  The old version
            # optimized one frozen objective but rejected using a different,
            # untrimmed Euclidean metric, causing deterministic reject loops.
            accepted = False
            best_u = u_old
            best_err = err_before
            best_terms = terms_before
            best_dynamic_E = E_before_value
            eta = 1.0

            for _bt in range(self.max_backtracks + 1):
                cand = self._apply_constraints((1.0 - eta) * u_old + eta * candidate)
                min_j = float(self.compute_current_J(cand).min().item())
                if min_j > self.min_accepted_j:
                    cand_state = self.build_outer_state(cand)
                    E_cand, _, terms_cand = self.energy_gradient(cand, cand_state)
                    E_cand_value = float(E_cand.item())
                    e_cand = self.compute_error_on_u(cand)

                    energy_limit = E_before_value * (1.0 + self.outer_energy_rel_tol)
                    error_limit = (
                        err_before["mean_sym_robust"]
                        * (1.0 + self.outer_error_rel_tol)
                    )

                    # Accept a physically valid step when the rebuilt total
                    # energy is not worse and registration error remains stable,
                    # or when the actual robust registration error improves.
                    energy_ok = E_cand_value <= energy_limit
                    error_ok = e_cand["mean_sym_robust"] <= error_limit
                    registration_improved = (
                        e_cand["mean_sym_robust"]
                        < err_before["mean_sym_robust"] - 1e-10
                    )

                    if (energy_ok and error_ok) or registration_improved:
                        accepted = True
                        best_u = cand
                        best_err = e_cand
                        best_terms = terms_cand
                        best_dynamic_E = E_cand_value
                        break

                eta *= self.backtrack_factor
                if eta < self.min_step:
                    break

            if accepted and inner_accepts > 0:
                self.u = best_u
                accepted_terms = best_terms
                reject_streak = 0
                hist.append(best_err["mean_sym_robust"])
                accepted_hist.append(best_err["mean_sym_robust"])
                um, ux = self.displacement_stats()
                log(
                    f"[Iter {it:03d}] accept | sym={err_before['mean_sym']:.6f}->{best_err['mean_sym']:.6f}, "
                    f"robust={err_before['mean_sym_robust']:.6f}->{best_err['mean_sym_robust']:.6f}, "
                    f"E={E_before_value:.6e}->{best_dynamic_E:.6e} "
                    f"(cor={accepted_terms['cor']:.3e}, vol={accepted_terms['vol']:.3e}, "
                    f"surf={accepted_terms['surface']:.3e}, match={accepted_terms['match']:.3e}), "
                    f"minJ={accepted_terms['min_J']:.6f}, "
                    f"inner={inner_accepts}, step={inner_step:.3e}, eta={eta:.3e}, "
                    f"u_mean={um:.4f}, u_max={ux:.4f}",
                    self.quiet,
                )
            else:
                self.u = u_old
                reject_streak += 1
                hist.append(err_before["mean_sym_robust"])
                log(
                    f"[Iter {it:03d}] reject | robust={err_before['mean_sym_robust']:.6f}, "
                    f"inner_accepts={inner_accepts}, reject_streak={reject_streak}",
                    self.quiet,
                )
                if reject_streak >= self.max_reject_streak:
                    stop_reason = f"plateau_reject_streak>={self.max_reject_streak}"
                    break

            if it >= min_iters and len(accepted_hist) >= 2:
                if abs(accepted_hist[-1] - accepted_hist[-2]) < tol:
                    stop_reason = f"delta_accepted_robust_mean<{tol}"
                    break
            if accepted and best_err["mean_sym"] < stop_mean_err:
                stop_reason = f"mean_err<{stop_mean_err}"
                break

        Vdef = self.volume_vertices_from_u(self.u).cpu().numpy()
        Sdef = Vdef[self.surf_map.cpu().numpy()]
        return {
            "history": np.asarray(hist, dtype=np.float64),
            "elapsed_sec": float(time.time() - t0),
            "stop_reason": stop_reason,
            "volume_vertices_final": Vdef,
            "surface_vertices_final": Sdef,
            "final_error": self.compute_error_on_u(self.u),
        }

# Export
# =============================================================================
def export_mesh(path: str, vertices: np.ndarray, faces_tri: np.ndarray):
    faces_pv = np.hstack([np.full((faces_tri.shape[0], 1), 3, dtype=np.int64), faces_tri]).ravel()
    mesh = pv.PolyData(np.asarray(vertices, dtype=np.float64), faces_pv)
    mesh.save(path)


@torch.no_grad()
def warp_points_by_tet_mesh(
    points_src: np.ndarray,
    grid_src: pv.UnstructuredGrid,
    grid_def: pv.UnstructuredGrid,
    clamp_outside: bool = True,
    device: str = "cuda",
    dtype: str = "float32",
    point_chunk: int = 200000,
    deg_eps: float = 1e-12,
) -> np.ndarray:
    points_src = np.asarray(points_src, dtype=np.float64)
    N = points_src.shape[0]
    if N == 0:
        return points_src.copy()

    cell_ids = grid_src.find_containing_cell(points_src)
    if clamp_outside and np.any(cell_ids < 0):
        bad = np.where(cell_ids < 0)[0]
        for i in bad:
            cell_ids[i] = grid_src.find_closest_cell(points_src[i])

    if np.any(cell_ids < 0):
        bad = np.where(cell_ids < 0)[0]
        raise RuntimeError(f"{bad.size} points not in any tet (even after clamp).")

    cell_ids = cell_ids.astype(np.int64)

    cells = np.asarray(grid_src.cells, dtype=np.int64).reshape(-1, 5)
    if not np.all(cells[:, 0] == 4):
        raise RuntimeError("grid_src is not pure tetra mesh (cells[:,0] != 4).")
    tets = cells[:, 1:5]
    tet_ids_np = tets[cell_ids]

    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    tdtype = torch.float32 if dtype == "float32" else torch.float64

    X = torch.as_tensor(np.asarray(grid_src.points, dtype=np.float64), device=dev, dtype=tdtype)
    Xd = torch.as_tensor(np.asarray(grid_def.points, dtype=np.float64), device=dev, dtype=tdtype)
    P = torch.as_tensor(points_src, device=dev, dtype=tdtype)
    tet_ids = torch.as_tensor(tet_ids_np, device=dev, dtype=torch.long)

    out = torch.empty((N, 3), device=dev, dtype=tdtype)

    for s in range(0, N, point_chunk):
        e = min(s + point_chunk, N)
        Pblk = P[s:e]
        ids = tet_ids[s:e]

        V0 = X[ids[:, 0]]
        V1 = X[ids[:, 1]]
        V2 = X[ids[:, 2]]
        V3 = X[ids[:, 3]]

        A = torch.stack((V1 - V0, V2 - V0, V3 - V0), dim=2)
        b = (Pblk - V0)

        detA = torch.linalg.det(A)
        valid = detA.abs() > deg_eps

        w = torch.zeros((e - s, 4), device=dev, dtype=tdtype)
        w[:, 0] = 1.0

        if valid.any():
            sol = torch.linalg.solve(A[valid], b[valid])
            w1 = sol[:, 0]
            w2 = sol[:, 1]
            w3 = sol[:, 2]
            w0 = 1.0 - w1 - w2 - w3
            w[valid, 0] = w0
            w[valid, 1] = w1
            w[valid, 2] = w2
            w[valid, 3] = w3

        VD0 = Xd[ids[:, 0]]
        VD1 = Xd[ids[:, 1]]
        VD2 = Xd[ids[:, 2]]
        VD3 = Xd[ids[:, 3]]

        out[s:e] = (
            w[:, 0:1] * VD0 +
            w[:, 1:2] * VD1 +
            w[:, 2:3] * VD2 +
            w[:, 3:4] * VD3
        )

    return out.detach().cpu().numpy().astype(np.float64)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 3)
    if a.shape != b.shape:
        raise ValueError(f"RMSE inputs shape mismatch: {a.shape} vs {b.shape}")
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def tre_stats(a: np.ndarray, b: np.ndarray):
    a = np.asarray(a, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 3)
    if a.shape != b.shape:
        raise ValueError(f"TRE inputs shape mismatch: {a.shape} vs {b.shape}")
    d = np.linalg.norm(a - b, axis=1)
    return {
        "tre_mean": float(d.mean()),
        "tre_std": float(d.std()),
        "tre_median": float(np.median(d)),
        "tre_max": float(d.max()),
        "rmse": float(np.sqrt(np.mean(d * d))),
        "n_points": int(d.shape[0]),
    }


def _surface_metrics_from_directed_distances(
    prediction_to_target: np.ndarray,
    target_to_prediction: np.ndarray,
):
    """Build resolution-normalized symmetric point-cloud surface metrics."""
    p2t = np.asarray(prediction_to_target, dtype=np.float64).reshape(-1)
    t2p = np.asarray(target_to_prediction, dtype=np.float64).reshape(-1)
    if p2t.size == 0 or t2p.size == 0:
        raise ValueError("Directed surface-distance arrays must be non-empty")
    if not np.all(np.isfinite(p2t)) or not np.all(np.isfinite(t2p)):
        raise ValueError("Directed surface distances contain NaN or Inf")

    p2t_mean = float(p2t.mean())
    t2p_mean = float(t2p.mean())
    p2t_hd95 = float(np.percentile(p2t, 95))
    t2p_hd95 = float(np.percentile(t2p, 95))
    return {
        # The supplied reference uses unsquared Euclidean distances. Normalize
        # each direction by its own point count so the value is not tied to
        # mesh resolution, then sum the two directional means.
        "chamfer_distance": float(p2t_mean + t2p_mean),
        "chamfer_symmetric_mean": float(0.5 * (p2t_mean + t2p_mean)),
        "chamfer_raw_sum": float(p2t.sum() + t2p.sum()),
        "hd95": float(max(p2t_hd95, t2p_hd95)),
        "hd95_concat": float(np.percentile(np.concatenate((p2t, t2p)), 95)),
        "prediction_to_target_mean": p2t_mean,
        "target_to_prediction_mean": t2p_mean,
        "prediction_to_target_hd95": p2t_hd95,
        "target_to_prediction_hd95": t2p_hd95,
        "prediction_to_target_max": float(p2t.max()),
        "target_to_prediction_max": float(t2p.max()),
    }


def point_cloud_surface_metrics(prediction: np.ndarray, target: np.ndarray):
    """Compute bidirectional unsquared CD and HD95 using scalable KD-trees.

    CD is mean(prediction -> target) + mean(target -> prediction). HD95 is
    max(P95(prediction -> target), P95(target -> prediction)). Coordinates are
    assumed to be millimetres, so all distance-valued outputs are in mm.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.ndim != 2 or prediction.shape[1] != 3:
        raise ValueError(f"prediction must have shape (N, 3), got {prediction.shape}")
    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError(f"target must have shape (M, 3), got {target.shape}")
    if prediction.shape[0] == 0 or target.shape[0] == 0:
        raise ValueError("prediction and target must be non-empty")
    if not np.all(np.isfinite(prediction)):
        raise ValueError("prediction contains NaN or Inf")
    if not np.all(np.isfinite(target)):
        raise ValueError("target contains NaN or Inf")

    prediction_to_target = cKDTree(target).query(prediction, k=1)[0]
    target_to_prediction = cKDTree(prediction).query(target, k=1)[0]
    metrics = _surface_metrics_from_directed_distances(
        prediction_to_target, target_to_prediction
    )
    metrics.update(
        {
            "prediction_points": int(prediction.shape[0]),
            "target_points": int(target.shape[0]),
            "distance_unit": "mm",
            "distance_type": "unsquared_euclidean",
            "chamfer_definition": "mean(prediction_to_target)+mean(target_to_prediction)",
            "hd95_definition": "max(P95(prediction_to_target),P95(target_to_prediction))",
        }
    )
    return metrics

def surface_negative_jacobian_stats(
    reference_volume_vertices: np.ndarray,
    deformed_volume_vertices: np.ndarray,
    tets: np.ndarray,
    surface_vertices: np.ndarray,
    surface_faces: np.ndarray,
    surf_map: np.ndarray,
    low_j_threshold: float = 0.10,
    det_eps: float = 1e-12,
):
    """
    使用四面体 FEM 网格计算 warped source surface 对应的负雅可比占比。

    注意：
    surface 三角形本身没有 3D 体积 Jacobian。因此，这里先计算每个
    四面体的真实形变梯度：

        F = Ds @ inverse(Dm)
        J = det(F)

    然后把每个 source surface 三角形与其相邻的边界四面体关联，
    使用该四面体的 J 作为该表面三角形的 Jacobian。

    Parameters
    ----------
    reference_volume_vertices:
        配准前四面体网格顶点，形状 (V, 3)。

    deformed_volume_vertices:
        配准后四面体网格顶点，形状 (V, 3)。

    tets:
        四面体顶点索引，形状 (T, 4)。

    surface_vertices:
        配准前 source surface 顶点，形状 (Ns, 3)。

    surface_faces:
        source surface 三角面，使用 surface 局部索引，形状 (F, 3)。

    surf_map:
        surface 顶点到 volume 顶点的映射，形状 (Ns,)。

    low_j_threshold:
        低 Jacobian 阈值，例如 0.10。

    Returns
    -------
    dict
        包括全体四面体和 surface 相邻四面体的 Jacobian 统计。
    """
    reference_volume_vertices = np.asarray(
        reference_volume_vertices,
        dtype=np.float64,
    )
    deformed_volume_vertices = np.asarray(
        deformed_volume_vertices,
        dtype=np.float64,
    )
    tets = np.asarray(tets, dtype=np.int64)
    surface_vertices = np.asarray(
        surface_vertices,
        dtype=np.float64,
    )
    surface_faces = np.asarray(surface_faces, dtype=np.int64)
    surf_map = np.asarray(surf_map, dtype=np.int64)

    if reference_volume_vertices.ndim != 2 or reference_volume_vertices.shape[1] != 3:
        raise ValueError(
            "reference_volume_vertices must have shape (V, 3), "
            f"got {reference_volume_vertices.shape}"
        )

    if deformed_volume_vertices.shape != reference_volume_vertices.shape:
        raise ValueError(
            "deformed_volume_vertices must have the same shape as "
            f"reference_volume_vertices: {reference_volume_vertices.shape}, "
            f"got {deformed_volume_vertices.shape}"
        )

    if tets.ndim != 2 or tets.shape[1] != 4:
        raise ValueError(f"tets must have shape (T, 4), got {tets.shape}")

    if surface_vertices.ndim != 2 or surface_vertices.shape[1] != 3:
        raise ValueError(
            f"surface_vertices must have shape (N, 3), got {surface_vertices.shape}"
        )

    if surface_faces.ndim != 2 or surface_faces.shape[1] != 3:
        raise ValueError(
            f"surface_faces must have shape (F, 3), got {surface_faces.shape}"
        )

    if surf_map.shape != (surface_vertices.shape[0],):
        raise ValueError(
            "surf_map must have one entry per surface vertex, "
            f"expected {(surface_vertices.shape[0],)}, got {surf_map.shape}"
        )

    if not np.all(np.isfinite(reference_volume_vertices)):
        raise ValueError("reference_volume_vertices contains NaN or Inf")

    if not np.all(np.isfinite(deformed_volume_vertices)):
        raise ValueError("deformed_volume_vertices contains NaN or Inf")

    # =========================================================
    # 1. 计算全部四面体的真实 Jacobian
    # =========================================================
    reference_tets = reference_volume_vertices[tets]
    deformed_tets = deformed_volume_vertices[tets]

    Dm = np.stack(
        [
            reference_tets[:, 1] - reference_tets[:, 0],
            reference_tets[:, 2] - reference_tets[:, 0],
            reference_tets[:, 3] - reference_tets[:, 0],
        ],
        axis=2,
    )

    Ds = np.stack(
        [
            deformed_tets[:, 1] - deformed_tets[:, 0],
            deformed_tets[:, 2] - deformed_tets[:, 0],
            deformed_tets[:, 3] - deformed_tets[:, 0],
        ],
        axis=2,
    )

    det_Dm = np.linalg.det(Dm)
    det_Ds = np.linalg.det(Ds)

    valid_tets = np.abs(det_Dm) > det_eps

    jacobian = np.full(
        (tets.shape[0],),
        np.nan,
        dtype=np.float64,
    )

    # 等价于 det(Ds @ inverse(Dm))
    jacobian[valid_tets] = (
        det_Ds[valid_tets] / det_Dm[valid_tets]
    )

    valid_jacobian = jacobian[valid_tets]

    if valid_jacobian.size == 0:
        raise RuntimeError(
            "No valid tetrahedral Jacobians were computed."
        )

    # =========================================================
    # 2. 将 surface 三角形映射到 volume 顶点索引
    # =========================================================
    surface_faces_volume = surf_map[surface_faces]

    # 一个 surface face 可能出现重复，因此一个 key 对应 face id 列表
    surface_face_lookup = {}

    for face_id, face in enumerate(surface_faces_volume):
        key = tuple(
            sorted(
                (
                    int(face[0]),
                    int(face[1]),
                    int(face[2]),
                )
            )
        )
        surface_face_lookup.setdefault(key, []).append(face_id)

    surface_face_owners = [
        [] for _ in range(surface_faces.shape[0])
    ]

    tet_local_faces = (
        (0, 1, 2),
        (0, 1, 3),
        (0, 2, 3),
        (1, 2, 3),
    )

    # =========================================================
    # 3. 寻找每个 surface face 相邻的四面体
    # =========================================================
    for tet_id, tet in enumerate(tets):
        for local_face in tet_local_faces:
            key = tuple(
                sorted(
                    (
                        int(tet[local_face[0]]),
                        int(tet[local_face[1]]),
                        int(tet[local_face[2]]),
                    )
                )
            )

            matched_face_ids = surface_face_lookup.get(key)

            if matched_face_ids is None:
                continue

            for face_id in matched_face_ids:
                surface_face_owners[face_id].append(tet_id)

    surface_face_jacobian = np.full(
        (surface_faces.shape[0],),
        np.nan,
        dtype=np.float64,
    )

    mapped_surface_faces = np.zeros(
        (surface_faces.shape[0],),
        dtype=bool,
    )

    boundary_tet_ids = []

    for face_id, owner_ids in enumerate(surface_face_owners):
        if not owner_ids:
            continue

        owner_ids = np.asarray(owner_ids, dtype=np.int64)
        owner_jacobian = jacobian[owner_ids]
        owner_jacobian = owner_jacobian[
            np.isfinite(owner_jacobian)
        ]

        if owner_jacobian.size == 0:
            continue

        # 正常边界面只有一个相邻四面体。
        # 若异常情况下有多个，保守地取最小 J。
        surface_face_jacobian[face_id] = float(
            np.min(owner_jacobian)
        )

        mapped_surface_faces[face_id] = True
        boundary_tet_ids.extend(owner_ids.tolist())

    if not np.any(mapped_surface_faces):
        raise RuntimeError(
            "No source surface triangle could be mapped to an "
            "adjacent tetrahedron. Check surface_faces and surf_map."
        )

    boundary_tet_ids = np.unique(
        np.asarray(boundary_tet_ids, dtype=np.int64)
    )

    boundary_tet_jacobian = jacobian[boundary_tet_ids]
    boundary_tet_jacobian = boundary_tet_jacobian[
        np.isfinite(boundary_tet_jacobian)
    ]

    mapped_surface_jacobian = surface_face_jacobian[
        mapped_surface_faces
    ]

    # =========================================================
    # 4. 计算 reference surface 三角形面积
    # =========================================================
    triangles = surface_vertices[surface_faces]

    triangle_area = 0.5 * np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )

    mapped_triangle_area = triangle_area[mapped_surface_faces]
    total_mapped_area = float(mapped_triangle_area.sum())

    # =========================================================
    # 5. 全体四面体统计
    # =========================================================
    volume_negative = valid_jacobian < 0.0
    volume_nonpositive = valid_jacobian <= 0.0
    volume_low = valid_jacobian <= low_j_threshold

    # =========================================================
    # 6. 边界四面体统计
    # =========================================================
    boundary_negative = boundary_tet_jacobian < 0.0
    boundary_nonpositive = boundary_tet_jacobian <= 0.0
    boundary_low = boundary_tet_jacobian <= low_j_threshold

    # =========================================================
    # 7. surface 三角形统计
    # =========================================================
    surface_negative = mapped_surface_jacobian < 0.0
    surface_nonpositive = mapped_surface_jacobian <= 0.0
    surface_low = mapped_surface_jacobian <= low_j_threshold

    if total_mapped_area > 0.0:
        surface_negative_area_fraction = float(
            mapped_triangle_area[surface_negative].sum()
            / total_mapped_area
        )

        surface_low_area_fraction = float(
            mapped_triangle_area[surface_low].sum()
            / total_mapped_area
        )
    else:
        surface_negative_area_fraction = 0.0
        surface_low_area_fraction = 0.0

    return {
        "evaluation_stage": "after_registration",

        "fit_mode": "exact_tetrahedral_deformation_gradient",

        "definition": (
            "Each source surface triangle inherits J=det(Ds @ inverse(Dm)) "
            "from its adjacent boundary tetrahedron."
        ),

        "low_j_threshold": float(low_j_threshold),

        # -----------------------------------------------------
        # 全体四面体
        # -----------------------------------------------------
        "tetrahedron_count": int(tets.shape[0]),

        "valid_tetrahedron_count": int(
            valid_jacobian.size
        ),

        "invalid_reference_tetrahedron_count": int(
            np.count_nonzero(~valid_tets)
        ),

        "volume_negative_jacobian_count": int(
            np.count_nonzero(volume_negative)
        ),

        "volume_negative_jacobian_fraction": float(
            np.mean(volume_negative)
        ),

        "volume_negative_jacobian_percentage": float(
            100.0 * np.mean(volume_negative)
        ),

        "volume_nonpositive_jacobian_count": int(
            np.count_nonzero(volume_nonpositive)
        ),

        "volume_nonpositive_jacobian_fraction": float(
            np.mean(volume_nonpositive)
        ),

        "volume_low_jacobian_count": int(
            np.count_nonzero(volume_low)
        ),

        "volume_low_jacobian_fraction": float(
            np.mean(volume_low)
        ),

        "volume_low_jacobian_percentage": float(
            100.0 * np.mean(volume_low)
        ),

        "volume_jacobian_min": float(
            np.min(valid_jacobian)
        ),

        "volume_jacobian_mean": float(
            np.mean(valid_jacobian)
        ),

        "volume_jacobian_median": float(
            np.median(valid_jacobian)
        ),

        "volume_jacobian_max": float(
            np.max(valid_jacobian)
        ),

        "volume_jacobian_percentiles_1_5_50": [
            float(value)
            for value in np.percentile(
                valid_jacobian,
                [1, 5, 50],
            )
        ],

        # -----------------------------------------------------
        # 与 surface 相邻的唯一边界四面体
        # -----------------------------------------------------
        "boundary_tetrahedron_count": int(
            boundary_tet_jacobian.size
        ),

        "boundary_negative_jacobian_count": int(
            np.count_nonzero(boundary_negative)
        ),

        "boundary_negative_jacobian_fraction": float(
            np.mean(boundary_negative)
            if boundary_tet_jacobian.size > 0
            else 0.0
        ),

        "boundary_negative_jacobian_percentage": float(
            100.0 * np.mean(boundary_negative)
            if boundary_tet_jacobian.size > 0
            else 0.0
        ),

        "boundary_nonpositive_jacobian_count": int(
            np.count_nonzero(boundary_nonpositive)
        ),

        "boundary_low_jacobian_count": int(
            np.count_nonzero(boundary_low)
        ),

        "boundary_low_jacobian_fraction": float(
            np.mean(boundary_low)
            if boundary_tet_jacobian.size > 0
            else 0.0
        ),

        "boundary_jacobian_min": float(
            np.min(boundary_tet_jacobian)
        ),

        "boundary_jacobian_percentiles_1_5_50": [
            float(value)
            for value in np.percentile(
                boundary_tet_jacobian,
                [1, 5, 50],
            )
        ],

        # -----------------------------------------------------
        # warped source surface 三角形
        # -----------------------------------------------------
        "surface_triangle_count": int(
            surface_faces.shape[0]
        ),

        "mapped_surface_triangle_count": int(
            np.count_nonzero(mapped_surface_faces)
        ),

        "unmapped_surface_triangle_count": int(
            np.count_nonzero(~mapped_surface_faces)
        ),

        "surface_negative_triangle_count": int(
            np.count_nonzero(surface_negative)
        ),

        "surface_negative_jacobian_fraction": float(
            np.mean(surface_negative)
        ),

        "surface_negative_jacobian_percentage": float(
            100.0 * np.mean(surface_negative)
        ),

        "surface_nonpositive_triangle_count": int(
            np.count_nonzero(surface_nonpositive)
        ),

        "surface_nonpositive_jacobian_fraction": float(
            np.mean(surface_nonpositive)
        ),

        "surface_low_jacobian_triangle_count": int(
            np.count_nonzero(surface_low)
        ),

        "surface_low_jacobian_fraction": float(
            np.mean(surface_low)
        ),

        "surface_low_jacobian_percentage": float(
            100.0 * np.mean(surface_low)
        ),

        "surface_negative_reference_area_fraction": (
            surface_negative_area_fraction
        ),

        "surface_negative_reference_area_percentage": float(
            100.0 * surface_negative_area_fraction
        ),

        "surface_low_reference_area_fraction": (
            surface_low_area_fraction
        ),

        "surface_low_reference_area_percentage": float(
            100.0 * surface_low_area_fraction
        ),

        "surface_jacobian_min": float(
            np.min(mapped_surface_jacobian)
        ),

        "surface_jacobian_mean": float(
            np.mean(mapped_surface_jacobian)
        ),

        "surface_jacobian_median": float(
            np.median(mapped_surface_jacobian)
        ),

        "surface_jacobian_max": float(
            np.max(mapped_surface_jacobian)
        ),

        "surface_jacobian_percentiles_1_5_50": [
            float(value)
            for value in np.percentile(
                mapped_surface_jacobian,
                [1, 5, 50],
            )
        ],
    }


def surface_negative_jacobian_stats_old(
    source_surface: np.ndarray,
    warped_surface: np.ndarray,
    k: int = 20,
):
    """Estimate final surface-point Jacobians by local KNN least squares.

    This follows the requested point-cloud definition: fit the displacement
    gradient around each source-surface vertex, form F = I + grad(U), and count
    det(F) < 0. It is an evaluation metric only and is separate from the exact
    per-tetrahedron Jacobians used by the FEM solver for step acceptance.
    """
    source = np.asarray(source_surface, dtype=np.float64)
    warped = np.asarray(warped_surface, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"source_surface must have shape (N, 3), got {source.shape}")
    if warped.shape != source.shape:
        raise ValueError(
            f"warped_surface must match source shape {source.shape}, got {warped.shape}"
        )
    if source.shape[0] < 2:
        raise ValueError("surface Jacobian evaluation requires at least two points")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(warped)):
        raise ValueError("surface points contain NaN or Inf")
    if k < 1:
        raise ValueError("k must be at least 1")

    actual_k = min(int(k), source.shape[0] - 1)
    displacement = warped - source
    _, neighbors = cKDTree(source).query(source, k=actual_k + 1)
    jacobian = np.empty((source.shape[0],), dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    for index in range(source.shape[0]):
        neighbor_ids = neighbors[index, 1:]
        delta_x = source[neighbor_ids] - source[index]
        delta_u = displacement[neighbor_ids] - displacement[index]
        gradient_transpose, _, _, _ = np.linalg.lstsq(delta_x, delta_u, rcond=None)
        deformation_gradient = identity + gradient_transpose.T
        jacobian[index] = np.linalg.det(deformation_gradient)

    negative_count = int(np.count_nonzero(jacobian < 0.0))
    point_count = int(jacobian.size)
    negative_fraction = float(negative_count / point_count)
    return {
        "surface_point_count": point_count,
        "negative_jacobian_count": negative_count,
        "negative_jacobian_fraction": negative_fraction,
        "negative_jacobian_percentage": float(100.0 * negative_fraction),
        "jacobian_min": float(jacobian.min()),
        "jacobian_percentiles_1_5_50": [
            float(value) for value in np.percentile(jacobian, [1, 5, 50])
        ],
        "k_neighbors": actual_k,
        "fit_mode": "local_knn_lstsq",
        "evaluation_stage": "after_registration_only",
        "definition": "100 * count(det(I + local_grad_U) < 0) / source_surface_point_count",
    }


def maybe_eval_tre(
    source_txt: str,
    gt_txt: str,
    grid_src: pv.UnstructuredGrid,
    grid_def: pv.UnstructuredGrid,
    device: str,
    dtype: str,
    warp_chunk: int,
    out_dir: str,
    quiet: bool = False,
):
    if (not source_txt) or (not gt_txt):
        return None

    if (not os.path.exists(source_txt)) or (not os.path.exists(gt_txt)):
        log(f"[WARN] Skip TRE: source_txt or gt_txt not found. source_txt={source_txt}, gt_txt={gt_txt}", quiet)
        return None

    src_txt_pts = np.loadtxt(source_txt).reshape(-1, 3).astype(np.float64)
    gt_txt_pts = np.loadtxt(gt_txt).reshape(-1, 3).astype(np.float64)

    stats_before = tre_stats(gt_txt_pts, src_txt_pts)
    log(
        f"[EVAL, BEFORE WARPING] TRE(mean)={stats_before['tre_mean']:.4f} mm, "
        f"TRE(std)={stats_before['tre_std']:.4f} mm, "
        f"TRE(median)={stats_before['tre_median']:.4f} mm, "
        f"TRE(max)={stats_before['tre_max']:.4f} mm, "
        f"RMSE={stats_before['rmse']:.4f} mm",
        quiet
    )

    warped = warp_points_by_tet_mesh(
        src_txt_pts,
        grid_src,
        grid_def,
        clamp_outside=True,
        device=device,
        dtype=dtype,
        point_chunk=warp_chunk,
    )

    out_txt = os.path.join(out_dir, "warped_points.txt")
    np.savetxt(out_txt, warped, fmt="%.6f")

    stats_after = tre_stats(gt_txt_pts, warped)
    log(
        f"[EVAL] TRE(mean)={stats_after['tre_mean']:.4f} mm, "
        f"TRE(std)={stats_after['tre_std']:.4f} mm, "
        f"TRE(median)={stats_after['tre_median']:.4f} mm, "
        f"TRE(max)={stats_after['tre_max']:.4f} mm, "
        f"RMSE={stats_after['rmse']:.4f} mm | saved -> {out_txt}",
        quiet
    )
    return {
        "tre_mean": stats_after["tre_mean"],
        "tre_std": stats_after["tre_std"],
        "tre_median": stats_after["tre_median"],
        "tre_max": stats_after["tre_max"],
        "rmse": stats_after["rmse"],
        "n_points": stats_after["n_points"],
        "before_tre_mean": stats_before["tre_mean"],
        "before_tre_std": stats_before["tre_std"],
        "before_tre_median": stats_before["tre_median"],
        "before_tre_max": stats_before["tre_max"],
        "before_rmse": stats_before["rmse"],
        "warped_points_path": out_txt,
    }
# =============================================================================

import sys
from contextlib import contextmanager


class _TeeIO:
    def __init__(self, *streams): self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data); s.flush()
    def flush(self):
        for s in self.streams: s.flush()
    def isatty(self): return False


@contextmanager
def tee_stdout_stderr(log_path: str, mode: str = "w"):
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    old_out, old_err = sys.stdout, sys.stderr
    f = open(log_path, mode, buffering=1, encoding="utf-8")
    try:
        sys.stdout = _TeeIO(old_out, f)
        sys.stderr = _TeeIO(old_err, f)
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        f.close()


# =============================================================================
# Hybrid strict-energy PCG solver
# =============================================================================

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pyvista as pv
import torch
from scipy.spatial import cKDTree



def log(message: str, quiet: bool = False) -> None:
    if not quiet:
        print(message, flush=True)


def barycentric_vertex_areas(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """One third of each incident triangle area, accumulated at its vertices."""
    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    tri = points[faces]
    area = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )
    out = np.zeros((points.shape[0],), dtype=np.float64)
    for corner in range(3):
        np.add.at(out, faces[:, corner], area / 3.0)
    return out


def _safe_point_normals(mesh: pv.PolyData) -> np.ndarray:
    mesh_n = mesh.compute_normals(
        point_normals=True,
        cell_normals=False,
        auto_orient_normals=True,
        consistent_normals=True,
        inplace=False,
    )
    normals = np.asarray(mesh_n.point_data["Normals"], dtype=np.float64)
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


def load_target_with_area(
    path: str, max_points: int, rng: np.random.Generator, quiet: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh = load_surface_mesh(path, quiet=quiet)
    points = np.asarray(mesh.points, dtype=np.float64)
    faces = polydata_to_faces_tri(mesh)
    normals = _safe_point_normals(mesh)
    area = barycentric_vertex_areas(points, faces)
    if points.shape[0] > max_points > 0:
        ids = rng.choice(points.shape[0], size=max_points, replace=False)
        points, normals, area = points[ids], normals[ids], area[ids]
    log(
        f"[Target] sampled_points={points.shape[0]}, sampled_area={area.sum():.6f}",
        quiet,
    )
    return points, normals, area


def save_tet_cache(
    path: str,
    surface_points: np.ndarray,
    surface_faces: np.ndarray,
    volume_points: np.ndarray,
    tets: np.ndarray,
    surf_map: np.ndarray,
    source_used: str,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        surface_points=np.asarray(surface_points, dtype=np.float64),
        surface_faces=np.asarray(surface_faces, dtype=np.int64),
        volume_points=np.asarray(volume_points, dtype=np.float64),
        tets=np.asarray(tets, dtype=np.int64),
        surf_map=np.asarray(surf_map, dtype=np.int64),
        source_used=np.asarray(os.path.abspath(source_used)),
    )


def load_tet_cache(path: str):
    with np.load(path, allow_pickle=False) as cache:
        source_used = str(cache["source_used"].item())
        return (
            cache["surface_points"],
            cache["surface_faces"],
            cache["volume_points"],
            cache["tets"],
            cache["surf_map"],
            source_used,
        )


def prepare_source_mesh(args, quiet: bool = False):
    """Use the legacy preprocessing settings and reuse the exact tet mesh."""
    cache_path = args.tet_cache or os.path.join(args.outdir, "source_tet_cache.npz")
    if os.path.exists(cache_path):
        values = load_tet_cache(cache_path)
        log(f"[Mesh cache] loaded exact surface/tets: {cache_path}", quiet)
        return (*values, cache_path)

    source_used = args.preprocessed_source
    if source_used:
        if not os.path.exists(source_used):
            if not args.blender_repair_source:
                raise FileNotFoundError(f"Preprocessed source does not exist: {source_used}")
            source_used = run_blender_repair(
                args.source,
                source_used,
                blender_bin=args.blender_bin,
                merge_dist=args.blender_merge_dist,
                voxel_size=args.blender_voxel_size,
                smooth_iters=args.blender_smooth_iters,
                smooth_factor=args.blender_smooth_factor,
                decimate_ratio=args.blender_decimate_ratio,
                no_apply_transform=args.blender_no_apply_transform,
                triangulate=args.blender_triangulate,
                use_cache=True,
            )
    elif args.blender_repair_source:
        source_used = os.path.splitext(cache_path)[0] + "_surface.ply"
        source_used = run_blender_repair(
            args.source,
            source_used,
            blender_bin=args.blender_bin,
            merge_dist=args.blender_merge_dist,
            voxel_size=args.blender_voxel_size,
            smooth_iters=args.blender_smooth_iters,
            smooth_factor=args.blender_smooth_factor,
            decimate_ratio=args.blender_decimate_ratio,
            no_apply_transform=args.blender_no_apply_transform,
            triangulate=args.blender_triangulate,
            use_cache=True,
        )
    else:
        source_used = args.source

    surface = load_surface_mesh(source_used, quiet=quiet)
    surface_points = np.asarray(surface.points, dtype=np.float64)
    surface_faces = polydata_to_faces_tri(surface)
    volume_points, tets = tetrahedralize_with_tetgen(
        surface, args.mindihedral, args.minratio, args.maxvolume, quiet=quiet
    )
    surf_map = match_surface_points_to_volume(
        surface_points, volume_points, tol=args.match_tol
    )
    save_tet_cache(
        cache_path,
        surface_points,
        surface_faces,
        volume_points,
        tets,
        surf_map,
        source_used,
    )
    log(f"[Mesh cache] saved exact surface/tets: {cache_path}", quiet)
    return (
        surface_points,
        surface_faces,
        volume_points,
        tets,
        surf_map,
        source_used,
        cache_path,
    )


class HybridProstateCORFEMReg(StrictProstateCORFEMReg):
    """Strict gradient plus a PSD FEM surrogate Hessian and PCG."""

    def __init__(
        self,
        *args,
        target_vertex_area: Optional[np.ndarray] = None,
        data_area_scale: float = 1.0,
        pcg_tol: float = 1e-5,
        pcg_max_iter: int = 500,
        barrier_curvature_cap: float = 1e5,
        outer_merit_rel_tol: float = 1e-8,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        source_area = barycentric_vertex_areas(
            self.Xsurf0.detach().cpu().numpy(), self.src_faces_tri
        )
        if target_vertex_area is None:
            target_vertex_area = np.ones((self.target.shape[0],), dtype=np.float64)
        target_vertex_area = np.asarray(target_vertex_area, dtype=np.float64)
        if target_vertex_area.shape != (self.target.shape[0],):
            raise ValueError("target_vertex_area must have one entry per target point")
        if np.any(target_vertex_area < 0.0) or target_vertex_area.sum() <= 0.0:
            raise ValueError("target vertex areas must be nonnegative with positive sum")
        self.source_area = torch.as_tensor(source_area, device=self.device, dtype=self.dtype)
        self.target_area = torch.as_tensor(
            target_vertex_area, device=self.device, dtype=self.dtype
        )
        self.source_total_area = self.source_area.sum().clamp_min(1e-12)
        self.data_area_scale = float(data_area_scale)
        self.pcg_tol = float(pcg_tol)
        self.pcg_max_iter = int(pcg_max_iter)
        self.barrier_curvature_cap = float(barrier_curvature_cap)
        self.outer_merit_rel_tol = max(0.0, float(outer_merit_rel_tol))
        self.pcg_fallback_count = 0
        self.accepted_iterations = 0
        self.rejected_iterations = 0
        log(
            f"[INFO] hybrid solver=exact-gradient+PSD-FEM-surrogate+PCG+Armijo, "
            f"A_ref={float(self.source_total_area.item()):.6f}",
            self.quiet,
        )

    @torch.no_grad()
    def _make_match_block(
        self,
        ids,
        targets,
        src_normals,
        tgt_normals,
        dist,
        direction_weight,
        area=None,
    ):
        mask = self._trim_mask(dist, src_normals, tgt_normals)
        if area is None:
            area = torch.ones_like(dist)
        return {
            "ids": ids[mask],
            "targets": targets[mask],
            "normals": self._blend_normals(src_normals[mask], tgt_normals[mask]),
            "area": area[mask],
            "direction_weight": float(direction_weight),
            "active_ratio": float(mask.float().mean().item()),
        }

    @torch.no_grad()
    def build_outer_state(self, u_vec: torch.Tensor):
        surf = self.surface_vertices_from_u(u_vec)
        normals = self._surface_normals_from_u(u_vec)
        idx_t2s, d2_t2s = knn1_gpu(
            self.target, surf, self.knn_p_chunk, self.knn_q_chunk
        )
        t2s = self._make_match_block(
            self.surf_map[idx_t2s],
            self.target,
            normals[idx_t2s],
            self.target_normals,
            torch.sqrt(d2_t2s.clamp_min(0.0)),
            self.w_t2s,
            self.target_area,
        )
        idx_s2t, d2_s2t = knn1_gpu(
            surf, self.target, self.knn_p_chunk, self.knn_q_chunk
        )
        s2t = self._make_match_block(
            self.surf_map,
            self.target[idx_s2t],
            normals,
            self.target_normals[idx_s2t],
            torch.sqrt(d2_s2t.clamp_min(0.0)),
            self.w_s2t,
            self.source_area,
        )

        hit = torch.zeros((surf.shape[0],), device=self.device, dtype=self.dtype)
        hit.index_add_(0, idx_t2s, torch.ones_like(idx_t2s, dtype=self.dtype))
        distance = torch.sqrt(d2_s2t.clamp_min(0.0))
        if self.cover_dist_end > self.cover_dist_start:
            coverage = torch.clamp(
                (distance - self.cover_dist_start)
                / max(self.cover_dist_end - self.cover_dist_start, 1e-12),
                0.0,
                1.0,
            )
        else:
            coverage = (distance >= self.cover_dist_start).to(self.dtype)
        coverage = torch.where(hit > 0, torch.zeros_like(coverage), coverage)
        if self.surface_edges.numel() > 0:
            edge = self.surface_edges
            edge_w = 0.5 * (coverage[edge[:, 0]] + coverage[edge[:, 1]])
        else:
            edge_w = torch.empty((0,), device=self.device, dtype=self.dtype)
        return {
            "matches": [t2s, s2t],
            "edge_w": edge_w,
            "beta": float(self.current_tangent_weight()),
            "t2s_active_ratio": t2s["active_ratio"],
            "s2t_active_ratio": s2t["active_ratio"],
        }

    @torch.no_grad()
    def _cor_vol_components(self, u_vec: torch.Tensor):
        x_nodes = self.volume_vertices_from_u(u_vec)
        grad_cor = torch.zeros_like(x_nodes)
        grad_vol = torch.zeros_like(x_nodes)
        energy_cor = torch.zeros((), device=self.device, dtype=self.dtype)
        energy_vol = torch.zeros((), device=self.device, dtype=self.dtype)
        min_j = float("inf")
        for start in range(0, int(self.tets.shape[0]), self.tet_chunk):
            end = min(start + self.tet_chunk, int(self.tets.shape[0]))
            tet = self.tets[start:end]
            xe = x_nodes[tet]
            ds = torch.stack(
                [xe[:, 1] - xe[:, 0], xe[:, 2] - xe[:, 0], xe[:, 3] - xe[:, 0]],
                dim=2,
            )
            deformation = ds @ self.invDm[start:end]
            jacobian = torch.linalg.det(deformation)
            min_j = min(min_j, float(jacobian.min().item()))
            rotation = polar_rotation_from_F(deformation)
            stretch = rotation.transpose(1, 2) @ deformation
            stretch = 0.5 * (stretch + stretch.transpose(1, 2))
            strain = stretch - self.I3
            trace = strain.diagonal(dim1=1, dim2=2).sum(dim=1)
            # Physical volume quadrature.  The strict baseline divided this by
            # total organ volume, which is incompatible with the surface-area
            # integral used by the hybrid data term and makes FEM forces far
            # too weak on a large organ.
            weight = self.vol[start:end]
            cor_density = self.mu * (strain * strain).sum(dim=(1, 2))
            cor_density += 0.5 * self.lam * trace * trace
            energy_cor += (weight * cor_density).sum()
            stress = 2.0 * self.mu * strain
            stress += self.lam * trace.view(-1, 1, 1) * self.I3
            p_cor = rotation @ stress

            j_range = self.vol_j_min - self.min_accepted_j
            active = jacobian < self.vol_j_min
            z = ((jacobian - self.min_accepted_j) / j_range).clamp_min(1e-12)
            barrier = torch.where(
                active, -torch.log(z) + z - 1.0, torch.zeros_like(jacobian)
            )
            vol_density = 0.5 * self.vol_preserve_k * (jacobian - 1.0) ** 2
            vol_density += self.vol_barrier_k * barrier
            energy_vol += (weight * vol_density).sum()
            dbarrier = torch.where(
                active,
                self.vol_barrier_k * (1.0 - 1.0 / z) / j_range,
                torch.zeros_like(jacobian),
            )
            coeff = self.vol_preserve_k * (jacobian - 1.0) + dbarrier
            # Candidates at/below the floor are rejected before evaluation.
            finv_t = torch.linalg.inv(deformation).transpose(1, 2)
            p_vol = (coeff * jacobian).view(-1, 1, 1) * finv_t
            grads = self.grads[start:end]
            for corner in range(4):
                gc = torch.bmm(p_cor, grads[:, corner].unsqueeze(2)).squeeze(2)
                gv = torch.bmm(p_vol, grads[:, corner].unsqueeze(2)).squeeze(2)
                grad_cor.index_add_(0, tet[:, corner], gc * weight.view(-1, 1))
                grad_vol.index_add_(0, tet[:, corner], gv * weight.view(-1, 1))
        return energy_cor, energy_vol, grad_cor, grad_vol, min_j

    @torch.no_grad()
    def energy_gradient_components(self, u_vec: torch.Tensor, state):
        e_cor, e_vol, g_cor, g_vol, min_j = self._cor_vol_components(u_vec)
        x_nodes = self.volume_vertices_from_u(u_vec)
        zero = torch.zeros_like(x_nodes)
        g_surface = zero.clone()
        g_match = zero.clone()
        e_surface = torch.zeros((), device=self.device, dtype=self.dtype)
        e_match = torch.zeros((), device=self.device, dtype=self.dtype)

        if self.surface_smooth_k > 0.0 and self.surface_edges.numel() > 0:
            xsurf = x_nodes[self.surf_map]
            edge = self.surface_edges
            delta = xsurf[edge[:, 0]] - xsurf[edge[:, 1]]
            delta0 = self.Xsurf0[edge[:, 0]] - self.Xsurf0[edge[:, 1]]
            length = torch.norm(delta, dim=1).clamp_min(1e-12)
            diff = length - torch.norm(delta0, dim=1)
            edge_w = state["edge_w"]
            e_surface = 0.5 * self.surface_smooth_k * (
                edge_w * diff * diff
            ).sum()
            edge_grad = (
                self.surface_smooth_k
                * edge_w.view(-1, 1)
                * (diff / length).view(-1, 1)
                * delta
            )
            surface_grad = torch.zeros_like(xsurf)
            surface_grad.index_add_(0, edge[:, 0], edge_grad)
            surface_grad.index_add_(0, edge[:, 1], -edge_grad)
            g_surface.index_add_(0, self.surf_map, surface_grad)

        beta = float(state["beta"])
        for block in state["matches"]:
            if block["ids"].numel() == 0 or block["direction_weight"] <= 0.0:
                continue
            ids = block["ids"]
            delta = block["targets"] - x_nodes[ids]
            normal = block["normals"]
            normal_delta = (delta * normal).sum(dim=1, keepdim=True) * normal
            tangent_delta = delta - normal_delta
            q = (normal_delta * normal_delta).sum(dim=1)
            q += beta * (tangent_delta * tangent_delta).sum(dim=1)
            area = block["area"]
            area_norm = area.sum().clamp_min(1e-12)
            scale = (
                self.penalty_k
                * self.data_area_scale
                * self.source_total_area
                * block["direction_weight"]
                / area_norm
            )
            if self.robust_sigma > 0.0:
                sigma2 = self.robust_sigma * self.robust_sigma
                robust_w = torch.exp(-q / (2.0 * sigma2))
                rho = sigma2 * (1.0 - robust_w)
            else:
                robust_w = torch.ones_like(q)
                rho = 0.5 * q
            e_match += scale * (area * rho).sum()
            force = scale * (area * robust_w).view(-1, 1)
            force = force * (normal_delta + beta * tangent_delta)
            g_match.index_add_(0, ids, -force)

        energies = {
            "cor": e_cor,
            "vol": e_vol,
            "surface": e_surface,
            "match": e_match,
        }
        gradients = {
            "cor": g_cor.reshape(-1),
            "vol": g_vol.reshape(-1),
            "surface": g_surface.reshape(-1),
            "match": g_match.reshape(-1),
        }
        if self.fixed_dofs.numel() > 0:
            for key in gradients:
                gradients[key] = gradients[key].clone()
                gradients[key][self.fixed_dofs] = 0.0
        return energies, gradients, min_j

    @torch.no_grad()
    def energy_gradient(self, u_vec: torch.Tensor, state):
        energies, gradients, min_j = self.energy_gradient_components(u_vec, state)
        total = sum(energies.values())
        grad = sum(gradients.values())
        terms = {key: float(value.item()) for key, value in energies.items()}
        terms["total"] = float(total.item())
        terms["min_J"] = float(min_j)
        return total, grad, terms

    @torch.no_grad()
    def _surrogate_matvec_and_diag(self, u_vec: torch.Tensor, state):
        """Build a PSD approximate Hessian; it is not the exact tangent."""
        x_nodes = self.volume_vertices_from_u(u_vec)
        tet = self.tets
        grads = self.grads
        # B_cor uses the same volume quadrature scale as the exact gradient.
        weight = self.vol
        diag_nodes = torch.zeros_like(x_nodes)
        elastic_a = self.lam + 2.0 * self.mu
        for corner in range(4):
            gi = grads[:, corner]
            gx2, gy2, gz2 = gi[:, 0] ** 2, gi[:, 1] ** 2, gi[:, 2] ** 2
            diagonal = torch.stack(
                [
                    elastic_a * gx2 + self.mu * (gy2 + gz2),
                    elastic_a * gy2 + self.mu * (gx2 + gz2),
                    elastic_a * gz2 + self.mu * (gx2 + gy2),
                ],
                dim=1,
            ) * weight.view(-1, 1)
            diag_nodes.index_add_(0, tet[:, corner], diagonal)

        xe = x_nodes[tet]
        ds = torch.stack(
            [xe[:, 1] - xe[:, 0], xe[:, 2] - xe[:, 0], xe[:, 3] - xe[:, 0]],
            dim=2,
        )
        deformation = ds @ self.invDm
        jacobian = torch.linalg.det(deformation)
        finv_t = torch.linalg.inv(deformation).transpose(1, 2)
        grad_j = jacobian.view(-1, 1, 1) * torch.bmm(
            finv_t, grads.transpose(1, 2)
        )
        grad_j = grad_j.transpose(1, 2)  # tet, corner, xyz
        j_range = self.vol_j_min - self.min_accepted_j
        z = ((jacobian - self.min_accepted_j) / j_range).clamp_min(1e-8)
        barrier_curvature = torch.where(
            jacobian < self.vol_j_min,
            self.vol_barrier_k / (j_range * j_range * z * z),
            torch.zeros_like(jacobian),
        )
        bulk = torch.clamp(
            self.vol_preserve_k + barrier_curvature,
            min=0.0,
            max=self.barrier_curvature_cap,
        )
        for corner in range(4):
            diagonal = (
                weight * bulk
            ).view(-1, 1) * grad_j[:, corner] ** 2
            diag_nodes.index_add_(0, tet[:, corner], diagonal)

        surface_data = []
        if self.surface_smooth_k > 0.0 and self.surface_edges.numel() > 0:
            edge = self.surface_edges
            xs = x_nodes[self.surf_map]
            delta = xs[edge[:, 0]] - xs[edge[:, 1]]
            direction = delta / torch.norm(delta, dim=1, keepdim=True).clamp_min(1e-12)
            stiffness = self.surface_smooth_k * state["edge_w"]
            surface_data = [edge, direction, stiffness]
            for side in range(2):
                contribution = stiffness.view(-1, 1) * direction**2
                diag_nodes.index_add_(0, self.surf_map[edge[:, side]], contribution)

        match_data = []
        beta = float(state["beta"])
        for block in state["matches"]:
            if block["ids"].numel() == 0 or block["direction_weight"] <= 0.0:
                continue
            ids = block["ids"]
            normal = block["normals"]
            delta = block["targets"] - x_nodes[ids]
            dn = (delta * normal).sum(dim=1, keepdim=True) * normal
            dt = delta - dn
            q = (dn * dn).sum(dim=1) + beta * (dt * dt).sum(dim=1)
            robust_w = (
                torch.exp(-q / (2.0 * self.robust_sigma**2))
                if self.robust_sigma > 0.0
                else torch.ones_like(q)
            )
            area = block["area"]
            scale = (
                self.penalty_k
                * self.data_area_scale
                * self.source_total_area
                * block["direction_weight"]
                / area.sum().clamp_min(1e-12)
            )
            stiffness = scale * area * robust_w
            match_data.append((ids, normal, stiffness))
            diagonal = stiffness.view(-1, 1) * (
                beta + (1.0 - beta) * normal**2
            )
            diag_nodes.index_add_(0, ids, diagonal)

        def matvec(vector: torch.Tensor) -> torch.Tensor:
            nodes = vector.view(self.V, 3)
            local = nodes[tet]
            grad_u = torch.einsum("tai,taj->tij", local, grads)
            strain = 0.5 * (grad_u + grad_u.transpose(1, 2))
            trace = strain.diagonal(dim1=1, dim2=2).sum(dim=1)
            stress = 2.0 * self.mu * strain
            stress += self.lam * trace.view(-1, 1, 1) * self.I3
            out = torch.zeros_like(nodes)
            for corner in range(4):
                value = torch.bmm(stress, grads[:, corner].unsqueeze(2)).squeeze(2)
                out.index_add_(0, tet[:, corner], value * weight.view(-1, 1))
            dj = (grad_j * local).sum(dim=(1, 2))
            for corner in range(4):
                value = (weight * bulk * dj).view(-1, 1) * grad_j[:, corner]
                out.index_add_(0, tet[:, corner], value)
            if surface_data:
                edge, direction, stiffness = surface_data
                surface_nodes = nodes[self.surf_map]
                relative = surface_nodes[edge[:, 0]] - surface_nodes[edge[:, 1]]
                edge_value = stiffness.view(-1, 1) * (
                    relative * direction
                ).sum(dim=1, keepdim=True) * direction
                out.index_add_(0, self.surf_map[edge[:, 0]], edge_value)
                out.index_add_(0, self.surf_map[edge[:, 1]], -edge_value)
            for ids, normal, stiffness in match_data:
                values = nodes[ids]
                projected = beta * values
                projected += (1.0 - beta) * (
                    values * normal
                ).sum(dim=1, keepdim=True) * normal
                out.index_add_(0, ids, stiffness.view(-1, 1) * projected)
            flat = out.reshape(-1)
            if self.fixed_dofs.numel() > 0:
                flat = flat.clone()
                flat[self.fixed_dofs] = vector[self.fixed_dofs]
            return flat

        diagonal = diag_nodes.reshape(-1).clamp_min(1e-12)
        if self.fixed_dofs.numel() > 0:
            diagonal[self.fixed_dofs] = 1.0
        return matvec, diagonal

    @torch.no_grad()
    def _pcg(self, matvec, rhs: torch.Tensor, inv_diag: torch.Tensor):
        x = torch.zeros_like(rhs)
        residual = rhs.clone()
        z = inv_diag * residual
        direction = z.clone()
        rz = torch.dot(residual, z)
        rhs_norm = torch.norm(rhs).clamp_min(1e-30)
        rel = float(torch.norm(residual).item() / rhs_norm.item())
        iterations = 0
        for iterations in range(1, self.pcg_max_iter + 1):
            bd = matvec(direction)
            denominator = torch.dot(direction, bd)
            if (not torch.isfinite(denominator)) or denominator <= 1e-30:
                break
            step = rz / denominator
            x += step * direction
            residual -= step * bd
            rel = float((torch.norm(residual) / rhs_norm).item())
            if rel <= self.pcg_tol:
                break
            z = inv_diag * residual
            rz_new = torch.dot(residual, z)
            if (not torch.isfinite(rz_new)) or rz_new <= 0.0:
                break
            direction = z + (rz_new / rz) * direction
            rz = rz_new
        return x, iterations, rel

    @torch.no_grad()
    def compute_search_direction(self, u_vec: torch.Tensor, state):
        energy, gradient, terms = self.energy_gradient(u_vec, state)
        matvec, diagonal = self._surrogate_matvec_and_diag(u_vec, state)
        rhs = -gradient
        if self.fixed_dofs.numel() > 0:
            rhs = rhs.clone()
            rhs[self.fixed_dofs] = 0.0
        direction, pcg_iterations, pcg_residual = self._pcg(
            matvec, rhs, 1.0 / diagonal
        )
        fallback = "none"
        gtp = torch.dot(gradient, direction)
        if (not torch.isfinite(gtp)) or gtp >= 0.0:
            direction = -gradient / diagonal
            if self.fixed_dofs.numel() > 0:
                direction[self.fixed_dofs] = 0.0
            fallback = "jacobi_steepest"
            gtp = torch.dot(gradient, direction)
        if (not torch.isfinite(gtp)) or gtp >= 0.0:
            direction = -gradient
            if self.fixed_dofs.numel() > 0:
                direction[self.fixed_dofs] = 0.0
            fallback = "steepest"
            gtp = torch.dot(gradient, direction)
        if fallback != "none":
            self.pcg_fallback_count += 1
        if self.max_delta_u > 0.0:
            max_node = torch.norm(direction.view(self.V, 3), dim=1).max()
            scale = min(1.0, self.max_delta_u / max(float(max_node.item()), 1e-30))
            direction *= scale
            gtp = torch.dot(gradient, direction)
        info = {
            "energy": energy,
            "gradient": gradient,
            "terms": terms,
            "dot_g_p": float(gtp.item()),
            "gradient_norm": float(torch.norm(gradient).item()),
            "pcg_iterations": int(pcg_iterations),
            "pcg_relative_residual": float(pcg_residual),
            "fallback": fallback,
        }
        return direction, info

    @torch.no_grad()
    def robust_surface_merit(self, u_vec: torch.Tensor):
        surf = self.surface_vertices_from_u(u_vec)
        _, d2_t2s = knn1_gpu(self.target, surf, self.knn_p_chunk, self.knn_q_chunk)
        _, d2_s2t = knn1_gpu(surf, self.target, self.knn_p_chunk, self.knn_q_chunk)
        if self.robust_sigma > 0.0:
            sigma2 = self.robust_sigma**2
            rho_t = sigma2 * (1.0 - torch.exp(-d2_t2s / (2.0 * sigma2)))
            rho_s = sigma2 * (1.0 - torch.exp(-d2_s2t / (2.0 * sigma2)))
        else:
            rho_t, rho_s = 0.5 * d2_t2s, 0.5 * d2_s2t
        mt = (self.target_area * rho_t).sum() / self.target_area.sum().clamp_min(1e-12)
        ms = (self.source_area * rho_s).sum() / self.source_total_area
        weight_sum = max(self.w_t2s + self.w_s2t, 1e-12)
        merit = (self.w_t2s * mt + self.w_s2t * ms) / weight_sum
        return float(merit.item())

    @torch.no_grad()
    def jacobian_stats(self, u_vec: Optional[torch.Tensor] = None):
        jacobian = self.compute_current_J(self.u if u_vec is None else u_vec)
        quantiles = torch.quantile(
            jacobian, torch.tensor([0.01, 0.05, 0.5], device=self.device, dtype=self.dtype)
        )
        return float(jacobian.min().item()), [float(x) for x in quantiles.cpu().tolist()]

    @torch.no_grad()
    def line_search(self, u_vec, state, direction, info, merit_before):
        energy = info["energy"]
        gtp = info["dot_g_p"]
        alpha = self.line_search_init
        outer_backtracks = 0
        for backtrack in range(self.max_backtracks + 1):
            candidate = self._apply_constraints(u_vec + alpha * direction)
            min_j = float(self.compute_current_J(candidate).min().item())
            if min_j > self.min_accepted_j:
                candidate_energy, _, candidate_terms = self.energy_gradient(candidate, state)
                armijo = candidate_energy <= energy + self.armijo_c1 * alpha * gtp
                if torch.isfinite(candidate_energy) and armijo:
                    merit = self.robust_surface_merit(candidate)
                    limit = merit_before * (1.0 + self.outer_merit_rel_tol) + 1e-12
                    if merit <= limit:
                        return candidate, candidate_terms, alpha, outer_backtracks, merit
                    outer_backtracks += 1
            alpha *= self.backtrack_factor
            if alpha < self.min_step:
                break
        return None, None, 0.0, outer_backtracks, merit_before

    @torch.no_grad()
    def fit(self, max_iters: int, tol: float, min_iters: int = 20, stop_mean_err: float = 3.0):
        history = []
        diagnostics = []
        stop_reason = "max_iters"
        start = time.time()
        initial_error = self.compute_error_on_u(self.u)
        initial_merit = self.robust_surface_merit(self.u)
        previous_accepted_merit = initial_merit
        reject_streak = 0
        for iteration in range(max_iters):
            self.iter_id = iteration
            error_before = self.compute_error_on_u(self.u)
            merit_before = self.robust_surface_merit(self.u)
            state = self.build_outer_state(self.u)
            direction, info = self.compute_search_direction(self.u, state)
            if info["gradient_norm"] <= self.grad_tol:
                stop_reason = f"gradient_norm<{self.grad_tol}"
                break
            candidate, terms, alpha, outer_bt, merit_after = self.line_search(
                self.u, state, direction, info, merit_before
            )
            accepted = candidate is not None
            if accepted:
                self.u = candidate
                self.accepted_iterations += 1
                reject_streak = 0
                error_after = self.compute_error_on_u(self.u)
                previous_delta = abs(previous_accepted_merit - merit_after)
                previous_accepted_merit = merit_after
            else:
                self.rejected_iterations += 1
                reject_streak += 1
                error_after = error_before
                terms = info["terms"]
                previous_delta = float("inf")
            min_j, j_percentiles = self.jacobian_stats()
            mean_u, max_u = self.displacement_stats()
            active_ratio = 0.5 * (
                state["t2s_active_ratio"] + state["s2t_active_ratio"]
            )
            record = {
                "iteration": iteration,
                "accepted": accepted,
                "symmetric_mean_distance": error_after["mean_sym"],
                "robust_symmetric_merit": merit_after if accepted else merit_before,
                "robust_symmetric_mean_distance": error_after["mean_sym_robust"],
                "t2s_mean": error_after["t2s_mean"],
                "s2t_mean": error_after["s2t_mean"],
                "total_energy": terms["total"],
                "corotational_energy": terms["cor"],
                "volume_energy": terms["vol"],
                "surface_energy": terms["surface"],
                "matching_energy": terms["match"],
                "gradient_norm": info["gradient_norm"],
                "dot_g_p": info["dot_g_p"],
                "pcg_iterations": info["pcg_iterations"],
                "pcg_relative_residual": info["pcg_relative_residual"],
                "search_direction_fallback": info["fallback"],
                "armijo_alpha": alpha,
                "outer_rematching_backtracks": outer_bt,
                "displacement_mean": mean_u,
                "displacement_max": max_u,
                "min_j": min_j,
                "j_percentiles": j_percentiles,
                "active_match_ratio": active_ratio,
            }
            diagnostics.append(record)
            history.append(error_after["mean_sym_robust"])
            status = "accept" if accepted else "reject"
            log(
                f"[Iter {iteration:03d}] {status} | sym={error_before['mean_sym']:.6f}"
                f"->{error_after['mean_sym']:.6f}, merit={merit_before:.6e}->{merit_after:.6e}, "
                f"E={info['terms']['total']:.6e}->{terms['total']:.6e}, "
                f"|g|={info['gradient_norm']:.3e}, gTp={info['dot_g_p']:.3e}, "
                f"PCG={info['pcg_iterations']}/{info['pcg_relative_residual']:.2e}, "
                f"fallback={info['fallback']}, alpha={alpha:.3e}, outer_bt={outer_bt}, "
                f"u={mean_u:.4f}/{max_u:.4f}, J={min_j:.5f} "
                f"q={j_percentiles}, active={active_ratio:.3f}",
                self.quiet,
            )
            if reject_streak >= self.max_reject_streak:
                stop_reason = f"reject_streak>={self.max_reject_streak}"
                break
            if accepted and iteration + 1 >= min_iters and previous_delta < tol:
                stop_reason = f"accepted_merit_delta<{tol}"
                break
            if accepted and error_after["mean_sym"] < stop_mean_err:
                stop_reason = f"mean_error<{stop_mean_err}"
                break
        volume_final = self.volume_vertices_from_u(self.u).cpu().numpy()
        final_error = self.compute_error_on_u(self.u)
        final_merit = self.robust_surface_merit(self.u)
        return {
            "history": np.asarray(history, dtype=np.float64),
            "diagnostics": diagnostics,
            "elapsed_sec": float(time.time() - start),
            "stop_reason": stop_reason,
            "volume_vertices_final": volume_final,
            "surface_vertices_final": volume_final[self.surf_map.cpu().numpy()],
            "initial_error": initial_error,
            "final_error": final_error,
            "initial_robust_merit": initial_merit,
            "final_robust_merit": final_merit,
            "accepted_iterations": self.accepted_iterations,
            "rejected_iterations": self.rejected_iterations,
        }


def evaluate_label_surface(
    source_path: str,
    target_path: str,
    grid_source: pv.UnstructuredGrid,
    grid_deformed: pv.UnstructuredGrid,
    device: str,
    dtype: str,
    chunk: int,
    warped_output_path: str = "",
    source_output_path: str = "",
    target_output_path: str = "",
):
    if not source_path or not target_path:
        return None
    if not os.path.exists(source_path) or not os.path.exists(target_path):
        return None
    source_mesh = load_surface_mesh(source_path, quiet=True)
    target_mesh = load_surface_mesh(target_path, quiet=True)
    source = np.asarray(source_mesh.points, dtype=np.float64)
    target = np.asarray(target_mesh.points, dtype=np.float64)
    warped = warp_points_by_tet_mesh(
        source,
        grid_source,
        grid_deformed,
        clamp_outside=True,
        device=device,
        dtype=dtype,
        point_chunk=chunk,
    )

    source_faces = polydata_to_faces_tri(source_mesh)
    target_faces = polydata_to_faces_tri(target_mesh)
    for output_path in (source_output_path, warped_output_path, target_output_path):
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if source_output_path:
        export_mesh(source_output_path, source, source_faces)
    if warped_output_path:
        export_mesh(warped_output_path, warped, source_faces)
    if target_output_path:
        export_mesh(target_output_path, target, target_faces)

    def distance_stats(distance):
        distance = np.asarray(distance, dtype=np.float64)
        return {
            "mean": float(distance.mean()),
            "std": float(distance.std()),
            "median": float(np.median(distance)),
            "max": float(distance.max()),
            "rmse": float(np.sqrt(np.mean(distance * distance))),
        }

    before_forward = cKDTree(target).query(source, k=1)[0]
    after_forward = cKDTree(target).query(warped, k=1)[0]
    before_reverse = cKDTree(source).query(target, k=1)[0]
    after_reverse = cKDTree(warped).query(target, k=1)[0]
    before_symmetric = 0.5 * (before_forward.mean() + before_reverse.mean())
    after_symmetric = 0.5 * (after_forward.mean() + after_reverse.mean())

    return {
        "mapping": "source_label_points_through_reference_to_deformed_main_label_tetrahedra",
        "before_tre": distance_stats(before_forward),
        "after_tre": distance_stats(after_forward),
        "before_symmetric_tre": float(before_symmetric),
        "after_symmetric_tre": float(after_symmetric),
        "tre_improved": bool(after_symmetric < before_symmetric),
        "source_points": int(source.shape[0]),
        "target_points": int(target.shape[0]),
        "source_before_surface": os.path.abspath(source_output_path) if source_output_path else None,
        "deformed_surface": os.path.abspath(warped_output_path) if warped_output_path else None,
        "target_surface": os.path.abspath(target_output_path) if target_output_path else None,
        # Backward-compatible alias used by earlier result readers.
        "warped_surface": os.path.abspath(warped_output_path) if warped_output_path else None,
    }


def _label_file_index(path: str) -> Optional[int]:
    match = re.fullmatch(r"label(\d+)\.stl", os.path.basename(path), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def discover_evaluation_label_pairs(
    source_dir: str,
    target_dir: str,
    registration_source: str,
    start_index: int = -1,
):
    """Find all common labelN surfaces after the main registration label."""
    if not source_dir or not target_dir:
        return {}
    source_files = {}
    target_files = {}
    for path in Path(source_dir).glob("label*.stl"):
        index = _label_file_index(str(path))
        if index is not None:
            source_files[index] = str(path)
    for path in Path(target_dir).glob("label*.stl"):
        index = _label_file_index(str(path))
        if index is not None:
            target_files[index] = str(path)
    main_index = _label_file_index(registration_source)
    if start_index < 0:
        start_index = (main_index + 1) if main_index is not None else 1
    pairs = {}
    for index in sorted(set(source_files) & set(target_files)):
        if index < start_index or index == main_index:
            continue
        pairs[f"label{index}"] = (source_files[index], target_files[index])
    return pairs


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        "Hybrid prostate COR-FEM: exact energy gradient + PSD FEM surrogate PCG"
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--GT", default="")
    parser.add_argument("--source_txt", default="")
    parser.add_argument("--gt_txt", default="")
    parser.add_argument("--warp_chunk", type=int, default=200000)
    parser.add_argument("--eval-label1-source", default="")
    parser.add_argument("--eval-label1-target", default="")
    parser.add_argument("--eval-label2-source", default="")
    parser.add_argument("--eval-label2-target", default="")
    parser.add_argument(
        "--eval-label-source-dir",
        default="",
        help="directory containing source labelN.stl files; defaults to source mesh directory",
    )
    parser.add_argument(
        "--eval-label-target-dir",
        default="",
        help="directory containing target labelN.stl files; defaults to target mesh directory",
    )
    parser.add_argument(
        "--eval-label-start-index",
        type=int,
        default=-1,
        help="first file label index to evaluate; -1 means the index after the registration label",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--w-t2s", type=float, default=1.0)
    parser.add_argument("--w-s2t", type=float, default=1.0)
    parser.add_argument("--preprocessed-source", default="")
    parser.add_argument("--tet-cache", default="")
    parser.add_argument("--blender-repair-source", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--blender-bin", default="blender")
    parser.add_argument("--blender-merge-dist", type=float, default=0.0)
    parser.add_argument("--blender-voxel-size", type=float, default=2.0)
    parser.add_argument("--blender-smooth-iters", type=int, default=5)
    parser.add_argument("--blender-smooth-factor", type=float, default=0.2)
    parser.add_argument("--blender-decimate-ratio", type=float, default=1.0)
    parser.add_argument("--blender-no-apply-transform", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--blender-triangulate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mindihedral", type=float, default=10.0)
    parser.add_argument("--minratio", type=float, default=1.5)
    parser.add_argument("--maxvolume", type=float, default=0.0)
    parser.add_argument("--match_tol", type=float, default=1e-5)
    parser.add_argument("--young", type=float, default=500.0)
    parser.add_argument("--poisson", type=float, default=0.45)
    parser.add_argument("--max-iters", type=int, default=100)
    parser.add_argument("--min-iters", type=int, default=10)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--stop_mean_err", type=float, default=1e-3)
    parser.add_argument("--penalty-k", type=float, default=100.0)
    parser.add_argument("--data-area-scale", type=float, default=1.0)
    parser.add_argument("--robust-sigma", type=float, default=20.0)
    parser.add_argument("--target-sample-count", type=int, default=12000)
    parser.add_argument("--tangent-weight", type=float, default=0.22)
    parser.add_argument("--tangent-weight-start", type=float, default=0.70)
    parser.add_argument("--tangent-anneal-iters", type=int, default=40)
    parser.add_argument("--trim-quantile", type=float, default=0.99)
    parser.add_argument("--trim-max-dist", type=float, default=50.0)
    parser.add_argument("--trim-normal-cos", type=float, default=0.25)
    parser.add_argument("--max-delta-u", type=float, default=1.0)
    parser.add_argument("--vol-preserve-k", type=float, default=10.0)
    parser.add_argument("--vol-barrier-k", type=float, default=40.0)
    parser.add_argument("--vol-j-min", type=float, default=0.35)
    parser.add_argument("--min-accepted-j", type=float, default=0.10)
    parser.add_argument("--barrier-curvature-cap", type=float, default=1e5)
    parser.add_argument("--surface-smooth-k", type=float, default=1.5)
    parser.add_argument("--cover-dist-start", type=float, default=4.0)
    parser.add_argument("--cover-dist-end", type=float, default=12.0)
    parser.add_argument("--inner-iters", type=int, default=1, help="kept for CLI compatibility; one frozen-state PCG direction is used")
    parser.add_argument("--lbfgs-history", type=int, default=8, help="deprecated compatibility option")
    parser.add_argument("--pcg-tol", type=float, default=1e-5)
    parser.add_argument("--pcg-max-iter", type=int, default=500)
    parser.add_argument("--grad-tol", type=float, default=1e-6)
    parser.add_argument("--armijo-c1", type=float, default=1e-4)
    parser.add_argument("--line-search-init", type=float, default=1.0)
    parser.add_argument("--backtrack-factor", type=float, default=0.5)
    parser.add_argument("--min-step", type=float, default=1e-5)
    parser.add_argument("--max-backtracks", type=int, default=14)
    parser.add_argument("--max-reject-streak", type=int, default=3)
    parser.add_argument("--outer-merit-rel-tol", type=float, default=1e-8)
    parser.add_argument("--outer-energy-rel-tol", type=float, default=0.0, help="deprecated; refreshed total energy is not an acceptance metric")
    parser.add_argument("--outer-error-rel-tol", type=float, default=0.0, help="deprecated; use --outer-merit-rel-tol")
    parser.add_argument("--boundary-mode", choices=["auto_posterior", "percentile_min", "percentile_max", "none"], default="none")
    parser.add_argument("--boundary-axis", choices=["x", "y", "z"], default="z")
    parser.add_argument("--boundary-percentile", type=float, default=0.5)
    parser.add_argument("--tet-chunk", type=int, default=65536)
    parser.add_argument("--knn-p-chunk", type=int, default=4096)
    parser.add_argument("--knn-q-chunk", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--registration-improvement-tol", type=float, default=1e-8)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    ensure_dir(args.outdir)
    with tee_stdout_stderr(os.path.join(args.outdir, "log.txt"), mode="w"):
        quiet = args.quiet
        (
            surface_points,
            surface_faces,
            volume_points,
            tets,
            surf_map,
            source_used,
            tet_cache,
        ) = prepare_source_mesh(args, quiet=quiet)
        target_points, target_normals, target_area = load_target_with_area(
            args.target, args.target_sample_count, rng, quiet=quiet
        )
        fixed = auto_fixed_vertex_ids(
            volume_points,
            args.boundary_mode,
            args.boundary_percentile,
            args.boundary_axis,
        )
        reg = HybridProstateCORFEMReg(
            vol_points=volume_points,
            vol_tets=tets,
            surf_map=surf_map,
            src_faces_tri=surface_faces,
            target_points=target_points,
            target_normals=target_normals,
            target_vertex_area=target_area,
            w_t2s=args.w_t2s,
            w_s2t=args.w_s2t,
            fixed_vertex_ids=fixed,
            young=args.young,
            poisson=args.poisson,
            penalty_k=args.penalty_k,
            data_area_scale=args.data_area_scale,
            robust_sigma=args.robust_sigma,
            tangent_weight=args.tangent_weight,
            tangent_weight_start=args.tangent_weight_start,
            tangent_anneal_iters=args.tangent_anneal_iters,
            trim_quantile=args.trim_quantile,
            trim_max_dist=args.trim_max_dist,
            trim_normal_cos=args.trim_normal_cos,
            max_delta_u=args.max_delta_u,
            vol_preserve_k=args.vol_preserve_k,
            vol_barrier_k=args.vol_barrier_k,
            vol_j_min=args.vol_j_min,
            min_accepted_j=args.min_accepted_j,
            barrier_curvature_cap=args.barrier_curvature_cap,
            surface_smooth_k=args.surface_smooth_k,
            cover_dist_start=args.cover_dist_start,
            cover_dist_end=args.cover_dist_end,
            inner_iters=args.inner_iters,
            lbfgs_history=args.lbfgs_history,
            pcg_tol=args.pcg_tol,
            pcg_max_iter=args.pcg_max_iter,
            grad_tol=args.grad_tol,
            armijo_c1=args.armijo_c1,
            line_search_init=args.line_search_init,
            backtrack_factor=args.backtrack_factor,
            min_step=args.min_step,
            max_backtracks=args.max_backtracks,
            max_reject_streak=args.max_reject_streak,
            outer_merit_rel_tol=args.outer_merit_rel_tol,
            outer_energy_rel_tol=args.outer_energy_rel_tol,
            outer_error_rel_tol=args.outer_error_rel_tol,
            device=args.device,
            dtype=args.dtype,
            tet_chunk=args.tet_chunk,
            knn_p_chunk=args.knn_p_chunk,
            knn_q_chunk=args.knn_q_chunk,
            quiet=quiet,
        )
        result = reg.fit(args.max_iters, args.tol, args.min_iters, args.stop_mean_err)
        actual_device = str(reg.device)
        surface_path = os.path.join(args.outdir, "deformed_surface.ply")
        export_mesh(surface_path, result["surface_vertices_final"], surface_faces)
        export_mesh(
            os.path.join(args.outdir, "deformed_surface.obj"),
            result["surface_vertices_final"],
            surface_faces,
        )
        target_metric_mesh = load_surface_mesh(args.target, quiet=True)
        target_metric_points = np.asarray(
                                            target_metric_mesh.points,
                                            dtype=np.float64,
                                        )

        pre_surface_metrics = point_cloud_surface_metrics(
            surface_points,
            target_metric_points,
        )

        pre_surface_metrics.update(
            {
                "evaluation_stage": "before_registration",
                "source_surface": os.path.abspath(source_used),
                "target_surface": os.path.abspath(args.target),
            }
        )

        # 配准后：warped source surface 与 target surface
        post_surface_metrics = point_cloud_surface_metrics(
            result["surface_vertices_final"],
            target_metric_points,
        )

        post_surface_metrics.update(
            {
                "evaluation_stage": "after_registration",
                "warped_source_surface": os.path.abspath(surface_path),
                "target_surface": os.path.abspath(args.target),
            }
        )

                # =====================================================
        # 使用 FEM 四面体 mesh 计算真实 Jacobian
        # =====================================================
        surface_jacobian = surface_negative_jacobian_stats(
            reference_volume_vertices=volume_points,
            deformed_volume_vertices=result["volume_vertices_final"],
            tets=tets,
            surface_vertices=surface_points,
            surface_faces=surface_faces,
            surf_map=surf_map,
            low_j_threshold=args.min_accepted_j,
        )

        # Jacobian 只属于配准后结果
        post_surface_metrics[
            "surface_negative_jacobian"
        ] = surface_jacobian

        # =====================================================
        # 整理配准前后 surface 指标
        # =====================================================
        surface_metrics = {
            "before_registration": pre_surface_metrics,

            "after_registration": post_surface_metrics,

            "improvement": {
                "chamfer_symmetric_mean_reduction": float(
                    pre_surface_metrics["chamfer_symmetric_mean"]
                    - post_surface_metrics["chamfer_symmetric_mean"]
                ),

                "chamfer_distance_reduction": float(
                    pre_surface_metrics["chamfer_distance"]
                    - post_surface_metrics["chamfer_distance"]
                ),

                "hd95_reduction": float(
                    pre_surface_metrics["hd95"]
                    - post_surface_metrics["hd95"]
                ),

                "chamfer_symmetric_mean_improved": bool(
                    post_surface_metrics["chamfer_symmetric_mean"]
                    < pre_surface_metrics["chamfer_symmetric_mean"]
                ),

                "hd95_improved": bool(
                    post_surface_metrics["hd95"]
                    < pre_surface_metrics["hd95"]
                ),
            },
        }

        surface_metrics_path = os.path.join(
            args.outdir,
            "surface_metrics.json",
        )

        with open(
            surface_metrics_path,
            "w",
            encoding="utf-8",
        ) as stream:
            json.dump(
                surface_metrics,
                stream,
                indent=2,
            )

        log(
            "[Surface test metrics] "
            f"CD="
            f"{pre_surface_metrics['chamfer_symmetric_mean']:.6f}"
            f"->{post_surface_metrics['chamfer_symmetric_mean']:.6f} mm, "
            f"HD95="
            f"{pre_surface_metrics['hd95']:.6f}"
            f"->{post_surface_metrics['hd95']:.6f} mm, "
            f"negative_surface_J="
            f"{surface_jacobian['surface_negative_triangle_count']}"
            f"/{surface_jacobian['mapped_surface_triangle_count']} "
            f"({surface_jacobian['surface_negative_jacobian_percentage']:.6f}%), "
            f"negative_surface_area="
            f"{surface_jacobian['surface_negative_reference_area_percentage']:.6f}%, "
            f"negative_boundary_tet_J="
            f"{surface_jacobian['boundary_negative_jacobian_count']}"
            f"/{surface_jacobian['boundary_tetrahedron_count']} "
            f"({surface_jacobian['boundary_negative_jacobian_percentage']:.6f}%), "
            f"negative_volume_tet_J="
            f"{surface_jacobian['volume_negative_jacobian_count']}"
            f"/{surface_jacobian['valid_tetrahedron_count']} "
            f"({surface_jacobian['volume_negative_jacobian_percentage']:.6f}%), "
            f"minJ={surface_jacobian['volume_jacobian_min']:.6f}",
            quiet,
        )

        grid_source = make_tet_ugrid(
            volume_points,
            tets,
        )

        grid_deformed = make_tet_ugrid(
            result["volume_vertices_final"],
            tets,
        )

        grid_deformed.save(
            os.path.join(
                args.outdir,
                "deformed_volume.vtu",
            )
        )
        np.save(os.path.join(args.outdir, "history.npy"), result["history"])
        with open(os.path.join(args.outdir, "diagnostics.json"), "w", encoding="utf-8") as stream:
            json.dump(result["diagnostics"], stream, indent=2)
        tre_eval = maybe_eval_tre(
            args.source_txt,
            args.gt_txt,
            grid_source,
            grid_deformed,
            actual_device,
            args.dtype,
            args.warp_chunk,
            args.outdir,
            quiet=quiet,
        )
        label_source_dir = args.eval_label_source_dir or os.path.dirname(args.source)
        label_target_dir = args.eval_label_target_dir or os.path.dirname(args.target)
        label_pairs = discover_evaluation_label_pairs(
            label_source_dir,
            label_target_dir,
            args.source,
            args.eval_label_start_index,
        )
        # Preserve the old explicit arguments for callers outside the batch
        # script; directory discovery is the default and supports label3/4/...
        if args.eval_label1_source and args.eval_label1_target:
            label_pairs["label1"] = (args.eval_label1_source, args.eval_label1_target)
        if args.eval_label2_source and args.eval_label2_target:
            label_pairs["label2"] = (args.eval_label2_source, args.eval_label2_target)
        labels = {}
        label_export_root = os.path.join(args.outdir, "label_surfaces")
        source_label_dir = os.path.join(label_export_root, "source_before")
        deformed_label_dir = os.path.join(label_export_root, "deformed")
        target_label_dir = os.path.join(label_export_root, "target")
        for label_name, (label_source, label_target) in sorted(
            label_pairs.items(), key=lambda item: int(item[0][5:])
        ):
            label_result = evaluate_label_surface(
                label_source,
                label_target,
                grid_source,
                grid_deformed,
                actual_device,
                args.dtype,
                args.warp_chunk,
                os.path.join(deformed_label_dir, f"{label_name}.ply"),
                os.path.join(source_label_dir, f"{label_name}.ply"),
                os.path.join(target_label_dir, f"{label_name}.ply"),
            )
            labels[label_name] = label_result
            log(
                f"[TRE] {label_name}: symmetric "
                f"{label_result['before_symmetric_tre']:.6f} -> "
                f"{label_result['after_symmetric_tre']:.6f}, "
                f"mapped_points={label_result['source_points']}",
                quiet,
            )
        Path(label_export_root).mkdir(parents=True, exist_ok=True)
        label_evaluation_path = os.path.join(label_export_root, "label_evaluation.json")
        with open(label_evaluation_path, "w", encoding="utf-8") as stream:
            json.dump(labels, stream, indent=2)
        min_j, j_percentiles = reg.jacobian_stats()
        mean_u, max_u = reg.displacement_stats()
        improved = (
            result["final_robust_merit"]
            < result["initial_robust_merit"] - args.registration_improvement_tol
        )
        meta = {
            "task": "prostate_bidirectional_registration",
            "process_completed": True,
            "registration_improved": bool(improved),
            "jacobian_valid": bool(min_j > args.min_accepted_j),
            "surface_error_reduction": result["initial_error"]["mean_sym"] - result["final_error"]["mean_sym"],
            "robust_error_reduction": result["initial_robust_merit"] - result["final_robust_merit"],
            "initial_error": result["initial_error"],
            "final_error": result["final_error"],
            "initial_robust_merit": result["initial_robust_merit"],
            "final_robust_merit": result["final_robust_merit"],
            "min_j": min_j,
            "j_percentiles": j_percentiles,
            # warped source surface 三角形统计
            "surface_negative_jacobian_count": (
                surface_jacobian[
                    "surface_negative_triangle_count"
                ]
            ),

            "surface_negative_jacobian_fraction": (
                surface_jacobian[
                    "surface_negative_jacobian_fraction"
                ]
            ),

            "surface_negative_jacobian_percentage": (
                surface_jacobian[
                    "surface_negative_jacobian_percentage"
                ]
            ),

            "surface_negative_jacobian_area_fraction": (
                surface_jacobian[
                    "surface_negative_reference_area_fraction"
                ]
            ),

            "surface_negative_jacobian_area_percentage": (
                surface_jacobian[
                    "surface_negative_reference_area_percentage"
                ]
            ),

            # 与 surface 相邻的唯一边界四面体统计
            "boundary_negative_jacobian_count": (
                surface_jacobian[
                    "boundary_negative_jacobian_count"
                ]
            ),

            "boundary_negative_jacobian_fraction": (
                surface_jacobian[
                    "boundary_negative_jacobian_fraction"
                ]
            ),

            "boundary_negative_jacobian_percentage": (
                surface_jacobian[
                    "boundary_negative_jacobian_percentage"
                ]
            ),

            # 整个 FEM 体网格统计
            "volume_negative_jacobian_count": (
                surface_jacobian[
                    "volume_negative_jacobian_count"
                ]
            ),

            "volume_negative_jacobian_fraction": (
                surface_jacobian[
                    "volume_negative_jacobian_fraction"
                ]
            ),

            "volume_negative_jacobian_percentage": (
                surface_jacobian[
                    "volume_negative_jacobian_percentage"
                ]
            ),

            "surface_jacobian_statistics": surface_jacobian,
            "pcg_fallback_count": reg.pcg_fallback_count,
            "mean_displacement": mean_u,
            "max_displacement": max_u,
            "elapsed_sec": result["elapsed_sec"],
            "accepted_iterations": result["accepted_iterations"],
            "rejected_iterations": result["rejected_iterations"],
            "stop_reason": result["stop_reason"],
            "solver": "exact_gradient_psd_fem_surrogate_pcg_armijo",
            "actual_device": actual_device,
            "source_original": os.path.abspath(args.source),
            "source_used_for_registration": os.path.abspath(source_used),
            "tet_cache": os.path.abspath(tet_cache),
            "target": os.path.abspath(args.target),
            "surface_metrics": surface_metrics,
            "surface_metrics_file": os.path.abspath(surface_metrics_path),
            "label_evaluation": labels,
            "label_evaluation_file": os.path.abspath(label_evaluation_path),
            "label_surface_directories": {
                "source_before": os.path.abspath(source_label_dir),
                "deformed": os.path.abspath(deformed_label_dir),
                "target": os.path.abspath(target_label_dir),
            },
            "tre_eval": tre_eval,
            "config": vars(args),
        }
        meta_path = os.path.join(args.outdir, "run_meta.json")
        with open(meta_path, "w", encoding="utf-8") as stream:
            json.dump(meta, stream, indent=2)
        log(f"[Done] mesh: {surface_path}", quiet)
        log(
            f"[Done] registration_improved={improved}, jacobian_valid={meta['jacobian_valid']}, meta={meta_path}",
            quiet,
        )
        return meta


if __name__ == "__main__":
    wall_start = time.time()
    main()
    print(f"\n[Time] Total runtime: {time.time() - wall_start:.2f} s")
