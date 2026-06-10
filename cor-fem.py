#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


def resolve_bidirectional(task: str, bidirectional_flag: Optional[bool]) -> bool:
    if bidirectional_flag is not None:
        return bool(bidirectional_flag)
    task = task.lower()
    if task == "liver":
        return False
    if task == "prostate":
        return True
    return False


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
# GPU FEM
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

    return (
        torch.cat(grads_list, dim=0),
        torch.cat(vol_list, dim=0),
        torch.cat(invDm_list, dim=0),
    )


@torch.no_grad()
def linear_elasticity_tangent_matvec(
    x,
    tets,
    grads,
    vol,
    lam,
    mu,
    V,
    add_diag_idx=None,
    add_diag_val=None,
):
    device, dtype = x.device, x.dtype
    x_nodes = x.view(V, 3)
    u = x_nodes[tets]
    g = grads

    dux_dx = (u[:, :, 0] * g[:, :, 0]).sum(dim=1)
    dux_dy = (u[:, :, 0] * g[:, :, 1]).sum(dim=1)
    dux_dz = (u[:, :, 0] * g[:, :, 2]).sum(dim=1)

    duy_dx = (u[:, :, 1] * g[:, :, 0]).sum(dim=1)
    duy_dy = (u[:, :, 1] * g[:, :, 1]).sum(dim=1)
    duy_dz = (u[:, :, 1] * g[:, :, 2]).sum(dim=1)

    duz_dx = (u[:, :, 2] * g[:, :, 0]).sum(dim=1)
    duz_dy = (u[:, :, 2] * g[:, :, 1]).sum(dim=1)
    duz_dz = (u[:, :, 2] * g[:, :, 2]).sum(dim=1)

    exx, eyy, ezz = dux_dx, duy_dy, duz_dz
    gxy = dux_dy + duy_dx
    gyz = duy_dz + duz_dy
    gxz = dux_dz + duz_dx

    sxx = (lam + 2.0 * mu) * exx + lam * eyy + lam * ezz
    syy = lam * exx + (lam + 2.0 * mu) * eyy + lam * ezz
    szz = lam * exx + lam * eyy + (lam + 2.0 * mu) * ezz
    sxy = mu * gxy
    syz = mu * gyz
    sxz = mu * gxz

    y_nodes = torch.zeros((V, 3), device=device, dtype=dtype)
    for i in range(4):
        gi = g[:, i, :]
        gx, gy, gz = gi[:, 0], gi[:, 1], gi[:, 2]

        fx = gx * sxx + gy * sxy + gz * sxz
        fy = gx * sxy + gy * syy + gz * syz
        fz = gx * sxz + gy * syz + gz * szz

        fi = torch.stack([fx, fy, fz], dim=1) * vol.view(-1, 1)
        y_nodes.index_add_(0, tets[:, i], fi)

    y = y_nodes.reshape(-1)

    if add_diag_idx is not None and add_diag_val is not None:
        y.index_add_(0, add_diag_idx, add_diag_val * x[add_diag_idx])

    return y


@torch.no_grad()
def build_jacobi_diag(tets, grads, vol, lam, mu, V, chunk=131072):
    device, dtype = grads.device, grads.dtype
    diag_nodes = torch.zeros((V, 3), device=device, dtype=dtype)
    a = (lam + 2.0 * mu)
    T = tets.shape[0]

    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        tet = tets[s:e]
        g = grads[s:e]
        vv = vol[s:e].view(-1, 1)

        for i in range(4):
            gi = g[:, i, :]
            gx2 = gi[:, 0] * gi[:, 0]
            gy2 = gi[:, 1] * gi[:, 1]
            gz2 = gi[:, 2] * gi[:, 2]

            dx = (a * gx2 + mu * (gy2 + gz2)) * vv[:, 0]
            dy = (a * gy2 + mu * (gx2 + gz2)) * vv[:, 0]
            dz = (a * gz2 + mu * (gx2 + gy2)) * vv[:, 0]

            diag_nodes.index_add_(0, tet[:, i], torch.stack([dx, dy, dz], dim=1))

    return diag_nodes.reshape(-1).clamp_min(1e-12)


@torch.no_grad()
def build_jacobi_diag_bulk(tets, grads, vol, bulk_per_tet, V, chunk=131072):
    device, dtype = grads.device, grads.dtype
    diag_nodes = torch.zeros((V, 3), device=device, dtype=dtype)
    T = tets.shape[0]

    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        tet = tets[s:e]
        g = grads[s:e]
        vv = vol[s:e].view(-1, 1)
        kb = bulk_per_tet[s:e].view(-1, 1)

        for i in range(4):
            gi = g[:, i, :]
            gx2 = gi[:, 0] * gi[:, 0]
            gy2 = gi[:, 1] * gi[:, 1]
            gz2 = gi[:, 2] * gi[:, 2]

            dx = (kb[:, 0] * gx2) * vv[:, 0]
            dy = (kb[:, 0] * gy2) * vv[:, 0]
            dz = (kb[:, 0] * gz2) * vv[:, 0]

            diag_nodes.index_add_(0, tet[:, i], torch.stack([dx, dy, dz], dim=1))

    return diag_nodes.reshape(-1).clamp_min(0.0)


@torch.no_grad()
def bulk_tangent_matvec(x, tets, grads, vol, bulk_per_tet, V):
    device, dtype = x.device, x.dtype
    x_nodes = x.view(V, 3)
    u = x_nodes[tets]
    g = grads

    dux_dx = (u[:, :, 0] * g[:, :, 0]).sum(dim=1)
    duy_dy = (u[:, :, 1] * g[:, :, 1]).sum(dim=1)
    duz_dz = (u[:, :, 2] * g[:, :, 2]).sum(dim=1)
    tr = dux_dx + duy_dy + duz_dz

    y_nodes = torch.zeros((V, 3), device=device, dtype=dtype)
    coeff = (bulk_per_tet.view(-1, 1) * vol.view(-1, 1))

    for i in range(4):
        gi = g[:, i, :]
        fx = gi[:, 0] * tr * coeff[:, 0]
        fy = gi[:, 1] * tr * coeff[:, 0]
        fz = gi[:, 2] * tr * coeff[:, 0]
        fi = torch.stack([fx, fy, fz], dim=1)
        y_nodes.index_add_(0, tets[:, i], fi)

    return y_nodes.reshape(-1)


@torch.no_grad()
def volume_preservation_internal_force(
    X0: torch.Tensor,
    u_vec: torch.Tensor,
    tets: torch.Tensor,
    grads_ref: torch.Tensor,
    vol_ref: torch.Tensor,
    invDm_ref: torch.Tensor,
    vol_preserve_k: float,
    vol_barrier_k: float,
    vol_j_min: float,
    V: int,
    chunk: int = 65536,
):
    device = X0.device
    dtype = X0.dtype

    if vol_preserve_k <= 0.0 and vol_barrier_k <= 0.0:
        return torch.zeros((V * 3,), device=device, dtype=dtype)

    x_nodes = X0 + u_vec.view(V, 3)
    f_nodes = torch.zeros((V, 3), device=device, dtype=dtype)
    I = torch.eye(3, device=device, dtype=dtype).view(1, 3, 3)

    T = tets.shape[0]
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        tet = tets[s:e]
        g = grads_ref[s:e]
        vv = vol_ref[s:e].view(-1, 1)
        invDm_blk = invDm_ref[s:e]

        x_e = x_nodes[tet]
        x0 = x_e[:, 0]
        x1 = x_e[:, 1]
        x2 = x_e[:, 2]
        x3 = x_e[:, 3]

        Ds = torch.stack([x1 - x0, x2 - x0, x3 - x0], dim=2)
        F = Ds @ invDm_blk
        J = torch.linalg.det(F)

        F_safe = F + 1e-8 * I.expand(e - s, 3, 3)
        FinvT = torch.linalg.inv(F_safe).transpose(1, 2)

        coeff = torch.zeros_like(J)
        if vol_preserve_k > 0.0:
            coeff = coeff + float(vol_preserve_k) * (J - 1.0)
        if vol_barrier_k > 0.0:
            coeff = coeff + float(vol_barrier_k) * torch.clamp(float(vol_j_min) - J, min=0.0)

        P = (coeff * J).view(-1, 1, 1) * FinvT

        for i in range(4):
            gi = g[:, i, :]
            fi = torch.bmm(P, gi.unsqueeze(2)).squeeze(2) * vv
            f_nodes.index_add_(0, tet[:, i], fi)

    return f_nodes.reshape(-1)


@torch.no_grad()
def pcg_solve_operator(matvec, b, M_inv=None, x0=None, tol=1e-6, max_iter=500):
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - matvec(x)
    z = r if M_inv is None else (M_inv * r)
    p = z.clone()
    rz_old = torch.dot(r, z)
    b_norm = torch.norm(b) + 1e-12

    for _ in range(max_iter):
        Ap = matvec(p)
        denom = torch.dot(p, Ap) + 1e-12
        alpha = rz_old / denom
        x = x + alpha * p
        r = r - alpha * Ap

        if torch.norm(r) / b_norm < tol:
            break

        z = r if M_inv is None else (M_inv * r)
        rz_new = torch.dot(r, z)
        beta = rz_new / (rz_old + 1e-12)
        p = z + beta * p
        rz_old = rz_new

    return x


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
            d2 = P2 + Q2 - 2.0 * (Pblk @ Qblk.transpose(0, 1))

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
    detR = torch.linalg.det(R)
    neg = detR < 0
    if neg.any():
        U_fix = U.clone()
        U_fix[neg, :, 2] *= -1.0
        R = U_fix @ Vh
    return R


@torch.no_grad()
def corotational_internal_force(
    X0: torch.Tensor,
    u_vec: torch.Tensor,
    tets: torch.Tensor,
    grads: torch.Tensor,
    vol: torch.Tensor,
    invDm: torch.Tensor,
    lam: float,
    mu: float,
    V: int,
    chunk: int = 65536,
):
    device = X0.device
    dtype = X0.dtype

    x_nodes = X0 + u_vec.view(V, 3)
    f_nodes = torch.zeros((V, 3), device=device, dtype=dtype)

    T = tets.shape[0]

    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        tet = tets[s:e]
        g = grads[s:e]
        vv = vol[s:e].view(-1, 1)
        invDm_blk = invDm[s:e]

        X_e = X0[tet]
        x_e = x_nodes[tet]

        x0 = x_e[:, 0]
        x1 = x_e[:, 1]
        x2 = x_e[:, 2]
        x3 = x_e[:, 3]

        Ds = torch.stack([x1 - x0, x2 - x0, x3 - x0], dim=2)
        F = Ds @ invDm_blk
        R = polar_rotation_from_F(F)
        Rt = R.transpose(1, 2)

        x_local = torch.bmm(x_e, R)
        u_local = x_local - X_e

        dux_dx = (u_local[:, :, 0] * g[:, :, 0]).sum(dim=1)
        dux_dy = (u_local[:, :, 0] * g[:, :, 1]).sum(dim=1)
        dux_dz = (u_local[:, :, 0] * g[:, :, 2]).sum(dim=1)

        duy_dx = (u_local[:, :, 1] * g[:, :, 0]).sum(dim=1)
        duy_dy = (u_local[:, :, 1] * g[:, :, 1]).sum(dim=1)
        duy_dz = (u_local[:, :, 1] * g[:, :, 2]).sum(dim=1)

        duz_dx = (u_local[:, :, 2] * g[:, :, 0]).sum(dim=1)
        duz_dy = (u_local[:, :, 2] * g[:, :, 1]).sum(dim=1)
        duz_dz = (u_local[:, :, 2] * g[:, :, 2]).sum(dim=1)

        exx, eyy, ezz = dux_dx, duy_dy, duz_dz
        gxy = dux_dy + duy_dx
        gyz = duy_dz + duz_dy
        gxz = dux_dz + duz_dx

        sxx = (lam + 2.0 * mu) * exx + lam * eyy + lam * ezz
        syy = lam * exx + (lam + 2.0 * mu) * eyy + lam * ezz
        szz = lam * exx + lam * eyy + (lam + 2.0 * mu) * ezz
        sxy = mu * gxy
        syz = mu * gyz
        sxz = mu * gxz

        for i in range(4):
            gi = g[:, i, :]
            gx, gy, gz = gi[:, 0], gi[:, 1], gi[:, 2]

            fx = gx * sxx + gy * sxy + gz * sxz
            fy = gx * sxy + gy * syy + gz * syz
            fz = gx * sxz + gy * syz + gz * szz

            fi_local = torch.stack([fx, fy, fz], dim=1) * vv
            fi_global = torch.bmm(fi_local.unsqueeze(1), Rt).squeeze(1)
            f_nodes.index_add_(0, tet[:, i], fi_global)

    return f_nodes.reshape(-1)


# =============================================================================
# Registration
# =============================================================================

class LiverFEMGPUReg:
    def __init__(
        self,
        vol_points: np.ndarray,
        vol_tets: np.ndarray,
        surf_map: np.ndarray,
        src_faces_tri: np.ndarray,
        target_points: np.ndarray,
        target_normals: np.ndarray,
        bidirectional: bool,
        w_t2s: float,
        w_s2t: float,
        fixed_vertex_ids: np.ndarray,
        young: float,
        poisson: float,
        penalty_k: float,
        penalty_scale_s: float,
        robust_sigma: float,
        max_force: float,
        data_stiffness_scale: float,
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
        surface_smooth_k: float,
        cover_dist_start: float,
        cover_dist_end: float,
        tangent_refresh_every: int,
        fix_penalty_scale: float,
        relax: float,
        relax_max: float,
        relax_growth: float,
        backtrack_factor: float,
        min_relax: float,
        max_backtracks: int,
        max_reject_streak: int,
        device: str,
        dtype: str,
        pcg_tol: float,
        pcg_max_iter: int,
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

        self.V = self.X0.shape[0]
        self.ndof = self.V * 3

        self.bidirectional = bool(bidirectional)
        self.w_t2s = float(w_t2s)
        self.w_s2t = float(w_s2t)

        self.young = float(young)
        self.poisson = float(poisson)

        self.penalty_k = float(penalty_k)
        self.penalty_scale_s = float(penalty_scale_s)
        self.robust_sigma = float(robust_sigma)
        self.max_force = float(max_force)

        self.data_stiffness_scale = float(data_stiffness_scale)
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
        self.surface_smooth_k = float(surface_smooth_k)
        self.cover_dist_start = float(cover_dist_start)
        self.cover_dist_end = float(cover_dist_end)
        self.tangent_refresh_every = max(1, int(tangent_refresh_every))

        self.relax = float(relax)
        self.relax_max = float(relax_max)
        self.relax_growth = float(relax_growth)
        self.backtrack_factor = float(backtrack_factor)
        self.min_relax = float(min_relax)
        self.max_backtracks = int(max_backtracks)
        self.max_reject_streak = int(max_reject_streak)

        self.pcg_tol = float(pcg_tol)
        self.pcg_max_iter = int(pcg_max_iter)
        self.knn_p_chunk = int(knn_p_chunk)
        self.knn_q_chunk = int(knn_q_chunk)
        self.tet_chunk = int(tet_chunk)

        self.iter_id = 0

        self.lam, self.mu = compute_lame(self.young, self.poisson)

        log("[INFO] Precomputing tet gradients / volumes / invDm ...", quiet)
        t0 = time.time()
        self.grads, self.vol, self.invDm = precompute_tet_ref(
            self.X0, self.tets, dtype=tdtype, chunk=int(tet_chunk)
        )
        log(f"[OK] precompute done in {time.time() - t0:.3f}s", quiet)

        fixed_vertex_ids = np.asarray(fixed_vertex_ids, dtype=np.int64)
        fixed_dofs = []
        for vid in fixed_vertex_ids:
            fixed_dofs.extend([3 * vid + 0, 3 * vid + 1, 3 * vid + 2])
        self.fixed_dofs = torch.tensor(np.asarray(fixed_dofs, dtype=np.int64), device=dev, dtype=torch.long)

        diag_ref = build_jacobi_diag(
            self.tets, self.grads, self.vol, self.lam, self.mu, self.V, chunk=int(tet_chunk)
        )
        max_diag = float(diag_ref.max().item())
        bc_val = max(1.0, max_diag) * float(fix_penalty_scale)

        if self.fixed_dofs.numel() > 0:
            self.add_diag_idx = self.fixed_dofs
            self.add_diag_val = torch.full(
                (self.fixed_dofs.numel(),), bc_val, device=dev, dtype=tdtype
            )
        else:
            self.add_diag_idx = None
            self.add_diag_val = None

        self.u = torch.zeros((self.ndof,), device=dev, dtype=tdtype)
        self.current_grads = self.grads.clone()
        self.current_vol = self.vol.clone()
        self.current_bulk_tangent = torch.full_like(self.vol, max(self.vol_preserve_k, 0.0))
        self.current_cover_weight = torch.zeros((self.surf_map.shape[0],), device=dev, dtype=tdtype)
        self.diag_internal = diag_ref.clone()
        self.M_inv = 1.0 / self.diag_internal.clamp_min(1e-12)
        self._last_linearization_iter = -10**9
        self.refresh_current_linearization(force=True)

        log(
            f"[INFO] device={self.device}, dtype={self.dtype}, "
            f"V={self.V}, T={self.tets.shape[0]}, bidirectional={self.bidirectional}, "
            f"fem_model=corotational+vol+surface_smooth, data_term=point-to-plane",
            quiet
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
        Vdef = self.volume_vertices_from_u(u_vec)
        return Vdef[self.surf_map]

    @torch.no_grad()
    def current_volume_vertices(self) -> torch.Tensor:
        return self.volume_vertices_from_u(self.u)

    @torch.no_grad()
    def current_surface_vertices(self) -> torch.Tensor:
        return self.surface_vertices_from_u(self.u)

    @torch.no_grad()
    def current_surface_normals(self) -> torch.Tensor:
        return self._surface_normals_from_u(self.u)

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

        nrm = torch.norm(vn, dim=1, keepdim=True).clamp_min(1e-12)
        vn = vn / nrm
        return vn

    @torch.no_grad()
    def compute_coverage_weights_from_u(self, u_vec: torch.Tensor) -> torch.Tensor:
        if self.surface_smooth_k <= 0.0 or self.surf_map.numel() == 0:
            return torch.zeros((self.surf_map.shape[0],), device=self.device, dtype=self.dtype)

        surf = self.surface_vertices_from_u(u_vec)
        surf_normals = self._surface_normals_from_u(u_vec)

        # 1) target -> surface: 哪些 surface 顶点被 target 真正“看见”
        t2s_idx, _ = knn1_gpu(self.target, surf, self.knn_p_chunk, self.knn_q_chunk)
        hit = torch.zeros((surf.shape[0],), device=self.device, dtype=self.dtype)
        hit.index_add_(0, t2s_idx, torch.ones_like(t2s_idx, dtype=self.dtype))
        hit = (hit > 0).to(self.dtype)

        # 2) surface -> target: 当前距离和法向一致性
        s2t_idx, s2t_d2 = knn1_gpu(surf, self.target, self.knn_p_chunk, self.knn_q_chunk)
        s2t_dist = torch.sqrt(s2t_d2.clamp_min(0.0))
        tgt_n = self.target_normals[s2t_idx]

        sn = self._normalize_rows(surf_normals)
        tn = self._normalize_rows(tgt_n)
        cos = torch.abs((sn * tn).sum(dim=1))

        # 3) 只把“真的没人观测到，并且确实离得远”的地方当 uncovered
        covered_by_target = hit > 0.0
        covered_by_proximity = torch.logical_and(
            s2t_dist <= self.cover_dist_end,
            cos >= max(self.trim_normal_cos, 0.15)
        )
        covered = torch.logical_or(covered_by_target, covered_by_proximity)

        # uncovered 权重：covered=0，uncovered 按距离渐进增强
        if self.cover_dist_end > self.cover_dist_start:
            ramp = torch.clamp(
                (s2t_dist - self.cover_dist_start) / max(self.cover_dist_end - self.cover_dist_start, 1e-12),
                0.0, 1.0
            )
        else:
            ramp = (s2t_dist >= self.cover_dist_start).to(self.dtype)

        w = ramp * (~covered).to(self.dtype)
        return torch.clamp(w, 0.0, 1.0)

    @torch.no_grad()
    def compute_coverage_weights_from_u_old(self, u_vec: torch.Tensor) -> torch.Tensor:
        if self.surface_smooth_k <= 0.0 or self.surf_map.numel() == 0:
            return torch.zeros((self.surf_map.shape[0],), device=self.device, dtype=self.dtype)

        surf = self.surface_vertices_from_u(u_vec)
        _, d2 = knn1_gpu(surf, self.target, self.knn_p_chunk, self.knn_q_chunk)
        dist = torch.sqrt(d2.clamp_min(0.0))

        if self.cover_dist_end <= self.cover_dist_start:
            return (dist >= self.cover_dist_start).to(self.dtype)

        w = (dist - self.cover_dist_start) / max(self.cover_dist_end - self.cover_dist_start, 1e-12)
        return torch.clamp(w, 0.0, 1.0)

    @torch.no_grad()
    def compute_current_J(self, u_vec: Optional[torch.Tensor] = None) -> torch.Tensor:
        if u_vec is None:
            u_vec = self.u
        x_nodes = self.volume_vertices_from_u(u_vec)
        x_e = x_nodes[self.tets]
        x0 = x_e[:, 0]
        x1 = x_e[:, 1]
        x2 = x_e[:, 2]
        x3 = x_e[:, 3]
        Ds = torch.stack([x1 - x0, x2 - x0, x3 - x0], dim=2)
        F = Ds @ self.invDm
        return torch.linalg.det(F)

    @torch.no_grad()
    def surface_smooth_internal_force(self, u_vec: torch.Tensor, cover_w: Optional[torch.Tensor] = None):
        if self.surface_smooth_k <= 0.0 or self.surface_edges.numel() == 0:
            return torch.zeros((self.ndof,), device=self.device, dtype=self.dtype)

        if cover_w is None:
            cover_w = self.compute_coverage_weights_from_u(u_vec)

        # 当前变形后的表面坐标
        xsurf = self.surface_vertices_from_u(u_vec)

        # 参考构型下的表面坐标
        Xsurf0 = self.X0[self.surf_map]

        e = self.surface_edges

        xi = xsurf[e[:, 0]]
        xj = xsurf[e[:, 1]]
        Xi = Xsurf0[e[:, 0]]
        Xj = Xsurf0[e[:, 1]]

        we = 0.5 * (cover_w[e[:, 0]] + cover_w[e[:, 1]])

        rest_vec = Xi - Xj
        cur_vec = xi - xj

        rest_len = torch.norm(rest_vec, dim=1, keepdim=True).clamp_min(1e-12)
        cur_len = torch.norm(cur_vec, dim=1, keepdim=True).clamp_min(1e-12)

        # 惩罚当前边长偏离参考边长
        edge_force = self.surface_smooth_k * we.view(-1, 1) * ((cur_len - rest_len) / cur_len) * cur_vec

        f_surf = torch.zeros_like(xsurf)
        f_surf.index_add_(0, e[:, 0], edge_force)
        f_surf.index_add_(0, e[:, 1], -edge_force)

        f_nodes = torch.zeros((self.V, 3), device=self.device, dtype=self.dtype)
        f_nodes.index_add_(0, self.surf_map, f_surf)
        return f_nodes.reshape(-1)

    @torch.no_grad()
    def surface_smooth_internal_force_old(self, u_vec: torch.Tensor, cover_w: Optional[torch.Tensor] = None):
        if self.surface_smooth_k <= 0.0 or self.surface_edges.numel() == 0:
            return torch.zeros((self.ndof,), device=self.device, dtype=self.dtype)

        if cover_w is None:
            cover_w = self.compute_coverage_weights_from_u(u_vec)

        us = u_vec.view(self.V, 3)[self.surf_map]
        e = self.surface_edges
        ui = us[e[:, 0]]
        uj = us[e[:, 1]]
        we = 0.5 * (cover_w[e[:, 0]] + cover_w[e[:, 1]])
        edge_force = self.surface_smooth_k * we.view(-1, 1) * (ui - uj)

        f_surf = torch.zeros_like(us)
        f_surf.index_add_(0, e[:, 0], edge_force)
        f_surf.index_add_(0, e[:, 1], -edge_force)

        f_nodes = torch.zeros((self.V, 3), device=self.device, dtype=self.dtype)
        f_nodes.index_add_(0, self.surf_map, f_surf)
        return f_nodes.reshape(-1)

    @torch.no_grad()
    def surface_smooth_tangent_matvec(self, x: torch.Tensor, cover_w: Optional[torch.Tensor] = None):
        if self.surface_smooth_k <= 0.0 or self.surface_edges.numel() == 0:
            return torch.zeros_like(x)

        if cover_w is None:
            cover_w = self.current_cover_weight

        xs = x.view(self.V, 3)[self.surf_map]
        e = self.surface_edges
        xi = xs[e[:, 0]]
        xj = xs[e[:, 1]]
        we = 0.5 * (cover_w[e[:, 0]] + cover_w[e[:, 1]])
        edge_y = self.surface_smooth_k * we.view(-1, 1) * (xi - xj)

        y_surf = torch.zeros_like(xs)
        y_surf.index_add_(0, e[:, 0], edge_y)
        y_surf.index_add_(0, e[:, 1], -edge_y)

        y_nodes = torch.zeros((self.V, 3), device=self.device, dtype=self.dtype)
        y_nodes.index_add_(0, self.surf_map, y_surf)
        return y_nodes.reshape(-1)

    @torch.no_grad()
    def surface_smooth_diag(self, cover_w: Optional[torch.Tensor] = None):
        if self.surface_smooth_k <= 0.0 or self.surface_edges.numel() == 0:
            return torch.zeros((self.ndof,), device=self.device, dtype=self.dtype)

        if cover_w is None:
            cover_w = self.current_cover_weight

        e = self.surface_edges
        we = 0.5 * (cover_w[e[:, 0]] + cover_w[e[:, 1]])
        diag_surf = torch.zeros((self.surf_map.shape[0], 1), device=self.device, dtype=self.dtype)
        contrib = (self.surface_smooth_k * we).view(-1, 1)
        diag_surf.index_add_(0, e[:, 0], contrib)
        diag_surf.index_add_(0, e[:, 1], contrib)

        diag_nodes = torch.zeros((self.V, 1), device=self.device, dtype=self.dtype)
        diag_nodes.index_add_(0, self.surf_map, diag_surf)
        return diag_nodes.expand(-1, 3).reshape(-1)

    @torch.no_grad()
    def refresh_current_linearization(self, force: bool = False):
        if (not force) and ((self.iter_id - self._last_linearization_iter) < self.tangent_refresh_every):
            return

        x_nodes = self.current_volume_vertices()
        self.current_grads, self.current_vol, _ = precompute_tet_ref(
            x_nodes, self.tets, dtype=self.dtype, chunk=self.tet_chunk
        )
        J_cur = self.compute_current_J(self.u)
        self.current_bulk_tangent = torch.full_like(J_cur, max(self.vol_preserve_k, 0.0))
        if self.vol_barrier_k > 0.0:
            self.current_bulk_tangent = self.current_bulk_tangent + self.vol_barrier_k * (J_cur < self.vol_j_min).to(self.dtype)

        self.current_cover_weight = self.compute_coverage_weights_from_u(self.u)

        diag_internal = build_jacobi_diag(
            self.tets, self.current_grads, self.current_vol, self.lam, self.mu, self.V, chunk=self.tet_chunk
        )

        if self.vol_preserve_k > 0.0 or self.vol_barrier_k > 0.0:
            diag_internal = diag_internal + build_jacobi_diag_bulk(
                self.tets, self.current_grads, self.current_vol, self.current_bulk_tangent, self.V, chunk=self.tet_chunk
            )

        if self.surface_smooth_k > 0.0:
            diag_internal = diag_internal + self.surface_smooth_diag(self.current_cover_weight)

        if self.add_diag_idx is not None and self.add_diag_val is not None:
            diag_internal = diag_internal.clone()
            diag_internal.index_add_(0, self.add_diag_idx, self.add_diag_val)

        self.diag_internal = diag_internal.clamp_min(1e-12)
        self.M_inv = 1.0 / self.diag_internal
        self._last_linearization_iter = int(self.iter_id)

    @torch.no_grad()
    def displacement_stats(self, u_vec: Optional[torch.Tensor] = None):
        if u_vec is None:
            u_vec = self.u
        disp = u_vec.view(self.V, 3).norm(dim=1)
        return float(disp.mean().item()), float(disp.max().item())

    @torch.no_grad()
    def gaussian_weight(self, d2: torch.Tensor) -> torch.Tensor:
        if self.robust_sigma <= 0:
            return torch.ones((d2.shape[0], 1), device=self.device, dtype=self.dtype)

        sigma2 = max(float(self.robust_sigma) * float(self.robust_sigma), 1e-12)
        return torch.exp(-d2.unsqueeze(1) / (2.0 * sigma2))

    @torch.no_grad()
    def _normalize_rows(self, x: torch.Tensor) -> torch.Tensor:
        return x / torch.norm(x, dim=1, keepdim=True).clamp_min(1e-12)

    @torch.no_grad()
    def _blend_normals(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = self._normalize_rows(a)
        b = self._normalize_rows(b)

        dot = (a * b).sum(dim=1, keepdim=True)
        b_aligned = torch.where(dot < 0.0, -b, b)
        n = a + b_aligned
        n_norm = torch.norm(n, dim=1, keepdim=True)

        fallback = a
        use_blend = n_norm.squeeze(1) > 1e-12
        n = torch.where(use_blend.view(-1, 1), n / n_norm.clamp_min(1e-12), fallback)
        return n

    @torch.no_grad()
    def _project_delta(self, delta: torch.Tensor, n_ref: torch.Tensor) -> torch.Tensor:
        tw = self.current_tangent_weight()
        if tw >= 0.999999:
            return delta

        n_ref = self._normalize_rows(n_ref)
        dn = (delta * n_ref).sum(dim=1, keepdim=True) * n_ref
        dt = delta - dn
        return dn + tw * dt

    @torch.no_grad()
    def _trim_mask(
        self,
        dist: torch.Tensor,
        src_normals: torch.Tensor,
        tgt_normals: torch.Tensor,
    ) -> torch.Tensor:
        d = dist.squeeze(1)
        mask = torch.ones((d.shape[0],), device=self.device, dtype=torch.bool)

        if 0.0 < self.trim_quantile < 1.0 and d.numel() > 8:
            thr_q = torch.quantile(d, self.trim_quantile)
            mask &= (d <= thr_q)

        if self.trim_max_dist > 0:
            mask &= (d <= self.trim_max_dist)

        if self.trim_normal_cos > 0:
            sn = self._normalize_rows(src_normals)
            tn = self._normalize_rows(tgt_normals)
            cos = torch.abs((sn * tn).sum(dim=1))
            mask &= (cos >= self.trim_normal_cos)

        if mask.sum().item() <= 0:
            mask = torch.ones((d.shape[0],), device=self.device, dtype=torch.bool)

        return mask

    @torch.no_grad()
    def match_target_to_surface(self, surf: torch.Tensor, surf_normals: torch.Tensor):
        idx, d2 = knn1_gpu(self.target, surf, self.knn_p_chunk, self.knn_q_chunk)

        matched_surface_pts = surf[idx]
        matched_surface_normals = surf_normals[idx]
        matched_vol_ids = self.surf_map[idx]

        delta_raw = self.target - matched_surface_pts
        dist = torch.sqrt(d2.clamp_min(1e-12)).unsqueeze(1)

        n_ref = self._blend_normals(matched_surface_normals, self.target_normals)
        delta_eff = self._project_delta(delta_raw, n_ref)

        mask = self._trim_mask(
            dist=dist,
            src_normals=matched_surface_normals,
            tgt_normals=self.target_normals,
        )

        mean_err = float(dist.mean().item())
        max_err = float(dist.max().item())
        active_ratio = float(mask.float().mean().item())

        return {
            "matched_vol_ids": matched_vol_ids,
            "delta_eff": delta_eff,
            "d2": d2,
            "dist": dist,
            "mean_err": mean_err,
            "max_err": max_err,
            "mask": mask,
            "active_ratio": active_ratio,
        }

    @torch.no_grad()
    def match_surface_to_target(self, surf: torch.Tensor, surf_normals: torch.Tensor):
        idx, d2 = knn1_gpu(surf, self.target, self.knn_p_chunk, self.knn_q_chunk)

        matched_target_pts = self.target[idx]
        matched_target_normals = self.target_normals[idx]
        matched_vol_ids = self.surf_map

        delta_raw = matched_target_pts - surf
        dist = torch.sqrt(d2.clamp_min(1e-12)).unsqueeze(1)

        n_ref = self._blend_normals(surf_normals, matched_target_normals)
        delta_eff = self._project_delta(delta_raw, n_ref)

        mask = self._trim_mask(
            dist=dist,
            src_normals=surf_normals,
            tgt_normals=matched_target_normals,
        )

        mean_err = float(dist.mean().item())
        max_err = float(dist.max().item())
        active_ratio = float(mask.float().mean().item())

        return {
            "matched_vol_ids": matched_vol_ids,
            "delta_eff": delta_eff,
            "d2": d2,
            "dist": dist,
            "mean_err": mean_err,
            "max_err": max_err,
            "mask": mask,
            "active_ratio": active_ratio,
        }

    @torch.no_grad()
    def combine_bidirectional_errors(
        self,
        t2s_mean: float,
        t2s_max: float,
        s2t_mean: float,
        s2t_max: float,
    ):
        wt = max(0.0, float(self.w_t2s))
        ws = max(0.0, float(self.w_s2t))

        if not self.bidirectional:
            wt, ws = 1.0, 0.0

        wsum = max(wt + ws, 1e-12)
        mean_sym = (wt * t2s_mean + ws * s2t_mean) / wsum
        max_sym = max(t2s_max, s2t_max)

        return float(mean_sym), float(max_sym)

    @torch.no_grad()
    def combine_bidirectional_mean_only(self, a: float, b: float) -> float:
        wt = max(0.0, float(self.w_t2s))
        ws = max(0.0, float(self.w_s2t))

        if not self.bidirectional:
            wt, ws = 1.0, 0.0

        wsum = max(wt + ws, 1e-12)
        return float((wt * a + ws * b) / wsum)

    @torch.no_grad()
    def build_force_nodes_from_matches(
        self,
        matched_vol_ids: torch.Tensor,
        delta_eff: torch.Tensor,
        d2: torch.Tensor,
        mask: torch.Tensor,
        weight: float,
    ):
        f_nodes = torch.zeros((self.V, 3), device=self.device, dtype=self.dtype)
        kdiag_nodes = torch.zeros((self.V, 1), device=self.device, dtype=self.dtype)

        if matched_vol_ids.numel() == 0 or weight <= 0:
            return f_nodes, kdiag_nodes

        if mask is not None and mask.any():
            ids = matched_vol_ids[mask]
            delta_eff = delta_eff[mask]
            d2 = d2[mask]
        else:
            ids = matched_vol_ids

        robust_w = self.gaussian_weight(d2)
        eff_w = robust_w * float(weight)

        fvec = self.penalty_k * eff_w * delta_eff
        f_nodes.index_add_(0, ids, fvec)
        kdiag_nodes.index_add_(0, ids, self.penalty_k * eff_w)

        if self.max_force > 0:
            fnorm = torch.norm(f_nodes, dim=1, keepdim=True).clamp_min(1e-12)
            scale = torch.clamp(self.max_force / fnorm, max=1.0)
            f_nodes = f_nodes * scale

        return f_nodes, kdiag_nodes

    @torch.no_grad()
    def compute_error_on_u(self, u_vec: torch.Tensor):
        surf = self.surface_vertices_from_u(u_vec)
        surf_normals = self._surface_normals_from_u(u_vec)

        mt2s = self.match_target_to_surface(surf, surf_normals)
        d2_t2s = mt2s["d2"]
        dist_t2s = mt2s["dist"]
        mask_t2s = mt2s["mask"]

        wt2s = self.gaussian_weight(d2_t2s).squeeze(1)
        if mask_t2s is not None and mask_t2s.any():
            wt2s = wt2s[mask_t2s]
            dist_t2s_obj = dist_t2s.squeeze(1)[mask_t2s]
        else:
            dist_t2s_obj = dist_t2s.squeeze(1)

        t2s_mean_robust = float(
            (wt2s * dist_t2s_obj).sum().item() / (wt2s.sum().item() + 1e-12)
        )

        if self.bidirectional:
            ms2t = self.match_surface_to_target(surf, surf_normals)
            d2_s2t = ms2t["d2"]
            dist_s2t = ms2t["dist"]
            mask_s2t = ms2t["mask"]

            ws2t = self.gaussian_weight(d2_s2t).squeeze(1)
            if mask_s2t is not None and mask_s2t.any():
                ws2t = ws2t[mask_s2t]
                dist_s2t_obj = dist_s2t.squeeze(1)[mask_s2t]
            else:
                dist_s2t_obj = dist_s2t.squeeze(1)

            s2t_mean_robust = float(
                (ws2t * dist_s2t_obj).sum().item() / (ws2t.sum().item() + 1e-12)
            )
            s2t_mean = ms2t["mean_err"]
            s2t_max = ms2t["max_err"]
            s2t_active_ratio = ms2t["active_ratio"]
        else:
            s2t_mean, s2t_max = 0.0, 0.0
            s2t_mean_robust = 0.0
            s2t_active_ratio = 0.0

        mean_sym, max_sym = self.combine_bidirectional_errors(
            t2s_mean=mt2s["mean_err"],
            t2s_max=mt2s["max_err"],
            s2t_mean=s2t_mean,
            s2t_max=s2t_max,
        )

        mean_sym_robust = self.combine_bidirectional_mean_only(
            t2s_mean_robust, s2t_mean_robust
        )

        return {
            "t2s_mean": float(mt2s["mean_err"]),
            "t2s_max": float(mt2s["max_err"]),
            "s2t_mean": float(s2t_mean),
            "s2t_max": float(s2t_max),
            "t2s_mean_robust": float(t2s_mean_robust),
            "s2t_mean_robust": float(s2t_mean_robust),
            "t2s_active_ratio": float(mt2s["active_ratio"]),
            "s2t_active_ratio": float(s2t_active_ratio),
            "mean_sym": float(mean_sym),
            "max_sym": float(max_sym),
            "mean_sym_robust": float(mean_sym_robust),
        }

    @torch.no_grad()
    def compute_external_force(self):
        surf = self.current_surface_vertices()
        surf_normals = self.current_surface_normals()

        mt2s = self.match_target_to_surface(surf, surf_normals)
        f_nodes_t2s, kdiag_t2s = self.build_force_nodes_from_matches(
            matched_vol_ids=mt2s["matched_vol_ids"],
            delta_eff=mt2s["delta_eff"],
            d2=mt2s["d2"],
            mask=mt2s["mask"],
            weight=self.w_t2s if self.bidirectional else 1.0,
        )

        if self.bidirectional:
            ms2t = self.match_surface_to_target(surf, surf_normals)
            f_nodes_s2t, kdiag_s2t = self.build_force_nodes_from_matches(
                matched_vol_ids=ms2t["matched_vol_ids"],
                delta_eff=ms2t["delta_eff"],
                d2=ms2t["d2"],
                mask=ms2t["mask"],
                weight=self.w_s2t,
            )
            s2t_mean = ms2t["mean_err"]
            s2t_max = ms2t["max_err"]
            s2t_active_ratio = ms2t["active_ratio"]
        else:
            f_nodes_s2t = torch.zeros((self.V, 3), device=self.device, dtype=self.dtype)
            kdiag_s2t = torch.zeros((self.V, 1), device=self.device, dtype=self.dtype)
            s2t_mean, s2t_max, s2t_active_ratio = 0.0, 0.0, 0.0

        f_nodes = f_nodes_t2s + f_nodes_s2t
        kdiag_nodes = kdiag_t2s + kdiag_s2t

        fext = f_nodes.reshape(-1)
        data_diag = kdiag_nodes.expand(-1, 3).reshape(-1)

        mean_sym, max_sym = self.combine_bidirectional_errors(
            t2s_mean=mt2s["mean_err"],
            t2s_max=mt2s["max_err"],
            s2t_mean=s2t_mean,
            s2t_max=s2t_max,
        )

        err_info = {
            "t2s_mean": float(mt2s["mean_err"]),
            "t2s_max": float(mt2s["max_err"]),
            "s2t_mean": float(s2t_mean),
            "s2t_max": float(s2t_max),
            "t2s_active_ratio": float(mt2s["active_ratio"]),
            "s2t_active_ratio": float(s2t_active_ratio),
            "mean_sym": float(mean_sym),
            "max_sym": float(max_sym),
        }

        return fext, data_diag, err_info

    @torch.no_grad()
    def compute_internal_force(self, u_vec: Optional[torch.Tensor] = None):
        if u_vec is None:
            u_vec = self.u

        use_cached = (u_vec.data_ptr() == self.u.data_ptr())
        cover_w = self.current_cover_weight if use_cached else self.compute_coverage_weights_from_u(u_vec)

        fint = corotational_internal_force(
            X0=self.X0,
            u_vec=u_vec,
            tets=self.tets,
            grads=self.grads,
            vol=self.vol,
            invDm=self.invDm,
            lam=self.lam,
            mu=self.mu,
            V=self.V,
            chunk=self.tet_chunk,
        )

        if self.vol_preserve_k > 0.0 or self.vol_barrier_k > 0.0:
            fint = fint + volume_preservation_internal_force(
                X0=self.X0,
                u_vec=u_vec,
                tets=self.tets,
                grads_ref=self.grads,
                vol_ref=self.vol,
                invDm_ref=self.invDm,
                vol_preserve_k=self.vol_preserve_k,
                vol_barrier_k=self.vol_barrier_k,
                vol_j_min=self.vol_j_min,
                V=self.V,
                chunk=self.tet_chunk,
            )

        if self.surface_smooth_k > 0.0:
            fint = fint + self.surface_smooth_internal_force(u_vec, cover_w=cover_w)

        if self.add_diag_idx is not None and self.add_diag_val is not None:
            fint = fint.clone()
            fint.index_add_(0, self.add_diag_idx, self.add_diag_val * u_vec[self.add_diag_idx])

        return fint

    @torch.no_grad()
    def matvec(self, x):
        y = linear_elasticity_tangent_matvec(
            x=x,
            tets=self.tets,
            grads=self.current_grads,
            vol=self.current_vol,
            lam=self.lam,
            mu=self.mu,
            V=self.V,
            add_diag_idx=self.add_diag_idx,
            add_diag_val=self.add_diag_val,
        )

        if self.vol_preserve_k > 0.0 or self.vol_barrier_k > 0.0:
            y = y + bulk_tangent_matvec(
                x=x,
                tets=self.tets,
                grads=self.current_grads,
                vol=self.current_vol,
                bulk_per_tet=self.current_bulk_tangent,
                V=self.V,
            )

        if self.surface_smooth_k > 0.0:
            y = y + self.surface_smooth_tangent_matvec(x, cover_w=self.current_cover_weight)

        return y

    @torch.no_grad()
    def propose_update(self):
        self.refresh_current_linearization(force=(self.tangent_refresh_every <= 1))
        fext, data_diag, err_info = self.compute_external_force()
        fint = self.compute_internal_force(self.u)
        residual = fext - fint

        total_diag = self.diag_internal + self.data_stiffness_scale * data_diag
        M_inv_total = 1.0 / total_diag.clamp_min(1e-12)

        def matvec_total(x):
            y = self.matvec(x)
            if self.data_stiffness_scale > 0:
                y = y + self.data_stiffness_scale * data_diag * x
            return y

        delta_u = pcg_solve_operator(
            matvec=matvec_total,
            b=residual,
            M_inv=M_inv_total,
            x0=None,
            tol=self.pcg_tol,
            max_iter=self.pcg_max_iter,
        )

        if self.max_delta_u > 0:
            delta_nodes = delta_u.view(self.V, 3)
            dnorm = torch.norm(delta_nodes, dim=1, keepdim=True).clamp_min(1e-12)
            scale = torch.clamp(self.max_delta_u / dnorm, max=1.0)
            delta_u = (delta_nodes * scale).reshape(-1)

        u_solve = self.u + delta_u
        return u_solve, err_info

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

            mean_before = err_before["mean_sym"]
            max_before = err_before["max_sym"]
            mean_before_robust = err_before["mean_sym_robust"]

            u_mean_before, u_max_before = self.displacement_stats(self.u)

            if self.bidirectional:
                log(
                    f"[Iter {it:03d}][state] "
                    f"sym_mean={mean_before:.6f}, "
                    f"sym_mean_robust={mean_before_robust:.6f}, "
                    f"sym_max={max_before:.6f}, "
                    f"t2s_mean={err_before['t2s_mean']:.6f}, "
                    f"s2t_mean={err_before['s2t_mean']:.6f}, "
                    f"t2s_active={err_before['t2s_active_ratio']:.3f}, "
                    f"s2t_active={err_before['s2t_active_ratio']:.3f}, "
                    f"tangent_w={self.current_tangent_weight():.3f}, "
                    f"u_mean={u_mean_before:.6f}, "
                    f"u_max={u_max_before:.6f}, "
                    f"relax={self.relax:.6f}",
                    self.quiet
                )
            else:
                log(
                    f"[Iter {it:03d}][state] "
                    f"mean_err={mean_before:.6f}, "
                    f"mean_err_robust={mean_before_robust:.6f}, "
                    f"max_err={max_before:.6f}, "
                    f"active={err_before['t2s_active_ratio']:.3f}, "
                    f"tangent_w={self.current_tangent_weight():.3f}, "
                    f"u_mean={u_mean_before:.6f}, "
                    f"u_max={u_max_before:.6f}, "
                    f"relax={self.relax:.6f}",
                    self.quiet
                )

            if mean_before < stop_mean_err:
                hist.append(mean_before_robust)
                accepted_hist.append(mean_before_robust)
                stop_reason = f"mean_err<{stop_mean_err}"
                log(
                    f"[Stop] current sym_mean={mean_before:.6f} < stop_mean_err={stop_mean_err:.6f}",
                    self.quiet
                )
                break

            u_old = self.u.clone()
            u_solve, _ = self.propose_update()

            best = None
            relax_try = float(self.relax)

            for bt in range(self.max_backtracks + 1):
                cand_u = (1.0 - relax_try) * u_old + relax_try * u_solve

                if self.fixed_dofs.numel() > 0:
                    cand_u[self.fixed_dofs] = 0.0

                err_after = self.compute_error_on_u(cand_u)
                mean_after_robust = err_after["mean_sym_robust"]

                if (best is None) or (mean_after_robust < best["err"]["mean_sym_robust"]):
                    best = {
                        "err": err_after,
                        "relax_used": float(relax_try),
                        "u": cand_u.clone(),
                        "backtracks": int(bt),
                    }

                if mean_after_robust <= mean_before_robust:
                    break

                next_relax = relax_try * self.backtrack_factor
                if next_relax < self.min_relax:
                    break
                relax_try = next_relax

            accepted = (best is not None) and (best["err"]["mean_sym_robust"] <= mean_before_robust)

            if accepted:
                self.u = best["u"]
                self.refresh_current_linearization(force=True)
                self.relax = min(
                    self.relax_max,
                    max(self.min_relax, best["relax_used"] * self.relax_growth)
                )
                reject_streak = 0

                best_err = best["err"]
                hist.append(best_err["mean_sym_robust"])
                accepted_hist.append(best_err["mean_sym_robust"])

                u_mean_after, u_max_after = self.displacement_stats(self.u)

                if self.bidirectional:
                    log(
                        f"[Iter {it:03d}][accept] "
                        f"sym_mean: {mean_before:.6f} -> {best_err['mean_sym']:.6f}, "
                        f"sym_mean_robust: {mean_before_robust:.6f} -> {best_err['mean_sym_robust']:.6f}, "
                        f"sym_max: {max_before:.6f} -> {best_err['max_sym']:.6f}, "
                        f"t2s_mean={best_err['t2s_mean']:.6f}, "
                        f"s2t_mean={best_err['s2t_mean']:.6f}, "
                        f"t2s_active={best_err['t2s_active_ratio']:.3f}, "
                        f"s2t_active={best_err['s2t_active_ratio']:.3f}, "
                        f"u_mean={u_mean_after:.6f}, "
                        f"u_max={u_max_after:.6f}, "
                        f"relax_used={best['relax_used']:.6f}, "
                        f"next_relax={self.relax:.6f}, "
                        f"backtracks={best['backtracks']}",
                        self.quiet
                    )
                else:
                    log(
                        f"[Iter {it:03d}][accept] "
                        f"mean_err: {mean_before:.6f} -> {best_err['mean_sym']:.6f}, "
                        f"mean_err_robust: {mean_before_robust:.6f} -> {best_err['mean_sym_robust']:.6f}, "
                        f"max_err: {max_before:.6f} -> {best_err['max_sym']:.6f}, "
                        f"active={best_err['t2s_active_ratio']:.3f}, "
                        f"u_mean={u_mean_after:.6f}, "
                        f"u_max={u_max_after:.6f}, "
                        f"relax_used={best['relax_used']:.6f}, "
                        f"next_relax={self.relax:.6f}, "
                        f"backtracks={best['backtracks']}",
                        self.quiet
                    )
            else:
                self.u = u_old
                self.relax = max(self.min_relax, relax_try * self.backtrack_factor)
                hist.append(mean_before_robust)
                reject_streak += 1

                log(
                    f"[Iter {it:03d}][reject] "
                    f"sym_mean stays {mean_before:.6f}, "
                    f"sym_mean_robust stays {mean_before_robust:.6f}, "
                    f"relax reduced to {self.relax:.6f}, "
                    f"reject_streak={reject_streak}",
                    self.quiet
                )

                if reject_streak >= self.max_reject_streak and self.relax <= self.min_relax + 1e-12:
                    stop_reason = f"plateau_reject_streak>={self.max_reject_streak}"
                    log(
                        f"[Stop] plateau detected: reject_streak={reject_streak}, "
                        f"relax={self.relax:.6f} at/below min_relax={self.min_relax:.6f}",
                        self.quiet
                    )
                    break

            if it >= min_iters and len(accepted_hist) >= 2:
                delta_hist = abs(accepted_hist[-1] - accepted_hist[-2])
                if delta_hist < tol and reject_streak == 0:
                    stop_reason = f"delta_accepted_robust_mean<{tol}"
                    log(
                        f"[Stop] |Δaccepted sym_mean_robust|={delta_hist:.6e} < tol={tol:.6e}",
                        self.quiet
                    )
                    break

            if best is not None and best["err"]["mean_sym"] < stop_mean_err:
                stop_reason = f"mean_err<{stop_mean_err}"
                log(
                    f"[Stop] updated sym_mean={best['err']['mean_sym']:.6f} < stop_mean_err={stop_mean_err:.6f}",
                    self.quiet
                )
                break

        elapsed = time.time() - t0
        Vdef = self.current_volume_vertices().detach().cpu().numpy()
        Sdef = Vdef[self.surf_map.detach().cpu().numpy()]
        final_err = self.compute_error_on_u(self.u)

        return {
            "history": np.asarray(hist, dtype=np.float64),
            "elapsed_sec": float(elapsed),
            "stop_reason": stop_reason,
            "volume_vertices_final": Vdef,
            "surface_vertices_final": Sdef,
            "final_error": final_err,
        }


# =============================================================================
# Export
# =============================================================================
def export_mesh(path: str, vertices: np.ndarray, faces_tri: np.ndarray):
    faces_pv = np.hstack([np.full((faces_tri.shape[0], 1), 3, dtype=np.int64), faces_tri]).ravel()
    mesh = pv.PolyData(np.asarray(vertices, dtype=np.float64), faces_pv)
    mesh.save(path)

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
            w1, w2, w3 = sol[:, 0], sol[:, 1], sol[:, 2]
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


def tre_mean(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 3)
    if a.shape != b.shape:
        raise ValueError(f"TRE inputs shape mismatch: {a.shape} vs {b.shape}")
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


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

    stats = tre_stats(gt_txt_pts, src_txt_pts)
    log(
        f"[EVAL, BEFORE WARPING] TRE(mean)={stats['tre_mean']:.4f} mm, "
        f"TRE(std)={stats['tre_std']:.4f} mm, "
        f"TRE(median)={stats['tre_median']:.4f} mm, "
        f"TRE(max)={stats['tre_max']:.4f} mm, ",
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

    stats = tre_stats(gt_txt_pts, warped)
    log(
        f"[EVAL] TRE(mean)={stats['tre_mean']:.4f} mm, "
        f"TRE(std)={stats['tre_std']:.4f} mm, "
        f"TRE(median)={stats['tre_median']:.4f} mm, "
        f"TRE(max)={stats['tre_max']:.4f} mm, "
        f"RMSE={stats['rmse']:.4f} mm | saved -> {out_txt}",
        quiet
    )
    return stats


# =============================================================================
# Main
# =============================================================================
def parse_args(i):
    p = argparse.ArgumentParser("GPU organ biomechanical registration (corotational FEM + point-to-plane)")

    p.add_argument("--task", type=str, default="prostate", choices=["liver", "prostate"],
                   help="task preset: liver defaults to unidirectional, prostate defaults to bidirectional")

    p.add_argument("--source", type=str,
                   default=f"/home/data20tb/pipeline/Project/0Dataset-for-CAS2026/MU-REG/RIGID-MESH/{i}MR.ply",
                   help="pre-op source mesh: stl / obj / ply")
    p.add_argument("--target", type=str,
                   default=f"/home/data20tb/pipeline/Project/0Dataset-for-CAS2026/MU-REG/RIGID-MESH/{i}US.ply",
                   help="intra-op target surface: stl / obj / ply")

    # p.add_argument("--source", type=str,
    #                default=f"/home/data20tb/pipeline/Project/0Dataset-for-CAS2026/opencas/simulation/{i}/source{i}.ply",
    #                help="pre-op source mesh: stl / obj / ply")
    # p.add_argument("--target", type=str,
    #                default=f"/home/data20tb/pipeline/Project/0Dataset-for-CAS2026/opencas/simulation/{i}/target{i}.ply",
    #                help="intra-op target partial surface: stl / obj / ply")
    p.add_argument("--source-txt", type=str,
                   default=f"/home/data20tb/pipeline/Project/0Dataset-for-CAS2026/MU-REG/RIGID-POINTS/{i}mr.txt",
                   help="source landmark / evaluation points txt")
    p.add_argument("--gt-txt", type=str,
                   default=f"/home/data20tb/pipeline/Project/0Dataset-for-CAS2026/MU-REG/RIGID-POINTS/{i}us.txt",
                   help="ground-truth target landmark / evaluation points txt")
    
    p.add_argument("--warp-chunk", type=int, default=200000)

    p.add_argument("--GT", type=str,
                   default=f"",
                   help="ground truth deformed surface: stl / obj / ply")

    p.add_argument("--outdir", type=str, default=f"./MUREG-optinua/{i}")

    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])

    p.add_argument("--bidirectional", action=argparse.BooleanOptionalAction, default=None,
                   help="override task default. liver default=False, prostate default=True")
    p.add_argument("--w-t2s", type=float, default=1.0,
                   help="weight for target -> surface direction")
    p.add_argument("--w-s2t", type=float, default=1.0,
                   help="weight for surface -> target direction")

    # Blender repair
    p.add_argument("--blender-repair-source", action=argparse.BooleanOptionalAction, default=True,
                   help="Repair source mesh with Blender before TetGen")
    p.add_argument("--blender-bin", type=str, default="blender",
                   help="Blender executable name or absolute path")
    p.add_argument("--blender-use-cache", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--blender-merge-dist", type=float, default=0.0,
                   help="Merge close vertices before remesh, in mesh units")
    p.add_argument("--blender-voxel-size", type=float, default=2, #1.5
                   help="Voxel remesh size, in mesh units; smaller = denser")
    p.add_argument("--blender-smooth-iters", type=int, default=5)
    p.add_argument("--blender-smooth-factor", type=float, default=0.2)
    p.add_argument("--blender-decimate-ratio", type=float, default=1.0, #1.0
                   help="1.0 means no decimation")
    p.add_argument("--blender-no-apply-transform", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--blender-triangulate", action=argparse.BooleanOptionalAction, default=True)

    # source->volume
    p.add_argument("--mindihedral", type=float, default=10.0)
    p.add_argument("--minratio", type=float, default=1.5)
    p.add_argument("--maxvolume", type=float, default=0.0)
    p.add_argument("--match-tol", type=float, default=1e-5)

    # material
    p.add_argument("--young", type=float, default=2000.0)
    # p.add_argument("--young", type=float, default=0.05)
    p.add_argument("--poisson", type=float, default=0.45)

    # registration
    p.add_argument("--max-iters", type=int, default=200)
    p.add_argument("--min-iters", type=int, default=60)
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--stop-mean-err", type=float, default=1e-3)

    # data term / force
    p.add_argument("--penalty-k", type=float, default=100)
    p.add_argument("--penalty-scale-s", type=float, default=0.05)
    p.add_argument("--robust-sigma", type=float, default=20.0)
    p.add_argument("--max-force", type=float, default=5000.0)
    p.add_argument("--target-sample-count", type=int, default=12000)

    p.add_argument("--data-stiffness-scale", type=float, default=0.25,
                   help="scale of approximate data-term stiffness added to linear solve")

    # mixed point-to-point -> point-to-plane schedule
    p.add_argument("--tangent-weight", type=float, default=0.20,
                   help="final tangent weight; 0 = point-to-plane, 1 = point-to-point")
    p.add_argument("--tangent-weight-start", type=float, default=0.60,
                   help="initial tangent weight for early iterations")
    p.add_argument("--tangent-anneal-iters", type=int, default=40,
                   help="anneal tangent weight from start to final in first N iterations")

    p.add_argument("--trim-quantile", type=float, default=0.99,
                   help="keep only correspondences within this distance quantile; set 1.0 to disable")
    p.add_argument("--trim-max-dist", type=float, default=50,
                   help="absolute max correspondence distance kept for force; <=0 disables")
    p.add_argument("--trim-normal-cos", type=float, default=0.0,
                   help="keep matches with |dot(n_src,n_tgt)| >= this threshold; <=0 disables")

    # update / backtracking
    p.add_argument("--relax", type=float, default=0.15)
    p.add_argument("--relax-max", type=float, default=0.35)
    p.add_argument("--relax-growth", type=float, default=1.05)
    p.add_argument("--backtrack-factor", type=float, default=0.5)
    p.add_argument("--min-relax", type=float, default=0.005)
    p.add_argument("--max-backtracks", type=int, default=8)
    p.add_argument("--max-delta-u", type=float, default=2.5,
                   help="cap nodal increment norm per iteration, in mesh units")
    p.add_argument("--vol-preserve-k", type=float, default=2.0,
                   help="quadratic volume-preservation stiffness on tet Jacobian J")
    p.add_argument("--vol-barrier-k", type=float, default=10.0,
                   help="extra barrier stiffness when J drops below --vol-j-min")
    p.add_argument("--vol-j-min", type=float, default=0.35,
                   help="compression barrier activates when tet J < this threshold")
    p.add_argument("--surface-smooth-k", type=float, default=2.0,
                   help="uncovered-region surface displacement smoothness weight")
    p.add_argument("--cover-dist-start", type=float, default=3.0,
                   help="surface->target distance where uncovered regularization starts to activate")
    p.add_argument("--cover-dist-end", type=float, default=12.0,
                   help="surface->target distance where uncovered regularization reaches full strength")
    p.add_argument("--tangent-refresh-every", type=int, default=1,
                   help="refresh current-configuration tangent every N iterations")
    p.add_argument("--max-reject-streak", type=int, default=8,
                   help="stop early when stuck at min_relax and repeatedly rejected")

    # boundary
    p.add_argument("--boundary-mode", type=str, default="none",
                   choices=["auto_posterior", "percentile_min", "percentile_max", "none"])
    p.add_argument("--boundary-axis", type=str, default="z", choices=["x", "y", "z"])
    p.add_argument("--boundary-percentile", type=float, default=0.5)
    p.add_argument("--fix-penalty-scale", type=float, default=1e3)

    # runtime
    p.add_argument("--pcg-tol", type=float, default=1e-5)
    p.add_argument("--pcg-max-iter", type=int, default=500)
    p.add_argument("--tet-chunk", type=int, default=65536)
    p.add_argument("--knn-p-chunk", type=int, default=4096)
    p.add_argument("--knn-q-chunk", type=int, default=32768)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


import sys
from contextlib import contextmanager
class _TeeIO:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return False

@contextmanager
def tee_stdout_stderr(log_path: str, mode: str = "w"):
    log_dir = os.path.dirname(os.path.abspath(log_path))
    os.makedirs(log_dir, exist_ok=True)
    old_out, old_err = sys.stdout, sys.stderr
    f = open(log_path, mode, buffering=1, encoding="utf-8")
    try:
        sys.stdout = _TeeIO(old_out, f)
        sys.stderr = _TeeIO(old_err, f)
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        f.close()

def main():
    for i in range(75, 108):
        t0 = time.time()
        args = parse_args(i)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        ensure_dir(args.outdir)

        log_path = os.path.join(args.outdir, "log.txt")
        with tee_stdout_stderr(log_path, mode="w"):
            quiet = args.quiet
            use_bidirectional = resolve_bidirectional(args.task, args.bidirectional)

            log(
                f"[INFO] task={args.task}, bidirectional={use_bidirectional}, "
                f"w_t2s={args.w_t2s}, w_s2t={args.w_s2t}, fem_model=corotational+vol+surface_smooth, "
                f"data_term=point-to-plane",
                quiet
            )

            source_mesh_for_reg = args.source
            if args.blender_repair_source:
                repaired_source = os.path.join(args.outdir, "source_repaired.ply")
                log(f"[INFO] Blender repair enabled for source: {args.source} -> {repaired_source}", quiet)

                source_mesh_for_reg = run_blender_repair(
                    inp_mesh=args.source,
                    out_mesh=repaired_source,
                    blender_bin=args.blender_bin,
                    merge_dist=args.blender_merge_dist,
                    voxel_size=args.blender_voxel_size,
                    smooth_iters=args.blender_smooth_iters,
                    smooth_factor=args.blender_smooth_factor,
                    decimate_ratio=args.blender_decimate_ratio,
                    no_apply_transform=args.blender_no_apply_transform,
                    triangulate=args.blender_triangulate,
                    use_cache=args.blender_use_cache,
                )

            # 1) source
            src = load_surface_mesh(source_mesh_for_reg, quiet=quiet)
            src_pts = np.asarray(src.points, dtype=np.float64)
            src_faces_tri = polydata_to_faces_tri(src)

            # 2) volume mesh
            log("[INFO] TetGen tetrahedralization ...", quiet)
            vol_pts, vol_tets = tetrahedralize_with_tetgen(
                surf=src,
                mindihedral=args.mindihedral,
                minratio=args.minratio,
                maxvolume=args.maxvolume,
                quiet=quiet,
            )
            log(f"[OK] volume V={vol_pts.shape[0]}, T={vol_tets.shape[0]}", quiet)

            # 3) map source surface vertices -> volume vertices
            surf_map = match_surface_points_to_volume(src_pts, vol_pts, tol=args.match_tol)

            # 4) target points + normals
            target_points, target_normals = load_target_points_with_normals(
                args.target, max_points=args.target_sample_count, quiet=quiet
            )

            # 4b) Load Ground Truth if provided
            gt_mesh = None
            if args.GT and os.path.exists(args.GT):
                log(f"[INFO] Loading Ground Truth: {args.GT}", quiet)
                gt_mesh = pv.read(args.GT)
                if not isinstance(gt_mesh, pv.PolyData):
                    gt_mesh = gt_mesh.extract_surface()
            else:
                log(f"[WARN] GT path not found or not provided: {args.GT}", quiet)

            # 5) boundary
            fixed_vertex_ids = auto_fixed_vertex_ids(
                vertices=vol_pts,
                mode=args.boundary_mode,
                percentile=args.boundary_percentile,
                axis=args.boundary_axis,
            )
            log(f"[Boundary] fixed vertices = {len(fixed_vertex_ids)}", quiet)

            # 6) registration
            reg = LiverFEMGPUReg(
                vol_points=vol_pts,
                vol_tets=vol_tets,
                surf_map=surf_map,
                src_faces_tri=src_faces_tri,
                target_points=target_points,
                target_normals=target_normals,
                bidirectional=use_bidirectional,
                w_t2s=args.w_t2s,
                w_s2t=args.w_s2t,
                fixed_vertex_ids=fixed_vertex_ids,
                young=args.young,
                poisson=args.poisson,
                penalty_k=args.penalty_k,
                penalty_scale_s=args.penalty_scale_s,
                robust_sigma=args.robust_sigma,
                max_force=args.max_force,
                data_stiffness_scale=args.data_stiffness_scale,
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
                surface_smooth_k=args.surface_smooth_k,
                cover_dist_start=args.cover_dist_start,
                cover_dist_end=args.cover_dist_end,
                tangent_refresh_every=args.tangent_refresh_every,
                fix_penalty_scale=args.fix_penalty_scale,
                relax=args.relax,
                relax_max=args.relax_max,
                relax_growth=args.relax_growth,
                backtrack_factor=args.backtrack_factor,
                min_relax=args.min_relax,
                max_backtracks=args.max_backtracks,
                max_reject_streak=args.max_reject_streak,
                device=args.device,
                dtype=args.dtype,
                pcg_tol=args.pcg_tol,
                pcg_max_iter=args.pcg_max_iter,
                tet_chunk=args.tet_chunk,
                knn_p_chunk=args.knn_p_chunk,
                knn_q_chunk=args.knn_q_chunk,
                quiet=quiet,
            )

            result = reg.fit(
                max_iters=args.max_iters,
                tol=args.tol,
                min_iters=args.min_iters,
                stop_mean_err=args.stop_mean_err,
            )

            # # 7) export
            # out_surface_obj = os.path.join(args.outdir, "deformed_surface.obj")
            # out_surface_ply = os.path.join(args.outdir, "deformed_surface.ply")
            # out_volume_vtu = os.path.join(args.outdir, "deformed_volume.vtu")
            # out_hist = os.path.join(args.outdir, "history.npy")
            # out_target = os.path.join(args.outdir, "target_points.npy")
            # out_meta = os.path.join(args.outdir, "run_meta.json")

            # export_mesh(out_surface_obj, result["surface_vertices_final"], src_faces_tri)
            # export_mesh(out_surface_ply, result["surface_vertices_final"], src_faces_tri)

            # ---------------------------------------------------------
            # 7) export
            # ---------------------------------------------------------
            out_surface_obj = os.path.join(args.outdir, "deformed_surface.obj")
            out_surface_ply = os.path.join(args.outdir, "deformed_surface.ply")
            out_gt_comp_ply = os.path.join(args.outdir, "ground_truth_comparison.ply")
            out_volume_vtu = os.path.join(args.outdir, "deformed_volume.vtu")
            out_hist = os.path.join(args.outdir, "history.npy")
            out_target = os.path.join(args.outdir, "target_points.npy")
            out_meta = os.path.join(args.outdir, "run_meta.json")

            # 正常导出变形后的单体表面
            res_pts = result["surface_vertices_final"]
            export_mesh(out_surface_obj, res_pts, src_faces_tri)
            export_mesh(out_surface_ply, res_pts, src_faces_tri)

            # 合并保存进 ground_truth_comparison.ply
            rmse = None
            if gt_mesh is not None:
                gt_pts = np.asarray(gt_mesh.points, dtype=np.float64)
                
                # 创建一个合并后的 PyVista 对象
                # 1. 变形后的结果 (Label = 1)
                faces_pv = np.hstack([np.full((src_faces_tri.shape[0], 1), 3, dtype=np.int64), src_faces_tri]).ravel()
                mesh_res = pv.PolyData(res_pts, faces_pv)
                mesh_res.point_data["ModelType"] = np.ones(res_pts.shape[0], dtype=np.int32) # 1 代表结果
                
                # 2. GT 结果 (Label = 2)
                # 注意：GT 的面片结构可能与 source 不同，所以直接使用 gt_mesh 本身的 faces
                mesh_gt = gt_mesh.copy()
                mesh_gt.point_data["ModelType"] = np.full(mesh_gt.n_points, 2, dtype=np.int32) # 2 代表 GT
                
                # 3. 合并两者
                combined = mesh_res.merge(mesh_gt)
                combined.save(out_gt_comp_ply)
                log(f"[Done] Combined comparison mesh: {out_gt_comp_ply}", quiet)

                # 计算 RMSE (如果顶点数一致)
                if gt_pts.shape == res_pts.shape:
                    rmse = np.sqrt(np.mean(np.sum((res_pts - gt_pts)**2, axis=1)))
                    log(f"[STAT] Final RMSE vs GT: {rmse:.6f}", quiet)
                else:
                    log("[WARN] GT and Result vertex counts differ, skipping direct RMSE calculation.", quiet)

            # # 导出 GT 副本以便对比
            # if gt_mesh is not None:
            #     out_gt_ply = os.path.join(args.outdir, "ground_truth_comparison.ply")
            #     gt_mesh.save(out_gt_ply)
            #     log(f"[Done] GT mesh: {out_gt_ply}", quiet)

            #     # 可选：如果你想在 metadata 中计算并记录最终结果与 GT 的 RMSE
            #     # 这里假设 GT 的顶点顺序与 Result 的表面顶点顺序一致 (常见于仿真数据)
            #     gt_pts = np.asarray(gt_mesh.points, dtype=np.float64)
            #     res_pts = result["surface_vertices_final"]
                
            #     if gt_pts.shape == res_pts.shape:
            #         rmse = np.sqrt(np.mean(np.sum((res_pts - gt_pts)**2, axis=1)))
            #         log(f"[STAT] Final RMSE vs GT: {rmse:.6f}", quiet)
            #         # 后面可以在 json.dump 中加入这个值
            #     else:
            #         log("[WARN] GT and Result vertex counts differ, skipping direct RMSE calculation.", quiet)

            grid_def = make_tet_ugrid(result["volume_vertices_final"], vol_tets)
            grid_def.save(out_volume_vtu)

            np.save(out_hist, result["history"])
            np.save(out_target, target_points)

            grid_src = make_tet_ugrid(vol_pts, vol_tets)

            tre_eval = maybe_eval_tre(
                source_txt=args.source_txt,
                gt_txt=args.gt_txt,
                grid_src=grid_src,
                grid_def=grid_def,
                device=args.device,
                dtype=args.dtype,
                warp_chunk=args.warp_chunk,
                out_dir=args.outdir,
                quiet=quiet,
            )

            final_err = result["final_error"]

            with open(out_meta, "w", encoding="utf-8") as f:
                json.dump({
                    "task": args.task,
                    "source_original": os.path.abspath(args.source),
                    "source_used_for_registration": os.path.abspath(source_mesh_for_reg),
                    "target": os.path.abspath(args.target),
                    "device": args.device,
                    "dtype": args.dtype,
                    "bidirectional": bool(use_bidirectional),
                    "fem_model": "corotational",
                    "data_term": "point_to_plane_trimmed",
                    "num_source_surface_vertices": int(src_pts.shape[0]),
                    "num_proxy_vertices": int(vol_pts.shape[0]),
                    "num_proxy_tets": int(vol_tets.shape[0]),
                    "num_target_points": int(target_points.shape[0]),
                    "num_fixed_vertices": int(len(fixed_vertex_ids)),
                    "elapsed_sec": float(result["elapsed_sec"]),
                    "stop_reason": result.get("stop_reason", None),

                    "final_symmetric_mean_error": float(final_err["mean_sym"]),
                    "final_symmetric_max_error": float(final_err["max_sym"]),
                    "final_t2s_mean_error": float(final_err["t2s_mean"]),
                    "final_t2s_max_error": float(final_err["t2s_max"]),
                    "final_s2t_mean_error": float(final_err["s2t_mean"]),
                    "final_s2t_max_error": float(final_err["s2t_max"]),

                    "final_symmetric_mean_error_robust": float(final_err["mean_sym_robust"]),
                    "final_t2s_mean_error_robust": float(final_err["t2s_mean_robust"]),
                    "vol_preserve_k": float(args.vol_preserve_k),
                    "vol_barrier_k": float(args.vol_barrier_k),
                    "vol_j_min": float(args.vol_j_min),
                    "surface_smooth_k": float(args.surface_smooth_k),
                    "cover_dist_start": float(args.cover_dist_start),
                    "cover_dist_end": float(args.cover_dist_end),
                    "tangent_refresh_every": int(args.tangent_refresh_every),
                    "final_min_J": float(reg.compute_current_J().min().item()),
                    "final_s2t_mean_error_robust": float(final_err["s2t_mean_robust"]),

                    "final_t2s_active_ratio": float(final_err["t2s_active_ratio"]),
                    "final_s2t_active_ratio": float(final_err["s2t_active_ratio"]),

                    # 在 json.dump 的字典里添加：
                    "gt_original": os.path.abspath(args.GT) if args.GT else None,
                    # "rmse_vs_gt": float(rmse) if 'rmse' in locals() else None,
                    "tre_eval": tre_eval,

                    "config": {
                        "young": args.young,
                        "poisson": args.poisson,
                        "mindihedral": args.mindihedral,
                        "minratio": args.minratio,
                        "maxvolume": args.maxvolume,
                        "match_tol": args.match_tol,

                        "max_iters": args.max_iters,
                        "min_iters": args.min_iters,
                        "tol": args.tol,
                        "stop_mean_err": args.stop_mean_err,

                        "penalty_k": args.penalty_k,
                        "penalty_scale_s": args.penalty_scale_s,
                        "robust_sigma": args.robust_sigma,
                        "max_force": args.max_force,
                        "target_sample_count": args.target_sample_count,

                        "data_stiffness_scale": args.data_stiffness_scale,
                        "tangent_weight": args.tangent_weight,
                        "tangent_weight_start": args.tangent_weight_start,
                        "tangent_anneal_iters": args.tangent_anneal_iters,
                        "trim_quantile": args.trim_quantile,
                        "trim_max_dist": args.trim_max_dist,
                        "trim_normal_cos": args.trim_normal_cos,

                        "relax": args.relax,
                        "relax_max": args.relax_max,
                        "relax_growth": args.relax_growth,
                        "backtrack_factor": args.backtrack_factor,
                        "min_relax": args.min_relax,
                        "max_backtracks": args.max_backtracks,
                        "max_delta_u": args.max_delta_u,
                        "max_reject_streak": args.max_reject_streak,

                        "boundary_mode": args.boundary_mode,
                        "boundary_axis": args.boundary_axis,
                        "boundary_percentile": args.boundary_percentile,
                        "fix_penalty_scale": args.fix_penalty_scale,

                        "pcg_tol": args.pcg_tol,
                        "pcg_max_iter": args.pcg_max_iter,
                        "tet_chunk": args.tet_chunk,
                        "knn_p_chunk": args.knn_p_chunk,
                        "knn_q_chunk": args.knn_q_chunk,

                        "task": args.task,
                        "bidirectional_arg": args.bidirectional,
                        "bidirectional_resolved": use_bidirectional,
                        "w_t2s": args.w_t2s,
                        "w_s2t": args.w_s2t,

                        "blender_repair_source": args.blender_repair_source,
                        "blender_bin": args.blender_bin,
                        "blender_use_cache": args.blender_use_cache,
                        "blender_merge_dist": args.blender_merge_dist,
                        "blender_voxel_size": args.blender_voxel_size,
                        "blender_smooth_iters": args.blender_smooth_iters,
                        "blender_smooth_factor": args.blender_smooth_factor,
                        "blender_decimate_ratio": args.blender_decimate_ratio,
                        "blender_no_apply_transform": args.blender_no_apply_transform,
                        "blender_triangulate": args.blender_triangulate,
                    }
                }, f, indent=2)

            log(f"[Done] mesh:   {out_surface_obj}", quiet)
            log(f"[Done] mesh:   {out_surface_ply}", quiet)
            log(f"[Done] volume: {out_volume_vtu}", quiet)
            log(f"[Done] meta:   {out_meta}", quiet)
            t1 = time.time()
            print(f"\n[Time] Total runtime: {t1 - t0:.2f} s")

if __name__ == "__main__":
    t0 = time.time()

    main()

    t1 = time.time()
    print(f"\n[Time] Total runtime: {t1 - t0:.2f} s")
