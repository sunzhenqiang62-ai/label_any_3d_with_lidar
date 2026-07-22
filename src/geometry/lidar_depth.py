"""
Build depth_map / PLY / cam_params from world-frame LiDAR + camera calibration.

Fuse optimizations (LiDAR + vision):
1. Soft confidence blend at LiDAR boundaries (no hard cut)
2. Distance-banded RANSAC / median scale alignment
3. Robust rasterize (foreground-cluster median) + neighborhood densify
4. Semantic priors (sky / ground / object) for fusion weights
5. Optional surround multi-view depth consistency
6. Edge-aware hole fill (RGB + depth gradients)
7. Small projection (du, dv) calibration refine against image edges
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import trimesh
from scipy.ndimage import (
    distance_transform_edt,
    maximum_filter,
    minimum_filter,
)


def load_calib(calib_path):
    """
    Load camera intrinsics and extrinsics from JSON.

    Required: K (3x3), width/height (or W/H).
    Extrinsics: world_to_cam (4x4) OR c2w (4x4, inverted to world_to_cam).
    Optional: sensor_to_world (4x4) if point cloud is in sensor frame.
    """
    with open(calib_path, "r") as f:
        data = json.load(f)

    K = np.array(data["K"], dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"K must be 3x3, got {K.shape}")

    if "world_to_cam" in data:
        world_to_cam = np.array(data["world_to_cam"], dtype=np.float64)
        if world_to_cam.shape != (4, 4):
            raise ValueError("world_to_cam must be 4x4")
        c2w = np.linalg.inv(world_to_cam)
    elif "c2w" in data:
        c2w = np.array(data["c2w"], dtype=np.float64)
        if c2w.shape != (4, 4):
            raise ValueError("c2w must be 4x4")
        world_to_cam = np.linalg.inv(c2w)
    else:
        raise ValueError("calib JSON must contain 'world_to_cam' or 'c2w'")

    W = int(data.get("width", data.get("W")))
    H = int(data.get("height", data.get("H")))
    if W <= 0 or H <= 0:
        raise ValueError("calib must specify positive width/height")

    sensor_to_world = None
    if "sensor_to_world" in data:
        sensor_to_world = np.array(data["sensor_to_world"], dtype=np.float64)

    points_in_sensor_frame = bool(data.get("points_in_sensor_frame", False))

    return {
        "K": K,
        "world_to_cam": world_to_cam,
        "c2w": c2w,
        "W": W,
        "H": H,
        "sensor_to_world": sensor_to_world,
        "points_in_sensor_frame": points_in_sensor_frame,
    }


def load_pointcloud(path):
    """Load Nx3 world (or sensor) points and optional Nx3 colors."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".npz":
        data = np.load(path)
        if "points" not in data:
            raise KeyError(f"{path} must contain 'points' array")
        points = np.asarray(data["points"], dtype=np.float64).reshape(-1, 3)
        colors = np.asarray(data["colors"], dtype=np.float64) if "colors" in data else None
        if colors is not None and colors.shape[0] != points.shape[0]:
            raise ValueError("colors length must match points")
        return points, colors

    loaded = trimesh.load(str(path))
    if isinstance(loaded, trimesh.Scene):
        chunks = []
        for geom in loaded.geometry.values():
            if hasattr(geom, "vertices") and len(geom.vertices) > 0:
                chunks.append(np.asarray(geom.vertices, dtype=np.float64))
        points = np.vstack(chunks) if chunks else np.empty((0, 3), dtype=np.float64)
        colors = None
    else:
        points = np.asarray(loaded.vertices, dtype=np.float64).reshape(-1, 3)
        colors = None
        if (
            hasattr(loaded, "visual")
            and loaded.visual is not None
            and hasattr(loaded.visual, "vertex_colors")
            and loaded.visual.vertex_colors is not None
        ):
            vc = np.asarray(loaded.visual.vertex_colors)
            if len(vc) == len(points):
                colors = vc[:, :3].astype(np.float64)

    return points, colors


def transform_points(points, transform_4x4):
    """Apply 4x4 homogeneous transform to Nx3 points."""
    if points.size == 0:
        return points.reshape(0, 3)
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    homogeneous = np.hstack([points, ones])
    return (transform_4x4 @ homogeneous.T).T[:, :3]


def project_points(points_cam, K, du: float = 0.0, dv: float = 0.0):
    """Project camera-frame points to pixel coordinates and depth."""
    X, Y, Z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
    valid = Z > 1e-6
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * X[valid] / Z[valid] + cx + du
    v = fy * Y[valid] / Z[valid] + cy + dv
    z = Z[valid]
    return u, v, z, valid


def _image_edge_magnitude(image_np: np.ndarray) -> np.ndarray:
    """Cheap RGB edge strength in [0, 1]."""
    if image_np.ndim == 3:
        gray = (
            0.299 * image_np[:, :, 0].astype(np.float32)
            + 0.587 * image_np[:, :, 1].astype(np.float32)
            + 0.114 * image_np[:, :, 2].astype(np.float32)
        )
    else:
        gray = image_np.astype(np.float32)
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    mag = np.sqrt(gx * gx + gy * gy)
    p99 = float(np.percentile(mag, 99)) if mag.size else 1.0
    return np.clip(mag / max(p99, 1e-3), 0.0, 1.0)


def refine_projection_shift(
    points_cam: np.ndarray,
    K: np.ndarray,
    image_np: np.ndarray,
    max_shift: int = 2,
) -> Tuple[float, float]:
    """
    Search a small (du, dv) that best aligns projected LiDAR hits with image edges.
    Returns best pixel shift applied to the projection.
    """
    if max_shift <= 0 or points_cam.shape[0] == 0 or image_np is None:
        return 0.0, 0.0

    H, W = image_np.shape[:2]
    edges = _image_edge_magnitude(image_np)
    # Subsample points for speed.
    pts = points_cam
    if pts.shape[0] > 80000:
        rng = np.random.default_rng(0)
        pts = pts[rng.choice(pts.shape[0], 80000, replace=False)]

    best_score = -1.0
    best = (0.0, 0.0)
    for dv in range(-max_shift, max_shift + 1):
        for du in range(-max_shift, max_shift + 1):
            u, v, z, _ = project_points(pts, K, du=float(du), dv=float(dv))
            ui = np.round(u).astype(np.int32)
            vi = np.round(v).astype(np.int32)
            in_bounds = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H) & (z > 0.5)
            if in_bounds.sum() < 50:
                continue
            # Prefer hits that land on image edges (occlusion / object boundaries).
            score = float(edges[vi[in_bounds], ui[in_bounds]].mean())
            # Slight preference for zero shift to avoid jitter.
            score -= 0.01 * (abs(du) + abs(dv))
            if score > best_score:
                best_score = score
                best = (float(du), float(dv))
    return best


def rasterize_depth(
    points_cam,
    K,
    H,
    W,
    *,
    mode: str = "median",
    densify_radius: int = 1,
    du: float = 0.0,
    dv: float = 0.0,
    return_count: bool = False,
):
    """
    Z-buffer rasterization with optional robust aggregation and densify.

    mode:
      - "min": classic nearest surface (minimum Z)
      - "median": foreground cluster around min-Z (trimmed mean ≈ median)

    densify_radius: fill empty pixels from nearest LiDAR neighbor within radius.
    Returns depth (H,W) with inf at empty pixels, and valid_mask.
    """
    depth = np.full((H, W), np.inf, dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)
    if points_cam.shape[0] == 0:
        valid_mask = np.zeros((H, W), dtype=bool)
        if return_count:
            return depth, valid_mask, count
        return depth, valid_mask

    u, v, z, _ = project_points(points_cam, K, du=du, dv=dv)
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    in_bounds = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui, vi, z = ui[in_bounds], vi[in_bounds], z[in_bounds].astype(np.float32)
    if ui.size == 0:
        valid_mask = np.zeros((H, W), dtype=bool)
        if return_count:
            return depth, valid_mask, count
        return depth, valid_mask

    flat_idx = vi * W + ui
    depth_flat = depth.ravel()
    np.minimum.at(depth_flat, flat_idx, z)
    depth = depth_flat.reshape(H, W)

    if mode == "median":
        # Accumulate points belonging to the near surface cluster at each pixel.
        z_min = depth[vi, ui]
        near = z <= (z_min * 1.03 + 0.2)
        sum_z = np.zeros((H * W,), dtype=np.float64)
        cnt_cluster = np.zeros((H * W,), dtype=np.float64)
        hit_cnt = np.zeros((H * W,), dtype=np.float64)
        np.add.at(sum_z, flat_idx[near], z[near].astype(np.float64))
        np.add.at(cnt_cluster, flat_idx[near], 1.0)
        np.add.at(hit_cnt, flat_idx, 1.0)
        count = hit_cnt.reshape(H, W).astype(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_z = (sum_z / np.maximum(cnt_cluster, 1.0)).reshape(H, W)
        has_cluster = cnt_cluster.reshape(H, W) >= 1
        depth = np.where(has_cluster, mean_z, depth).astype(np.float32)
    else:
        hit_cnt = np.zeros((H * W,), dtype=np.float64)
        np.add.at(hit_cnt, flat_idx, 1.0)
        count = hit_cnt.reshape(H, W).astype(np.float32)

    valid_mask = np.isfinite(depth) & (depth < np.inf)

    if densify_radius and densify_radius > 0 and valid_mask.any() and (~valid_mask).any():
        # Spread valid depths into nearby empty pixels (min of neighborhood).
        k = 2 * int(densify_radius) + 1
        depth_fill = depth.copy()
        depth_fill[~valid_mask] = np.inf
        neigh_min = minimum_filter(depth_fill, size=k, mode="nearest")
        neigh_has = maximum_filter(valid_mask.astype(np.uint8), size=k, mode="nearest") > 0
        fill = (~valid_mask) & neigh_has & np.isfinite(neigh_min) & (neigh_min < np.inf)
        depth[fill] = neigh_min[fill]
        count[fill] = np.maximum(count[fill], 0.35)
        valid_mask = np.isfinite(depth) & (depth < np.inf)

    if return_count:
        return depth, valid_mask, count
    return depth, valid_mask


def lidar_confidence_map(
    valid_mask: np.ndarray,
    count: Optional[np.ndarray] = None,
    blend_radius: int = 3,
) -> np.ndarray:
    """
    Soft confidence in [0, 1]: 1 at dense LiDAR cores, decaying toward hole edges.
    """
    valid = np.asarray(valid_mask, dtype=bool)
    conf = np.zeros(valid.shape, dtype=np.float32)
    if not valid.any():
        return conf

    # Distance to nearest invalid pixel → high inside LiDAR islands.
    dist_in = distance_transform_edt(valid).astype(np.float32)
    # Distance to nearest valid pixel → used for soft ring outside hits.
    dist_out = distance_transform_edt(~valid).astype(np.float32)

    r = max(1.0, float(blend_radius))
    # Interior: ramp up over ~blend_radius pixels from the LiDAR boundary.
    interior = np.clip(dist_in / r, 0.0, 1.0)
    # Exterior soft skirt (for optional blend into vision just outside hits).
    exterior = np.clip(1.0 - dist_out / r, 0.0, 1.0)
    conf = np.where(valid, 0.55 + 0.45 * interior, 0.35 * exterior).astype(np.float32)

    if count is not None:
        c = np.asarray(count, dtype=np.float32)
        c_n = np.clip(c / 3.0, 0.0, 1.0)
        conf = np.clip(conf * (0.65 + 0.35 * c_n), 0.0, 1.0)

    return conf


def estimate_semantic_priors(image_np: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Lightweight RGB priors (no VLM): sky / ground / object-ish regions.
    Used to modulate LiDAR vs vision trust.
    """
    H, W = image_np.shape[:2]
    img = image_np.astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    bright = (r + g + b) / 3.0
    # Sky-ish: upper image, bright, blue-ish or low saturation gray.
    yy = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]
    blue_dom = (b > r + 0.02) & (b > g)
    low_sat = (np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)) < 0.08
    sky = (yy < 0.42) & (bright > 0.45) & (blue_dom | low_sat)
    # Ground-ish: lower image, darker / textured less blue.
    ground = (yy > 0.55) & (bright < 0.65) & (~blue_dom)
    edges = _image_edge_magnitude(image_np)
    objectish = edges > 0.25
    return {
        "sky": sky.astype(bool),
        "ground": ground.astype(bool),
        "object": objectish.astype(bool),
        "edge": edges.astype(np.float32),
    }


def edge_aware_fill(
    depth: np.ndarray,
    seed_mask: np.ndarray,
    image_np: Optional[np.ndarray] = None,
    max_radius: int = 24,
    depth_jump: float = 0.08,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fill invalid pixels from nearest seed, but stop across strong RGB/depth edges.
    """
    depth_out = depth.copy().astype(np.float32)
    valid = np.asarray(seed_mask, dtype=bool) & np.isfinite(depth_out) & (depth_out > 1e-6)
    if not valid.any() or valid.all():
        return depth_out, valid

    if image_np is not None:
        edge = _image_edge_magnitude(image_np)
    else:
        edge = np.zeros(depth_out.shape, dtype=np.float32)

    # EDT nearest seed index.
    _, ind = distance_transform_edt(~valid, return_distances=True, return_indices=True)
    src_y, src_x = ind[0], ind[1]
    dist = distance_transform_edt(~valid).astype(np.float32)
    yy, xx = np.indices(depth_out.shape)
    cand = (~valid) & (dist <= float(max_radius))
    if not cand.any():
        return depth_out, valid

    src_d = depth_out[src_y, src_x]
    # Edge cost along the straight path approximated by max edge at dest/src.
    edge_block = (edge > 0.55) | (edge[src_y, src_x] > 0.55)
    # Relative depth jump between neighbor seeds and candidate (use local 3x3 max).
    local_max = maximum_filter(np.where(valid, depth_out, 0.0), size=3)
    local_min = minimum_filter(np.where(valid, depth_out, 1e6), size=3)
    jump = (local_max - local_min) / np.maximum(local_min, 1e-3)
    jump_block = jump > depth_jump * 4.0

    allow = cand & (~edge_block) & (~jump_block) & np.isfinite(src_d) & (src_d > 1e-6)
    # Soften: also reject if relative |src_d - median nearby| huge — skip for speed.
    depth_out[allow] = src_d[allow]
    new_valid = valid | allow
    return depth_out, new_valid


def fill_depth_holes(depth, valid_mask, method="nearest"):
    """Fill invalid depth pixels using nearest-neighbor propagation."""
    if method == "none" or valid_mask.all():
        return depth.copy(), valid_mask.copy()

    depth_out = depth.copy().astype(np.float32)
    invalid = ~valid_mask
    if not invalid.any():
        return depth_out, valid_mask

    fill_src = np.where(valid_mask, depth_out, np.nan)
    ind = distance_transform_edt(np.isnan(fill_src), return_distances=False, return_indices=True)
    filled = fill_src[ind[0], ind[1]].astype(np.float32)
    new_mask = np.isfinite(filled) & (filled < np.inf)
    return filled, new_mask


def align_depth_banded(
    estimate: np.ndarray,
    lidar: np.ndarray,
    mask: np.ndarray,
    apply_mask: Optional[np.ndarray] = None,
    bands: Sequence[Tuple[float, float]] = (
        (0.5, 15.0),
        (15.0, 40.0),
        (40.0, 80.0),
        (80.0, 200.0),
    ),
    min_points: int = 40,
) -> np.ndarray:
    """
    Fit an independent positive scale per distance band on LiDAR overlap,
    then apply scales to vision depth (band assignment by estimate depth).
    Falls back to global median scale when a band is too sparse.
    """
    estimate = np.asarray(estimate, dtype=np.float32)
    lidar = np.asarray(lidar, dtype=np.float32)
    fit = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(estimate)
        & np.isfinite(lidar)
        & (estimate > 1e-6)
        & (lidar > 1e-6)
        & (estimate < 1e4)
        & (lidar < 1e4)
    )
    if apply_mask is None:
        apply_mask = np.isfinite(estimate) & (estimate > 1e-6)
    apply_mask = np.asarray(apply_mask, dtype=bool)

    out = np.full_like(estimate, 10000.0)
    if fit.sum() < min_points:
        return estimate.copy()

    global_scale = float(np.median(lidar[fit] / estimate[fit]))
    global_scale = float(np.clip(global_scale, 0.05, 20.0))

    band_scales: List[Tuple[float, float, float]] = []
    for z0, z1 in bands:
        band = fit & (lidar >= z0) & (lidar < z1)
        if band.sum() >= min_points:
            s = float(np.median(lidar[band] / estimate[band]))
            s = float(np.clip(s, 0.05, 20.0))
        else:
            s = global_scale
        band_scales.append((z0, z1, s))

    # Assign by *aligned* estimate proxy: first apply global, then refine by band of result.
    aligned_global = estimate * global_scale
    for z0, z1, s in band_scales:
        sel = apply_mask & (aligned_global >= z0) & (aligned_global < z1)
        out[sel] = estimate[sel] * s
    # Anything outside bands
    covered = np.zeros_like(apply_mask, dtype=bool)
    for z0, z1, _ in band_scales:
        covered |= apply_mask & (aligned_global >= z0) & (aligned_global < z1)
    rest = apply_mask & (~covered)
    out[rest] = estimate[rest] * global_scale
    return out.astype(np.float32)


def sample_colors_from_image(points_cam, K, image_np):
    """Sample RGB from image at projected pixel locations."""
    H, W = image_np.shape[:2]
    n = points_cam.shape[0]
    colors = np.full((n, 3), 128, dtype=np.uint8)
    X, Y, Z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
    ok = Z > 1e-6
    if not ok.any():
        return colors
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = np.round(fx * X[ok] / Z[ok] + cx).astype(np.int32)
    v = np.round(fy * Y[ok] / Z[ok] + cy).astype(np.int32)
    ok_idx = np.where(ok)[0]
    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    colors[ok_idx[in_img]] = image_np[v[in_img], u[in_img]]
    return colors


def _subsample_points(points, colors, max_points=500000, seed=0):
    if points.shape[0] <= max_points:
        return points, colors
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], max_points, replace=False)
    points = points[idx]
    if colors is not None:
        colors = colors[idx]
    return points, colors


def compute_lidar_depth(
    image_np,
    points_world,
    calib,
    colors=None,
    depth_fill="none",
    *,
    raster_mode: str = "median",
    densify_radius: int = 1,
    calib_refine: bool = True,
    calib_max_shift: int = 2,
):
    """Rasterize LiDAR to a metric depth map (no file I/O)."""
    K = calib["K"]
    world_to_cam = calib["world_to_cam"]
    c2w = calib["c2w"]
    H = calib.get("H", calib.get("height"))
    W = calib.get("W", calib.get("width"))
    if H is None or W is None:
        raise KeyError("Calibration must include H/W or height/width")

    points = points_world
    if calib.get("points_in_sensor_frame") and calib.get("sensor_to_world") is not None:
        points = transform_points(points, calib["sensor_to_world"])

    points_cam = transform_points(points, world_to_cam)

    du = dv = 0.0
    if calib_refine and image_np is not None:
        du, dv = refine_projection_shift(
            points_cam, K, image_np, max_shift=int(calib_max_shift)
        )

    depth_map, valid_mask, count = rasterize_depth(
        points_cam,
        K,
        H,
        W,
        mode=raster_mode,
        densify_radius=int(densify_radius),
        du=du,
        dv=dv,
        return_count=True,
    )
    if depth_fill and depth_fill != "none":
        if depth_fill == "edge":
            depth_map, valid_mask = edge_aware_fill(
                depth_map, valid_mask, image_np=image_np
            )
        else:
            depth_map, valid_mask = fill_depth_holes(
                depth_map, valid_mask, method=depth_fill
            )

    if colors is None:
        colors = sample_colors_from_image(points_cam, K, image_np)
    elif colors.max() <= 1.0:
        colors = (colors * 255).astype(np.uint8)

    return {
        "depth_map": depth_map,
        "valid_mask": valid_mask,
        "hit_count": count,
        "K": K,
        "c2w": c2w,
        "H": int(H),
        "W": int(W),
        "points_cam": points_cam,
        "colors": colors,
        "proj_shift": (du, dv),
    }


def fuse_lidar_with_estimate(
    lidar_depth,
    valid_mask,
    estimate_depth,
    align=True,
    align_fn=None,
    *,
    image_np: Optional[np.ndarray] = None,
    hit_count: Optional[np.ndarray] = None,
    soft_blend: bool = True,
    blend_radius: int = 3,
    banded_align: bool = True,
    semantic_guide: bool = True,
    edge_fill: bool = True,
):
    """
    LiDAR-constrained depth with soft blend, banded scale, semantic weights,
    and edge-aware vision fill.
    """
    lidar = np.asarray(lidar_depth, dtype=np.float32)
    estimate = np.asarray(estimate_depth, dtype=np.float32)
    valid = (
        np.asarray(valid_mask, dtype=bool)
        & np.isfinite(lidar)
        & (lidar < np.inf)
        & (lidar > 1e-6)
    )

    estimate_finite = np.isfinite(estimate) & (estimate > 1e-6) & (estimate < np.inf)
    fused = estimate.copy()

    if align and valid.sum() > 50:
        if banded_align and align_fn is None:
            fused = align_depth_banded(
                estimate, lidar, mask=valid, apply_mask=estimate_finite
            )
        elif align_fn is not None:
            fused = align_fn(
                estimate,
                lidar,
                mask=valid,
                apply_mask=estimate_finite,
            ).astype(np.float32)
        else:
            fused = align_depth_banded(
                estimate, lidar, mask=valid, apply_mask=estimate_finite
            )

    # Edge-aware fill for remaining invalid vision pixels using aligned seeds.
    if edge_fill:
        seed = estimate_finite & np.isfinite(fused) & (fused < 5000)
        fused, _ = edge_aware_fill(
            fused,
            seed,
            image_np=image_np,
            max_radius=32,
        )

    conf = lidar_confidence_map(valid, count=hit_count, blend_radius=blend_radius)

    if semantic_guide and image_np is not None:
        priors = estimate_semantic_priors(image_np)
        # Sky: distrust vision metric depth (often collapsed); prefer far clamp later.
        fused = fused.copy()
        sky = priors["sky"] & (~valid)
        if sky.any() and valid.any():
            z_far = float(np.percentile(lidar[valid], 95))
            fused[sky] = np.maximum(fused[sky], z_far)
            conf[sky] = np.minimum(conf[sky], 0.05)
        # Ground: slightly prefer LiDAR when present.
        ground = priors["ground"] & valid
        conf[ground] = np.clip(conf[ground] + 0.15, 0.0, 1.0)
        # Object edges: soften LiDAR hard edges, trust vision structure more.
        obj = priors["object"]
        conf[obj & valid] = conf[obj & valid] * 0.85

    if soft_blend:
        alpha = conf
        # Harder lock in dense cores.
        alpha = np.where(valid & (conf >= 0.92), 1.0, alpha)
        vision = np.where(np.isfinite(fused) & (fused < 5000), fused, estimate)
        vision = np.where(np.isfinite(vision) & (vision > 1e-6), vision, lidar)
        lidar_safe = np.where(valid, lidar, vision)
        mixed = alpha * lidar_safe + (1.0 - alpha) * vision
        # Outside LiDAR support keep vision (with tiny exterior skirt already in conf).
        fused = np.where(valid | (conf > 0.05), mixed, vision).astype(np.float32)
        # Ensure pure LiDAR at highest-confidence hits.
        hard = valid & (conf >= 0.98)
        fused[hard] = lidar[hard]
    else:
        fused[valid] = lidar[valid]

    # Clamp vision-filled holes to a LiDAR-informed metric range.
    if valid.sum() > 20:
        z_hi = float(np.percentile(lidar[valid], 99.5)) * 1.5
        z_hi = float(np.clip(z_hi, 40.0, 200.0))
    else:
        z_hi = 120.0
    z_lo = 0.3
    finite = np.isfinite(fused)
    fused[finite & (fused < z_lo)] = z_lo
    fused[finite & (fused > z_hi)] = z_hi

    fused_valid = valid | (
        np.isfinite(fused) & (fused < np.inf) & (fused > 1e-6) & (fused < 5000)
    )
    return fused, fused_valid


def warp_depth_camera_to_camera(
    depth_src: np.ndarray,
    K_src: np.ndarray,
    c2w_src: np.ndarray,
    K_dst: np.ndarray,
    c2w_dst: np.ndarray,
    H_dst: int,
    W_dst: int,
    max_points: int = 250000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Unproject src depth to world, reproject into dst camera (z-buffer)."""
    H_src, W_src = depth_src.shape[:2]
    depth_src = np.asarray(depth_src, dtype=np.float32)
    valid = np.isfinite(depth_src) & (depth_src > 0.3) & (depth_src < 200.0)
    ys, xs = np.where(valid)
    if ys.size == 0:
        return (
            np.full((H_dst, W_dst), np.inf, dtype=np.float32),
            np.zeros((H_dst, W_dst), dtype=bool),
        )
    if ys.size > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(ys.size, max_points, replace=False)
        ys, xs = ys[idx], xs[idx]
    z = depth_src[ys, xs].astype(np.float64)
    fx, fy, cx, cy = K_src[0, 0], K_src[1, 1], K_src[0, 2], K_src[1, 2]
    X = (xs.astype(np.float64) - cx) * z / fx
    Y = (ys.astype(np.float64) - cy) * z / fy
    pts_cam = np.stack([X, Y, z], axis=1)
    ones = np.ones((pts_cam.shape[0], 1), dtype=np.float64)
    pts_h = np.hstack([pts_cam, ones])
    pts_world = (c2w_src @ pts_h.T).T[:, :3]
    w2c_dst = np.linalg.inv(c2w_dst)
    pts_dst = transform_points(pts_world, w2c_dst)
    return rasterize_depth(pts_dst, K_dst, H_dst, W_dst, mode="min", densify_radius=0)


def multiview_consistency_refine(
    depth: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    neighbor_depths: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    blend: float = 0.35,
) -> np.ndarray:
    """
    Soft-average current depth with warps from neighboring surround cameras
    where both views agree within a relative tolerance.
    neighbor_depths: list of (depth, K, c2w)
    """
    if not neighbor_depths:
        return depth.astype(np.float32)

    H, W = depth.shape[:2]
    acc = depth.astype(np.float64).copy()
    wsum = np.ones((H, W), dtype=np.float64)
    base_valid = np.isfinite(depth) & (depth > 0.3) & (depth < 200)

    for nd, nK, nc2w in neighbor_depths:
        warped, wvalid = warp_depth_camera_to_camera(
            nd, nK, nc2w, K, c2w, H, W
        )
        agree = (
            base_valid
            & wvalid
            & np.isfinite(warped)
            & (np.abs(warped - depth) / np.maximum(depth, 1e-3) < 0.12)
        )
        if agree.sum() < 100:
            continue
        acc[agree] += float(blend) * warped[agree]
        wsum[agree] += float(blend)

    out = (acc / wsum).astype(np.float32)
    return out


def write_depth_artifacts(
    out_dir,
    depth_map,
    valid_mask,
    K,
    c2w,
    points_cam,
    colors,
    max_ply_points=500000,
):
    """Write depth_map.npy, cam_params.json, and scene PLY files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    pts_export, col_export = _subsample_points(points_cam, colors, max_points=max_ply_points)
    trimesh.PointCloud(pts_export, col_export).export(out_dir / "depth_scene.ply")
    trimesh.PointCloud(pts_export, col_export).export(out_dir / "depth_scene_no_edge.ply")

    depth_save = depth_map.copy().astype(np.float32)
    depth_save[~valid_mask] = np.inf
    np.save(out_dir / "depth_map.npy", depth_save)

    cam_params = {
        "K": np.asarray(K).tolist(),
        "c2w": np.asarray(c2w).tolist(),
        "W": int(valid_mask.shape[1]),
        "H": int(valid_mask.shape[0]),
    }
    with open(out_dir / "cam_params.json", "w") as fp:
        json.dump(cam_params, fp)


def build_scene_outputs(
    out_dir,
    image_np,
    points_world,
    calib,
    colors=None,
    depth_fill="none",
    max_ply_points=500000,
    **lidar_kwargs,
):
    """
    Transform LiDAR to camera frame, rasterize depth, write pipeline artifacts.

    Writes: depth_map.npy, cam_params.json, depth_scene.ply, depth_scene_no_edge.ply
    """
    out_dir = Path(out_dir)
    computed = compute_lidar_depth(
        image_np,
        points_world,
        calib,
        colors=colors,
        depth_fill=depth_fill,
        **lidar_kwargs,
    )
    write_depth_artifacts(
        out_dir,
        computed["depth_map"],
        computed["valid_mask"],
        computed["K"],
        computed["c2w"],
        computed["points_cam"],
        computed["colors"],
        max_ply_points=max_ply_points,
    )
    return (
        computed["depth_map"],
        computed["valid_mask"],
        computed["K"],
        computed["c2w"],
    )


def refine_scene_multiview_depths(scene_root: Path, camera_keys: Sequence[str]) -> int:
    """
    Post-pass: for each camera under scene_root, blend with warped neighbor depths.
    Returns number of cameras updated.
    """
    scene_root = Path(scene_root)
    loaded = []
    for cam in camera_keys:
        d = scene_root / cam
        depth_path = d / "depth_map.npy"
        cam_path = d / "cam_params.json"
        if not depth_path.exists() or not cam_path.exists():
            continue
        depth = np.load(depth_path)
        with open(cam_path, "r") as f:
            params = json.load(f)
        K = np.asarray(params["K"], dtype=np.float64)
        c2w = np.asarray(params["c2w"], dtype=np.float64)
        loaded.append((cam, d, depth, K, c2w))

    if len(loaded) < 2:
        return 0

    updated = 0
    for i, (cam, d, depth, K, c2w) in enumerate(loaded):
        neighbors = [
            (depth_j, Kj, c2wj)
            for j, (_, _, depth_j, Kj, c2wj) in enumerate(loaded)
            if j != i
        ]
        # Use up to 2 spatial neighbors by name proximity when possible.
        refined = multiview_consistency_refine(depth, K, c2w, neighbors[:3], blend=0.3)
        valid = np.isfinite(refined) & (refined > 0.3) & (refined < 200)
        # Preserve inf holes marker for pipeline.
        out = refined.copy()
        out[~valid] = np.inf
        np.save(d / "depth_map.npy", out.astype(np.float32))
        updated += 1
    return updated
