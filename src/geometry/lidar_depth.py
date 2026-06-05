"""
Build depth_map / PLY / cam_params from world-frame LiDAR + camera calibration.
"""

import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt


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


def project_points(points_cam, K):
    """Project camera-frame points to pixel coordinates and depth."""
    X, Y, Z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
    valid = Z > 1e-6
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * X[valid] / Z[valid] + cx
    v = fy * Y[valid] / Z[valid] + cy
    z = Z[valid]
    return u, v, z, valid


def rasterize_depth(points_cam, K, H, W):
    """
    Z-buffer rasterization: per-pixel minimum positive depth (OpenCV camera, Z forward).
    Returns depth (H,W) with inf at empty pixels, and valid_mask.
    """
    depth = np.full((H, W), np.inf, dtype=np.float32)
    if points_cam.shape[0] == 0:
        valid_mask = np.zeros((H, W), dtype=bool)
        return depth, valid_mask

    u, v, z, _ = project_points(points_cam, K)
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    in_bounds = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui, vi, z = ui[in_bounds], vi[in_bounds], z[in_bounds].astype(np.float32)

    flat_idx = vi * W + ui
    depth_flat = depth.ravel()
    np.minimum.at(depth_flat, flat_idx, z)
    depth = depth_flat.reshape(H, W)
    valid_mask = np.isfinite(depth) & (depth < np.inf)
    return depth, valid_mask


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


def compute_lidar_depth(image_np, points_world, calib, colors=None, depth_fill="none"):
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
    depth_map, valid_mask = rasterize_depth(points_cam, K, H, W)
    depth_map, valid_mask = fill_depth_holes(depth_map, valid_mask, method=depth_fill)

    if colors is None:
        colors = sample_colors_from_image(points_cam, K, image_np)
    elif colors.max() <= 1.0:
        colors = (colors * 255).astype(np.uint8)

    return {
        "depth_map": depth_map,
        "valid_mask": valid_mask,
        "K": K,
        "c2w": c2w,
        "H": int(H),
        "W": int(W),
        "points_cam": points_cam,
        "colors": colors,
    }


def fuse_lidar_with_estimate(
    lidar_depth,
    valid_mask,
    estimate_depth,
    align=True,
    align_fn=None,
):
    """
    LiDAR-constrained depth: keep metric LiDAR where valid, fill holes with vision.

    Args:
        lidar_depth: (H, W) metric depth from point cloud
        valid_mask: (H, W) bool, LiDAR-covered pixels
        estimate_depth: (H, W) dense metric depth from MoGe+DepthPro
        align: RANSAC-scale estimate to LiDAR in overlap before fusion
        align_fn: optional align_depth(relative, metric, mask) callable
    """
    lidar = np.asarray(lidar_depth, dtype=np.float32)
    estimate = np.asarray(estimate_depth, dtype=np.float32)
    valid = (
        np.asarray(valid_mask, dtype=bool)
        & np.isfinite(lidar)
        & (lidar < np.inf)
        & (lidar > 1e-6)
    )

    fused = estimate.copy()
    if align and valid.sum() > 50 and align_fn is not None:
        estimate_finite = np.isfinite(estimate) & (estimate > 1e-6) & (estimate < np.inf)
        fused = align_fn(
            estimate,
            lidar,
            mask=valid,
            apply_mask=estimate_finite,
        ).astype(np.float32)

    fused[valid] = lidar[valid]
    fused_valid = valid | (
        np.isfinite(fused) & (fused < np.inf) & (fused > 1e-6)
    )
    return fused, fused_valid


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
):
    """
    Transform LiDAR to camera frame, rasterize depth, write pipeline artifacts.

    Writes: depth_map.npy, cam_params.json, depth_scene.ply, depth_scene_no_edge.ply
    """
    out_dir = Path(out_dir)
    computed = compute_lidar_depth(
        image_np, points_world, calib, colors=colors, depth_fill=depth_fill
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
