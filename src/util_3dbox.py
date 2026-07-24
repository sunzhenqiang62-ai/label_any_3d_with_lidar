"""
3D bounding box estimation utilities.

Functions for estimating oriented 3D bounding boxes from point clouds
with ground plane alignment.
"""

import numpy as np
import trimesh
import os
import json
import math
from sklearn.decomposition import PCA
from pathlib import Path
from PIL import Image


# =============================================================================
# Basic Geometry Functions
# =============================================================================

def normalize(v):
    """Normalize a vector."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def rotate_y(yaw):
    """Generate a rotation matrix for yaw (around the y-axis)."""
    return np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)],
    ])


def rotation_matrix_from_vectors(vec1, vec2):
    """Compute rotation matrix that rotates vec1 to vec2."""
    vec1 = normalize(vec1)
    vec2 = normalize(vec2)

    axis = np.cross(vec1, vec2)
    cos_theta = np.dot(vec1, vec2)
    axis_norm = np.linalg.norm(axis)

    if not np.isfinite(cos_theta):
        return np.eye(3)
    if axis_norm < 1e-8:
        # Parallel or anti-parallel vectors: avoid division by zero.
        if cos_theta > 0:
            return np.eye(3)
        # 180-degree flip around a stable orthogonal axis.
        ortho = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(vec1, ortho)) > 0.9:
            ortho = np.array([0.0, 0.0, 1.0])
        axis = normalize(np.cross(vec1, ortho))
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-8:
            return np.eye(3)

    skew_symmetric = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    rotation_matrix = (
        np.eye(3) + skew_symmetric +
        np.dot(skew_symmetric, skew_symmetric) * (1 - cos_theta) / (axis_norm ** 2)
    )

    return rotation_matrix


def point_to_plane_distance(plane, x, y, z):
    """Calculate the shortest distance from a point to a plane."""
    plane = np.array(plane)
    a, b, c, d = plane
    numerator = abs(a * x + b * y + c * z + d)
    denominator = np.sqrt(a**2 + b**2 + c**2)
    return numerator / denominator


# =============================================================================
# Bounding Box Functions
# =============================================================================

def convert_box_vertices(center_x, center_y, center_z, l, w, h, yaw):
    """
    Generate 8 corner vertices of a 3D bounding box.

    Args:
        center_x, center_y, center_z: Box center coordinates
        l, w, h: Box dimensions (length, width, height)
        yaw: Rotation angle around y-axis

    Returns:
        8x3 array of corner vertices
    """
    local_corners = np.array([
        [-l / 2, -w / 2, -h / 2],
        [l / 2, -w / 2, -h / 2],
        [l / 2, w / 2, -h / 2],
        [-l / 2, w / 2, -h / 2],
        [-l / 2, -w / 2, h / 2],
        [l / 2, -w / 2, h / 2],
        [l / 2, w / 2, h / 2],
        [-l / 2, w / 2, h / 2]
    ])

    rotation_matrix = np.array([
        [math.cos(yaw), 0, math.sin(yaw)],
        [0, 1, 0],
        [-math.sin(yaw), 0, math.cos(yaw)]
    ])

    rotated_corners = np.dot(local_corners, rotation_matrix.T)
    global_corners = rotated_corners + np.array([center_x, center_y, center_z])

    return global_corners


def estimate_bbox(in_pc, cat_name=None, ground_equ=None, method='pca'):
    """
    Estimate oriented bounding box from point cloud.

    Args:
        in_pc: Input point cloud (N, 3)
        cat_name: Category name (unused, kept for compatibility)
        ground_equ: Ground plane equation [a, b, c, d] or canonical upright direction
        method: 'pca' or 'convex_hull' for yaw estimation

    Returns:
        vertices: 8 bbox vertices in camera coordinates
        center_cam: bbox center in camera coordinates
        dimension: [depth, height, width]
        R_cam: Rotation matrix from canonical to camera coordinates
    """
    # Subsample input point cloud if needed
    if in_pc.shape[0] > 500:
        rand_ind = np.random.randint(0, in_pc.shape[0], 500)
        in_pc = in_pc[rand_ind]

    # Rotate the point cloud to align with the ground plane
    valid_ground = (
        ground_equ is not None
        and np.isfinite(np.asarray(ground_equ[:3], dtype=np.float64)).all()
        and np.linalg.norm(np.asarray(ground_equ[:3], dtype=np.float64)) > 1e-8
    )
    if valid_ground:
        ground_equ = np.asarray(ground_equ, dtype=np.float64)
        dot_product = np.dot([0, -1, 0], ground_equ[:3])
        if dot_product <= 0:
            ground_equ = -ground_equ
        rotation_matrix = rotation_matrix_from_vectors([0, -1, 0], ground_equ[:3])
    else:
        rotation_matrix = np.eye(3)

    rotated_pc = np.dot(in_pc, rotation_matrix)

    # Remove invalid points
    valid_mask = np.isfinite(rotated_pc).all(axis=1)
    rotated_pc = rotated_pc[valid_mask]

    if len(rotated_pc) == 0:
        raise ValueError("No valid points after removing NaN values")

    # Determine yaw using selected method
    if method == 'convex_hull':
        yaw = _estimate_yaw_convex_hull(rotated_pc)
    elif method == 'pca':
        yaw = _estimate_yaw_pca(rotated_pc)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pca' or 'convex_hull'")

    # Rotate the point cloud to align with the x-axis and z-axis
    rotated_pc_2 = rotate_y(yaw) @ rotated_pc.T
    x_min, x_max = rotated_pc_2[0, :].min(), rotated_pc_2[0, :].max()
    y_min, y_max = rotated_pc_2[1, :].min(), rotated_pc_2[1, :].max()
    z_min, z_max = rotated_pc_2[2, :].min(), rotated_pc_2[2, :].max()

    dx, dy, dz = x_max - x_min, y_max - y_min, z_max - z_min
    cx, cy, cz = (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2

    print(f"[{method}] dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f}")

    # Generate vertices in aligned space
    vertices = convert_box_vertices(cx, cy, cz, dx, dy, dz, 0).astype(np.float16)

    # Transform vertices back to camera space
    vertices = np.dot(rotate_y(-yaw), vertices.T).T
    vertices = np.dot(vertices, rotation_matrix.T)

    # Use the final corner geometry as the single source of truth for the
    # returned center. This keeps center_cam consistent with bbox3D_cam after
    # yaw/ground alignment transforms.
    center_cam = vertices.mean(axis=0)

    dimension = [dz, dy, dx]
    R_cam = rotation_matrix.T @ rotate_y(-yaw)

    return vertices, center_cam, dimension, R_cam


def _estimate_yaw_pca(rotated_pc):
    """Estimate yaw angle using PCA."""
    pca = PCA(2)
    pca.fit(rotated_pc[:, [0, 2]])
    yaw_vec = pca.components_[0, :]
    return np.arctan2(yaw_vec[1], yaw_vec[0])


def _estimate_yaw_convex_hull(rotated_pc):
    """Estimate yaw angle using minimum area bounding box from convex hull."""
    from scipy.spatial import ConvexHull

    points_2d = rotated_pc[:, [0, 2]]  # X and Z coordinates

    try:
        hull = ConvexHull(points_2d)
        hull_points = points_2d[hull.vertices]

        min_area = float('inf')
        best_yaw = 0.0

        for i in range(len(hull_points)):
            edge = hull_points[(i + 1) % len(hull_points)] - hull_points[i]
            # Match rotate_y: x'=c*x+s*z, z'=-s*x+c*z  => yaw = atan2(edge_x, edge_z)
            # when aligning edge to +Z, or use candidate angles from edge.
            for yaw in (
                np.arctan2(edge[0], edge[1]),
                np.arctan2(edge[0], edge[1]) + 0.5 * np.pi,
            ):
                aligned = (rotate_y(yaw) @ rotated_pc.T).T
                dx = float(aligned[:, 0].max() - aligned[:, 0].min())
                dz = float(aligned[:, 2].max() - aligned[:, 2].min())
                area = dx * dz
                if area < min_area:
                    min_area = area
                    best_yaw = float(yaw)

        return best_yaw

    except Exception as e:
        print(f"ConvexHull failed: {e}, falling back to PCA")
        return _estimate_yaw_pca(rotated_pc)


VEHICLE_CATEGORIES = {
    "car",
    "truck",
    "bus",
    "van",
    "vehicle",
    "automobile",
    "trailer",
    "construction_vehicle",
    "vehicle_sedancar",
    "vehicle_truck",
    "vehicle_bus",
    "sedan",
    "sed_car",
}


def _upright_rotation(ground_equ):
    """Rotation that maps camera points into an upright frame (Y down)."""
    valid_ground = (
        ground_equ is not None
        and np.isfinite(np.asarray(ground_equ[:3], dtype=np.float64)).all()
        and np.linalg.norm(np.asarray(ground_equ[:3], dtype=np.float64)) > 1e-8
    )
    if not valid_ground:
        return np.eye(3)
    ground_equ = np.asarray(ground_equ, dtype=np.float64)
    if np.dot([0, -1, 0], ground_equ[:3]) <= 0:
        ground_equ = -ground_equ
    return rotation_matrix_from_vectors([0, -1, 0], ground_equ[:3])


def _normalize_yaw(yaw: float) -> float:
    return float((yaw + np.pi) % (2 * np.pi) - np.pi)


def _estimate_yaw_min_area_sweep(rotated_pc, n_angles: int = 180) -> float:
    """Brute-force min-area yaw in [0, π) using rotate_y convention."""
    best_yaw = 0.0
    min_area = float("inf")
    for yaw in np.linspace(0.0, np.pi, n_angles, endpoint=False):
        aligned = rotate_y(yaw) @ rotated_pc.T
        dx = float(aligned[0].max() - aligned[0].min())
        dz = float(aligned[2].max() - aligned[2].min())
        area = dx * dz
        if area < min_area:
            min_area = area
            best_yaw = float(yaw)
    return best_yaw


def estimate_bev_yaw(
    points_cam,
    upright=None,
    *,
    make_length_along_z: bool = True,
) -> float:
    """
    Estimate yaw from BEV (X–Z) min-area rectangle on camera-frame points.

    After ``rotate_y(yaw)``, the long horizontal axis is aligned with +Z when
    ``make_length_along_z`` is True (matches ``estimate_bbox`` length=dz).
    """
    points_cam = np.asarray(points_cam, dtype=np.float64)
    if points_cam.ndim != 2 or points_cam.shape[0] < 16 or points_cam.shape[1] != 3:
        raise ValueError("Need >=16 finite 3D points for BEV yaw")
    finite = np.isfinite(points_cam).all(axis=1)
    points_cam = points_cam[finite]
    if len(points_cam) < 16:
        raise ValueError("Too few finite points for BEV yaw")

    rotation_matrix = _upright_rotation(upright)
    rotated_pc = points_cam @ rotation_matrix
    yaw = _estimate_yaw_min_area_sweep(rotated_pc)
    aligned = (rotate_y(yaw) @ rotated_pc.T).T
    dx = float(aligned[:, 0].max() - aligned[:, 0].min())
    dz = float(aligned[:, 2].max() - aligned[:, 2].min())
    if make_length_along_z and dx > dz:
        yaw = yaw + 0.5 * np.pi
    return _normalize_yaw(yaw)


def disambiguate_yaw_180(yaw, center_cam, upright=None, prefer="away_from_camera"):
    """
    Resolve length-axis 180° ambiguity.

    prefer:
      - away_from_camera: heading has positive dot with object center ray (XZ)
      - camera_forward: heading has positive camera +Z component
    """
    rotation_matrix = _upright_rotation(upright)
    R_cam = rotation_matrix.T @ rotate_y(-float(yaw))
    heading = R_cam @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    heading_xz = np.array([heading[0], heading[2]], dtype=np.float64)
    n = np.linalg.norm(heading_xz)
    if n < 1e-8:
        return _normalize_yaw(yaw)
    heading_xz /= n

    center = np.asarray(center_cam, dtype=np.float64).reshape(3)
    if prefer == "camera_forward":
        if heading_xz[1] < 0:
            yaw = yaw + np.pi
    else:
        center_xz = np.array([center[0], center[2]], dtype=np.float64)
        cn = np.linalg.norm(center_xz)
        if cn > 1e-6:
            center_xz /= cn
            if float(np.dot(heading_xz, center_xz)) < 0:
                yaw = yaw + np.pi
        elif heading_xz[1] < 0:
            yaw = yaw + np.pi
    return _normalize_yaw(yaw)


def oriented_box_from_dims_yaw(
    center_cam,
    dimensions,
    yaw,
    upright=None,
):
    """
    Build box corners / R_cam from center, dims=[length, height, width], yaw.

    Length is the local +Z axis after upright align + yaw (same as estimate_bbox).
    """
    length, height, width = [float(v) for v in dimensions]
    rotation_matrix = _upright_rotation(upright)
    center = np.asarray(center_cam, dtype=np.float64).reshape(3)
    c_aligned = center @ rotation_matrix
    # Same as estimate_bbox: yaw about origin, AABB center expressed in yaw frame.
    c_yaw = rotate_y(float(yaw)) @ c_aligned
    corners = convert_box_vertices(
        float(c_yaw[0]),
        float(c_yaw[1]),
        float(c_yaw[2]),
        width,
        height,
        length,
        0.0,
    ).astype(np.float64)
    corners = (rotate_y(-float(yaw)) @ corners.T).T
    corners = corners @ rotation_matrix.T
    R_cam = rotation_matrix.T @ rotate_y(-float(yaw))
    return corners, center, [length, height, width], R_cam


def backproject_mask_depth_points(mask, depth, K, max_points=4000):
    """Back-project mask pixels with valid depth to camera XYZ (OpenCV)."""
    mask = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    valid = mask & np.isfinite(depth) & (depth > 0)
    ys, xs = np.where(valid)
    if len(xs) < 16:
        return None
    z = depth[ys, xs]
    # Drop far depth plateaus that dominate yaw.
    lo, hi = np.percentile(z, [5, 95])
    keep = (z >= lo) & (z <= hi)
    if keep.sum() < 16:
        keep = np.ones_like(z, dtype=bool)
    xs, ys, z = xs[keep], ys[keep], z[keep]
    if len(xs) > max_points:
        idx = np.random.choice(len(xs), max_points, replace=False)
        xs, ys, z = xs[idx], ys[idx], z[idx]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (xs.astype(np.float64) - cx) * z / fx
    y = (ys.astype(np.float64) - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def filter_object_points(
    points_cam,
    *,
    center_cam=None,
    depth_rel_band: float = 0.12,
    depth_abs_band: float = 1.25,
    xz_mad_thresh: float = 3.5,
    max_points: int = 3000,
):
    """
    Keep only near-object depth points for yaw/size (drop road / far fill).

    1) Depth band around median (or around center_z if given)
    2) XZ robust outlier rejection (MAD)
    """
    pts = np.asarray(points_cam, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 16:
        return None
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if len(pts) < 16:
        return None

    z = pts[:, 2]
    if center_cam is not None:
        z0 = float(np.asarray(center_cam, dtype=np.float64).reshape(3)[2])
    else:
        z0 = float(np.median(z))
    band = max(depth_abs_band, depth_rel_band * max(z0, 1.0))
    keep = np.abs(z - z0) <= band
    if keep.sum() < 16:
        # Widen once.
        keep = np.abs(z - z0) <= (2.0 * band)
    if keep.sum() < 16:
        return None
    pts = pts[keep]

    xz = pts[:, [0, 2]]
    med = np.median(xz, axis=0)
    mad = np.median(np.abs(xz - med), axis=0)
    mad = np.maximum(mad, 1e-3)
    keep = np.all(np.abs(xz - med) <= (xz_mad_thresh * 1.4826 * mad), axis=1)
    if keep.sum() < 16:
        # Fallback: radius around median in XZ.
        rad = np.linalg.norm(xz - med, axis=1)
        keep = rad <= max(2.5, float(np.percentile(rad, 70)))
    if keep.sum() < 16:
        return None
    pts = pts[keep]
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts = pts[idx]
    return pts


def vehicle_size_priors(category=None):
    """Reasonable metric size ranges for passenger vehicles (meters)."""
    cat = (category or "car").split(",")[0].strip().lower().replace(" ", "_")
    if cat in ("truck", "bus", "trailer", "construction_vehicle"):
        return {
            "length": (6.0, 12.0),
            "width": (2.0, 3.0),
            "height": (2.0, 3.5),
            "default": (8.0, 2.5, 2.8),
        }
    return {
        "length": (3.6, 5.2),
        "width": (1.6, 2.15),
        "height": (1.35, 2.0),
        "default": (4.5, 1.85, 1.55),
    }


def _clamp(v, lo, hi):
    return float(min(max(v, lo), hi))


def load_object_depth_points(scene_dir, obj_stem, max_points=4000, center_cam=None):
    """Load camera-frame points for one crop object from mask + depth_map."""
    from util import restore_mask_from_crop

    scene_dir = Path(scene_dir)
    cam_path = scene_dir / "cam_params.json"
    depth_path = scene_dir / "depth_map.npy"
    crop_path = scene_dir / "crops" / f"{obj_stem}_reproj.png"
    crop_params_path = scene_dir / "crops" / f"{obj_stem}_crop_params.npy"
    if not (cam_path.exists() and depth_path.exists() and crop_path.exists() and crop_params_path.exists()):
        return None
    with open(cam_path, "r") as fp:
        cam = json.load(fp)
    K = np.asarray(cam["K"], dtype=np.float64)
    depth = np.load(depth_path)
    H, W = depth.shape[:2]
    crop_params = np.load(crop_params_path)
    crop = Image.open(crop_path)
    mask_local = np.array(crop)[:, :, 3] > 127
    mask = restore_mask_from_crop(
        mask_local, crop_params[0], crop_params[1], crop_params[2], (H, W)
    )
    pts = backproject_mask_depth_points(mask, depth, K, max_points=max_points)
    if pts is None:
        return None
    return filter_object_points(pts, center_cam=center_cam, max_points=max_points)


def yaw_along_ego_forward(c2w, upright=None, ego_forward_world=None):
    """
    Yaw that aligns box length (+Z) with ego forward expressed in camera frame.

    Default ego_forward_world=+X matches this bad_case / loopify FRONT optical axis.
    """
    c2w = np.asarray(c2w, dtype=np.float64).reshape(4, 4)
    if ego_forward_world is None:
        ego_forward_world = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        ego_forward_world = np.asarray(ego_forward_world, dtype=np.float64).reshape(3)
    f_cam = c2w[:3, :3].T @ ego_forward_world
    R_up = _upright_rotation(upright)
    f_up = f_cam @ R_up
    if abs(f_up[0]) + abs(f_up[2]) < 1e-8:
        return 0.0
    return _normalize_yaw(float(np.arctan2(f_up[0], f_up[2])))


def refine_box_yaw_with_bev(
    center_cam,
    dimensions,
    upright,
    points_cam,
    *,
    category=None,
    prefer="away_from_camera",
    elong_ratio_min: float = 1.35,
    c2w=None,
):
    """
    Refine vehicle orientation (and L/W) from filtered depth points.

    - If BEV footprint is clearly elongated: min-area yaw + 180° disambiguation
    - Else: align heading with ego-forward (from c2w) or camera +Z
    - Clamp L/W/H to vehicle priors; prefer filtered BEV extents when sane
    """
    priors = vehicle_size_priors(category)
    center = np.asarray(center_cam, dtype=np.float64).reshape(3)
    mesh_length, mesh_height, mesh_width = [float(v) for v in dimensions]
    length0, height0, width0 = mesh_length, mesh_height, mesh_width

    pts = filter_object_points(points_cam, center_cam=center)
    if c2w is not None:
        yaw = yaw_along_ego_forward(c2w, upright=upright)
        yaw_source = "ego_forward"
    else:
        yaw = 0.0
        yaw_source = "camera_forward"

    used_bev_size = False
    if pts is not None and len(pts) >= 16:
        rotation_matrix = _upright_rotation(upright)
        rotated = pts @ rotation_matrix
        xz = rotated[:, [0, 2]]
        xz_c = xz - xz.mean(axis=0)
        cov = (xz_c.T @ xz_c) / max(len(xz_c), 1)
        eigvals = np.linalg.eigvalsh(cov)
        elong = float(eigvals[-1] / max(eigvals[0], 1e-6))
        span = xz.max(axis=0) - xz.min(axis=0)
        max_span = float(max(span[0], span[1]))

        if max_span <= 8.0 and elong >= elong_ratio_min:
            try:
                yaw = estimate_bev_yaw(pts, upright=upright)
                yaw = disambiguate_yaw_180(
                    yaw, center, upright=upright, prefer=prefer
                )
                yaw_source = "bev_depth"
            except Exception:
                pass

        aligned = (rotate_y(yaw) @ rotated.T).T
        dx = float(aligned[:, 0].max() - aligned[:, 0].min())
        dy = float(aligned[:, 1].max() - aligned[:, 1].min())
        dz = float(aligned[:, 2].max() - aligned[:, 2].min())
        if dx > dz:
            dx, dz = dz, dx
        if 1.2 <= dz <= 7.5 and 1.2 <= dx <= 3.0:
            length0, width0 = dz, dx
            used_bev_size = True
        if 1.0 <= dy <= 3.0:
            height0 = dy

    length = _clamp(length0, *priors["length"])
    width = _clamp(width0, *priors["width"])
    height = _clamp(height0, *priors["height"])
    if (
        not used_bev_size
        and (
            mesh_length > priors["length"][1] * 1.35
            or mesh_width > priors["width"][1] * 1.35
            or mesh_height > priors["height"][1] * 1.35
        )
    ):
        length, width, height = priors["default"]

    corners, center_out, dims, R_cam = oriented_box_from_dims_yaw(
        center, [length, height, width], yaw, upright=upright
    )
    return corners, center_out, dims, R_cam, yaw, yaw_source


# =============================================================================
# Scene Processing Functions
# =============================================================================

def save_3d_with_ground_alignment_bbox(scene_dir, bbox_method='pca', refine_yaw_bev=True):
    """
    Save 3D bounding boxes with ground alignment for all objects in a scene.

    Args:
        scene_dir: Scene directory path
        bbox_method: Method for bbox estimation - 'pca' (default) or 'convex_hull'
        refine_yaw_bev: For vehicles, replace yaw using mask+depth BEV min-area rect

    Returns:
        List of bounding box dictionaries
    """
    recons_dir = os.path.join(scene_dir, "reconstruction")
    files_and_dirs = os.listdir(recons_dir)
    objs = [
        item for item in files_and_dirs
        if item not in ['full_scene.glb', 'background.ply'] and item.endswith('.glb')
    ]
    bbox_list = []

    for obj in objs:
        obj_dict = {}
        parts = obj.split("_", 1)
        obj_id = parts[0]
        category, _ = parts[1].split(".", 1)
        obj_stem = obj.split(".", 1)[0]

        mesh = trimesh.load(os.path.join(recons_dir, obj))
        canonical_upright = np.load(
            os.path.join(recons_dir, f"{obj_stem}_canonical_upright.npy")
        )

        if isinstance(mesh, trimesh.Scene):
            meshes = mesh.dump()
            mesh = meshes[0]

        if mesh.is_empty or mesh.area == 0 or len(mesh.faces) == 0:
            print(f"Invalid mesh at {os.path.join(recons_dir, obj)}, skipping.")
            continue

        verts = np.asarray(mesh.vertices, dtype=np.float64)
        if verts.ndim != 2 or verts.shape[1] != 3:
            print(f"Invalid vertex array for {obj}, skipping.")
            continue
        verts = verts[np.isfinite(verts).all(axis=1)]
        if len(verts) == 0:
            print(f"No finite vertices for {obj}, skipping.")
            continue
        if len(verts) > 5000:
            idx = np.random.choice(len(verts), 5000, replace=False)
            verts = verts[idx]

        try:
            boxes3d, center_cam, dimensions, R_cam = estimate_bbox(
                verts,
                category,
                canonical_upright,
                method=bbox_method
            )
        except Exception as e:
            print(f"Error estimating bbox for {obj}: {e}")
            continue

        yaw_source = bbox_method
        cat_key = category.split(",")[0].strip().lower().replace(" ", "_")
        if refine_yaw_bev and (
            cat_key in VEHICLE_CATEGORIES or cat_key.endswith("car")
        ):
            # Raw backproject first; refine_* applies robust filtering.
            points = load_object_depth_points(scene_dir, obj_stem, center_cam=center_cam)
            # load_object_depth_points already filters; also try unfiltered path via crops
            if points is None:
                from util import restore_mask_from_crop

                cam_path = Path(scene_dir) / "cam_params.json"
                depth_path = Path(scene_dir) / "depth_map.npy"
                crop_path = Path(scene_dir) / "crops" / f"{obj_stem}_reproj.png"
                crop_params_path = Path(scene_dir) / "crops" / f"{obj_stem}_crop_params.npy"
                if cam_path.exists() and depth_path.exists() and crop_path.exists():
                    with open(cam_path, "r") as fp:
                        cam = json.load(fp)
                    K = np.asarray(cam["K"], dtype=np.float64)
                    depth = np.load(depth_path)
                    H, W = depth.shape[:2]
                    crop_params = np.load(crop_params_path)
                    crop = Image.open(crop_path)
                    mask_local = np.array(crop)[:, :, 3] > 127
                    mask = restore_mask_from_crop(
                        mask_local, crop_params[0], crop_params[1], crop_params[2], (H, W)
                    )
                    points = backproject_mask_depth_points(mask, depth, K)
            if points is not None and len(points) >= 16:
                try:
                    c2w = None
                    cam_path = Path(scene_dir) / "cam_params.json"
                    if cam_path.exists():
                        with open(cam_path, "r") as fp:
                            cam = json.load(fp)
                        if cam.get("c2w") is not None:
                            c2w = np.asarray(cam["c2w"], dtype=np.float64)
                    boxes3d, center_cam, dimensions, R_cam, yaw, yaw_source = refine_box_yaw_with_bev(
                        center_cam,
                        dimensions,
                        canonical_upright,
                        points,
                        category=category,
                        c2w=c2w,
                    )
                    print(
                        f"[bev_yaw] {obj_stem}: yaw={yaw:.3f} rad "
                        f"src={yaw_source} dims={[round(d,2) for d in dimensions]}"
                    )
                except Exception as e:
                    print(f"[bev_yaw] {obj_stem}: refine failed ({e}); keeping {bbox_method}")
                    yaw_source = bbox_method

        obj_dict["obj_id"] = obj_id
        obj_dict["category_name"] = category
        obj_dict["center_cam"] = np.asarray(center_cam, dtype=np.float64).tolist()
        obj_dict["R_cam"] = np.asarray(R_cam, dtype=np.float64).tolist()
        obj_dict["dimensions"] = list(dimensions)
        obj_dict["bbox3D_cam"] = np.asarray(boxes3d, dtype=np.float64).tolist()
        obj_dict["yaw_source"] = yaw_source
        bbox_list.append(obj_dict)

    with open(os.path.join(scene_dir, '3dbbox_ground.json'), 'w') as json_file:
        json.dump(bbox_list, json_file)

    return bbox_list


def save_3d_bbox_from_depth_fallback(
    scene_dir,
    exclude_obj_ids=None,
    write_json=True,
):
    """
    Fallback 3D bbox generation from 2D masks + depth when mesh-based bbox fails.

    Args:
        scene_dir: Scene directory path
        exclude_obj_ids: Optional set of obj_id / stem names already covered by mesh bboxes
        write_json: If True, overwrite 3dbbox_ground.json with this fallback-only list
    """
    from util import restore_mask_from_crop

    scene_dir = Path(scene_dir)
    exclude = {str(x) for x in (exclude_obj_ids or [])}
    cam_path = scene_dir / "cam_params.json"
    depth_path = scene_dir / "depth_map.npy"
    crops_dir = scene_dir / "crops"
    if not cam_path.exists() or not depth_path.exists() or not crops_dir.exists():
        return []

    with open(cam_path, "r") as fp:
        cam = json.load(fp)
    K = np.asarray(cam["K"], dtype=np.float64)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    depth = np.load(depth_path)
    H, W = depth.shape[:2]
    global_valid_depth = depth[np.isfinite(depth) & (depth > 0)]
    z_default = float(np.median(global_valid_depth)) if global_valid_depth.size > 0 else 10.0

    bbox_list = []
    crop_paths = sorted(crops_dir.glob("*_reproj.png"))
    for crop_path in crop_paths:
        obj_stem = crop_path.stem.replace("_reproj", "")
        if "_" in obj_stem:
            numeric_id, category = obj_stem.split("_", 1)
        else:
            numeric_id, category = obj_stem, "object"
        if numeric_id in exclude or obj_stem in exclude:
            continue

        crop_params_path = crops_dir / f"{obj_stem}_crop_params.npy"
        if not crop_params_path.exists():
            continue
        crop_params = np.load(crop_params_path)
        crop = Image.open(crop_path)
        mask_local = np.array(crop)[:, :, 3] > 127
        mask = restore_mask_from_crop(
            mask_local, crop_params[0], crop_params[1], crop_params[2], (H, W)
        )
        valid = mask & np.isfinite(depth) & (depth > 0)
        ys, xs = np.where(valid)
        if len(xs) < 20:
            continue

        z_vals = depth[valid]
        z = float(np.median(z_vals)) if z_vals.size > 0 else z_default
        x1, x2 = float(xs.min()), float(xs.max())
        y1, y2 = float(ys.min()), float(ys.max())
        uc = 0.5 * (x1 + x2)
        vc = 0.5 * (y1 + y2)

        center_x = (uc - cx) * z / fx
        center_y = (vc - cy) * z / fy
        center_z = z

        points = backproject_mask_depth_points(mask, depth, K)
        center = [center_x, center_y, center_z]
        cat_key = category.split(",")[0].strip().lower().replace(" ", "_").replace("-", "_")
        is_vehicle = (
            cat_key in VEHICLE_CATEGORIES
            or cat_key.endswith("car")
            or "truck" in cat_key
            or "bus" in cat_key
        )

        if is_vehicle and points is not None and len(points) >= 16:
            # Use shared refine path: filtered BEV + ego-forward fallback + size priors.
            c2w = np.asarray(cam["c2w"], dtype=np.float64) if cam.get("c2w") is not None else None
            priors = vehicle_size_priors(category)
            seed_dims = list(priors["default"])  # [L, H, W]
            try:
                vertices, center_arr, dims, R_cam, yaw, yaw_source = refine_box_yaw_with_bev(
                    center,
                    seed_dims,
                    upright=None,
                    points_cam=points,
                    category=category,
                    c2w=c2w,
                )
                bbox_list.append(
                    {
                        "obj_id": numeric_id,
                        "category_name": category,
                        "center_cam": np.asarray(center_arr, dtype=np.float64).tolist(),
                        "R_cam": np.asarray(R_cam, dtype=np.float64).tolist(),
                        "dimensions": list(dims),
                        "bbox3D_cam": np.asarray(vertices, dtype=np.float64).tolist(),
                        "source": f"depth_fallback_{yaw_source}",
                        "yaw_source": yaw_source,
                    }
                )
                continue
            except Exception as e:
                print(f"[depth_fallback] refine failed for {obj_stem}: {e}")

        # Non-vehicle / refine-failed: simple backproject box with optional prior clamp.
        yaw = 0.0
        yaw_source = "depth_fallback"
        dx = max((x2 - x1) * z / fx, 0.05)
        dy = max((y2 - y1) * z / fy, 0.05)
        dz = max(0.5 * (dx + dy), 0.05)
        if is_vehicle:
            priors = vehicle_size_priors(category)
            length, height, width = priors["default"]
            dx, dy, dz = width, height, length
            yaw_source = "depth_fallback"
            if cam.get("c2w") is not None:
                try:
                    yaw = yaw_along_ego_forward(cam["c2w"], upright=None)
                    yaw_source = "ego_forward"
                except Exception:
                    yaw = 0.0
            else:
                yaw = 0.0
            R_cam = rotate_y(-yaw)
            vertices = convert_box_vertices(center_x, center_y, center_z, dx, dy, dz, yaw)
            dims = [dz, dy, dx]
        else:
            yaw = 0.0
            yaw_source = "depth_fallback"
            R_cam = np.eye(3)
            vertices = convert_box_vertices(center_x, center_y, center_z, dx, dy, dz, 0.0)
            dims = [dz, dy, dx]

        bbox_list.append(
            {
                "obj_id": numeric_id,
                "category_name": category,
                "center_cam": [center_x, center_y, center_z],
                "R_cam": np.asarray(R_cam, dtype=np.float64).tolist(),
                "dimensions": dims,
                "bbox3D_cam": np.asarray(vertices, dtype=np.float64).tolist(),
                "source": yaw_source,
                "yaw_source": yaw_source,
            }
        )

    # Last-resort fallback: use 2D bboxes when crop masks are unusable.
    # Only used when nothing was produced for remaining crops (and no excludes
    # already cover the scene) — skip if we already have some depth boxes or
    # mesh exclusions imply a partial merge.
    if len(bbox_list) == 0 and not exclude:
        bbox2d_path = scene_dir / "bboxes.json"
        if bbox2d_path.exists():
            with open(bbox2d_path, "r") as fp:
                bbox2d = json.load(fp)
            obj_names = [p.stem.replace("_reproj", "") for p in crop_paths]
            for obj_name, box in zip(sorted(obj_names), bbox2d):
                if len(box) != 4:
                    continue
                x1, y1, x2, y2 = box
                x1 = int(max(0, min(W - 1, round(x1))))
                x2 = int(max(0, min(W - 1, round(x2))))
                y1 = int(max(0, min(H - 1, round(y1))))
                y2 = int(max(0, min(H - 1, round(y2))))
                if x2 <= x1 or y2 <= y1:
                    continue
                depth_roi = depth[y1:y2 + 1, x1:x2 + 1]
                valid = np.isfinite(depth_roi) & (depth_roi > 0)
                z = float(np.median(depth_roi[valid])) if valid.sum() >= 5 else z_default
                uc = 0.5 * (x1 + x2)
                vc = 0.5 * (y1 + y2)
                center_x = (uc - cx) * z / fx
                center_y = (vc - cy) * z / fy
                center_z = z
                dx = max((x2 - x1) * z / fx, 0.05)
                dy = max((y2 - y1) * z / fy, 0.05)
                dz = max(0.5 * (dx + dy), 0.05)
                vertices = convert_box_vertices(center_x, center_y, center_z, dx, dy, dz, 0.0)
                if "_" in obj_name:
                    obj_id, category = obj_name.split("_", 1)
                else:
                    obj_id, category = obj_name, "object"
                if obj_id in exclude or obj_name in exclude:
                    continue
                bbox_list.append(
                    {
                        "obj_id": obj_id,
                        "category_name": category,
                        "center_cam": [center_x, center_y, center_z],
                        "R_cam": np.eye(3).tolist(),
                        "dimensions": [dz, dy, dx],
                        "bbox3D_cam": vertices.tolist(),
                        "source": "depth_fallback_2d",
                    }
                )

    if write_json:
        with open(scene_dir / "3dbbox_ground.json", "w") as fp:
            json.dump(bbox_list, fp)
    return bbox_list
