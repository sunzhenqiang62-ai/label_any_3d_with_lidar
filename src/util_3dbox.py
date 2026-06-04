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
        best_yaw = 0

        for i in range(len(hull_points)):
            edge = hull_points[(i + 1) % len(hull_points)] - hull_points[i]
            yaw = np.arctan2(edge[1], edge[0])

            rot_2d = np.array([
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw), np.cos(yaw)]
            ])
            rotated_2d = (rot_2d @ points_2d.T).T

            x_min, x_max = rotated_2d[:, 0].min(), rotated_2d[:, 0].max()
            z_min, z_max = rotated_2d[:, 1].min(), rotated_2d[:, 1].max()
            area = (x_max - x_min) * (z_max - z_min)

            if area < min_area:
                min_area = area
                best_yaw = yaw

        return best_yaw

    except Exception as e:
        print(f"ConvexHull failed: {e}, falling back to PCA")
        return _estimate_yaw_pca(rotated_pc)


# =============================================================================
# Scene Processing Functions
# =============================================================================

def save_3d_with_ground_alignment_bbox(scene_dir, bbox_method='pca'):
    """
    Save 3D bounding boxes with ground alignment for all objects in a scene.

    Args:
        scene_dir: Scene directory path
        bbox_method: Method for bbox estimation - 'pca' (default) or 'convex_hull'

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

        mesh = trimesh.load(os.path.join(recons_dir, obj))
        canonical_upright = np.load(
            os.path.join(recons_dir, f"{obj.split('.', 1)[0]}_canonical_upright.npy")
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

        obj_dict["obj_id"] = obj_id
        obj_dict["category_name"] = category
        obj_dict["center_cam"] = center_cam.tolist()
        obj_dict["R_cam"] = R_cam.tolist()
        obj_dict["dimensions"] = dimensions
        obj_dict["bbox3D_cam"] = boxes3d.tolist()
        bbox_list.append(obj_dict)

    with open(os.path.join(scene_dir, '3dbbox_ground.json'), 'w') as json_file:
        json.dump(bbox_list, json_file)

    return bbox_list


def save_3d_bbox_from_depth_fallback(scene_dir):
    """
    Fallback 3D bbox generation from 2D masks + depth when mesh-based bbox fails.
    """
    from util import restore_mask_from_crop

    scene_dir = Path(scene_dir)
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
        obj_id = crop_path.stem.replace("_reproj", "")
        if "_" in obj_id:
            _, category = obj_id.split("_", 1)
        else:
            category = "object"

        crop_params_path = crops_dir / f"{obj_id}_crop_params.npy"
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

        dx = max((x2 - x1) * z / fx, 0.05)
        dy = max((y2 - y1) * z / fy, 0.05)
        dz = max(0.5 * (dx + dy), 0.05)

        vertices = convert_box_vertices(center_x, center_y, center_z, dx, dy, dz, 0.0)
        bbox_list.append(
            {
                "obj_id": obj_id.split("_", 1)[0] if "_" in obj_id else obj_id,
                "category_name": category,
                "center_cam": [center_x, center_y, center_z],
                "R_cam": np.eye(3).tolist(),
                "dimensions": [dz, dy, dx],
                "bbox3D_cam": vertices.tolist(),
            }
        )

    # Last-resort fallback: use 2D bboxes when crop masks are unusable.
    if len(bbox_list) == 0:
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
                bbox_list.append(
                    {
                        "obj_id": obj_id,
                        "category_name": category,
                        "center_cam": [center_x, center_y, center_z],
                        "R_cam": np.eye(3).tolist(),
                        "dimensions": [dz, dy, dx],
                        "bbox3D_cam": vertices.tolist(),
                    }
                )

    with open(scene_dir / "3dbbox_ground.json", "w") as fp:
        json.dump(bbox_list, fp)
    return bbox_list
