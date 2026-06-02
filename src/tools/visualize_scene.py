"""
Lightweight scene visualization (no Blender required).

Usage (from src/):
    python tools/visualize_scene.py --scene_dir ../experimental_results/nuScenes/nuscenes_val/<id> --mode compose
    python tools/visualize_scene.py --root ../experimental_results/nuScenes/nuscenes_val --mode gt_2d,depth
    python tools/visualize_scene.py --scene_dir <dir> --backend blender
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

sys.path = ["./"] + sys.path


def _import_util():
    from util import read_bounding_boxes_segmentations, render_bbox_overlay
    return read_bounding_boxes_segmentations, render_bbox_overlay


def _viz_dir(scene_dir: Path, out_subdir: str = "viz") -> Path:
    d = scene_dir / out_subdir
    d.mkdir(parents=True, exist_ok=True)
    return d


def render_gt_2d(scene_dir: Path, out_dir: Path) -> Optional[Path]:
    img_path = scene_dir / "input.png"
    ann_path = scene_dir / "nuscenes_annotations.json"
    if not img_path.exists():
        return None
    image = cv2.imread(str(img_path))
    if image is None:
        return None
    if ann_path.exists():
        read_boxes, _render_bbox = _import_util()
        with open(ann_path, "r") as f:
            annotations = json.load(f)
        w, h = image.shape[1], image.shape[0]
        bboxes, masks, object_ids, _labels = read_boxes(
            annotations, (w, h)
        )
        for idx in object_ids:
            mask = masks[idx].astype(np.uint8)
            if mask.ndim == 2:
                colored = np.zeros_like(image)
                colored[:, :, 1] = mask * 180
                image = cv2.addWeighted(image, 1.0, colored, 0.35, 0)
            x, y, bw, bh = [int(v) for v in bboxes[idx]]
            cv2.rectangle(image, (x, y), (x + bw, y + bh), (0, 255, 255), 2)
    out_path = out_dir / "gt_overlay.png"
    cv2.imwrite(str(out_path), image)
    return out_path


def render_depth(scene_dir: Path, out_dir: Path) -> List[Path]:
    depth_path = scene_dir / "depth_map.npy"
    if not depth_path.exists():
        return []
    depth = np.load(depth_path)
    valid = depth[np.isfinite(depth) & (depth > 0)]
    if valid.size == 0:
        return []
    vmin, vmax = np.percentile(valid, [2, 98])
    depth_norm = np.clip((depth - vmin) / (vmax - vmin + 1e-8), 0, 1)
    depth_u8 = (depth_norm * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    depth_color[~np.isfinite(depth) | (depth <= 0)] = 0

    paths = []
    p1 = out_dir / "depth_colormap.png"
    cv2.imwrite(str(p1), depth_color)
    paths.append(p1)

    rgb_path = scene_dir / "input.png"
    if rgb_path.exists():
        rgb = cv2.imread(str(rgb_path))
        if rgb is not None and rgb.shape[:2] == depth_color.shape[:2]:
            panel = np.hstack([rgb, depth_color])
            p2 = out_dir / "rgb_depth.png"
            cv2.imwrite(str(p2), panel)
            paths.append(p2)
    return paths


def render_crops_grid(scene_dir: Path, out_dir: Path, max_cols: int = 4) -> Optional[Path]:
    crop_dir = scene_dir / "crops"
    if not crop_dir.exists():
        return None
    crops = sorted(crop_dir.glob("*_reproj.png"))[:16]
    if not crops:
        return None
    thumbs = []
    for p in crops:
        im = cv2.imread(str(p))
        if im is not None:
            thumbs.append(cv2.resize(im, (128, 128)))
    if not thumbs:
        return None
    cols = min(max_cols, len(thumbs))
    rows = math.ceil(len(thumbs) / cols)
    grid = np.zeros((rows * 128, cols * 128, 3), dtype=np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        grid[r * 128 : (r + 1) * 128, c * 128 : (c + 1) * 128] = t
    out_path = out_dir / "crops_grid.png"
    cv2.imwrite(str(out_path), grid)
    return out_path


def render_bbox_3d(scene_dir: Path, out_dir: Path) -> Optional[Path]:
    bbox_path = scene_dir / "3dbbox.json"
    if not bbox_path.exists():
        bbox_path = scene_dir / "3dbbox_ground.json"
    if not bbox_path.exists():
        return None
    is_ground = bbox_path.name == "3dbbox_ground.json"
    _read_boxes, render_bbox_overlay = _import_util()
    image = render_bbox_overlay(scene_dir, is_ground=is_ground, bbox_file=bbox_path.name)
    out_path = out_dir / "bbox3d_overlay.png"
    cv2.imwrite(str(out_path), image)
    return out_path


def render_pointcloud_projection(
    scene_dir: Path,
    out_dir: Path,
    max_points: int = 120000,
) -> Optional[Path]:
    """Project depth_scene point cloud to image and overlay points."""
    img_path = scene_dir / "input.png"
    cam_path = scene_dir / "cam_params.json"
    ply_path = scene_dir / "depth_scene.ply"
    if not img_path.exists() or not cam_path.exists() or not ply_path.exists():
        return None

    import trimesh

    image = cv2.imread(str(img_path))
    if image is None:
        return None
    with open(cam_path, "r") as f:
        cam = json.load(f)
    K = np.asarray(cam["K"], dtype=np.float64)
    H, W = image.shape[:2]

    cloud = trimesh.load(str(ply_path))
    if isinstance(cloud, trimesh.Scene):
        geos = [g for g in cloud.geometry.values() if hasattr(g, "vertices")]
        if not geos:
            return None
        pts = np.concatenate([np.asarray(g.vertices) for g in geos], axis=0)
    else:
        pts = np.asarray(cloud.vertices)
    if pts.size == 0:
        return None

    # Keep rendering responsive for large clouds.
    if pts.shape[0] > max_points:
        idx = np.linspace(0, pts.shape[0] - 1, max_points, dtype=np.int64)
        pts = pts[idx]

    def _project_points(points_cam: np.ndarray):
        zc = points_cam[:, 2]
        valid = np.isfinite(points_cam).all(axis=1) & (zc > 1e-6)
        if not np.any(valid):
            return None
        p = points_cam[valid]
        z = p[:, 2]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        u = fx * p[:, 0] / z + cx
        v = fy * p[:, 1] / z + cy
        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not np.any(in_bounds):
            return None
        u = np.round(u[in_bounds]).astype(np.int32)
        v = np.round(v[in_bounds]).astype(np.int32)
        z = z[in_bounds]
        u = np.clip(u, 0, W - 1)
        v = np.clip(v, 0, H - 1)
        return u, v, z

    # depth_scene.ply is exported by lidar_depth.build_scene_outputs in OpenCV
    # camera coordinates, so project it directly with K.
    proj = _project_points(pts)
    if proj is None:
        return None
    mode, u, v, z = ("camera_frame",) + proj

    # Depth-based coloring (near=red, far=blue in TURBO colormap).
    z_lo, z_hi = np.percentile(z, [2, 98])
    z_norm = np.clip((z - z_lo) / (z_hi - z_lo + 1e-8), 0, 1)
    z_u8 = (z_norm * 255).astype(np.uint8)
    colors = cv2.applyColorMap(z_u8.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)

    point_radius = 2
    overlay = image.copy()
    if point_radius <= 0:
        overlay[v, u] = colors
    else:
        # Rasterize then dilate to make projected points easier to inspect.
        point_layer = np.zeros_like(image)
        point_mask = np.zeros((H, W), dtype=np.uint8)
        point_layer[v, u] = colors
        point_mask[v, u] = 255
        kernel_size = 2 * point_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        point_layer = cv2.dilate(point_layer, kernel, iterations=1)
        point_mask = cv2.dilate(point_mask, kernel, iterations=1)
        active = point_mask > 0
        overlay[active] = point_layer[active]
    blended = cv2.addWeighted(image, 0.55, overlay, 0.75, 0.0)
    cv2.putText(
        blended,
        f"Projected points: {len(u)} ({mode}, r={point_radius})",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    out_path = out_dir / "pointcloud_projection.png"
    cv2.imwrite(str(out_path), blended)
    return out_path


def render_bev_3d(scene_dir: Path, out_dir: Path, canvas: int = 800) -> Optional[Path]:
    """Render a simple bird's-eye-view (X-Z) of 3D boxes."""
    bbox_path = scene_dir / "3dbbox.json"
    if not bbox_path.exists():
        bbox_path = scene_dir / "3dbbox_ground.json"
    if not bbox_path.exists():
        return None
    with open(bbox_path, "r") as f:
        boxes = json.load(f)
    if not boxes:
        return None

    # Gather XZ extents from all 3D box vertices.
    xz_all = []
    for b in boxes:
        verts = np.asarray(b.get("bbox3D_cam", []), dtype=np.float32)
        if verts.ndim == 2 and verts.shape[1] == 3 and len(verts) >= 4:
            xz_all.append(verts[:, [0, 2]])
    if not xz_all:
        return None
    xz = np.vstack(xz_all)

    # Optional BEV point cloud overlay from depth_scene.
    pc_xz = None
    ply_path = scene_dir / "depth_scene.ply"
    if ply_path.exists():
        try:
            import trimesh

            cloud = trimesh.load(str(ply_path))
            if isinstance(cloud, trimesh.Scene):
                geos = [g for g in cloud.geometry.values() if hasattr(g, "vertices")]
                if geos:
                    pts = np.concatenate([np.asarray(g.vertices) for g in geos], axis=0)
                else:
                    pts = np.zeros((0, 3), dtype=np.float32)
            else:
                pts = np.asarray(cloud.vertices)
            if pts.size > 0:
                finite = np.isfinite(pts).all(axis=1)
                pts = pts[finite]
                if pts.shape[0] > 150000:
                    idx = np.linspace(0, pts.shape[0] - 1, 150000, dtype=np.int64)
                    pts = pts[idx]
                if pts.size > 0:
                    pc_xz = pts[:, [0, 2]]
                    xz = np.vstack([xz, pc_xz])
        except Exception:
            pc_xz = None

    xmin, zmin = np.min(xz, axis=0)
    xmax, zmax = np.max(xz, axis=0)
    dx = float(max(xmax - xmin, 1e-6))
    dz = float(max(zmax - zmin, 1e-6))
    pad_ratio = 0.08
    xmin -= dx * pad_ratio
    xmax += dx * pad_ratio
    zmin -= dz * pad_ratio
    zmax += dz * pad_ratio

    bev = np.full((canvas, canvas, 3), 245, dtype=np.uint8)
    # Light grid.
    for t in np.linspace(0, 1, 11):
        x = int(t * (canvas - 1))
        cv2.line(bev, (x, 0), (x, canvas - 1), (230, 230, 230), 1, cv2.LINE_AA)
        cv2.line(bev, (0, x), (canvas - 1, x), (230, 230, 230), 1, cv2.LINE_AA)

    def to_px(x: float, z: float) -> tuple[int, int]:
        u = (x - xmin) / (xmax - xmin + 1e-8)
        v = (z - zmin) / (zmax - zmin + 1e-8)
        px = int(np.clip(u * (canvas - 1), 0, canvas - 1))
        py = int(np.clip((1.0 - v) * (canvas - 1), 0, canvas - 1))
        return px, py

    # Draw point cloud first (faint gray), then 3D boxes on top.
    if pc_xz is not None and pc_xz.shape[0] > 0:
        for x, z in pc_xz:
            px, py = to_px(float(x), float(z))
            bev[py, px] = (170, 170, 170)

    color = (30, 80, 220)
    for b in boxes:
        verts = np.asarray(b.get("bbox3D_cam", []), dtype=np.float32)
        if not (verts.ndim == 2 and verts.shape[1] == 3 and len(verts) >= 8):
            continue
        # Bottom face in convert_box_vertices order: indices 0..3.
        pts = np.array([to_px(float(verts[i, 0]), float(verts[i, 2])) for i in [0, 1, 2, 3]], dtype=np.int32)
        cv2.polylines(bev, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
        center = b.get("center_cam", [0, 0, 0])
        cpx, cpy = to_px(float(center[0]), float(center[2]))
        cv2.circle(bev, (cpx, cpy), 3, (20, 20, 20), -1)
        label = f"{b.get('obj_id', '?')}_{b.get('category_name', 'obj')}"
        cv2.putText(bev, label, (cpx + 4, cpy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (20, 20, 20), 1, cv2.LINE_AA)

    title = "BEV (X-Z): point cloud + 3D boxes" if pc_xz is not None else "BEV (X-Z) with 3D boxes"
    cv2.putText(bev, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)
    out_path = out_dir / "bev_3d.png"
    cv2.imwrite(str(out_path), bev)
    return out_path


def render_mesh_overlay(
    scene_dir: Path,
    out_dir: Path,
    max_faces_per_mesh: int = 1200,
    max_vertices_per_mesh: int = 4000,
) -> Optional[Path]:
    """Project reconstructed camera-frame meshes onto the RGB image."""
    img_path = scene_dir / "input.png"
    cam_path = scene_dir / "cam_params.json"
    recons_dir = scene_dir / "reconstruction"
    if not img_path.exists() or not cam_path.exists() or not recons_dir.exists():
        return None

    import trimesh

    image = cv2.imread(str(img_path))
    if image is None:
        return None
    with open(cam_path, "r") as f:
        cam = json.load(f)
    K = np.asarray(cam["K"], dtype=np.float64)
    H, W = image.shape[:2]

    mesh_paths = sorted(
        p for p in recons_dir.glob("*.glb")
        if p.name != "full_scene.glb" and not p.name.startswith("background")
    )
    if not mesh_paths:
        return None

    rng = np.random.default_rng(0)
    overlay = image.copy()
    palette = [
        (40, 220, 255),
        (255, 180, 40),
        (120, 255, 80),
        (255, 80, 180),
        (180, 120, 255),
        (80, 180, 255),
    ]
    rendered = 0

    def _project(points_cam: np.ndarray):
        z = points_cam[:, 2]
        valid = np.isfinite(points_cam).all(axis=1) & (z > 1e-6)
        uv = np.full((points_cam.shape[0], 2), np.nan, dtype=np.float64)
        if not np.any(valid):
            return uv, valid
        p = points_cam[valid]
        z = p[:, 2]
        uv[valid, 0] = K[0, 0] * p[:, 0] / z + K[0, 2]
        uv[valid, 1] = K[1, 1] * p[:, 1] / z + K[1, 2]
        return uv, valid

    for mesh_idx, mesh_path in enumerate(mesh_paths):
        loaded = trimesh.load(str(mesh_path), force="scene")
        meshes = loaded.dump() if isinstance(loaded, trimesh.Scene) else [loaded]
        color = palette[mesh_idx % len(palette)]
        for mesh in meshes:
            if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
                continue
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            faces = np.asarray(getattr(mesh, "faces", []), dtype=np.int64)
            uv, valid = _project(vertices)

            if faces.ndim == 2 and faces.shape[1] == 3 and len(faces) > 0:
                face_valid = valid[faces].all(axis=1)
                face_uv = uv[faces]
                in_margin = (
                    (face_uv[:, :, 0] >= -W) & (face_uv[:, :, 0] <= 2 * W) &
                    (face_uv[:, :, 1] >= -H) & (face_uv[:, :, 1] <= 2 * H)
                ).all(axis=1)
                draw_faces = faces[face_valid & in_margin]
                if len(draw_faces) > max_faces_per_mesh:
                    idx = rng.choice(len(draw_faces), max_faces_per_mesh, replace=False)
                    draw_faces = draw_faces[idx]
                for face in draw_faces:
                    pts = np.round(uv[face]).astype(np.int32)
                    pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
                    pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
                    cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)
                rendered += len(draw_faces)

            vert_idx = np.where(valid)[0]
            if len(vert_idx) > max_vertices_per_mesh:
                vert_idx = rng.choice(vert_idx, max_vertices_per_mesh, replace=False)
            if len(vert_idx) > 0:
                pts = np.round(uv[vert_idx]).astype(np.int32)
                in_img = (pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)
                pts = pts[in_img]
                for u, v in pts:
                    cv2.circle(overlay, (int(u), int(v)), 1, color, -1, cv2.LINE_AA)

    blended = cv2.addWeighted(image, 0.65, overlay, 0.75, 0.0)
    cv2.putText(
        blended,
        f"Projected reconstructed meshes: {len(mesh_paths)} objects, {rendered} faces",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    out_path = out_dir / "mesh_overlay.png"
    cv2.imwrite(str(out_path), blended)
    return out_path


def render_compose(scene_dir: Path, out_dir: Path, modes: List[str]) -> Optional[Path]:
    panels = []
    for mode in modes:
        if mode == "gt_2d":
            p = render_gt_2d(scene_dir, out_dir)
        elif mode == "depth":
            ps = render_depth(scene_dir, out_dir)
            p = ps[0] if ps else None
        elif mode == "crops":
            p = render_crops_grid(scene_dir, out_dir)
        elif mode == "bbox_3d":
            p = render_bbox_3d(scene_dir, out_dir)
        elif mode == "pc_proj":
            p = render_pointcloud_projection(scene_dir, out_dir)
        elif mode == "bev_3d":
            p = render_bev_3d(scene_dir, out_dir)
        elif mode in ("mesh", "mesh_overlay"):
            p = render_mesh_overlay(scene_dir, out_dir)
        else:
            p = None
        if p and p.exists():
            panels.append(cv2.imread(str(p)))

    if not panels:
        return None

    target_h = 360
    resized = []
    for im in panels:
        scale = target_h / im.shape[0]
        resized.append(cv2.resize(im, (int(im.shape[1] * scale), target_h)))
    summary = np.hstack(resized)
    out_path = out_dir / "summary.png"
    cv2.imwrite(str(out_path), summary)
    return out_path


def run_blender_scene(scene_dir: Path, verbose: bool = False) -> bool:
    bpy_script = Path(__file__).resolve().parent.parent / "bpy_render" / "bpy_load_blender_pointmap_plot.py"
    if not bpy_script.exists():
        print(f"Blender script not found: {bpy_script}")
        return False

    ply = scene_dir / "depth_scene_no_edge.ply"
    if not ply.exists():
        ply = scene_dir / "depth_scene.ply"
    if not ply.exists():
        print(f"No PLY in {scene_dir}")
        return False

    all_dirs = sorted(
        [p for p in scene_dir.parent.iterdir() if p.is_dir()],
        key=lambda p: p.name,
    )
    if scene_dir not in all_dirs:
        print(f"Scene dir not under parent: {scene_dir}")
        return False
    idx = all_dirs.index(scene_dir)

    blender = os.environ.get("BLENDER_BIN", "blender")
    cmd = [
        blender,
        "--background",
        "--python",
        str(bpy_script),
        "--",
        "--root",
        str(scene_dir.parent),
        "--start_idx",
        str(idx),
        "--end_idx",
        str(idx),
    ]
    if verbose:
        cmd.append("--verbose")
    try:
        subprocess.run(cmd, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Blender render failed: {e}")
        return False


def visualize_scene(
    scene_dir: Path,
    modes: List[str],
    backend: str = "preview",
    viz_subdir: str = "viz",
    verbose: bool = False,
) -> None:
    out_dir = _viz_dir(scene_dir, viz_subdir)
    if "compose" in modes:
        sub_modes = [m for m in modes if m != "compose"] or ["gt_2d", "depth", "crops", "bbox_3d", "mesh_overlay", "pc_proj", "bev_3d"]
        render_compose(scene_dir, out_dir, sub_modes)
    else:
        for mode in modes:
            if mode == "gt_2d":
                render_gt_2d(scene_dir, out_dir)
            elif mode == "depth":
                render_depth(scene_dir, out_dir)
            elif mode == "crops":
                render_crops_grid(scene_dir, out_dir)
            elif mode == "bbox_3d":
                render_bbox_3d(scene_dir, out_dir)
            elif mode == "pc_proj":
                render_pointcloud_projection(scene_dir, out_dir)
            elif mode == "bev_3d":
                render_bev_3d(scene_dir, out_dir)
            elif mode in ("mesh", "mesh_overlay"):
                render_mesh_overlay(scene_dir, out_dir)

    if backend in ("blender", "both"):
        run_blender_scene(scene_dir, verbose=verbose)


def list_scene_dirs(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)


def main():
    parser = argparse.ArgumentParser(description="Visualize LabelAny3D scene outputs")
    parser.add_argument("--scene_dir", type=str, default=None, help="Single scene directory")
    parser.add_argument("--root", type=str, default=None, help="Root with multiple scene subdirs")
    parser.add_argument(
        "--mode",
        type=str,
        default="compose",
        help="Comma-separated: gt_2d,depth,crops,bbox_3d,mesh_overlay,pc_proj,bev_3d,compose",
    )
    parser.add_argument(
        "--backend",
        choices=["preview", "blender", "both"],
        default="preview",
    )
    parser.add_argument("--viz_subdir", type=str, default="viz")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    modes = [m.strip() for m in args.mode.split(",") if m.strip()]
    if not args.scene_dir and not args.root:
        parser.error("Provide --scene_dir or --root")

    if args.scene_dir:
        scenes = [Path(args.scene_dir)]
    else:
        scenes = list_scene_dirs(Path(args.root))

    for scene_dir in scenes:
        print(f"Visualizing {scene_dir}")
        visualize_scene(
            scene_dir,
            modes=modes,
            backend=args.backend,
            viz_subdir=args.viz_subdir,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
