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
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

sys.path = ["./"] + sys.path

from integrations.py123d.nuscenes_adapter import (
    NUSCENES_SURROUND_CAMERAS,
    discover_camera_view_dirs,
    get_primary_camera_view,
)


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


def _det2d_labels_from_crops(scene_dir: Path, n_boxes: int) -> List[str]:
    """Best-effort labels aligned with bboxes.json order (crops use reverse index)."""
    crop_dir = scene_dir / "crops"
    if not crop_dir.exists() or n_boxes <= 0:
        return [f"det_{i}" for i in range(n_boxes)]
    # get_crops_enhanced iterates masks reversed, so crop id j maps to selected order.
    by_id: Dict[int, str] = {}
    for path in crop_dir.glob("*_reproj.png"):
        stem = path.stem.replace("_reproj", "")
        if "_" not in stem:
            continue
        idx_str, label = stem.split("_", 1)
        if idx_str.isdigit():
            by_id[int(idx_str)] = label.replace("_", " ")
    labels = []
    for i in range(n_boxes):
        # selected_bboxes append order follows reverse iteration: first saved ~= highest id
        # Fall back to matching by sorted ids when counts differ.
        labels.append(by_id.get(i, by_id.get(n_boxes - 1 - i, f"det_{i}")))
    if by_id and len(by_id) == n_boxes:
        # Prefer explicit id order 0..n-1 when all ids present.
        labels = [by_id[i] for i in sorted(by_id.keys())]
    return labels


def render_det_2d(scene_dir: Path, out_dir: Path) -> Optional[Path]:
    """Draw pipeline 2D detections (crops / bboxes.json) on RGB."""
    img_path = scene_dir / "input.png"
    if not img_path.exists():
        return None
    image = cv2.imread(str(img_path))
    if image is None:
        return None
    h, w = image.shape[:2]
    color = (255, 140, 0)  # BGR orange for Pred-2D
    n_drawn = 0

    crop_dir = scene_dir / "crops"
    drawn_from_crops = False
    if crop_dir.exists():
        try:
            from util import restore_mask_from_crop

            overlay = image.copy()
            for path in sorted(crop_dir.glob("*_reproj.png")):
                stem = path.stem.replace("_reproj", "")
                params_path = crop_dir / f"{stem}_crop_params.npy"
                if not params_path.exists():
                    continue
                params = np.load(params_path)
                crop = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if crop is None or crop.ndim < 3 or crop.shape[2] < 4:
                    continue
                mask = restore_mask_from_crop(
                    crop[:, :, 3] > 127,
                    float(params[0]),
                    float(params[1]),
                    float(params[2]),
                    (h, w),
                )
                ys, xs = np.where(mask)
                if ys.size == 0:
                    continue
                x1, x2 = int(xs.min()), int(xs.max())
                y1, y2 = int(ys.min()), int(ys.max())
                overlay[mask] = (
                    0.55 * overlay[mask].astype(np.float32)
                    + 0.45 * np.array(color, dtype=np.float32)
                ).astype(np.uint8)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                label = stem.replace("_", " ")
                cv2.putText(
                    overlay,
                    label,
                    (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )
                n_drawn += 1
            if n_drawn > 0:
                image = overlay
                drawn_from_crops = True
        except Exception:
            drawn_from_crops = False

    if not drawn_from_crops:
        bbox_path = scene_dir / "bboxes.json"
        if not bbox_path.exists():
            return None
        with open(bbox_path, "r") as f:
            bboxes = json.load(f)
        if not isinstance(bboxes, list) or not bboxes:
            return None
        labels = _det2d_labels_from_crops(scene_dir, len(bboxes))
        for i, box in enumerate(bboxes):
            if len(box) < 4:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
            # Support xywh
            if x2 < x1 or y2 < y1:
                x2, y2 = x1 + max(1, int(box[2])), y1 + max(1, int(box[3]))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                image,
                labels[i] if i < len(labels) else f"det_{i}",
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
            n_drawn += 1

    if n_drawn == 0:
        return None
    out_path = out_dir / "det2d_overlay.png"
    cv2.imwrite(str(out_path), image)
    return out_path


_LA_BOX_COLORS = {
    "person": (0, 165, 255),
    "car": (0, 200, 0),
    "truck": (0, 128, 255),
    "bus": (255, 128, 0),
    "motorcycle": (255, 0, 255),
    "bicycle": (255, 255, 0),
    "traffic_light": (200, 0, 200),
    "traffic_sign": (180, 0, 180),
}


def _la_color_for_label(label: str) -> tuple:
    key = label.strip().lower().replace(" ", "_")
    if key in _LA_BOX_COLORS:
        return _LA_BOX_COLORS[key]
    for cat, color in _LA_BOX_COLORS.items():
        if cat in key or key in cat:
            return color
    return (255, 140, 0)


def _collect_saved_la_detections(scene_dir: Path) -> List[Dict]:
    """Collect LocateAnything (or pipeline) 2D dets from crops / bboxes.json."""
    dets: List[Dict] = []
    crop_dir = scene_dir / "crops"
    if crop_dir.exists():
        try:
            from util import restore_mask_from_crop

            img_path = scene_dir / "input.png"
            image = cv2.imread(str(img_path)) if img_path.exists() else None
            h, w = (image.shape[:2] if image is not None else (0, 0))
            for path in sorted(crop_dir.glob("*_reproj.png")):
                stem = path.stem.replace("_reproj", "")
                if "_" not in stem:
                    continue
                idx_str, label = stem.split("_", 1)
                params_path = crop_dir / f"{stem}_crop_params.npy"
                if not params_path.exists() or image is None:
                    continue
                params = np.load(params_path)
                crop = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if crop is None or crop.ndim < 3 or crop.shape[2] < 4:
                    continue
                mask = restore_mask_from_crop(
                    crop[:, :, 3] > 127,
                    float(params[0]),
                    float(params[1]),
                    float(params[2]),
                    (h, w),
                )
                ys, xs = np.where(mask)
                if ys.size == 0:
                    continue
                dets.append(
                    {
                        "id": int(idx_str) if idx_str.isdigit() else len(dets),
                        "label": label.replace("_", " "),
                        "bbox_xyxy": [
                            float(xs.min()),
                            float(ys.min()),
                            float(xs.max()),
                            float(ys.max()),
                        ],
                    }
                )
            if dets:
                dets.sort(key=lambda d: d.get("id", 0))
                return dets
        except Exception:
            dets = []

    bbox_path = scene_dir / "bboxes.json"
    if not bbox_path.exists():
        return []
    with open(bbox_path, "r") as f:
        boxes = json.load(f)
    if not isinstance(boxes, list) or not boxes:
        return []
    labels = _det2d_labels_from_crops(scene_dir, len(boxes))
    for i, box in enumerate(boxes):
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        if x2 < x1 or y2 < y1:
            x2, y2 = x1 + max(1.0, float(box[2])), y1 + max(1.0, float(box[3]))
        dets.append(
            {
                "id": i,
                "label": labels[i] if i < len(labels) else f"det_{i}",
                "bbox_xyxy": [x1, y1, x2, y2],
            }
        )
    return dets


def _draw_la_overlay(image_bgr: np.ndarray, dets: List[Dict], title: str) -> np.ndarray:
    overlay = image_bgr.copy()
    h, w = overlay.shape[:2]
    for det in dets:
        box = det.get("bbox_xyxy") or []
        if len(box) < 4:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        label = str(det.get("label", "obj"))
        color = _la_color_for_label(label)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        tag = label
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ty = max(th + 4, y1)
        cv2.rectangle(overlay, (x1, ty - th - 6), (x1 + tw + 6, ty + 2), color, -1)
        cv2.putText(
            overlay,
            tag,
            (x1 + 3, ty - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    cv2.rectangle(overlay, (0, 0), (w, 28), (20, 20, 20), -1)
    cv2.putText(
        overlay,
        title,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    return overlay


def render_locateanything_2d(
    scene_dir: Path,
    out_dir: Path,
    categories: Optional[List[str]] = None,
    model_path: Optional[str] = None,
    device: str = "cuda",
    generation_mode: str = "hybrid",
    run_infer: bool = False,
) -> Optional[Path]:
    """Draw LocateAnything detections (saved crops/bboxes by default; optional re-infer)."""
    img_path = scene_dir / "input.png"
    if not img_path.exists():
        return None
    image_bgr = cv2.imread(str(img_path))
    if image_bgr is None:
        return None

    if categories is None:
        categories = [
            "person",
            "car",
            "truck",
            "bus",
            "motorcycle",
            "bicycle",
            "traffic_light",
            "traffic_sign",
        ]

    dets: List[Dict] = []
    source = "saved"
    if os.environ.get("LA3D_LA_INFER", "").strip() in ("1", "true", "True"):
        run_infer = True
    if not run_infer:
        dets = _collect_saved_la_detections(scene_dir)

    if run_infer or not dets:
        from integrations.locateanything.detect import detect_boxes

        image_pil = Image.open(img_path).convert("RGB")
        labels, boxes = detect_boxes(
            image_pil,
            categories=categories,
            device=device,
            model_path=model_path,
            generation_mode=generation_mode,
            allowed_categories=None,
            min_box_area=400,
        )
        dets = []
        for label, box in zip(labels, boxes):
            if len(box) < 4:
                continue
            dets.append(
                {
                    "label": label,
                    "bbox_xyxy": [float(v) for v in box[:4]],
                }
            )
        source = "infer"

    if not dets:
        return None

    title = f"LocateAnything ({len(dets)} boxes, {source})"
    overlay = _draw_la_overlay(image_bgr, dets, title)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "locateanything_dets.json", "w") as f:
        json.dump(
            {"categories": categories, "source": source, "detections": dets},
            f,
            indent=2,
        )
    out_path = out_dir / "locateanything_overlay.png"
    cv2.imwrite(str(out_path), overlay)
    return out_path


def render_surround_locateanything(
    scene_dir: Path,
    out_dir: Path,
    categories: Optional[List[str]] = None,
    model_path: Optional[str] = None,
    device: str = "cuda",
    run_infer: bool = False,
) -> Optional[Path]:
    """Surround grid of LocateAnything overlays for all cameras."""

    def _panel(view_dir: Path, sub_out: Path) -> Optional[Path]:
        return render_locateanything_2d(
            view_dir,
            sub_out,
            categories=categories,
            model_path=model_path,
            device=device,
            run_infer=run_infer,
        )

    panels = _collect_surround_panels(scene_dir, out_dir, _panel)
    grid = _compose_surround_spatial_grid(panels, target_h=240)
    if grid is None:
        return None
    out_path = out_dir / "surround_locateanything.png"
    cv2.imwrite(str(out_path), grid)
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


def _load_depth_colormap(
    scene_dir: Path,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Optional[tuple]:
    """Return (depth, depth_color_bgr, vmin, vmax) or None."""
    depth_path = scene_dir / "depth_map.npy"
    if not depth_path.exists():
        return None
    depth = np.load(depth_path)
    valid = depth[np.isfinite(depth) & (depth > 0)]
    if valid.size == 0:
        return None
    if vmin is None or vmax is None:
        vmin, vmax = np.percentile(valid, [2, 98])
    depth_norm = np.clip((depth - vmin) / (vmax - vmin + 1e-8), 0, 1)
    depth_u8 = (depth_norm * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    depth_color[~np.isfinite(depth) | (depth <= 0)] = 0
    return depth, depth_color, float(vmin), float(vmax)


def _project_depth_scene_points(
    scene_dir: Path,
    max_points: int = 120000,
) -> Optional[tuple]:
    """Project depth_scene.ply into image coords. Returns (u, v, z, K, H, W) or None."""
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
    if pts.shape[0] > max_points:
        idx = np.linspace(0, pts.shape[0] - 1, max_points, dtype=np.int64)
        pts = pts[idx]

    zc = pts[:, 2]
    valid = np.isfinite(pts).all(axis=1) & (zc > 1e-6)
    if not np.any(valid):
        return None
    p = pts[valid]
    z = p[:, 2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * p[:, 0] / z + cx
    v = fy * p[:, 1] / z + cy
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not np.any(in_bounds):
        return None
    u = np.clip(np.round(u[in_bounds]).astype(np.int32), 0, W - 1)
    v = np.clip(np.round(v[in_bounds]).astype(np.int32), 0, H - 1)
    z = z[in_bounds]
    return u, v, z, K, H, W


def _draw_points_on(
    base: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    colors: np.ndarray,
    point_radius: int = 2,
    blend: float = 0.7,
) -> np.ndarray:
    """Overlay colored points onto an image."""
    H, W = base.shape[:2]
    overlay = base.copy()
    if point_radius <= 0:
        overlay[v, u] = colors
        return overlay
    point_layer = np.zeros_like(base)
    point_mask = np.zeros((H, W), dtype=np.uint8)
    point_layer[v, u] = colors
    point_mask[v, u] = 255
    kernel_size = 2 * point_radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    point_layer = cv2.dilate(point_layer, kernel, iterations=1)
    point_mask = cv2.dilate(point_mask, kernel, iterations=1)
    active = point_mask > 0
    overlay[active] = (
        (1.0 - blend) * overlay[active].astype(np.float32)
        + blend * point_layer[active].astype(np.float32)
    ).astype(np.uint8)
    return overlay


def _panel_banner(im: np.ndarray, title: str) -> np.ndarray:
    panel = im.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 30), (16, 16, 16), -1)
    cv2.putText(
        panel,
        title,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return panel


def render_depth_pc_combo(
    scene_dir: Path,
    out_dir: Path,
    max_points: int = 120000,
) -> List[Path]:
    """Combine dense depth estimate with projected scene point cloud.

    Outputs:
      - depth_pc_combo.png: RGB | depth colormap | depth+PC overlay | residual
      - depth_pc_overlay.png: compact depth colormap with PC points (shared scale)
    """
    depth_pack = _load_depth_colormap(scene_dir)
    if depth_pack is None:
        return []
    depth, depth_color, vmin, vmax = depth_pack

    rgb_path = scene_dir / "input.png"
    rgb = cv2.imread(str(rgb_path)) if rgb_path.exists() else None
    if rgb is None or rgb.shape[:2] != depth_color.shape[:2]:
        # Fall back to depth-only if RGB missing / mismatched.
        rgb = depth_color.copy()

    proj = _project_depth_scene_points(scene_dir, max_points=max_points)
    paths: List[Path] = []
    if proj is None:
        # Still emit depth-only combo so surround/compose keep working.
        panel = np.hstack([
            _panel_banner(rgb, "RGB"),
            _panel_banner(depth_color, f"Depth est. [{vmin:.1f},{vmax:.1f}]m"),
        ])
        p = out_dir / "depth_pc_combo.png"
        cv2.imwrite(str(p), panel)
        paths.append(p)
        p2 = out_dir / "depth_pc_overlay.png"
        cv2.imwrite(str(p2), _panel_banner(depth_color, "Depth (no PC)"))
        paths.append(p2)
        return paths

    u, v, z, _K, _H, _W = proj
    z_norm = np.clip((z - vmin) / (vmax - vmin + 1e-8), 0, 1)
    z_u8 = (z_norm * 255).astype(np.uint8)
    pc_colors = cv2.applyColorMap(z_u8.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)

    # Panel A: RGB with PC (depth-colored, shared scale with dense depth).
    rgb_pc = _draw_points_on(rgb, u, v, pc_colors, point_radius=2, blend=0.85)
    rgb_pc = _panel_banner(rgb_pc, f"RGB + PC ({len(u)} pts)")

    # Panel B: dense depth estimate.
    depth_panel = _panel_banner(depth_color, f"Depth est. [{vmin:.1f},{vmax:.1f}]m")

    # Panel C: depth colormap + PC points (same TURBO scale) — fusion view.
    depth_pc = _draw_points_on(depth_color, u, v, pc_colors, point_radius=2, blend=0.95)
    # Draw PC as slightly brighter rings so they stand out on dense depth.
    for ui, vi in zip(u[:: max(1, len(u) // 4000)], v[:: max(1, len(u) // 4000)]):
        cv2.circle(depth_pc, (int(ui), int(vi)), 3, (255, 255, 255), 1, cv2.LINE_AA)
    depth_pc = _panel_banner(depth_pc, "Depth + PC (shared scale)")

    # Panel D: residual at PC hits — est depth vs LiDAR/scene PC z.
    residual_vis = np.zeros_like(rgb)
    est_at = depth[v, u]
    valid_res = np.isfinite(est_at) & (est_at > 0) & np.isfinite(z)
    err = est_at[valid_res] - z[valid_res]
    if np.any(valid_res):
        # Diverging: blue = PC farther than est, red = PC closer; green ~ match.
        abs_hi = float(np.percentile(np.abs(err), 95)) if err.size else 1.0
        abs_hi = max(abs_hi, 0.5)
        err_clip = np.clip(err / abs_hi, -1.0, 1.0)
        # Map [-1,1] -> BGR: blue<-0, green~0, red>0
        b = np.clip((-err_clip).clip(0, 1) * 255, 0, 255).astype(np.uint8)
        r = np.clip(err_clip.clip(0, 1) * 255, 0, 255).astype(np.uint8)
        g = np.clip((1.0 - np.abs(err_clip)) * 200, 0, 255).astype(np.uint8)
        res_colors = np.stack([b, g, r], axis=1)  # BGR
        residual_vis = _draw_points_on(
            rgb * 0 + 32,
            u[valid_res],
            v[valid_res],
            res_colors,
            point_radius=2,
            blend=1.0,
        )
        med = float(np.median(np.abs(err)))
        p90 = float(np.percentile(np.abs(err), 90))
        residual_vis = _panel_banner(
            residual_vis,
            f"Residual est-PC |med|={med:.2f}m p90={p90:.2f}m (red:est>PC)",
        )
    else:
        residual_vis = _panel_banner(residual_vis, "Residual (no valid PC hits)")

    combo = np.hstack([rgb_pc, depth_panel, depth_pc, residual_vis])
    p_combo = out_dir / "depth_pc_combo.png"
    cv2.imwrite(str(p_combo), combo)
    paths.append(p_combo)

    # Compact overlay for surround grids (not too wide).
    compact = _draw_points_on(depth_color, u, v, pc_colors, point_radius=2, blend=0.9)
    compact = _panel_banner(compact, f"Depth+PC [{vmin:.1f},{vmax:.1f}]m n={len(u)}")
    p_overlay = out_dir / "depth_pc_overlay.png"
    cv2.imwrite(str(p_overlay), compact)
    paths.append(p_overlay)

    # Persist residual stats for quick inspection.
    if np.any(valid_res):
        stats = {
            "n_pc": int(len(u)),
            "n_residual": int(np.count_nonzero(valid_res)),
            "depth_vmin": vmin,
            "depth_vmax": vmax,
            "abs_err_median": float(np.median(np.abs(err))),
            "abs_err_p90": float(np.percentile(np.abs(err), 90)),
            "abs_err_mean": float(np.mean(np.abs(err))),
            "signed_err_mean": float(np.mean(err)),
        }
        with open(out_dir / "depth_pc_residual.json", "w") as f:
            json.dump(stats, f, indent=2)

    return paths


def _write_ascii_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> Path:
    """Write ASCII PLY (xyz + rgb) for VS Code / Cursor 3D viewer plugins."""
    pts = np.asarray(points, dtype=np.float64)
    cols = np.asarray(colors)
    if cols.ndim != 2 or cols.shape[0] != pts.shape[0]:
        raise ValueError("colors must be (N,3) or (N,4) matching points")
    if cols.shape[1] >= 3:
        cols = cols[:, :3]
    if cols.dtype != np.uint8:
        if cols.max() <= 1.0:
            cols = (np.clip(cols, 0, 1) * 255).astype(np.uint8)
        else:
            cols = np.clip(cols, 0, 255).astype(np.uint8)
    n = pts.shape[0]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    # Vectorized body write (much faster than per-point Python loops).
    body = np.column_stack([pts, cols.astype(np.float64)])
    with open(path, "w") as f:
        f.write(header)
        np.savetxt(f, body, fmt="%.6f %.6f %.6f %d %d %d")
    return path


def _normalize_rgb_u8(colors: np.ndarray, n: int) -> np.ndarray:
    cols = np.asarray(colors)
    if cols.ndim != 2 or cols.shape[0] != n:
        raise ValueError("colors must be (N,3) or (N,4) matching points")
    if cols.shape[1] >= 3:
        cols = cols[:, :3]
    if cols.dtype != np.uint8:
        if cols.max() <= 1.0:
            cols = (np.clip(cols, 0, 1) * 255).astype(np.uint8)
        else:
            cols = np.clip(cols, 0, 255).astype(np.uint8)
    return cols


def _write_pcd(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    *,
    binary: bool = True,
) -> Path:
    """Write PCD (xyz + rgb). Prefer Open3D; fall back to ASCII PCD."""
    pts = np.asarray(points, dtype=np.float64)
    cols = _normalize_rgb_u8(colors, pts.shape[0])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64) / 255.0)
        o3d.io.write_point_cloud(str(path), pcd, write_ascii=not binary, compressed=False)
        return path
    except Exception:
        pass

    # ASCII PCD fallback (FIELDS x y z rgb with packed float rgb).
    n = pts.shape[0]
    rgb_u32 = (
        (cols[:, 0].astype(np.uint32) << 16)
        | (cols[:, 1].astype(np.uint32) << 8)
        | cols[:, 2].astype(np.uint32)
    )
    rgb_f32 = rgb_u32.view(np.float32)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z rgb\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA ascii\n"
    )
    body = np.column_stack([pts, rgb_f32.astype(np.float64)])
    with open(path, "w") as f:
        f.write(header)
        np.savetxt(f, body, fmt="%.6f %.6f %.6f %.8e")
    return path


def _write_points(path_stem: Path, points: np.ndarray, colors: np.ndarray) -> List[Path]:
    """Write both ASCII PLY and PCD for the same cloud."""
    ply = _write_ascii_ply(path_stem.with_suffix(".ply"), points, colors)
    pcd = _write_pcd(path_stem.with_suffix(".pcd"), points, colors, binary=True)
    return [ply, pcd]

def _write_colored_mesh_ply(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    faces: np.ndarray,
) -> Path:
    """Binary PLY mesh with vertex colors — works with vscode-3dviewer 'colors' material."""
    import trimesh

    pts = np.asarray(points, dtype=np.float64)
    cols = _normalize_rgb_u8(colors, pts.shape[0])
    rgba = np.concatenate([cols, np.full((cols.shape[0], 1), 255, dtype=np.uint8)], axis=1)
    mesh = trimesh.Trimesh(vertices=pts, faces=np.asarray(faces, dtype=np.int64), process=False)
    mesh.visual.vertex_colors = rgba
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path))
    return path


def _write_colored_mesh_glb(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    faces: np.ndarray,
) -> Path:
    """GLB mesh with vertex colors (ohzi GLB viewer / most 3D tools)."""
    import trimesh

    pts = np.asarray(points, dtype=np.float64)
    cols = _normalize_rgb_u8(colors, pts.shape[0])
    rgba = np.concatenate([cols, np.full((cols.shape[0], 1), 255, dtype=np.uint8)], axis=1)
    mesh = trimesh.Trimesh(vertices=pts, faces=np.asarray(faces, dtype=np.int64), process=False)
    mesh.visual.vertex_colors = rgba
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path))
    return path


def _write_mesh_bundle(
    stem: Path,
    points: np.ndarray,
    colors: np.ndarray,
    faces: np.ndarray,
) -> List[Path]:
    """Write PLY + GLB for one colored mesh."""
    return [
        _write_colored_mesh_ply(stem.with_suffix(".ply"), points, colors, faces),
        _write_colored_mesh_glb(stem.with_suffix(".glb"), points, colors, faces),
    ]


def _render_mesh_preview(
    points: np.ndarray,
    colors: np.ndarray,
    faces: np.ndarray,
    out_path: Path,
    width: int = 1280,
    height: int = 720,
) -> Optional[Path]:
    """Mesh preview PNG. Uses CPU painter fallback (no EGL/GPU required)."""
    pts = np.asarray(points, dtype=np.float64).copy()
    cols = _normalize_rgb_u8(colors, pts.shape[0]).astype(np.float64) / 255.0
    faces = np.asarray(faces, dtype=np.int32)
    if pts.shape[0] == 0 or faces.shape[0] == 0:
        return None

    # OpenCV (Y down) → display Y-up.
    pts[:, 1] *= -1.0
    return _render_mesh_preview_fallback(pts, cols, faces, out_path, width, height)

def _render_mesh_preview_fallback(
    pts_yup: np.ndarray,
    cols01: np.ndarray,
    faces: np.ndarray,
    out_path: Path,
    width: int,
    height: int,
) -> Optional[Path]:
    """Simple painter's-algorithm mesh preview without GPU."""
    center = pts_yup.mean(axis=0)
    pts = pts_yup - center
    # Look from +Z-ish after Y-up convert (camera behind looking +Z in OpenCV was - after flip...).
    # Use PCA-ish: place camera along -Z of centered cloud.
    z = pts[:, 2]
    order = np.argsort(-(pts[faces].mean(axis=1)[:, 2]))  # far to near
    # Orthographic-ish projection onto XY.
    xy = pts[:, :2]
    span = np.percentile(np.abs(xy), 98, axis=0)
    span = np.maximum(span, 1e-3)
    scale = 0.45 * min(width, height) / float(np.max(span))
    u = (xy[:, 0] * scale + width * 0.5).astype(np.int32)
    v = (-xy[:, 1] * scale + height * 0.5).astype(np.int32)
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    cols_u8 = (np.clip(cols01, 0, 1) * 255).astype(np.uint8)
    for fi in order:
        i0, i1, i2 = faces[fi]
        tri = np.array([[u[i0], v[i0]], [u[i1], v[i1]], [u[i2], v[i2]]], dtype=np.int32)
        color = tuple(
            int(x)
            for x in (
                (cols_u8[i0].astype(np.int32) + cols_u8[i1] + cols_u8[i2]) // 3
            )[::-1]
        )  # BGR
        cv2.fillConvexPoly(canvas, tri, color, lineType=cv2.LINE_AA)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    return out_path


def _depth_grid_mesh(
    depth: np.ndarray,
    K: np.ndarray,
    stride: int = 4,
    max_edge: float = 3.0,
    rgb_image: Optional[np.ndarray] = None,
) -> Optional[tuple]:
    """Build camera-frame depth surface mesh.

    Returns (pts, turbo_rgb, faces, rgb_or_None) or None.
    rgb_image: optional BGR uint8 matching depth size for vertex RGB colors.
    """
    H, W = depth.shape[:2]
    ys = np.arange(0, H, stride, dtype=np.int32)
    xs = np.arange(0, W, stride, dtype=np.int32)
    if ys.size < 2 or xs.size < 2:
        return None
    uu, vv = np.meshgrid(xs, ys)
    z = depth[vv, uu].astype(np.float64)
    valid = np.isfinite(z) & (z > 1e-6) & (z < 1e5)
    if valid.sum() < 16:
        return None

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (uu.astype(np.float64) - cx) * z / fx
    y = (vv.astype(np.float64) - cy) * z / fy
    x = np.where(valid, x, np.nan)
    y = np.where(valid, y, np.nan)
    z = np.where(valid, z, np.nan)

    gh, gw = z.shape
    pts = np.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], axis=1)
    uu_f = uu.reshape(-1)
    vv_f = vv.reshape(-1)
    keep = np.isfinite(pts).all(axis=1)
    new_id = -np.ones(gh * gw, dtype=np.int64)
    new_id[keep] = np.arange(int(keep.sum()), dtype=np.int64)
    pts_keep = pts[keep]
    uu_keep = uu_f[keep]
    vv_keep = vv_f[keep]

    valid_d = depth[np.isfinite(depth) & (depth > 0) & (depth < 1e5)]
    vmin, vmax = (np.percentile(valid_d, [2, 98]) if valid_d.size else (0.0, 1.0))
    turbo = _depth_turbo_colors(pts_keep[:, 2], float(vmin), float(vmax))

    rgb_cols = None
    if rgb_image is not None and rgb_image.shape[:2] == (H, W):
        bgr = rgb_image[vv_keep, uu_keep]
        rgb_cols = bgr[:, ::-1].copy()

    i00 = (np.arange(gh - 1)[:, None] * gw + np.arange(gw - 1)[None, :]).reshape(-1)
    i10 = i00 + 1
    i01 = i00 + gw
    i11 = i00 + gw + 1
    cell_ok = keep[i00] & keep[i10] & keep[i01] & keep[i11]
    zs = pts[:, 2]
    zs_stack = np.stack([zs[i00], zs[i10], zs[i01], zs[i11]], axis=0)
    cell_ok &= (np.nanmax(zs_stack, axis=0) - np.nanmin(zs_stack, axis=0)) <= max_edge
    if not np.any(cell_ok):
        return None
    a = new_id[i00[cell_ok]]
    b = new_id[i10[cell_ok]]
    c = new_id[i01[cell_ok]]
    d = new_id[i11[cell_ok]]
    faces = np.vstack([np.stack([a, b, d], axis=1), np.stack([a, d, c], axis=1)])
    return pts_keep, turbo, faces, rgb_cols


def export_depth_meshes(
    scene_dir: Path,
    out_dir: Path,
    stride: int = 4,
    max_edge: float = 4.0,
) -> List[Path]:
    """Export depth as colored meshes (PLY/GLB) + preview PNG.

    Outputs:
      - depth_est_mesh.{ply,glb}       TURBO depth colors (incl. ground)
      - depth_est_mesh_rgb.{ply,glb}   image RGB colors
      - depth_est_mesh_preview.png     offscreen render
    """
    depth_path = scene_dir / "depth_map.npy"
    cam_path = scene_dir / "cam_params.json"
    if not depth_path.exists() or not cam_path.exists():
        return []

    depth = np.load(depth_path)
    with open(cam_path, "r") as f:
        cam = json.load(f)
    K = np.asarray(cam["K"], dtype=np.float64)
    rgb_bgr = None
    rgb_path = scene_dir / "input.png"
    if rgb_path.exists():
        rgb_bgr = cv2.imread(str(rgb_path))

    pack = _depth_grid_mesh(
        depth, K, stride=stride, max_edge=max_edge, rgb_image=rgb_bgr
    )
    if pack is None:
        return []
    pts, turbo, faces, rgb_cols = pack

    paths: List[Path] = []
    paths.extend(_write_mesh_bundle(out_dir / "depth_est_mesh", pts, turbo, faces))
    if rgb_cols is not None:
        paths.extend(_write_mesh_bundle(out_dir / "depth_est_mesh_rgb", pts, rgb_cols, faces))

    preview = _render_mesh_preview(pts, turbo, faces, out_dir / "depth_est_mesh_preview.png")
    if preview is not None:
        paths.append(preview)
    if rgb_cols is not None:
        preview_rgb = _render_mesh_preview(
            pts, rgb_cols, faces, out_dir / "depth_est_mesh_rgb_preview.png"
        )
        if preview_rgb is not None:
            paths.append(preview_rgb)

    meta = {
        "frame": "opencv_camera",
        "n_vertices": int(pts.shape[0]),
        "n_faces": int(faces.shape[0]),
        "stride": int(stride),
        "max_edge_m": float(max_edge),
        "files": [p.name for p in paths],
        "viewer_note": (
            "Open depth_est_mesh.glb in GLB viewer, or depth_est_mesh.ply in 3D Viewer "
            "(Materials → colors). Ground is included."
        ),
    }
    with open(out_dir / "depth_est_mesh.json", "w") as f:
        json.dump(meta, f, indent=2)
    return paths


def export_scene_depth_mesh_ego(
    scene_dir: Path,
    out_dir: Path,
    stride: int = 6,
    max_edge: float = 4.0,
) -> Optional[Path]:
    """Merge per-camera depth meshes into one ego-centered mesh (PLY/GLB + preview)."""
    view_dirs = discover_camera_view_dirs(scene_dir)
    if len(view_dirs) <= 1:
        return None

    all_pts: List[np.ndarray] = []
    all_turbo: List[np.ndarray] = []
    all_rgb: List[np.ndarray] = []
    all_faces: List[np.ndarray] = []
    have_rgb = True
    offset = 0
    for view_dir in view_dirs:
        depth_path = view_dir / "depth_map.npy"
        cam_path = view_dir / "cam_params.json"
        if not depth_path.exists() or not cam_path.exists():
            continue
        depth = np.load(depth_path)
        with open(cam_path, "r") as f:
            cam = json.load(f)
        K = np.asarray(cam["K"], dtype=np.float64)
        c2w = np.asarray(cam["c2w"], dtype=np.float64)
        rgb_bgr = cv2.imread(str(view_dir / "input.png")) if (view_dir / "input.png").exists() else None
        pack = _depth_grid_mesh(
            depth, K, stride=stride, max_edge=max_edge, rgb_image=rgb_bgr
        )
        if pack is None:
            continue
        pts_c, turbo, faces, rgb_cols = pack
        pts_h = np.concatenate([pts_c, np.ones((pts_c.shape[0], 1))], axis=1)
        pts_w = (c2w @ pts_h.T).T[:, :3]
        all_pts.append(pts_w)
        all_turbo.append(turbo)
        all_faces.append(faces + offset)
        if rgb_cols is None:
            have_rgb = False
        else:
            all_rgb.append(rgb_cols)
        offset += pts_w.shape[0]

    if not all_pts:
        return None

    pts = np.concatenate(all_pts, axis=0)
    turbo = np.concatenate(all_turbo, axis=0)
    faces = np.concatenate(all_faces, axis=0)
    center = np.median(pts, axis=0)
    pts_local = pts - center

    paths = _write_mesh_bundle(out_dir / "scene_depth_mesh_ego", pts_local, turbo, faces)
    if have_rgb and len(all_rgb) == len(all_pts):
        rgb = np.concatenate(all_rgb, axis=0)
        paths.extend(_write_mesh_bundle(out_dir / "scene_depth_mesh_ego_rgb", pts_local, rgb, faces))

    preview = _render_mesh_preview(
        pts_local, turbo, faces, out_dir / "scene_depth_mesh_ego_preview.png",
        width=1600, height=900,
    )
    if preview is not None:
        paths.append(preview)

    with open(out_dir / "scene_depth_mesh_ego.json", "w") as f:
        json.dump(
            {
                "frame": "world_minus_median",
                "median_xyz": center.tolist(),
                "n_vertices": int(pts.shape[0]),
                "n_faces": int(faces.shape[0]),
                "cameras": [v.name for v in view_dirs],
                "files": [p.name for p in paths],
            },
            f,
            indent=2,
        )
    return paths[0]


def _backproject_depth(
    depth: np.ndarray,
    K: np.ndarray,
    stride: int = 4,
    max_points: int = 400000,
) -> tuple:
    """Backproject dense depth map to camera-frame XYZ. Returns (pts, u, v)."""
    H, W = depth.shape[:2]
    ys = np.arange(0, H, stride, dtype=np.int32)
    xs = np.arange(0, W, stride, dtype=np.int32)
    uu, vv = np.meshgrid(xs, ys)
    uu = uu.reshape(-1)
    vv = vv.reshape(-1)
    z = depth[vv, uu].astype(np.float64)
    valid = np.isfinite(z) & (z > 1e-6) & (z < 1e5)
    uu, vv, z = uu[valid], vv[valid], z[valid]
    if uu.size == 0:
        return np.zeros((0, 3)), uu, vv
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (uu.astype(np.float64) - cx) * z / fx
    y = (vv.astype(np.float64) - cy) * z / fy
    pts = np.stack([x, y, z], axis=1)
    if pts.shape[0] > max_points:
        idx = np.linspace(0, pts.shape[0] - 1, max_points, dtype=np.int64)
        pts, uu, vv = pts[idx], uu[idx], vv[idx]
    return pts, uu, vv


def _depth_turbo_colors(z: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    z_norm = np.clip((z - vmin) / (vmax - vmin + 1e-8), 0, 1)
    z_u8 = (z_norm * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(z_u8.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)
    return bgr[:, ::-1].copy()  # RGB


def export_depth_pc_clouds(
    scene_dir: Path,
    out_dir: Path,
    stride: int = 4,
    max_dense: int = 350000,
    max_lidar: int = 200000,
) -> List[Path]:
    """Export depth estimate + LiDAR as ASCII PLY for 3D viewer plugins.

    Files (camera frame, OpenCV: X right, Y down, Z forward):
      - depth_est_rgb.{ply,pcd}     dense depth colored by image RGB
      - depth_est_turbo.{ply,pcd}   dense depth colored by TURBO depth
      - lidar_pc.{ply,pcd}          sparse LiDAR (Z>0) with image/original colors
      - depth_pc_fused.{ply,pcd}    dense (turbo) + LiDAR (magenta highlight)
      - depth_est_mesh.ply          colored depth mesh (for vscode-3dviewer)
    """
    depth_path = scene_dir / "depth_map.npy"
    cam_path = scene_dir / "cam_params.json"
    if not depth_path.exists() or not cam_path.exists():
        return []

    depth = np.load(depth_path)
    with open(cam_path, "r") as f:
        cam = json.load(f)
    K = np.asarray(cam["K"], dtype=np.float64)

    valid = depth[np.isfinite(depth) & (depth > 0) & (depth < 1e5)]
    if valid.size == 0:
        return []
    vmin, vmax = np.percentile(valid, [2, 98])

    pts_d, uu, vv = _backproject_depth(depth, K, stride=stride, max_points=max_dense)
    if pts_d.shape[0] == 0:
        return []

    rgb = None
    rgb_path = scene_dir / "input.png"
    if rgb_path.exists():
        bgr = cv2.imread(str(rgb_path))
        if bgr is not None and bgr.shape[:2] == depth.shape[:2]:
            rgb = bgr[vv, uu][:, ::-1].copy()  # RGB

    turbo = _depth_turbo_colors(pts_d[:, 2], vmin, vmax)
    if rgb is None:
        rgb = turbo.copy()

    paths: List[Path] = []
    paths.extend(_write_points(out_dir / "depth_est_rgb", pts_d, rgb))
    paths.extend(_write_points(out_dir / "depth_est_turbo", pts_d, turbo))

    # LiDAR / scene PC: keep only points in front of camera.
    lidar_pts = None
    lidar_cols = None
    ply_src = scene_dir / "depth_scene.ply"
    if ply_src.exists():
        import trimesh

        cloud = trimesh.load(str(ply_src))
        if isinstance(cloud, trimesh.Scene):
            geos = [g for g in cloud.geometry.values() if hasattr(g, "vertices")]
            if geos:
                lidar_pts = np.concatenate([np.asarray(g.vertices) for g in geos], axis=0)
                cols_list = []
                for g in geos:
                    if hasattr(g, "colors") and g.colors is not None and len(g.colors) == len(g.vertices):
                        cols_list.append(np.asarray(g.colors)[:, :3])
                    else:
                        cols_list.append(np.full((len(g.vertices), 3), 200, dtype=np.uint8))
                lidar_cols = np.concatenate(cols_list, axis=0)
        else:
            lidar_pts = np.asarray(cloud.vertices)
            if hasattr(cloud, "colors") and cloud.colors is not None and len(cloud.colors) == len(lidar_pts):
                lidar_cols = np.asarray(cloud.colors)[:, :3]
            else:
                lidar_cols = np.full((lidar_pts.shape[0], 3), 200, dtype=np.uint8)

    if lidar_pts is not None and lidar_pts.size:
        front = np.isfinite(lidar_pts).all(axis=1) & (lidar_pts[:, 2] > 1e-6)
        lidar_pts = lidar_pts[front]
        lidar_cols = np.asarray(lidar_cols)[front]
        if lidar_cols.dtype != np.uint8:
            if lidar_cols.max() <= 1.0:
                lidar_cols = (np.clip(lidar_cols, 0, 1) * 255).astype(np.uint8)
            else:
                lidar_cols = np.clip(lidar_cols, 0, 255).astype(np.uint8)
        if lidar_pts.shape[0] > max_lidar:
            idx = np.linspace(0, lidar_pts.shape[0] - 1, max_lidar, dtype=np.int64)
            lidar_pts, lidar_cols = lidar_pts[idx], lidar_cols[idx]
        paths.extend(_write_points(out_dir / "lidar_pc", lidar_pts, lidar_cols))

        # Fused: dense turbo + LiDAR in magenta so plugins show both sources.
        magenta = np.tile(np.array([[255, 0, 220]], dtype=np.uint8), (lidar_pts.shape[0], 1))
        fused_pts = np.concatenate([pts_d, lidar_pts], axis=0)
        fused_cols = np.concatenate([turbo, magenta], axis=0)
        paths.extend(_write_points(out_dir / "depth_pc_fused", fused_pts, fused_cols))
    else:
        paths.extend(_write_points(out_dir / "depth_pc_fused", pts_d, turbo))

    # Colored depth mesh for vscode-3dviewer / GLB plugins.
    mesh_paths = export_depth_meshes(scene_dir, out_dir, stride=max(stride, 4), max_edge=4.0)
    paths.extend(mesh_paths)

    meta = {
        "frame": "opencv_camera",
        "axes": "X right, Y down, Z forward",
        "depth_range_m": [float(vmin), float(vmax)],
        "n_dense": int(pts_d.shape[0]),
        "n_lidar": int(0 if lidar_pts is None else lidar_pts.shape[0]),
        "stride": int(stride),
        "files": [p.name for p in paths],
        "viewer_note": (
            "Prefer depth_est_mesh.glb / depth_est_mesh.ply for mesh view "
            "(Materials → colors in vscode-3dviewer). PCD for CloudCompare/PCL."
        ),
    }
    with open(out_dir / "depth_pc_clouds.json", "w") as f:
        json.dump(meta, f, indent=2)
    return paths


def export_scene_depth_pc_ego(
    scene_dir: Path,
    out_dir: Path,
    stride: int = 6,
    max_per_cam: int = 120000,
) -> Optional[Path]:
    """Merge per-camera dense depth into one world-frame ASCII PLY for surround viewing."""
    view_dirs = discover_camera_view_dirs(scene_dir)
    if len(view_dirs) <= 1:
        return None

    all_pts: List[np.ndarray] = []
    all_cols: List[np.ndarray] = []
    all_mesh_pts: List[np.ndarray] = []
    all_mesh_cols: List[np.ndarray] = []
    all_mesh_faces: List[np.ndarray] = []
    mesh_offset = 0
    for view_dir in view_dirs:
        depth_path = view_dir / "depth_map.npy"
        cam_path = view_dir / "cam_params.json"
        if not depth_path.exists() or not cam_path.exists():
            continue
        depth = np.load(depth_path)
        with open(cam_path, "r") as f:
            cam = json.load(f)
        K = np.asarray(cam["K"], dtype=np.float64)
        c2w = np.asarray(cam["c2w"], dtype=np.float64)
        pts_c, uu, vv = _backproject_depth(depth, K, stride=stride, max_points=max_per_cam)
        if pts_c.shape[0] == 0:
            continue
        valid = depth[np.isfinite(depth) & (depth > 0) & (depth < 1e5)]
        vmin, vmax = (np.percentile(valid, [2, 98]) if valid.size else (0.0, 1.0))
        cols = _depth_turbo_colors(pts_c[:, 2], float(vmin), float(vmax))
        pts_h = np.concatenate([pts_c, np.ones((pts_c.shape[0], 1))], axis=1)
        pts_w = (c2w @ pts_h.T).T[:, :3]
        all_pts.append(pts_w)
        all_cols.append(cols)

        mesh_pack = _depth_grid_mesh(depth, K, stride=max(stride, 6), max_edge=4.0)
        if mesh_pack is not None:
            m_pts, m_cols, m_faces, _rgb = mesh_pack
            m_h = np.concatenate([m_pts, np.ones((m_pts.shape[0], 1))], axis=1)
            m_w = (c2w @ m_h.T).T[:, :3]
            all_mesh_pts.append(m_w)
            all_mesh_cols.append(m_cols)
            all_mesh_faces.append(m_faces + mesh_offset)
            mesh_offset += m_w.shape[0]

    if not all_pts:
        return None
    pts = np.concatenate(all_pts, axis=0)
    cols = np.concatenate(all_cols, axis=0)
    # Center near ego for easier plugin navigation (world coords are large for nuScenes).
    center = np.median(pts, axis=0)
    pts_local = pts - center
    out_path = out_dir / "scene_depth_pc_ego.ply"
    _write_ascii_ply(out_path, pts_local, cols)
    pcd_path = out_dir / "scene_depth_pc_ego.pcd"
    _write_pcd(pcd_path, pts_local, cols, binary=True)

    mesh_path = None
    if all_mesh_pts:
        m_pts = np.concatenate(all_mesh_pts, axis=0) - center
        m_cols = np.concatenate(all_mesh_cols, axis=0)
        m_faces = np.concatenate(all_mesh_faces, axis=0)
        mesh_bundle = _write_mesh_bundle(out_dir / "scene_depth_mesh_ego", m_pts, m_cols, m_faces)
        mesh_path = mesh_bundle[0]
        _render_mesh_preview(
            m_pts, m_cols, m_faces, out_dir / "scene_depth_mesh_ego_preview.png",
            width=1600, height=900,
        )

    with open(out_dir / "scene_depth_pc_ego.json", "w") as f:
        json.dump(
            {
                "frame": "world_minus_median",
                "median_xyz": center.tolist(),
                "n_points": int(pts.shape[0]),
                "cameras": [v.name for v in view_dirs],
                "point_cloud_ply": out_path.name,
                "point_cloud_pcd": pcd_path.name,
                "colored_mesh": None if mesh_path is None else mesh_path.name,
                "viewer_note": (
                    "Use scene_depth_pc_ego.pcd in CloudCompare/PCL. "
                    "For vscode-3dviewer open scene_depth_mesh_ego.ply and Materials → colors."
                ),
            },
            f,
            indent=2,
        )
    return out_path
BEV_CANVAS_SIZE = 1920
# Fixed ego-centric BEV window (meters): X left/right, Z behind/forward.
# Override via --bev_extent xmin,xmax,zmin,zmax
BEV_DEFAULT_EXTENT = (-50.0, 50.0, -50.0, 100.0)
DEFAULT_COMPOSE_MODES = [
    "gt_2d",
    "det_2d",
    "depth",
    "depth_pc",
    "depth_mesh",
    "crops",
    "bbox_3d",
    "mesh_overlay",
    "pc_proj",
    "bev_3d",
]
COMPOSE_PANEL_FILES = {
    "gt_2d": ["gt_overlay.png"],
    "det_2d": ["det2d_overlay.png"],
    "la_2d": ["locateanything_overlay.png"],
    "locateanything": ["locateanything_overlay.png"],
    "depth": ["rgb_depth.png", "depth_colormap.png"],
    "depth_pc": ["depth_pc_combo.png", "depth_pc_overlay.png"],
    "depth_mesh": ["depth_est_mesh_preview.png", "depth_est_mesh_rgb_preview.png"],
    "crops": ["crops_grid.png"],
    "bbox_3d": ["bbox3d_overlay.png"],
    "mesh_overlay": ["mesh_overlay.png"],
    "mesh": ["mesh_overlay.png"],
    "pc_proj": ["pointcloud_projection.png"],
    "bev_3d": ["bev_3d.png"],
}
BEV_PRED_COLOR = (220, 80, 30)
# Ego-centric surround: front on top, left/right beside BEV, back at bottom.
SURROUND_SPATIAL_SLOTS = [
    ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"],
    ["CAM_BACK_LEFT", "BEV", "CAM_BACK_RIGHT"],
    [None, "CAM_BACK", None],
]
BEV_GT_COLORS = {
    "car": (0, 180, 0),
    "vehicle": (0, 180, 0),
    "person": (0, 140, 255),
    "motorcycle": (255, 180, 0),
    "bicycle": (255, 120, 0),
    "traffic_light": (200, 0, 200),
    "traffic_sign": (180, 0, 180),
    "default": (40, 170, 170),
}


def _load_pred_3dbbox(scene_dir: Path) -> List[dict]:
    for name in ("3dbbox.json", "3dbbox_ground.json"):
        path = scene_dir / name
        if path.exists():
            with open(path, "r") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
    return []


def _select_bev_view(scene_dir: Path, view_dirs: List[Path], primary: Path) -> Path:
    """Prefer a camera that has predicted 3D boxes so surround BEV is informative."""
    if _load_pred_3dbbox(primary):
        return primary
    for view in view_dirs:
        if view == primary:
            continue
        if _load_pred_3dbbox(view):
            return view
    return primary


def _py123d_dataset_split_candidates(data_root: str) -> List[tuple[str, str]]:
    logs_dir = Path(data_root) / "logs"
    if not logs_dir.is_dir():
        return [("nuscenes-mini", "val"), ("nuscenes", "val")]
    out: List[tuple[str, str]] = []
    for entry in sorted(logs_dir.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        if name.endswith("_val"):
            out.append((name[: -len("_val")], "val"))
        elif name.endswith("_train"):
            out.append((name[: -len("_train")], "train"))
        elif name.endswith("_test"):
            out.append((name[: -len("_test")], "test"))
    return out or [("nuscenes-mini", "val"), ("nuscenes", "val")]


def _fetch_gt_3dbbox_py123d(scene_dir: Path) -> List[dict]:
    data_root = os.environ.get("PY123D_DATA_ROOT")
    if not data_root:
        return []
    try:
        from integrations.py123d.nuscenes_adapter import Py123dNuScenesLoader

        scene_id = scene_dir.name
        for dataset_name, split_type in _py123d_dataset_split_candidates(data_root):
            try:
                loader = Py123dNuScenesLoader(
                    data_root=data_root,
                    dataset_name=dataset_name,
                    split_type=split_type,
                    max_scenes=None,
                )
            except RuntimeError:
                continue
            for i in range(len(loader)):
                sample = loader.extract_sample(i)
                if sample["scene_id"] == scene_id:
                    gt = sample.get("gt_3dbbox", [])
                    gt_path = scene_dir / "nuscenes_gt_3dbbox.json"
                    with open(gt_path, "w") as f:
                        json.dump(gt, f)
                    return gt
    except Exception:
        return []
    return []


def _load_gt_3dbbox(scene_dir: Path) -> List[dict]:
    gt_path = scene_dir / "nuscenes_gt_3dbbox.json"
    if gt_path.exists():
        with open(gt_path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    return _fetch_gt_3dbbox_py123d(scene_dir)


def _collect_valid_xz(boxes: List[dict]) -> List[np.ndarray]:
    chunks = []
    for b in boxes:
        verts = np.asarray(b.get("bbox3D_cam", []), dtype=np.float32)
        if verts.ndim == 2 and verts.shape[1] == 3 and len(verts) >= 4:
            if not np.isfinite(verts).all() or np.min(verts[:, 2]) <= 1e-6:
                continue
            chunks.append(verts[:, [0, 2]])
    return chunks


def _bev_gt_color(category_name: str) -> tuple[int, int, int]:
    return BEV_GT_COLORS.get((category_name or "object").lower(), BEV_GT_COLORS["default"])


def _nice_distance_step(span: float, target_ticks: int = 8) -> float:
    if span <= 0:
        return 1.0
    raw = span / max(target_ticks, 1)
    magnitude = 10 ** math.floor(math.log10(max(raw, 1e-6)))
    for mult in (1.0, 2.0, 5.0, 10.0):
        step = mult * magnitude
        if span / step <= target_ticks * 1.2:
            return float(step)
    return float(10.0 * magnitude)


def _format_distance_m(value: float) -> str:
    if abs(value) < 1e-6:
        return "0m"
    if abs(value - round(value)) < 1e-3:
        return f"{int(round(value))}m"
    return f"{value:.1f}m"


def _put_text_with_bg(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    font_scale: float,
    color: tuple[int, int, int],
    thickness: int,
    *,
    bg: tuple[int, int, int] = (245, 245, 245),
) -> None:
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = org
    cv2.rectangle(img, (x - 2, y - th - 2), (x + tw + 2, y + baseline + 2), bg, -1)
    cv2.putText(
        img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA
    )


def _draw_bev_distance_axes(
    bev: np.ndarray,
    xmin: float,
    xmax: float,
    zmin: float,
    zmax: float,
    to_px,
    *,
    scale: float,
) -> None:
    """Draw meter grid lines and distance tick labels on the BEV canvas."""
    canvas_h, canvas_w = bev.shape[:2]
    tick_font = 0.42 * scale
    tick_th = max(1, int(round(1.2 * scale)))
    axis_color = (40, 40, 40)
    major_grid = (175, 175, 175)
    minor_grid = (210, 210, 210)
    margin = max(8, int(10 * scale))

    x_step = _nice_distance_step(xmax - xmin, target_ticks=6)
    z_step = _nice_distance_step(zmax - zmin, target_ticks=6)
    minor_x = x_step / 2.0
    minor_z = z_step / 2.0

    x_val = math.ceil(xmin / minor_x) * minor_x
    while x_val <= xmax + 1e-6:
        px, _ = to_px(float(x_val), zmin)
        is_major = abs(round(x_val / x_step) * x_step - x_val) < 1e-6 * max(1.0, abs(x_step))
        cv2.line(
            bev,
            (px, 0),
            (px, canvas_h - 1),
            major_grid if is_major else minor_grid,
            2 if is_major else 1,
            cv2.LINE_AA,
        )
        if is_major:
            label = _format_distance_m(x_val)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, tick_font, tick_th)
            tx = int(np.clip(px - tw // 2, margin, canvas_w - tw - margin))
            _put_text_with_bg(
                bev, label, (tx, canvas_h - margin), tick_font, axis_color, tick_th
            )
        x_val += minor_x

    z_val = math.ceil(zmin / minor_z) * minor_z
    while z_val <= zmax + 1e-6:
        _, py = to_px(xmin, float(z_val))
        is_major = abs(round(z_val / z_step) * z_step - z_val) < 1e-6 * max(1.0, abs(z_step))
        cv2.line(
            bev,
            (0, py),
            (canvas_w - 1, py),
            major_grid if is_major else minor_grid,
            2 if is_major else 1,
            cv2.LINE_AA,
        )
        if is_major:
            label = _format_distance_m(z_val)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, tick_font, tick_th)
            ty = int(np.clip(py + th // 2, margin + th, canvas_h - margin - int(28 * scale)))
            _put_text_with_bg(
                bev, label, (margin, ty), tick_font, axis_color, tick_th
            )
        z_val += minor_z

    if xmin <= 0.0 <= xmax and zmin <= 0.0 <= zmax:
        ox, oy = to_px(0.0, 0.0)
        # Ego / camera wedge pointing toward +Z (up in image).
        size = max(14, int(18 * scale))
        wedge = np.array(
            [
                [ox, oy - size],
                [ox - int(0.55 * size), oy + int(0.45 * size)],
                [ox + int(0.55 * size), oy + int(0.45 * size)],
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(bev, wedge, (35, 35, 35), lineType=cv2.LINE_AA)
        cv2.polylines(bev, [wedge], True, (250, 250, 250), max(1, int(scale)), cv2.LINE_AA)
        _put_text_with_bg(
            bev,
            "ego",
            (ox + int(8 * scale), oy - int(8 * scale)),
            tick_font,
            (35, 35, 35),
            tick_th,
        )

    _put_text_with_bg(
        bev,
        "X (m)  right+",
        (canvas_w - margin - int(110 * scale), canvas_h - margin - int(20 * scale)),
        tick_font,
        axis_color,
        tick_th,
    )
    _put_text_with_bg(
        bev,
        "Z (m)  forward+",
        (margin, margin + int(16 * scale)),
        tick_font,
        axis_color,
        tick_th,
    )


def _bev_footprint_polygon(verts: np.ndarray, to_px) -> Optional[np.ndarray]:
    if verts.shape[0] < 4:
        return None
    hull = cv2.convexHull(verts[:, [0, 2]].astype(np.float32).reshape(-1, 1, 2))
    if hull is None or len(hull) < 3:
        return None
    return np.array(
        [to_px(float(p[0]), float(p[1])) for p in hull.reshape(-1, 2)],
        dtype=np.int32,
    )


def _bev_heading_tip(verts: np.ndarray) -> Optional[tuple[float, float]]:
    """Return XZ tip of a short heading arrow from box center toward front."""
    xz = verts[:, [0, 2]].astype(np.float64)
    center = xz.mean(axis=0)
    # Prefer the bottom-face edge with largest |z| midpoint as forward.
    if len(verts) >= 4:
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        best = None
        best_score = -1e9
        for i, j in edges:
            mid = 0.5 * (xz[i] + xz[j])
            score = mid[1]  # prefer forward (+Z)
            length = float(np.linalg.norm(xz[i] - xz[j]))
            score = score + 0.05 * length
            if score > best_score:
                best_score = score
                best = mid
        if best is not None:
            direction = best - center
            n = float(np.linalg.norm(direction))
            if n > 1e-6:
                direction = direction / n
                extent = max(0.6, 0.35 * float(np.linalg.norm(xz.max(0) - xz.min(0))))
                tip = center + direction * extent
                return float(tip[0]), float(tip[1])
    return None


def _compute_bev_extent(
    box_xz: List[np.ndarray],
    pc_xz: Optional[np.ndarray],
    *,
    pad_ratio: float = 0.18,
    min_half_span: float = 12.0,
    max_span: float = 90.0,
) -> tuple[float, float, float, float]:
    """Fit BEV window to boxes (preferred), not the full sparse point cloud."""
    chunks = list(box_xz)
    if not chunks and pc_xz is not None and pc_xz.size > 0:
        # Robust fallback: keep central mass of the cloud.
        lo = np.percentile(pc_xz, 5, axis=0)
        hi = np.percentile(pc_xz, 95, axis=0)
        chunks = [np.vstack([lo, hi])]
    if not chunks:
        return -min_half_span, min_half_span, 0.0, 2.0 * min_half_span

    xz = np.vstack(chunks)
    xmin, zmin = np.min(xz, axis=0).astype(float)
    xmax, zmax = np.max(xz, axis=0).astype(float)
    # Always keep the camera origin in view.
    xmin = min(xmin, -1.0)
    xmax = max(xmax, 1.0)
    zmin = min(zmin, -1.0)
    zmax = max(zmax, 1.0)

    dx = max(xmax - xmin, 1e-6)
    dz = max(zmax - zmin, 1e-6)
    if dx < 2.0 * min_half_span:
        mid = 0.5 * (xmin + xmax)
        xmin, xmax = mid - min_half_span, mid + min_half_span
        dx = xmax - xmin
    if dz < 2.0 * min_half_span:
        mid = 0.5 * (zmin + zmax)
        zmin, zmax = mid - min_half_span, mid + min_half_span
        dz = zmax - zmin

    xmin -= dx * pad_ratio
    xmax += dx * pad_ratio
    zmin -= dz * pad_ratio
    zmax += dz * pad_ratio

    # Cap overly large windows so objects stay readable.
    cx = 0.5 * (xmin + xmax)
    cz = 0.5 * (zmin + zmax)
    if (xmax - xmin) > max_span:
        xmin, xmax = cx - 0.5 * max_span, cx + 0.5 * max_span
    if (zmax - zmin) > max_span:
        zmin, zmax = cz - 0.5 * max_span, cz + 0.5 * max_span
    return float(xmin), float(xmax), float(zmin), float(zmax)


def _draw_bev_boxes(
    bev: np.ndarray,
    boxes: List[dict],
    to_px,
    *,
    color: tuple[int, int, int],
    line_th: int,
    center_r: int,
    label_scale: float,
    label_th: int,
    label_scale_offset: float,
    label_prefix: str = "",
    draw_labels: bool = True,
    use_convex_hull: bool = False,
    bottom_face_indices: Optional[List[int]] = None,
    fill_alpha: float = 0.28,
) -> None:
    for b in boxes:
        verts = np.asarray(b.get("bbox3D_cam", []), dtype=np.float32)
        if not (verts.ndim == 2 and verts.shape[1] == 3 and len(verts) >= 4):
            continue
        if not np.isfinite(verts).all():
            continue
        # Allow boxes behind ego (negative Z) for BEV windows that include rear.
        if float(np.max(verts[:, 2])) <= 1e-6:
            continue
        if use_convex_hull and len(verts) >= 4:
            pts = _bev_footprint_polygon(verts, to_px)
        else:
            idx = bottom_face_indices or [0, 1, 2, 3]
            if len(verts) < max(idx) + 1:
                continue
            pts = np.array(
                [to_px(float(verts[i, 0]), float(verts[i, 2])) for i in idx],
                dtype=np.int32,
            )
        if pts is None or len(pts) < 3:
            continue

        if fill_alpha > 0:
            overlay = bev.copy()
            cv2.fillPoly(overlay, [pts], color, lineType=cv2.LINE_AA)
            cv2.addWeighted(overlay, fill_alpha, bev, 1.0 - fill_alpha, 0, bev)

        cv2.polylines(bev, [pts], isClosed=True, color=color, thickness=line_th, lineType=cv2.LINE_AA)
        # White outline for contrast on dense point clouds.
        cv2.polylines(
            bev,
            [pts],
            isClosed=True,
            color=(255, 255, 255),
            thickness=max(1, line_th - 2),
            lineType=cv2.LINE_AA,
        )
        cv2.polylines(bev, [pts], isClosed=True, color=color, thickness=max(1, line_th - 1), lineType=cv2.LINE_AA)

        center = b.get("center_cam", verts.mean(axis=0).tolist())
        cpx, cpy = to_px(float(center[0]), float(center[2]))
        tip = _bev_heading_tip(verts)
        if tip is not None:
            tpx, tpy = to_px(tip[0], tip[1])
            cv2.arrowedLine(
                bev,
                (cpx, cpy),
                (tpx, tpy),
                color,
                max(1, line_th - 1),
                tipLength=0.35,
                line_type=cv2.LINE_AA,
            )
        cv2.circle(bev, (cpx, cpy), center_r, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(bev, (cpx, cpy), center_r + 1, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        if draw_labels:
            label_off = max(6, int(6 * label_scale_offset))
            text = f"{label_prefix}{b.get('obj_id', '?')}_{b.get('category_name', 'obj')}"
            _put_text_with_bg(
                bev,
                text,
                (cpx + label_off, cpy - label_off),
                label_scale,
                color,
                label_th,
                bg=(250, 250, 250),
            )


def render_bev_3d(
    scene_dir: Path,
    out_dir: Path,
    canvas: int = BEV_CANVAS_SIZE,
    extent: Optional[tuple[float, float, float, float]] = None,
) -> Optional[Path]:
    """Render bird's-eye-view (X-Z): point cloud, dataset GT boxes, and predicted boxes.

    extent: optional (xmin, xmax, zmin, zmax) in meters. Defaults to BEV_DEFAULT_EXTENT.
    """
    pred_boxes = _load_pred_3dbbox(scene_dir)
    gt_boxes = _load_gt_3dbbox(scene_dir)
    if not pred_boxes and not gt_boxes:
        return None

    box_xz = _collect_valid_xz(gt_boxes) + _collect_valid_xz(pred_boxes)

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
                max_pts = min(500000, max(200000, canvas * canvas // 3))
                if pts.shape[0] > max_pts:
                    idx = np.linspace(0, pts.shape[0] - 1, max_pts, dtype=np.int64)
                    pts = pts[idx]
                if pts.size > 0:
                    pc_xz = pts[:, [0, 2]]
        except Exception:
            pc_xz = None

    if not box_xz and pc_xz is None:
        return None

    if extent is None:
        env_ext = os.environ.get("LA3D_BEV_EXTENT", "").strip()
        if env_ext:
            vals = [float(v) for v in env_ext.split(",")]
            if len(vals) == 4:
                extent = (vals[0], vals[1], vals[2], vals[3])
        if extent is None:
            extent = BEV_DEFAULT_EXTENT
    xmin, xmax, zmin, zmax = [float(v) for v in extent]

    # Non-square canvas so meter scales match on X and Z.
    x_span = max(xmax - xmin, 1e-3)
    z_span = max(zmax - zmin, 1e-3)
    if x_span >= z_span:
        canvas_w = canvas
        canvas_h = max(64, int(round(canvas * z_span / x_span)))
    else:
        canvas_h = canvas
        canvas_w = max(64, int(round(canvas * x_span / z_span)))

    scale = min(canvas_w, canvas_h) / 800.0
    line_th = max(3, int(3 * scale))
    center_r = max(4, int(4 * scale))
    label_scale = 0.48 * scale
    label_th = max(1, int(round(1.2 * scale)))
    title_scale = 0.72 * scale
    title_th = max(2, int(2 * scale))
    title_y = max(28, int(28 * scale))
    title_x = max(14, int(14 * scale))

    # Soft cool background improves contrast vs white labels / boxes.
    bev = np.full((canvas_h, canvas_w, 3), (236, 239, 242), dtype=np.uint8)

    def to_px(x: float, z: float) -> tuple[int, int]:
        u = (x - xmin) / (xmax - xmin + 1e-8)
        v = (z - zmin) / (zmax - zmin + 1e-8)
        px = int(np.clip(u * (canvas_w - 1), 0, canvas_w - 1))
        py = int(np.clip((1.0 - v) * (canvas_h - 1), 0, canvas_h - 1))
        return px, py

    _draw_bev_distance_axes(bev, xmin, xmax, zmin, zmax, to_px, scale=scale)

    def xz_to_px_array(xz_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u = (xz_pts[:, 0] - xmin) / (xmax - xmin + 1e-8)
        v = (xz_pts[:, 1] - zmin) / (zmax - zmin + 1e-8)
        px = np.clip((u * (canvas_w - 1)).astype(np.int32), 0, canvas_w - 1)
        py = np.clip(((1.0 - v) * (canvas_h - 1)).astype(np.int32), 0, canvas_h - 1)
        return px, py

    if pc_xz is not None and pc_xz.shape[0] > 0:
        # Keep only points inside the focused window.
        inside = (
            (pc_xz[:, 0] >= xmin)
            & (pc_xz[:, 0] <= xmax)
            & (pc_xz[:, 1] >= zmin)
            & (pc_xz[:, 1] <= zmax)
        )
        pc_view = pc_xz[inside]
        if pc_view.shape[0] > 0:
            px, py = xz_to_px_array(pc_view.astype(np.float64))
            dens = np.zeros((canvas_h, canvas_w), dtype=np.float32)
            np.add.at(dens, (py, px), 1.0)
            if dens.max() > 0:
                dens = dens / dens.max()
                # Darker where denser; still leave room for box overlays.
                shade = (35 + 95 * dens).astype(np.uint8)
                mask = dens > 0
                for c in range(3):
                    channel = bev[:, :, c]
                    channel[mask] = np.minimum(channel[mask], shade[mask])
                    bev[:, :, c] = channel
                # Slight blur so sparse single-pixel hits read as a surface.
                mist = bev.copy()
                mist = cv2.GaussianBlur(mist, (0, 0), sigmaX=0.8)
                blend = dens > 0.02
                bev[blend] = (
                    0.55 * bev[blend].astype(np.float32) + 0.45 * mist[blend].astype(np.float32)
                ).astype(np.uint8)

    gt_th = max(line_th, int(line_th * 1.2))
    for b in gt_boxes:
        _draw_bev_boxes(
            bev, [b], to_px,
            color=_bev_gt_color(b.get("category_name", "object")),
            line_th=gt_th, center_r=center_r, label_scale=label_scale,
            label_th=label_th, label_scale_offset=scale,
            draw_labels=False, use_convex_hull=True, fill_alpha=0.22,
        )
    if pred_boxes:
        _draw_bev_boxes(
            bev, pred_boxes, to_px, color=BEV_PRED_COLOR,
            line_th=line_th + 1, center_r=center_r, label_scale=label_scale,
            label_th=label_th, label_scale_offset=scale, label_prefix="Pred:",
            use_convex_hull=True, fill_alpha=0.32,
        )

    parts = [
        "BEV (X-Z)",
        f"X[{xmin:.0f},{xmax:.0f}]",
        f"Z[{zmin:.0f},{zmax:.0f}]",
    ]
    if pc_xz is not None:
        parts.append("point cloud")
    if gt_boxes:
        parts.append(f"GT={len(gt_boxes)}")
    if pred_boxes:
        parts.append(f"Pred={len(pred_boxes)}")
    _put_text_with_bg(
        bev,
        ", ".join(parts),
        (title_x, title_y),
        title_scale,
        (20, 20, 20),
        title_th,
    )
    legend_y = title_y + int(32 * scale)
    lx = title_x
    for text, col in [("Pred", BEV_PRED_COLOR), ("GT car", BEV_GT_COLORS["car"]),
                      ("GT person", BEV_GT_COLORS["person"]), ("GT other", BEV_GT_COLORS["default"])]:
        cv2.rectangle(bev, (lx, legend_y - 12), (lx + 22, legend_y + 6), col, -1)
        cv2.rectangle(bev, (lx, legend_y - 12), (lx + 22, legend_y + 6), (255, 255, 255), 1)
        _put_text_with_bg(
            bev,
            text,
            (lx + 28, legend_y),
            label_scale * 0.95,
            (30, 30, 30),
            label_th,
        )
        lx += int(150 * scale)
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


def _resize_to_height(im: np.ndarray, target_h: int) -> np.ndarray:
    scale = target_h / im.shape[0]
    return cv2.resize(im, (int(im.shape[1] * scale), target_h))


def _pad_panel_to_width(im: np.ndarray, width: int) -> np.ndarray:
    h, w = im.shape[:2]
    if w == width:
        return im
    if w > width:
        return im[:, (w - width) // 2 : (w - width) // 2 + width]
    pad = width - w
    return cv2.copyMakeBorder(im, 0, 0, pad // 2, pad - pad // 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def _compose_summary_grid(
    panels: List[np.ndarray],
    bev_panel: Optional[np.ndarray],
    *,
    target_h: int = 360,
    cols: int = 3,
) -> np.ndarray:
    resized = [_resize_to_height(im, target_h) for im in panels]
    cell_w = max(im.shape[1] for im in resized) if resized else target_h
    row_width = cell_w * cols
    rows: List[np.ndarray] = []
    for i in range(0, len(resized), cols):
        row_cells = [_pad_panel_to_width(im, cell_w) for im in resized[i : i + cols]]
        while len(row_cells) < cols:
            row_cells.append(np.zeros((target_h, cell_w, 3), dtype=np.uint8))
        row = np.hstack(row_cells)
        if row.shape[1] < row_width:
            row = _pad_panel_to_width(row, row_width)
        rows.append(row)
    if bev_panel is not None:
        bev_h = max(1, int(bev_panel.shape[0] * row_width / bev_panel.shape[1]))
        interp = cv2.INTER_AREA if row_width < bev_panel.shape[1] else cv2.INTER_CUBIC
        rows.append(cv2.resize(bev_panel, (row_width, bev_h), interpolation=interp))
    return np.vstack(rows)


def _load_compose_panel(out_dir: Path, mode: str, rendered: Optional[Path]) -> Optional[np.ndarray]:
    candidates: List[Path] = []
    if rendered is not None:
        candidates.append(rendered)
    candidates.extend(out_dir / name for name in COMPOSE_PANEL_FILES.get(mode, []))
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen or not path.exists():
            continue
        seen.add(key)
        im = cv2.imread(str(path))
        if im is not None:
            return im
    return None


def _render_compose_panel(scene_dir: Path, out_dir: Path, mode: str) -> Optional[Path]:
    if mode == "gt_2d":
        return render_gt_2d(scene_dir, out_dir)
    if mode == "det_2d":
        return render_det_2d(scene_dir, out_dir)
    if mode in ("la_2d", "locateanything"):
        return render_locateanything_2d(scene_dir, out_dir)
    if mode == "depth":
        ps = render_depth(scene_dir, out_dir)
        if not ps:
            return None
        for p in ps:
            if p.name == "rgb_depth.png":
                return p
        return ps[0]
    if mode == "depth_pc":
        ps = render_depth_pc_combo(scene_dir, out_dir)
        export_depth_pc_clouds(scene_dir, out_dir)
        if not ps:
            return None
        for p in ps:
            if p.name == "depth_pc_combo.png":
                return p
        return ps[0]
    if mode == "depth_mesh":
        ps = export_depth_meshes(scene_dir, out_dir)
        if not ps:
            return None
        for p in ps:
            if p.name == "depth_est_mesh_preview.png":
                return p
        for p in ps:
            if p.suffix == ".png":
                return p
        return ps[0]
    if mode == "crops":
        return render_crops_grid(scene_dir, out_dir)
    if mode == "bbox_3d":
        return render_bbox_3d(scene_dir, out_dir)
    if mode == "pc_proj":
        return render_pointcloud_projection(scene_dir, out_dir)
    if mode == "bev_3d":
        return render_bev_3d(scene_dir, out_dir)
    if mode in ("mesh", "mesh_overlay"):
        return render_mesh_overlay(scene_dir, out_dir)
    return None


def _label_camera_panel(im: np.ndarray, label: str) -> np.ndarray:
    panel = im.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 28), (20, 20, 20), -1)
    cv2.putText(
        panel,
        label,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    return panel


def _blank_surround_cell(target_h: int, cell_w: int) -> np.ndarray:
    return np.full((target_h, cell_w, 3), 24, dtype=np.uint8)


def _fit_panel_letterbox(im: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Resize with letterbox into a fixed canvas."""
    scale = min(out_w / max(im.shape[1], 1), out_h / max(im.shape[0], 1))
    nw = max(1, int(round(im.shape[1] * scale)))
    nh = max(1, int(round(im.shape[0] * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(im, (nw, nh), interpolation=interp)
    canvas = np.full((out_h, out_w, 3), 24, dtype=np.uint8)
    y0 = (out_h - nh) // 2
    x0 = (out_w - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def _stack_camera_layers(
    layers: List[Tuple[str, np.ndarray]],
    *,
    target_w: int,
    layer_h: int = 160,
) -> Optional[np.ndarray]:
    """Vertically stack labeled layers for one camera."""
    if not layers:
        return None
    strips: List[np.ndarray] = []
    for label, im in layers:
        if im is None or im.size == 0:
            continue
        h = max(48, int(round(im.shape[0] * target_w / max(im.shape[1], 1))))
        h = min(h, layer_h)
        interp = cv2.INTER_AREA if target_w < im.shape[1] else cv2.INTER_LINEAR
        resized = cv2.resize(im, (target_w, h), interpolation=interp)
        strips.append(_label_camera_panel(resized, label))
    if not strips:
        return None
    return np.vstack(strips)


def _compose_surround_spatial_grid(
    panels: Dict[str, np.ndarray],
    *,
    bev_panel: Optional[np.ndarray] = None,
    target_h: int = 220,
    bev_height_scale: float = 2.4,
    fit_side_to_bev: bool = False,
) -> Optional[np.ndarray]:
    """
    Arrange camera panels around BEV.

    Layout (SURROUND_SPATIAL_SLOTS):
      FRONT_LEFT | FRONT | FRONT_RIGHT
      BACK_LEFT  | BEV   | BACK_RIGHT
      (empty)    | BACK  | (empty)
    """
    if not panels and bev_panel is None:
        return None

    side_w = int(target_h * 16 / 9)
    for camera_key in NUSCENES_SURROUND_CAMERAS:
        im = panels.get(camera_key)
        if im is None:
            continue
        # Tall stacked composites: derive a readable side width.
        if fit_side_to_bev and im.shape[0] > im.shape[1]:
            side_w = max(side_w, min(480, max(240, im.shape[1])))
        else:
            side_w = max(side_w, _resize_to_height(im, target_h).shape[1])

    if bev_panel is not None:
        bev_h = max(target_h, int(target_h * bev_height_scale))
        bev_w = max(1, int(round(bev_panel.shape[1] * bev_h / max(bev_panel.shape[0], 1))))
        # Keep BEV large in the center; sides stay narrower.
        if fit_side_to_bev:
            side_w = min(side_w, max(260, bev_w // 3))
            bev_h = max(bev_h, int(target_h * 3.2))
            bev_w = max(1, int(round(bev_panel.shape[1] * bev_h / max(bev_panel.shape[0], 1))))
    else:
        bev_h = target_h
        bev_w = side_w

    grid_rows: List[np.ndarray] = []
    for row_slots in SURROUND_SPATIAL_SLOTS:
        row_has_bev = any(slot == "BEV" for slot in row_slots)
        row_h = bev_h if row_has_bev else target_h
        row_cells: List[np.ndarray] = []
        for idx, slot in enumerate(row_slots):
            is_center = idx == 1
            cell_w = bev_w if (slot == "BEV" or (slot == "CAM_BACK" and is_center)) else side_w
            if slot == "BEV":
                if bev_panel is not None:
                    cell = _fit_panel_letterbox(bev_panel, bev_w, row_h)
                    cell = _label_camera_panel(cell, "BEV")
                else:
                    cell = _label_camera_panel(_blank_surround_cell(row_h, bev_w), "BEV")
            elif slot is None:
                cell = _blank_surround_cell(row_h, cell_w)
            elif slot in panels:
                im = panels[slot]
                if fit_side_to_bev:
                    cell = _fit_panel_letterbox(im, cell_w, row_h)
                else:
                    resized = _resize_to_height(im, row_h)
                    cell = _pad_panel_to_width(resized, cell_w)
                    if cell.shape[1] > cell_w:
                        cell = cell[:, :cell_w]
                cell = _label_camera_panel(cell, slot.replace("CAM_", ""))
            else:
                cell = _label_camera_panel(
                    _blank_surround_cell(row_h, cell_w), slot.replace("CAM_", "")
                )
            row_cells.append(cell)

        row_h_final = max(c.shape[0] for c in row_cells)
        normalized = []
        for c in row_cells:
            if c.shape[0] < row_h_final:
                pad = row_h_final - c.shape[0]
                c = cv2.copyMakeBorder(
                    c, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24)
                )
            normalized.append(c)
        grid_rows.append(np.hstack(normalized))

    width = max(r.shape[1] for r in grid_rows)
    return np.vstack([_pad_panel_to_width(r, width) for r in grid_rows])


def _build_camera_summary_layers(
    scene_dir: Path,
    out_dir: Path,
    modes: List[str],
) -> Dict[str, np.ndarray]:
    """Per-camera stacked overlays for the summary surround ring."""
    preferred: List[Tuple[str, str]] = []
    candidates = [
        ("gt_2d", "GT"),
        ("det_2d", "Pred-2D"),
        ("la_2d", "LA"),
        ("depth", "Depth"),
        ("bbox_3d", "Pred-3D"),
    ]
    for mode, label in candidates:
        if mode == "la_2d" and "locateanything" in modes and "la_2d" not in modes:
            preferred.append((mode, label))
            continue
        if mode in modes or mode in ("gt_2d", "det_2d", "depth", "bbox_3d"):
            preferred.append((mode, label))
        if len(preferred) >= 3:
            break
    if not preferred:
        preferred = [("gt_2d", "GT"), ("depth", "Depth"), ("bbox_3d", "Pred-3D")]

    composites: Dict[str, np.ndarray] = {}
    for view_dir in discover_camera_view_dirs(scene_dir):
        view_out = out_dir / view_dir.name
        view_out.mkdir(parents=True, exist_ok=True)
        layers: List[Tuple[str, np.ndarray]] = []
        used: set[str] = set()
        for mode, label in preferred:
            if label in used:
                continue
            rendered = _render_compose_panel(view_dir, view_out, mode)
            im = _load_compose_panel(view_out, mode, rendered)
            if im is None:
                continue
            short = view_dir.name.replace("CAM_", "")
            layers.append((f"{short} · {label}", im))
            used.add(label)
            if len(layers) >= 3:
                break
        if not layers:
            rgb = cv2.imread(str(view_dir / "input.png"))
            if rgb is not None:
                layers.append((view_dir.name.replace("CAM_", ""), rgb))
        stacked = _stack_camera_layers(layers, target_w=400, layer_h=150)
        if stacked is not None:
            composites[view_dir.name] = stacked
    return composites


def _compose_surround_grid(
    panels: Dict[str, np.ndarray],
    *,
    target_h: int = 220,
    cols: int = 3,
) -> Optional[np.ndarray]:
    return _compose_surround_spatial_grid(panels, target_h=target_h)


def render_surround_rgb(scene_dir: Path, out_dir: Path, target_h: int = 220) -> Optional[Path]:
    view_dirs = discover_camera_view_dirs(scene_dir)
    if len(view_dirs) <= 1:
        return None
    panels: Dict[str, np.ndarray] = {}
    for view_dir in view_dirs:
        img_path = view_dir / "input.png"
        image = cv2.imread(str(img_path))
        if image is not None:
            panels[view_dir.name] = image
    grid = _compose_surround_grid(panels, target_h=target_h)
    if grid is None:
        return None
    out_path = out_dir / "surround_rgb.png"
    cv2.imwrite(str(out_path), grid)
    return out_path


def render_surround_gt_2d(scene_dir: Path, out_dir: Path, target_h: int = 220) -> Optional[Path]:
    view_dirs = discover_camera_view_dirs(scene_dir)
    if len(view_dirs) <= 1:
        return None
    panels: Dict[str, np.ndarray] = {}
    for view_dir in view_dirs:
        sub_out = out_dir / view_dir.name
        sub_out.mkdir(parents=True, exist_ok=True)
        rendered = render_gt_2d(view_dir, sub_out)
        if rendered is None:
            continue
        image = cv2.imread(str(rendered))
        if image is not None:
            panels[view_dir.name] = image
    grid = _compose_surround_grid(panels, target_h=target_h)
    if grid is None:
        return None
    out_path = out_dir / "surround_gt_2d.png"
    cv2.imwrite(str(out_path), grid)
    return out_path


def render_surround_depth(scene_dir: Path, out_dir: Path, target_h: int = 220) -> Optional[Path]:
    view_dirs = discover_camera_view_dirs(scene_dir)
    if len(view_dirs) <= 1:
        return None
    panels: Dict[str, np.ndarray] = {}
    for view_dir in view_dirs:
        sub_out = out_dir / view_dir.name
        sub_out.mkdir(parents=True, exist_ok=True)
        depth_paths = render_depth(view_dir, sub_out)
        chosen = None
        for path in depth_paths:
            if path.name == "rgb_depth.png":
                chosen = path
                break
        if chosen is None and depth_paths:
            chosen = depth_paths[0]
        if chosen is None:
            continue
        image = cv2.imread(str(chosen))
        if image is not None:
            panels[view_dir.name] = image
    grid = _compose_surround_grid(panels, target_h=target_h)
    if grid is None:
        return None
    out_path = out_dir / "surround_depth.png"
    cv2.imwrite(str(out_path), grid)
    return out_path


def render_surround_depth_pc(scene_dir: Path, out_dir: Path, target_h: int = 220) -> Optional[Path]:
    """Surround grid of compact depth + point-cloud overlays."""

    def _panel(view_dir: Path, sub_out: Path) -> Optional[Path]:
        paths = render_depth_pc_combo(view_dir, sub_out)
        for path in paths:
            if path.name == "depth_pc_overlay.png":
                return path
        return paths[0] if paths else None

    panels = _collect_surround_panels(scene_dir, out_dir, _panel)
    grid = _compose_surround_spatial_grid(panels, target_h=target_h)
    if grid is None:
        return None
    out_path = out_dir / "surround_depth_pc.png"
    cv2.imwrite(str(out_path), grid)
    return out_path


def _collect_surround_panels(
    scene_dir: Path,
    out_dir: Path,
    render_fn,
) -> Dict[str, np.ndarray]:
    panels: Dict[str, np.ndarray] = {}
    for view_dir in discover_camera_view_dirs(scene_dir):
        sub_out = out_dir / view_dir.name
        sub_out.mkdir(parents=True, exist_ok=True)
        rendered = render_fn(view_dir, sub_out)
        if rendered is None:
            continue
        image = cv2.imread(str(rendered))
        if image is not None:
            panels[view_dir.name] = image
    return panels


def render_scene_compose(scene_dir: Path, out_dir: Path, modes: List[str]) -> Optional[Path]:
    """Compose surround grids plus primary-camera detail panels and BEV."""
    view_dirs = discover_camera_view_dirs(scene_dir)
    if len(view_dirs) <= 1:
        return render_compose(scene_dir, out_dir, modes)

    primary = get_primary_camera_view(scene_dir)
    if primary is None:
        primary = view_dirs[0]
    primary_out = out_dir / primary.name
    primary_out.mkdir(parents=True, exist_ok=True)

    bev_panel: Optional[np.ndarray] = None
    # Prefer BEV whenever available so the surround summary has a center.
    bev_view = _select_bev_view(scene_dir, view_dirs, primary)
    bev_out = out_dir / bev_view.name
    bev_out.mkdir(parents=True, exist_ok=True)
    rendered = _render_compose_panel(bev_view, bev_out, "bev_3d")
    bev_panel = _load_compose_panel(bev_out, "bev_3d", rendered)

    gt_panels = _collect_surround_panels(scene_dir, out_dir, render_gt_2d)
    spatial_gt = _compose_surround_spatial_grid(gt_panels, bev_panel=bev_panel)
    if spatial_gt is not None:
        cv2.imwrite(str(out_dir / "surround_gt_2d.png"), spatial_gt)

    det_panels = _collect_surround_panels(scene_dir, out_dir, render_det_2d)
    spatial_det = _compose_surround_spatial_grid(det_panels, bev_panel=bev_panel)
    if spatial_det is not None:
        cv2.imwrite(str(out_dir / "surround_det_2d.png"), spatial_det)

    def _depth_panel(view_dir: Path, sub_out: Path) -> Optional[Path]:
        depth_paths = render_depth(view_dir, sub_out)
        for path in depth_paths:
            if path.name == "rgb_depth.png":
                return path
        return depth_paths[0] if depth_paths else None

    depth_panels = _collect_surround_panels(scene_dir, out_dir, _depth_panel)
    spatial_depth = _compose_surround_spatial_grid(depth_panels)
    if spatial_depth is not None:
        cv2.imwrite(str(out_dir / "surround_depth.png"), spatial_depth)

    def _depth_pc_panel(view_dir: Path, sub_out: Path) -> Optional[Path]:
        paths = render_depth_pc_combo(view_dir, sub_out)
        export_depth_pc_clouds(view_dir, sub_out)
        for path in paths:
            if path.name == "depth_pc_overlay.png":
                return path
        return paths[0] if paths else None

    depth_pc_panels = _collect_surround_panels(scene_dir, out_dir, _depth_pc_panel)
    spatial_depth_pc = _compose_surround_spatial_grid(depth_pc_panels)
    if spatial_depth_pc is not None:
        cv2.imwrite(str(out_dir / "surround_depth_pc.png"), spatial_depth_pc)

    if "depth_pc" in modes:
        export_scene_depth_pc_ego(scene_dir, out_dir)
    if "depth_mesh" in modes:
        export_scene_depth_mesh_ego(scene_dir, out_dir)
        # Surround grid of per-camera mesh previews.
        def _mesh_preview_panel(view_dir: Path, sub_out: Path) -> Optional[Path]:
            paths = export_depth_meshes(view_dir, sub_out)
            for path in paths:
                if path.name == "depth_est_mesh_preview.png":
                    return path
            for path in paths:
                if path.suffix == ".png":
                    return path
            return None

        mesh_panels = _collect_surround_panels(scene_dir, out_dir, _mesh_preview_panel)
        spatial_mesh = _compose_surround_spatial_grid(mesh_panels)
        if spatial_mesh is not None:
            cv2.imwrite(str(out_dir / "surround_depth_mesh.png"), spatial_mesh)

    rgb_panels: Dict[str, np.ndarray] = {}
    for view_dir in view_dirs:
        image = cv2.imread(str(view_dir / "input.png"))
        if image is not None:
            rgb_panels[view_dir.name] = image
    spatial_rgb = _compose_surround_spatial_grid(rgb_panels)
    if spatial_rgb is not None:
        cv2.imwrite(str(out_dir / "surround_rgb.png"), spatial_rgb)

    # Summary = one BEV-centered ring; each camera slot stacks GT / Pred / Depth.
    cam_composites = _build_camera_summary_layers(scene_dir, out_dir, modes)
    summary = _compose_surround_spatial_grid(
        cam_composites,
        bev_panel=bev_panel,
        target_h=280,
        bev_height_scale=3.6,
        fit_side_to_bev=True,
    )
    if summary is None:
        return None
    out_path = out_dir / "summary.png"
    cv2.imwrite(str(out_path), summary)
    return out_path


def render_compose(scene_dir: Path, out_dir: Path, modes: List[str]) -> Optional[Path]:
    panels: List[np.ndarray] = []
    bev_panel: Optional[np.ndarray] = None
    for mode in modes:
        rendered = _render_compose_panel(scene_dir, out_dir, mode)
        im = _load_compose_panel(out_dir, mode, rendered)
        if im is None:
            continue
        if mode == "bev_3d":
            bev_panel = im
        else:
            panels.append(im)
    if not panels and bev_panel is None:
        return None
    summary = _compose_summary_grid(panels, bev_panel)
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
    multi_camera = len(discover_camera_view_dirs(scene_dir)) > 1
    if "compose" in modes:
        extra_modes = [m for m in modes if m not in ("compose",) and m not in DEFAULT_COMPOSE_MODES]
        sub_modes = list(dict.fromkeys(DEFAULT_COMPOSE_MODES + extra_modes))
        if multi_camera:
            render_scene_compose(scene_dir, out_dir, sub_modes)
        else:
            render_compose(scene_dir, out_dir, sub_modes)
    else:
        if multi_camera:
            render_surround_rgb(scene_dir, out_dir)
            render_surround_gt_2d(scene_dir, out_dir)
            render_surround_depth(scene_dir, out_dir)
            if "depth_pc" in modes or "pc_proj" in modes:
                render_surround_depth_pc(scene_dir, out_dir)
            if "depth_pc" in modes:
                export_scene_depth_pc_ego(scene_dir, out_dir)
            if "depth_mesh" in modes:
                export_scene_depth_mesh_ego(scene_dir, out_dir)

                def _mesh_preview_panel(view_dir: Path, sub_out: Path) -> Optional[Path]:
                    paths = export_depth_meshes(view_dir, sub_out)
                    for path in paths:
                        if path.name == "depth_est_mesh_preview.png":
                            return path
                    for path in paths:
                        if path.suffix == ".png":
                            return path
                    return None

                mesh_panels = _collect_surround_panels(scene_dir, out_dir, _mesh_preview_panel)
                spatial_mesh = _compose_surround_spatial_grid(mesh_panels)
                if spatial_mesh is not None:
                    cv2.imwrite(str(out_dir / "surround_depth_mesh.png"), spatial_mesh)
            # Also emit Pred-2D surround when detections exist.
            det_panels = _collect_surround_panels(scene_dir, out_dir, render_det_2d)
            spatial_det = _compose_surround_spatial_grid(det_panels)
            if spatial_det is not None:
                cv2.imwrite(str(out_dir / "surround_det_2d.png"), spatial_det)
        for view_dir in discover_camera_view_dirs(scene_dir) or [scene_dir]:
            view_out = out_dir / view_dir.name if view_dir != scene_dir else out_dir
            if view_dir != scene_dir:
                view_out.mkdir(parents=True, exist_ok=True)
            for mode in modes:
                if mode == "gt_2d":
                    render_gt_2d(view_dir, view_out)
                elif mode == "det_2d":
                    render_det_2d(view_dir, view_out)
                elif mode in ("la_2d", "locateanything"):
                    render_locateanything_2d(view_dir, view_out)
                elif mode == "depth":
                    render_depth(view_dir, view_out)
                elif mode == "depth_pc":
                    render_depth_pc_combo(view_dir, view_out)
                    export_depth_pc_clouds(view_dir, view_out)
                elif mode == "depth_mesh":
                    export_depth_meshes(view_dir, view_out)
                elif mode == "crops":
                    render_crops_grid(view_dir, view_out)
                elif mode == "bbox_3d":
                    render_bbox_3d(view_dir, view_out)
                elif mode == "pc_proj":
                    render_pointcloud_projection(view_dir, view_out)
                elif mode == "bev_3d":
                    render_bev_3d(view_dir, view_out)
                elif mode in ("mesh", "mesh_overlay"):
                    render_mesh_overlay(view_dir, view_out)

        if any(m in ("la_2d", "locateanything") for m in modes):
            render_surround_locateanything(scene_dir, out_dir)

    if backend in ("blender", "both"):
        primary = get_primary_camera_view(scene_dir) or scene_dir
        run_blender_scene(primary, verbose=verbose)


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
        help="Comma-separated: gt_2d,det_2d,la_2d,depth,depth_pc,depth_mesh,crops,bbox_3d,mesh_overlay,pc_proj,bev_3d,compose",
    )
    parser.add_argument(
        "--backend",
        choices=["preview", "blender", "both"],
        default="preview",
    )
    parser.add_argument("--viz_subdir", type=str, default="viz")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--la_infer",
        action="store_true",
        help="Re-run LocateAnything model instead of using saved crops/bboxes",
    )
    parser.add_argument(
        "--bev_extent",
        type=str,
        default=None,
        help="Fixed BEV window xmin,xmax,zmin,zmax in meters (default: -50,50,-50,100)",
    )
    args = parser.parse_args()

    modes = [m.strip() for m in args.mode.split(",") if m.strip()]
    if not args.scene_dir and not args.root:
        parser.error("Provide --scene_dir or --root")

    if args.scene_dir:
        scenes = [Path(args.scene_dir)]
    else:
        scenes = list_scene_dirs(Path(args.root))

    # Optional: force LA re-inference via env consumed by wrappers if needed later.
    if args.la_infer:
        os.environ["LA3D_LA_INFER"] = "1"
    if args.bev_extent:
        os.environ["LA3D_BEV_EXTENT"] = args.bev_extent
    elif "LA3D_BEV_EXTENT" not in os.environ:
        # Keep fixed ego window unless caller overrides.
        os.environ["LA3D_BEV_EXTENT"] = ",".join(str(v) for v in BEV_DEFAULT_EXTENT)

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
