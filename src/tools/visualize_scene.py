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
        sub_modes = [m for m in modes if m != "compose"] or ["gt_2d", "depth", "crops", "bbox_3d"]
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
        help="Comma-separated: gt_2d,depth,crops,bbox_3d,compose",
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
