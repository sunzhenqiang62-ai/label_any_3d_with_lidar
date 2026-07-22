#!/usr/bin/env python3
"""
Render per-frame surround summaries for consecutive py123d nuScenes frames,
then stitch them into a 2 Hz video.

Usage (from src/):
  python tools/render_summary_video.py \\
    --save_dir ../experimental_results/nuScenes_summary_video \\
    --num_frames 50 --fps 2 --scene_index 0

nuScenes-mini scenes usually have ~40 keyframes; the script caps automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_opening
from tqdm import tqdm

sys.path = ["./"] + sys.path

from geometry.lidar_depth import build_scene_outputs
from integrations.py123d.nuscenes_adapter import (
    Py123dNuScenesLoader,
    discover_camera_view_dirs,
    resolve_camera_keys,
    scene_camera_output_dir,
    write_cameras_manifest,
)
from util import crop_object, read_bounding_boxes_segmentations
from util_3dbox import save_3d_bbox_from_depth_fallback
from tools.visualize_scene import visualize_scene


def _ensure_enhanced(view_dir: Path) -> None:
    src = view_dir / "input.png"
    dst_dir = view_dir / "enhanced"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "input.png"
    if src.exists() and (not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime):
        shutil.copy2(src, dst)


def _write_frame_depth(
    loader: Py123dNuScenesLoader,
    scene_index: int,
    frame_index: int,
    scene_root: Path,
    depth_fill: str = "nearest",
) -> None:
    loader.frame_index = frame_index
    samples = loader.extract_samples(scene_index)
    multi = len(loader.camera_keys) > 1
    scene_root.mkdir(parents=True, exist_ok=True)
    if multi:
        write_cameras_manifest(scene_root, loader.camera_keys)

    for sample in samples:
        cam = sample.get("camera_key", loader.camera_keys[0])
        out_dir = scene_camera_output_dir(scene_root, cam, multi)
        out_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("crops", "object_space", "reconstruction", "enhanced"):
            (out_dir / sub).mkdir(exist_ok=True)

        Image.fromarray(sample["image_np"].astype(np.uint8)).save(out_dir / "input.png")
        build_scene_outputs(
            out_dir,
            sample["image_np"],
            sample["points_world"],
            sample["calib"],
            depth_fill=depth_fill,
            densify_radius=1,
            raster_mode="median",
            calib_refine=False,  # speed for multi-frame
        )
        with open(out_dir / "nuscenes_annotations.json", "w") as f:
            json.dump(sample["annotations"], f)
        with open(out_dir / "nuscenes_gt_3dbbox.json", "w") as f:
            json.dump(sample.get("gt_3dbbox", []), f)
        _ensure_enhanced(out_dir)
        with open(out_dir / "frame_meta.json", "w") as f:
            json.dump(
                {
                    "scene_id": sample["scene_id"],
                    "camera_key": cam,
                    "frame_index": frame_index,
                    "iteration": sample.get("iteration"),
                },
                f,
            )


def _gt_crops_for_view(view_dir: Path, min_mask_area: int = 800, crop_size: int = 256) -> int:
    """Write GT instance crops + bboxes.json (native resolution, enhance_scale=1)."""
    ann_path = view_dir / "nuscenes_annotations.json"
    img_path = view_dir / "input.png"
    if not ann_path.exists() or not img_path.exists():
        return 0
    image = np.array(Image.open(img_path).convert("RGB"))
    h, w = image.shape[:2]
    with open(ann_path, "r") as f:
        annotations = json.load(f)
    if not annotations:
        return 0

    bboxes, masks, object_ids, instance_labels = read_bounding_boxes_segmentations(
        annotations, (w, h)
    )
    crop_dir = view_dir / "crops"
    crop_dir.mkdir(exist_ok=True)
    # Clear stale crops when regenerating.
    for old in crop_dir.glob("*"):
        old.unlink()

    selected = []
    n = 0
    for j in range(len(masks) - 1, -1, -1):
        label = instance_labels[object_ids[j]]
        label = label.replace(" (", ", ").replace(")", "")
        obj_id = f"{j}_{label.replace(' ', '_')}"
        mask = binary_opening(masks[j], np.ones((5, 5)))
        if mask.sum() < min_mask_area:
            continue
        ys, xs = np.where(mask)
        if ys.size == 0:
            continue
        crop, crop_params = crop_object(image, mask, crop_size)
        crop.save(crop_dir / f"{obj_id}_reproj.png")
        # enhance_scale = 1 → params already in native coords
        np.save(crop_dir / f"{obj_id}_crop_params.npy", np.asarray(crop_params, dtype=np.float64))
        # Also save rgba for completeness.
        crop.save(crop_dir / f"{obj_id}_rgba.png")
        bb = bboxes[object_ids[j]]
        # annotations may be xywh
        if len(bb) >= 4:
            x, y, bw, bh = [float(v) for v in bb[:4]]
            selected.append([x, y, x + bw, y + bh])
        n += 1
    with open(view_dir / "bboxes.json", "w") as f:
        json.dump(selected, f)
    return n


def _depth_fallback_boxes(view_dir: Path) -> int:
    bbox_path = view_dir / "3dbbox.json"
    if bbox_path.exists():
        bbox_path.unlink()
    ground = view_dir / "3dbbox_ground.json"
    if ground.exists():
        ground.unlink()
    boxes = save_3d_bbox_from_depth_fallback(view_dir, write_json=True)
    if ground.exists():
        ground.replace(bbox_path)
    elif boxes:
        with open(bbox_path, "w") as f:
            json.dump(boxes, f)
    return len(boxes)


def _encode_video(frame_paths: List[Path], out_mp4: Path, fps: float = 2.0) -> Path:
    if not frame_paths:
        raise RuntimeError("No summary frames to encode")
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise RuntimeError(f"Failed to read {frame_paths[0]}")
    h, w = first.shape[:2]
    # Ensure even dims for many codecs.
    w2, h2 = w - (w % 2), h - (h % 2)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_mp4), fourcc, float(fps), (w2, h2))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed for {out_mp4}")
    for fp in frame_paths:
        im = cv2.imread(str(fp))
        if im is None:
            continue
        if im.shape[1] != w2 or im.shape[0] != h2:
            im = cv2.resize(im, (w2, h2), interpolation=cv2.INTER_AREA)
        writer.write(im)
    writer.release()

    # Also try ffmpeg re-encode to H.264 for better player support.
    h264 = out_mp4.with_name(out_mp4.stem + "_h264.mp4")
    try:
        import subprocess

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(out_mp4),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "20",
                str(h264),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if h264.exists() and h264.stat().st_size > 0:
            return h264
    except Exception:
        pass
    return out_mp4


def main():
    parser = argparse.ArgumentParser(description="Multi-frame summary video at 2 Hz")
    parser.add_argument("--save_dir", type=str, default="../experimental_results/nuScenes_summary_video")
    parser.add_argument("--split", type=str, default="nuscenes_val")
    parser.add_argument("--py123d_data_root", type=str, default=None)
    parser.add_argument("--py123d_dataset", type=str, default="nuscenes-mini")
    parser.add_argument("--py123d_split", type=str, default="val")
    parser.add_argument("--scene_index", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=50)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--camera_keys", type=str, default="all")
    parser.add_argument("--depth_fill", type=str, default="nearest")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--viz_only", action="store_true")
    parser.add_argument("--bev_extent", type=str, default="-50,50,-50,100")
    args = parser.parse_args()

    os.environ["LA3D_BEV_EXTENT"] = args.bev_extent

    data_root = args.py123d_data_root or os.environ.get("PY123D_DATA_ROOT")
    camera_keys = resolve_camera_keys(args.camera_keys, args.camera_keys)
    loader = Py123dNuScenesLoader(
        data_root=data_root,
        split_type=args.py123d_split,
        dataset_name=args.py123d_dataset,
        max_scenes=args.scene_index + 1,
        camera_key=camera_keys[0],
        camera_keys=camera_keys,
        lidar_key="merged",
        frame_index=0,
    )
    if args.scene_index >= len(loader):
        raise IndexError(f"scene_index {args.scene_index} out of range (n={len(loader)})")

    scene_api = loader.scenes[args.scene_index]
    meta = scene_api.get_scene_metadata()
    available = int(meta.total_iterations) - int(meta.num_history_iterations)
    n_frames = min(args.num_frames, max(0, available - args.start_frame))
    if n_frames < args.num_frames:
        print(
            f"Warning: requested {args.num_frames} frames, scene has {available} "
            f"keyframes from start_frame={args.start_frame}; using {n_frames}."
        )
    scene_id = meta.initial_uuid.replace("/", "_").replace("-", "_")
    print(
        f"Scene {scene_id}: frames {args.start_frame}..{args.start_frame + n_frames - 1} "
        f"-> {args.fps} Hz video"
    )

    save_root = Path(args.save_dir) / args.split
    save_root.mkdir(parents=True, exist_ok=True)
    summary_paths: List[Path] = []

    for i in tqdm(range(n_frames), desc="frames"):
        frame_index = args.start_frame + i
        scene_root = save_root / f"{scene_id}_f{frame_index:04d}"
        summary_path = scene_root / "viz" / "summary.png"

        if not args.viz_only:
            have_depth = (scene_root / "CAM_FRONT" / "depth_map.npy").exists()
            if not (args.skip_existing and have_depth):
                _write_frame_depth(
                    loader,
                    args.scene_index,
                    frame_index,
                    scene_root,
                    depth_fill=args.depth_fill,
                )

            for view_dir in discover_camera_view_dirs(scene_root):
                crops = list((view_dir / "crops").glob("*_reproj.png"))
                if not (args.skip_existing and crops):
                    n = _gt_crops_for_view(view_dir)
                    print(f"  {view_dir.name}: {n} GT crops")
                n_box = _depth_fallback_boxes(view_dir)
                print(f"  {view_dir.name}: {n_box} depth-fallback boxes")

        if args.skip_existing and summary_path.exists():
            summary_paths.append(summary_path)
            continue

        visualize_scene(
            scene_root,
            modes=["compose"],
            backend="preview",
            viz_subdir="viz",
            verbose=False,
        )
        if summary_path.exists():
            # Burn frame index onto summary for readability.
            im = cv2.imread(str(summary_path))
            if im is not None:
                cv2.putText(
                    im,
                    f"frame {frame_index:04d}  |  {args.fps:g} Hz",
                    (16, im.shape[0] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (20, 20, 20),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imwrite(str(summary_path), im)
            summary_paths.append(summary_path)
        else:
            print(f"Warning: missing summary for {scene_root.name}")

    out_mp4 = Path(args.save_dir) / f"{scene_id}_summary_{n_frames}f_{args.fps:g}hz.mp4"
    written = _encode_video(summary_paths, out_mp4, fps=args.fps)
    print(f"Wrote {len(summary_paths)} frames -> {written}")
    print(f"Duration ≈ {len(summary_paths) / args.fps:.1f}s at {args.fps} Hz")


if __name__ == "__main__":
    main()
