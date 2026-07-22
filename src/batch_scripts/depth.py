"""
Depth estimation pipeline.

Modes:
  estimate (default): MoGe + DepthPro + RANSAC alignment
  lidar: world-frame LiDAR + calibration -> depth_map / PLY / cam_params
  py123d: nuScenes (etc.) via py123d Arrow logs -> depth_map + nuscenes_annotations.json
  fuse: LiDAR metric depth where valid + MoGe/DepthPro fill for holes (py123d or manifest)

Usage:
    python batch_scripts/depth.py --start_index 0 --end_index 100 --split val
    python batch_scripts/depth.py --depth_source lidar --manifest ../dataset/lidar/manifest.json --config configs/lidar.yaml
    python batch_scripts/depth.py --depth_source py123d --config configs/py123d_nuscenes.yaml --end_index -1
"""
import argparse
from omegaconf import OmegaConf
import sys
import os
import shutil
from tqdm import tqdm
import torch
from dataset_model import get_scene
from pathlib import Path
import numpy as np
import json
import trimesh
from PIL import Image

from util import depth_to_points
from batch_scripts.coconut_loader import CoconutLoader, get_dataset_paths
from batch_scripts.lidar_loader import LidarManifestLoader, get_lidar_save_split
from geometry.lidar_depth import (
    build_scene_outputs,
    compute_lidar_depth,
    fuse_lidar_with_estimate,
    load_calib,
    load_pointcloud,
    write_depth_artifacts,
)


def save_moge_data(image, points, depth, mask, save_path):
    """Save MoGe output as PLY mesh with edges removed."""
    import utils3d_moge
    from moge.utils.io import save_ply

    height, width = image.shape[:2]
    normals, normals_mask = utils3d_moge.numpy.points_to_normals(points, mask=mask)

    faces, vertices, vertex_colors, vertex_uvs = utils3d_moge.numpy.image_mesh(
        points,
        image.astype(np.float32) / 255,
        utils3d_moge.numpy.image_uv(width=width, height=height),
        mask=mask & ~(utils3d_moge.numpy.depth_edge(depth, rtol=0.03, mask=mask) &
                      utils3d_moge.numpy.normals_edge(normals, tol=5, mask=normals_mask)),
        tri=True
    )
    save_ply(save_path / 'depth_scene_no_edge.ply', vertices, faces, vertex_colors)


def align_depth(
    relative_depth,
    metric_depth,
    mask=None,
    apply_mask=None,
    min_samples=0.2,
    max_valid_depth=400.0,
):
    """
    Align scale-invariant depth to metric depth using RANSAC linear regression.

    ``mask`` selects pixels used to fit the scale. ``apply_mask`` selects where
    the fitted scale is written; defaults to ``mask`` (MoGe+DepthPro path) or all
    finite relative depths when ``mask`` is None.
    """
    from sklearn.linear_model import RANSACRegressor, LinearRegression

    regressor = RANSACRegressor(estimator=LinearRegression(fit_intercept=False), min_samples=min_samples)

    fit_valid = (~np.isinf(relative_depth)) & (metric_depth < max_valid_depth)
    if mask is not None:
        fit_valid &= mask

    if fit_valid.sum() == 0:
        print("Warning: No valid points for alignment. Returning metric depth.")
        return metric_depth

    try:
        regressor.fit(relative_depth[fit_valid].reshape(-1, 1), metric_depth[fit_valid].reshape(-1, 1))
    except Exception as e:
        print(f"Error fitting RANSACRegressor: {e}, using metric depth directly")
        return metric_depth

    if apply_mask is None:
        apply_mask = mask if mask is not None else ~np.isinf(relative_depth)

    depth = np.full_like(relative_depth, 10000.0)
    apply_mask = np.asarray(apply_mask, dtype=bool)
    if apply_mask.any():
        masked_pred = regressor.predict(relative_depth[apply_mask].reshape(-1, 1)).flatten()
        depth[apply_mask] = masked_pred

    return depth


def prepare_output_dirs(out_dir, scene):
    """Create scene output tree and save input.png if missing."""
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    (out_dir / "crops").mkdir(exist_ok=True)
    (out_dir / "object_space").mkdir(exist_ok=True)
    (out_dir / "reconstruction").mkdir(exist_ok=True)
    if not os.path.exists(out_dir / 'input.png'):
        scene.image_pil.save(out_dir / 'input.png')
    return out_dir


def depth_already_done(out_dir):
    return os.path.exists(out_dir / 'depth_map.npy') and os.path.exists(out_dir / 'cam_params.json')


def run_lidar_depth(scene, out_dir, depth_fill="none"):
    """Build depth artifacts from external LiDAR + calibration."""
    points, colors = load_pointcloud(scene.pointcloud_path)
    calib = load_calib(scene.calib_path)
    build_scene_outputs(
        out_dir,
        scene.image_np,
        points,
        calib,
        colors=colors,
        depth_fill=depth_fill,
    )


def _moge_depthpro_depth(scene, out_dir, depthpro_model, depthpro_transform, f_px=None):
    sys.path = ['./', '../external/MoGe'] + [p for p in sys.path if p not in ('./', '../external/MoGe')]
    from infer_moge import infer_geometry_on_image

    _, moge_depth_map, moge_mask, K_img = infer_geometry_on_image(f'{out_dir}/input.png', out_dir)
    if f_px is None:
        f_px = float(K_img[0, 0])
    img = depthpro_transform(scene.image_pil)
    f_px_tensor = (
        torch.tensor([float(f_px)], dtype=torch.float32, device=img.device)
        if f_px is not None
        else None
    )
    prediction = depthpro_model.infer(img, f_px=f_px_tensor)
    pro_depth_map = prediction["depth"].cpu().numpy()
    depth_map = align_depth(moge_depth_map, pro_depth_map, mask=moge_mask)
    return depth_map.astype(np.float32), moge_mask, K_img


def _resize_depth_to_image(depth_map, mask, image_np):
    H, W = image_np.shape[:2]
    if depth_map.shape == (H, W):
        return depth_map, mask
    depth_map = np.array(
        Image.fromarray(depth_map.astype(np.float32)).resize((W, H), Image.NEAREST)
    )
    mask = np.array(
        Image.fromarray(mask.astype(np.uint8)).resize((W, H), Image.NEAREST)
    ) > 0
    return depth_map, mask


def run_estimate_depth_maps(scene, out_dir, depthpro_model, depthpro_transform, f_px=None):
    """MoGe + DepthPro; returns dense metric depth (H,W) aligned to image size."""
    depth_map, moge_mask, _ = _moge_depthpro_depth(
        scene, out_dir, depthpro_model, depthpro_transform, f_px=f_px
    )
    depth_map, moge_mask = _resize_depth_to_image(depth_map, moge_mask, scene.image_np)
    return depth_map, moge_mask


def run_estimate_depth(scene, out_dir, depthpro_model, depthpro_transform):
    """MoGe + DepthPro depth estimation (original pipeline)."""
    depth_map, moge_mask, K_img = _moge_depthpro_depth(
        scene, out_dir, depthpro_model, depthpro_transform
    )
    depth_map, moge_mask = _resize_depth_to_image(depth_map, moge_mask, scene.image_np)

    pts3d = depth_to_points(depth_map[None], K_img)
    save_moge_data(scene.image_np, pts3d, depth_map, moge_mask, out_dir)
    np.save(out_dir / 'depth_map.npy', depth_map)
    trimesh.PointCloud(pts3d.reshape(-1, 3), scene.image_np.reshape(-1, 3)).export(out_dir / 'depth_scene.ply')

    pose = np.eye(4)
    cam_params = {
        'K': K_img.tolist(),
        'c2w': pose.tolist(),
        'W': scene.image_pil.width,
        'H': scene.image_pil.height,
    }
    with open(out_dir / 'cam_params.json', 'w') as fp:
        json.dump(cam_params, fp)


def run_fuse_depth(
    scene,
    out_dir,
    points_world,
    calib,
    depthpro_model,
    depthpro_transform,
    depth_fill="none",
    fuse_align=True,
    colors=None,
    fuse_opts=None,
):
    """Fuse LiDAR point-cloud depth with dense vision depth."""
    fuse_opts = dict(fuse_opts or {})
    lidar_kwargs = {
        "raster_mode": fuse_opts.get("raster_mode", "median"),
        "densify_radius": int(fuse_opts.get("densify_radius", 1)),
        "calib_refine": bool(fuse_opts.get("calib_refine", True)),
        "calib_max_shift": int(fuse_opts.get("calib_max_shift", 2)),
    }
    lidar = compute_lidar_depth(
        scene.image_np,
        points_world,
        calib,
        colors=colors,
        depth_fill=depth_fill,
        **lidar_kwargs,
    )
    f_px = float(lidar["K"][0, 0])
    estimate_depth, _ = run_estimate_depth_maps(
        scene, out_dir, depthpro_model, depthpro_transform, f_px=f_px
    )
    fused, fused_valid = fuse_lidar_with_estimate(
        lidar["depth_map"],
        lidar["valid_mask"],
        estimate_depth,
        align=fuse_align,
        align_fn=align_depth if fuse_align and not fuse_opts.get("banded_align", True) else None,
        image_np=scene.image_np,
        hit_count=lidar.get("hit_count"),
        soft_blend=bool(fuse_opts.get("soft_blend", True)),
        blend_radius=int(fuse_opts.get("blend_radius", 3)),
        banded_align=bool(fuse_opts.get("banded_align", True)),
        semantic_guide=bool(fuse_opts.get("semantic_guide", True)),
        edge_fill=bool(fuse_opts.get("edge_fill", True)),
    )
    write_depth_artifacts(
        out_dir,
        fused,
        fused_valid,
        lidar["K"],
        lidar["c2w"],
        lidar["points_cam"],
        lidar["colors"],
    )
    shift = lidar.get("proj_shift", (0.0, 0.0))
    if abs(shift[0]) + abs(shift[1]) > 0:
        print(f"  calib refine shift (du,dv)=({shift[0]:.0f},{shift[1]:.0f})")


def run_fuse_py123d_depth(
    sample,
    out_dir,
    depthpro_model,
    depthpro_transform,
    depth_fill="none",
    fuse_align=True,
    fuse_opts=None,
):
    """Fuse py123d LiDAR with vision depth and write nuScenes annotations."""
    scene = _NumpyImageScene(sample["image_np"])
    run_fuse_depth(
        scene,
        out_dir,
        sample["points_world"],
        sample["calib"],
        depthpro_model,
        depthpro_transform,
        depth_fill=depth_fill,
        fuse_align=fuse_align,
        fuse_opts=fuse_opts,
    )
    ann_path = Path(out_dir) / "nuscenes_annotations.json"
    with open(ann_path, "w") as f:
        json.dump(sample["annotations"], f)
    gt_path = Path(out_dir) / "nuscenes_gt_3dbbox.json"
    with open(gt_path, "w") as f:
        json.dump(sample.get("gt_3dbbox", []), f)


class _NumpyImageScene:
    """Minimal scene wrapper for prepare_output_dirs from numpy RGB."""

    def __init__(self, image_np):
        self.image_np = image_np
        self.image_pil = Image.fromarray(image_np.astype(np.uint8))


def run_py123d_depth(sample, out_dir, depth_fill="nearest"):
    """Build depth + annotations from py123d nuScenes sample dict."""
    scene = _NumpyImageScene(sample["image_np"])
    build_scene_outputs(
        out_dir,
        sample["image_np"],
        sample["points_world"],
        sample["calib"],
        depth_fill=depth_fill,
    )
    ann_path = Path(out_dir) / "nuscenes_annotations.json"
    with open(ann_path, "w") as f:
        json.dump(sample["annotations"], f)
    gt_path = Path(out_dir) / "nuscenes_gt_3dbbox.json"
    with open(gt_path, "w") as f:
        json.dump(sample.get("gt_3dbbox", []), f)


def copy_optional_annotations(scene_entry, out_dir):
    """Copy manifest annotations file into scene folder as bboxes.json if provided."""
    ann_path = scene_entry.get("annotations_path")
    if ann_path and os.path.exists(ann_path):
        dest = Path(out_dir) / "bboxes.json"
        if not dest.exists():
            shutil.copy2(ann_path, dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to the yaml config file", default='configs/image.yaml', type=str)
    parser.add_argument('--gpu_idx', type=int, default=0, help='GPU index (estimate mode only)')
    parser.add_argument('--start_index', type=int, default=0, help='Scene index to start processing')
    parser.add_argument('--end_index', type=int, default=1, help='Scene index to end processing')
    parser.add_argument("--split", help="dataset split name for output subfolder", default="val", type=str)
    parser.add_argument("--save_dir", help="save directory", default="../experimental_results/COCO/", type=str)
    parser.add_argument(
        "--depth_source",
        choices=["estimate", "lidar", "py123d", "fuse"],
        default=None,
        help="depth backend: estimate, lidar, py123d, or fuse (LiDAR + vision)",
    )
    parser.add_argument(
        "--manifest",
        help="path to LiDAR manifest JSON (required for depth_source=lidar)",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--depth_fill",
        choices=["none", "nearest"],
        default=None,
        help="hole filling for sparse LiDAR depth maps",
    )
    parser.add_argument("--py123d_data_root", default=None, help="PY123D_DATA_ROOT override")
    parser.add_argument("--py123d_dataset", default=None, help="py123d dataset name (default nuscenes)")
    parser.add_argument("--py123d_split", default=None, help="py123d split type: train/val/test")
    parser.add_argument("--camera_key", default=None, help="py123d camera id/name (e.g. CAM_FRONT)")
    parser.add_argument("--lidar_key", default=None, help="py123d lidar id (default merged)")
    parser.add_argument("--py123d_max_scenes", type=int, default=None, help="cap scenes from py123d filter")

    args, extras = parser.parse_known_args()
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    depth_source = args.depth_source or opt.run.depth.get("source", "estimate")
    depth_fill = args.depth_fill or opt.run.depth.get("fill", "none")
    fuse_align = bool(opt.run.depth.get("fuse_align", True))
    fuse_opts = {
        "soft_blend": bool(opt.run.depth.get("soft_blend", True)),
        "blend_radius": int(opt.run.depth.get("blend_radius", 3)),
        "banded_align": bool(opt.run.depth.get("banded_align", True)),
        "semantic_guide": bool(opt.run.depth.get("semantic_guide", True)),
        "edge_fill": bool(opt.run.depth.get("edge_fill", True)),
        "raster_mode": str(opt.run.depth.get("raster_mode", "median")),
        "densify_radius": int(opt.run.depth.get("densify_radius", 1)),
        "calib_refine": bool(opt.run.depth.get("calib_refine", True)),
        "calib_max_shift": int(opt.run.depth.get("calib_max_shift", 2)),
        "multiview_refine": bool(opt.run.depth.get("multiview_refine", True)),
    }
    end_index = None if args.end_index < 0 else args.end_index
    has_py123d = bool(
        args.py123d_data_root
        or os.environ.get("PY123D_DATA_ROOT")
        or opt.run.get("py123d")
    )

    if depth_source == "py123d" or (depth_source == "fuse" and has_py123d):
        from integrations.py123d.nuscenes_adapter import (
            Py123dNuScenesLoader,
            is_multi_camera_keys,
            resolve_camera_keys,
            scene_camera_output_dir,
            write_cameras_manifest,
        )

        py_cfg = opt.run.get("py123d", {})
        camera_keys = resolve_camera_keys(
            args.camera_key or py_cfg.get("camera_key", "CAM_FRONT"),
            py_cfg.get("camera_keys"),
        )
        multi_camera = is_multi_camera_keys(camera_keys)
        loader = Py123dNuScenesLoader(
            data_root=args.py123d_data_root,
            split_type=args.py123d_split or py_cfg.get("split_type", "val"),
            dataset_name=args.py123d_dataset or py_cfg.get("dataset", "nuscenes"),
            max_scenes=args.py123d_max_scenes or py_cfg.get("max_scenes"),
            camera_key=camera_keys[0],
            camera_keys=camera_keys,
            lidar_key=args.lidar_key or py_cfg.get("lidar_key", "merged"),
            frame_index=py_cfg.get("frame_index"),
        )
        split = args.split if args.split not in ("val",) else loader.split
        indices = range(args.start_index, end_index if end_index is not None else len(loader))

        depthpro_model = depthpro_transform = None
        if depth_source == "fuse":
            sys.path = ['./', '../external/MoGe'] + sys.path
            import depth_pro

            assert torch.cuda.is_available()
            device = f"cuda:{args.gpu_idx}"
            print("Loading DepthPro for fuse mode...")
            depthpro_model, depthpro_transform = depth_pro.create_model_and_transforms(
                device=device, precision=torch.float16
            )
            depthpro_model.eval()

        desc = "fuse depth (py123d+vision)" if depth_source == "fuse" else "py123d depth"
        for i in tqdm(indices, desc=desc):
            samples = loader.extract_samples(i)
            scene_root = Path(args.save_dir) / split / loader.output_dir_name(samples[0])
            if multi_camera:
                write_cameras_manifest(scene_root, camera_keys)
            for sample in samples:
                camera_key = sample.get("camera_key", camera_keys[0])
                out_dir = scene_camera_output_dir(scene_root, camera_key, multi_camera)
                scene = _NumpyImageScene(sample["image_np"])
                out_dir = prepare_output_dirs(out_dir, scene)
                print(f"Saving to {out_dir}")
                if depth_already_done(out_dir):
                    if not (out_dir / "nuscenes_annotations.json").exists():
                        with open(out_dir / "nuscenes_annotations.json", "w") as f:
                            json.dump(sample["annotations"], f)
                    if not (out_dir / "nuscenes_gt_3dbbox.json").exists():
                        with open(out_dir / "nuscenes_gt_3dbbox.json", "w") as f:
                            json.dump(sample.get("gt_3dbbox", []), f)
                    continue
                if depth_source == "fuse":
                    run_fuse_py123d_depth(
                        sample,
                        out_dir,
                        depthpro_model,
                        depthpro_transform,
                        depth_fill=depth_fill,
                        fuse_align=fuse_align,
                        fuse_opts=fuse_opts,
                    )
                else:
                    run_py123d_depth(sample, out_dir, depth_fill=depth_fill)
            if (
                depth_source == "fuse"
                and multi_camera
                and fuse_opts.get("multiview_refine", True)
            ):
                from geometry.lidar_depth import refine_scene_multiview_depths

                n_upd = refine_scene_multiview_depths(scene_root, camera_keys)
                if n_upd:
                    print(f"Multi-view depth refine updated {n_upd} cameras in {scene_root.name}")


    elif depth_source in ("lidar", "fuse"):
        if not args.manifest:
            raise ValueError(
                "--manifest is required for depth_source=lidar/fuse "
                "(or use fuse with py123d config + PY123D_DATA_ROOT)"
            )
        loader = LidarManifestLoader(args.manifest)
        split = get_lidar_save_split(loader, args.split)
        indices = range(args.start_index, end_index if end_index is not None else len(loader))

        depthpro_model = depthpro_transform = None
        if depth_source == "fuse":
            sys.path = ['./', '../external/MoGe'] + sys.path
            import depth_pro

            assert torch.cuda.is_available()
            device = f"cuda:{args.gpu_idx}"
            print("Loading DepthPro for fuse mode...")
            depthpro_model, depthpro_transform = depth_pro.create_model_and_transforms(
                device=device, precision=torch.float16
            )
            depthpro_model.eval()

        desc = "fuse depth (LiDAR+vision)" if depth_source == "fuse" else "LiDAR depth"
        for i in tqdm(indices, desc=desc):
            scene_entry = loader.get_scene_by_index(i)
            output_dir = os.path.join(
                args.save_dir, split, loader.output_dir_name(scene_entry)
            )

            opt.scene.type = "PointCloudScene"
            opt.scene.attributes.img_path = scene_entry["image_path"]
            opt.scene.attributes.pointcloud_path = scene_entry["pointcloud_path"]
            opt.scene.attributes.calib_path = scene_entry["calib_path"]
            scene = get_scene(opt.scene.type, opt.scene.attributes)

            out_dir = prepare_output_dirs(output_dir, scene)
            print(f"Saving to {out_dir}")

            if depth_already_done(out_dir):
                continue

            if depth_source == "fuse":
                points, colors = load_pointcloud(scene.pointcloud_path)
                calib = load_calib(scene.calib_path)
                run_fuse_depth(
                    scene,
                    out_dir,
                    points,
                    calib,
                    depthpro_model,
                    depthpro_transform,
                    depth_fill=depth_fill,
                    fuse_align=fuse_align,
                    colors=colors,
                    fuse_opts=fuse_opts,
                )
            else:
                run_lidar_depth(scene, out_dir, depth_fill=depth_fill)
            copy_optional_annotations(scene_entry, out_dir)

    elif depth_source == "estimate":
        sys.path = ['./', '../external/MoGe'] + sys.path
        import depth_pro

        dataset_root, annotations_dir = get_dataset_paths(args.split)
        loader = CoconutLoader(split=args.split, annotations_dir=annotations_dir)

        assert torch.cuda.is_available()
        device = f"cuda:{args.gpu_idx}"

        print("Loading DepthPro model...")
        depthpro_model, depthpro_transform = depth_pro.create_model_and_transforms(
            device=device, precision=torch.float16
        )
        depthpro_model.eval()
        print("DepthPro model loaded.")

        coco_end = end_index if end_index is not None else len(loader)
        for i in tqdm(range(args.start_index, coco_end), desc="Estimate depth"):
            image_info = loader.get_image_by_index(i)
            img_name = image_info["file_name"]
            image_path = os.path.join(dataset_root, img_name)
            output_dir = os.path.join(
                args.save_dir,
                args.split,
                img_name.split(".")[0].replace("/", "_").replace("-", "_"),
            )

            opt.scene.attributes.img_path = image_path
            scene = get_scene(opt.scene.type, opt.scene.attributes)

            out_dir = prepare_output_dirs(output_dir, scene)
            print(f"Saving to {out_dir}")

            if depth_already_done(out_dir):
                continue

            run_estimate_depth(scene, out_dir, depthpro_model, depthpro_transform)
