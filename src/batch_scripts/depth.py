"""
Depth estimation pipeline.

Modes:
  estimate (default): MoGe + DepthPro + RANSAC alignment
  lidar: world-frame LiDAR + calibration -> depth_map / PLY / cam_params

Usage:
    python batch_scripts/depth.py --start_index 0 --end_index 100 --split val
    python batch_scripts/depth.py --depth_source lidar --manifest ../dataset/lidar/manifest.json --config configs/lidar.yaml
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

from util import depth_to_points
from batch_scripts.coconut_loader import CoconutLoader, get_dataset_paths
from batch_scripts.lidar_loader import LidarManifestLoader, get_lidar_save_split
from geometry.lidar_depth import build_scene_outputs, load_calib, load_pointcloud


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


def align_depth(relative_depth, metric_depth, mask=None, min_samples=0.2, max_valid_depth=400.0):
    """
    Align scale-invariant depth to metric depth using RANSAC linear regression.
    """
    from sklearn.linear_model import RANSACRegressor, LinearRegression

    regressor = RANSACRegressor(estimator=LinearRegression(fit_intercept=False), min_samples=min_samples)

    valid = (~np.isinf(relative_depth)) & (metric_depth < max_valid_depth)
    if mask is not None:
        valid &= mask

    if valid.sum() == 0:
        print("Warning: No valid points for alignment. Returning metric depth.")
        return metric_depth

    try:
        regressor.fit(relative_depth[valid].reshape(-1, 1), metric_depth[valid].reshape(-1, 1))
    except Exception as e:
        print(f"Error fitting RANSACRegressor: {e}, using metric depth directly")
        return metric_depth

    depth = np.full_like(relative_depth, 10000.0)

    if mask is not None:
        masked_pred = regressor.predict(relative_depth[mask].reshape(-1, 1)).flatten()
        depth[mask] = masked_pred
    else:
        valid_mask = ~np.isinf(relative_depth)
        masked_pred = regressor.predict(relative_depth[valid_mask].reshape(-1, 1)).flatten()
        depth[valid_mask] = masked_pred

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


def run_estimate_depth(scene, out_dir, depthpro_model, depthpro_transform):
    """MoGe + DepthPro depth estimation (original pipeline)."""
    sys.path = ['./', '../external/MoGe'] + [p for p in sys.path if p not in ('./', '../external/MoGe')]
    from infer_moge import infer_geometry_on_image

    _, moge_depth_map, moge_mask, K_img = infer_geometry_on_image(f'{out_dir}/input.png', out_dir)

    img = depthpro_transform(scene.image_pil)
    prediction = depthpro_model.infer(img, f_px=K_img[0, 0])
    pro_depth_map = prediction["depth"].cpu().numpy()

    depth_map = align_depth(moge_depth_map, pro_depth_map, mask=moge_mask)
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
        choices=["estimate", "lidar"],
        default=None,
        help="depth backend: estimate (MoGe+DepthPro) or lidar (external point cloud)",
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

    args, extras = parser.parse_known_args()
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    depth_source = args.depth_source or opt.run.depth.get("source", "estimate")
    depth_fill = args.depth_fill or opt.run.depth.get("fill", "none")
    end_index = None if args.end_index < 0 else args.end_index

    if depth_source == "lidar":
        if not args.manifest:
            raise ValueError("--manifest is required when --depth_source=lidar")
        loader = LidarManifestLoader(args.manifest)
        split = get_lidar_save_split(loader, args.split)
        indices = range(args.start_index, end_index if end_index is not None else len(loader))

        for i in tqdm(indices, desc="LiDAR depth"):
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

            run_lidar_depth(scene, out_dir, depth_fill=depth_fill)
            copy_optional_annotations(scene_entry, out_dir)

    else:
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
