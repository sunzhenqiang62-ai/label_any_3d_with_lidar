import argparse
from omegaconf import OmegaConf
import sys
import os
import json
from tqdm import tqdm
import torch
import trimesh
import cv2
sys.path = ["./"] + sys.path
from dataset_model import get_scene
from pathlib import Path
import numpy as np
from PIL import Image
from util import restore_mask_from_crop, align_to_depth_match, draw_cube
from util_3dbox import save_3d_with_ground_alignment_bbox, save_3d_bbox_from_depth_fallback
from matching.process_image_space import load_model
from batch_scripts.pipeline_loader import setup_pipeline_loop


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to the yaml config file", default="configs/image.yaml", type=str)
    parser.add_argument("--gpu_idx", type=int, default=0, help="GPU index")
    parser.add_argument("--start_index", type=int, default=0, help="Object index to start processing")
    parser.add_argument("--end_index", type=int, default=-1, help="Object index to end processing (-1 = all)")
    parser.add_argument("--split", help="split", default="val", type=str)
    parser.add_argument("--save_dir", help="save directory", default="../experimental_results/COCO/", type=str)
    parser.add_argument(
        "--data_backend",
        choices=["coco", "py123d"],
        default=None,
        help="coco: COCONUT; py123d: scene dirs from depth.py --depth_source py123d",
    )

    args, extras = parser.parse_known_args()
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    data_backend, split, loader, indices = setup_pipeline_loop(
        args,
        opt,
        require_files=("input.png", "depth_map.npy", "cam_params.json"),
    )

    assert torch.cuda.is_available()
    device = f"cuda:{args.gpu_idx}"

    mast3r_model = load_model(device)
    for i in tqdm(indices):
        scene_entry = loader.get_scene_by_index(i)
        output_dir = scene_entry["scene_dir"]
        opt.scene.attributes.img_path = scene_entry["image_path"]
        opt.run.amodal_completion = "our"
        scene = get_scene(opt.scene.type, opt.scene.attributes)

        out_dir = Path(output_dir)
        print(f"Saving to {out_dir}")
        out_dir.mkdir(exist_ok=True, parents=True)
        (out_dir / "crops").mkdir(exist_ok=True)
        (out_dir / "object_space").mkdir(exist_ok=True)
        (out_dir / "reconstruction").mkdir(exist_ok=True)

        if (out_dir / "3dbbox.json").exists():
            continue
        depth_map = np.load(out_dir / "depth_map.npy")

        scene_mesh = trimesh.Scene([None])
        crop_root = out_dir / "crops"
        crop_paths = list(crop_root.glob("*_reproj.png"))
        for crop_path in reversed(crop_paths):
            obj_id = crop_path.stem.replace("_reproj", "")

            crop = Image.open(crop_path)
            crop_params_path = out_dir / "crops" / f"{obj_id}_crop_params.npy"
            if not crop_params_path.exists():
                continue
            crop_params = np.load(crop_params_path)
            resized_mask = np.array(crop)[:, :, 3] > 127
            mask = restore_mask_from_crop(
                resized_mask,
                crop_params[0],
                crop_params[1],
                crop_params[2],
                scene.image_np.shape[:2],
            )
            full_crop_path = out_dir / "crops" / f"{obj_id}_rgba.png"
            if not full_crop_path.exists():
                full_crop_path = out_dir / "crops" / f"{obj_id}_reproj.png"

            elevation_path = out_dir / "object_space" / f"{obj_id}" / "estimated_elevation.npy"
            object_space_path = out_dir / "object_space" / f"{obj_id}.glb"
            if not os.path.exists(object_space_path):
                print(f"Object space file {object_space_path} does not exist")
                continue
            obj_mesh = trimesh.load(object_space_path)
            if isinstance(obj_mesh, trimesh.Scene):
                meshes = obj_mesh.dump()
                obj_mesh = meshes[0]

            project_root = out_dir
            try:
                transform = align_to_depth_match(mask, depth_map, obj_id, project_root, mast3r_model)
            except Exception as e:
                print(f"Error aligning {obj_id}: {e}")
                continue
            obj_mesh.apply_transform(transform)
            # align_to_depth_match uses PyTorch3D camera coordinates. Convert the
            # mesh to OpenCV camera coordinates expected by cam_params/K and
            # downstream bbox3D_cam consumers.
            convention_transform = np.array(
                [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            )
            obj_mesh.apply_transform(convention_transform)

            obj_mesh.export(out_dir / "reconstruction" / f"{obj_id}.glb")
            scene_mesh.add_geometry([obj_mesh])
            print(f"Saved, {obj_id}.glb")
            canonical_upright = (convention_transform @ transform)[:, 1]
            np.save(
                out_dir / "reconstruction" / f"{obj_id}_canonical_upright.npy",
                canonical_upright,
            )
        if len(scene_mesh.geometry) > 0:
            scene_mesh.export(out_dir / "reconstruction" / "full_scene.glb")

            print("Going to save ground aligned bbox")
            bbox_list = save_3d_with_ground_alignment_bbox(out_dir)
            if len(bbox_list) == 0:
                print("Mesh-based bbox is empty; using depth fallback bbox.")
                bbox_list = save_3d_bbox_from_depth_fallback(out_dir)
            draw_cube(out_dir, is_ground=True)

            if os.path.exists(out_dir / "3dbbox_ground.json"):
                os.rename(out_dir / "3dbbox_ground.json", out_dir / "3dbbox.json")
