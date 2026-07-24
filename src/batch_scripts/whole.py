import argparse
from omegaconf import OmegaConf
import sys
import os
import json
from tqdm import tqdm
import torch
import trimesh
import cv2
sys.path = [
    "./",
    "../external/mast3r",
    "../external/mast3r/dust3r",
] + sys.path
from dataset_model import get_scene
from pathlib import Path
import numpy as np
from PIL import Image
from util import restore_mask_from_crop, align_to_depth_match, draw_cube
from util_3dbox import save_3d_with_ground_alignment_bbox, save_3d_bbox_from_depth_fallback
from matching.process_image_space import load_model
from batch_scripts.pipeline_loader import setup_pipeline_loop


# Taxonomy crop stems (e.g. 0_vehicle_sedancar) → legacy TRELLIS asset stems (0_car).
_MESH_CATEGORY_ALIASES = {
    "vehicle_sedancar": ["car", "sedan", "sed_car"],
    "vehicle_truck": ["truck", "ltrucks"],
    "vehicle_bus": ["bus"],
    "person_person": ["person", "pedestrian"],
    "non-vehicle_bicycle": ["bicycle", "cyclist", "bike"],
    "non_vehicle_bicycle": ["bicycle", "cyclist", "bike"],
    "non-vehicle_tricycle": ["tricycle", "tricyclist"],
    "non_vehicle_tricycle": ["tricycle", "tricyclist"],
}


def _resolve_object_space_stem(out_dir: Path, obj_id: str) -> str:
    """Prefer exact stem; fall back to legacy category aliases for existing .glb."""
    exact = out_dir / "object_space" / f"{obj_id}.glb"
    if exact.exists():
        return obj_id
    if "_" not in obj_id:
        return obj_id
    numeric, category = obj_id.split("_", 1)
    cat_key = category.strip().lower().replace(" ", "_")
    for alias in _MESH_CATEGORY_ALIASES.get(cat_key, []):
        cand = f"{numeric}_{alias}"
        if (out_dir / "object_space" / f"{cand}.glb").exists():
            print(f"Using legacy mesh asset {cand}.glb for crop {obj_id}")
            return cand
    return obj_id


def _ensure_mesh_assets_for_crop(out_dir: Path, obj_id: str, mesh_stem: str) -> str:
    """
    Make object_space/{obj_id}.glb (+ elevation dir) available for taxonomy crop ids
    by linking/copying legacy assets, and ensure crops/{obj_id}_rgba.png exists.
    Returns the stem to use with process_object (taxonomy obj_id when linked).
    """
    import shutil

    obj_space = out_dir / "object_space"
    crop_dir = out_dir / "crops"
    # Ensure rgba for taxonomy crop id (amodal may be skipped).
    rgba = crop_dir / f"{obj_id}_rgba.png"
    if not rgba.exists():
        reproj = crop_dir / f"{obj_id}_reproj.png"
        if reproj.exists():
            shutil.copy2(reproj, rgba)

    target_glb = obj_space / f"{obj_id}.glb"
    src_glb = obj_space / f"{mesh_stem}.glb"
    if not target_glb.exists() and src_glb.exists() and mesh_stem != obj_id:
        try:
            target_glb.symlink_to(src_glb.name)
        except OSError:
            shutil.copy2(src_glb, target_glb)

    src_elev_dir = obj_space / mesh_stem
    dst_elev_dir = obj_space / obj_id
    if src_elev_dir.is_dir() and not dst_elev_dir.exists() and mesh_stem != obj_id:
        try:
            dst_elev_dir.symlink_to(src_elev_dir.name)
        except OSError:
            shutil.copytree(src_elev_dir, dst_elev_dir)

    if (obj_space / f"{obj_id}.glb").exists():
        return obj_id
    return mesh_stem


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
        choices=["coco", "py123d", "bad_case_pkl"],
        default=None,
        help="coco / py123d / bad_case_pkl scene dirs under save_dir/split",
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

            mesh_stem = _resolve_object_space_stem(out_dir, obj_id)
            align_stem = _ensure_mesh_assets_for_crop(out_dir, obj_id, mesh_stem)
            elevation_path = out_dir / "object_space" / f"{align_stem}" / "estimated_elevation.npy"
            object_space_path = out_dir / "object_space" / f"{align_stem}.glb"
            if not os.path.exists(object_space_path):
                print(f"Object space file {object_space_path} does not exist")
                continue
            obj_mesh = trimesh.load(object_space_path)
            if isinstance(obj_mesh, trimesh.Scene):
                meshes = obj_mesh.dump()
                obj_mesh = meshes[0]

            project_root = out_dir
            try:
                transform = align_to_depth_match(
                    mask, depth_map, align_stem, project_root, mast3r_model
                )
            except Exception as e:
                print(f"Error aligning {obj_id} (stem={align_stem}): {e}")
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
            mesh_bbox_list = save_3d_with_ground_alignment_bbox(out_dir)
        else:
            print("No reconstructed meshes; using depth fallback for 3D boxes.")
            mesh_bbox_list = []

        # Mesh recon may be capped (e.g. SMOKE_MAX_RECON_OBJECTS). Fill remaining
        # crops with depth-based 3D boxes so Pred count tracks 2D detections.
        have_ids = {str(b.get("obj_id")) for b in mesh_bbox_list}
        fallback_bbox_list = save_3d_bbox_from_depth_fallback(
            out_dir, exclude_obj_ids=have_ids, write_json=False
        )
        if fallback_bbox_list:
            print(
                f"Depth fallback added {len(fallback_bbox_list)} boxes "
                f"(mesh={len(mesh_bbox_list)}, exclude={sorted(have_ids)})"
            )
        bbox_list = list(mesh_bbox_list) + list(fallback_bbox_list)
        if len(bbox_list) == 0:
            print("No mesh or depth boxes; retrying full depth fallback.")
            bbox_list = save_3d_bbox_from_depth_fallback(out_dir, write_json=False)

        with open(out_dir / "3dbbox_ground.json", "w") as fp:
            json.dump(bbox_list, fp)
        draw_cube(out_dir, is_ground=True)

        if os.path.exists(out_dir / "3dbbox_ground.json"):
            os.rename(out_dir / "3dbbox_ground.json", out_dir / "3dbbox.json")
