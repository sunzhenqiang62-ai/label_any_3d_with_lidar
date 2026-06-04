import argparse
from omegaconf import OmegaConf
import sys
import os
import json
from tqdm import tqdm
import torch
import cv2
sys.path = ['./',] + sys.path
from dataset_model import get_scene
from pathlib import Path
import numpy as np
from PIL import Image
from util import read_bounding_boxes_segmentations, crop_object
from scipy.ndimage import binary_opening
from detectron2.structures import BoxMode
from batch_scripts.pipeline_loader import setup_pipeline_loop


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to the yaml config file", default='configs/image.yaml', type=str)
    parser.add_argument('--gpu_idx', type=int, default=0, help='GPU index')
    parser.add_argument('--start_index', type=int, default=0, help='Object index to start processing')
    parser.add_argument('--end_index', type=int, default=-1, help='Object index to end processing (-1 = all)')
    parser.add_argument("--split", help="split", default="val", type=str)
    parser.add_argument("--save_dir", help="save directory", default="../experimental_results/COCO/", type=str)
    parser.add_argument(
        "--data_backend",
        choices=["coco", "py123d"],
        default=None,
        help="coco: COCONUT annotations; py123d: scene dirs from depth.py --depth_source py123d",
    )

    args, extras = parser.parse_known_args()
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    data_backend, split, loader, indices = setup_pipeline_loop(
        args,
        opt,
        require_files=("input.png",),
        require_annotations=True,
    )
    crop_size = 512

    for i in tqdm(indices):
        scene_entry = loader.get_scene_by_index(i)
        output_dir = scene_entry["scene_dir"]
        opt.scene.attributes.img_path = scene_entry["image_path"]
        scene = get_scene(opt.scene.type, opt.scene.attributes)
        annotations = scene_entry["annotations"]
        img_name = scene_entry["file_name"]

        out_dir = Path(output_dir)
        print(f"Saving to {out_dir}")
        out_dir.mkdir(exist_ok=True, parents=True)
        (out_dir / "crops").mkdir(exist_ok=True)
        (out_dir / "object_space").mkdir(exist_ok=True)
        (out_dir / "reconstruction").mkdir(exist_ok=True)

        if annotations:
            bboxes, masks, object_ids, instance_labels = read_bounding_boxes_segmentations(
                annotations, scene.image_pil.size
            )
            if len(masks[object_ids]) == 0:
                print(f"No valid objects found in {img_name}")
                continue
        else:
            print(f"No annotations found for {img_name}")
            continue

        bboxes = BoxMode.convert(np.array(bboxes), BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)

        scaled_masks = []
        for mask in masks:
            mask = mask.astype(np.uint8)
            new_size = (mask.shape[1] * 4, mask.shape[0] * 4)
            scaled_mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)
            scaled_masks.append(scaled_mask)
        masks = np.array(scaled_masks)

        enhanced_path = out_dir / "enhanced" / "input.png"
        if not enhanced_path.exists():
            print(f"Missing {enhanced_path}; run enhance.py first")
            continue
        enhanced_image = Image.open(enhanced_path)
        scene.image_pil = enhanced_image.convert("RGB")
        scene.image_np = np.array(enhanced_image)
        selected_bboxes = []
        for j in range(len(masks[object_ids]) - 1, -1, -1):
            label = instance_labels[object_ids[j]]
            label = label.replace(" (", ", ").replace(")", "")
            obj_id = f"{j}_{label.replace(' ', '_')}"

            mask = binary_opening(masks[object_ids][j], np.ones((7, 7)))
            if mask.sum() < 6400:
                print(f"Skipped too small object: {obj_id}")
                continue
            selected_bboxes.append(bboxes[object_ids[j]])
            crop_path = out_dir / "crops" / f"{obj_id}_reproj.png"
            crop_params_path = out_dir / "crops" / f"{obj_id}_crop_params.npy"
            if not crop_path.exists() or not crop_params_path.exists():
                crop, crop_params = crop_object(scene.image_np, mask, crop_size)
                crop.save(crop_path)
                crop_params = np.array([crop_params[0] / 4, crop_params[1] / 4, crop_params[2] * 4])
                np.save(crop_params_path, crop_params)
        with open(out_dir / "bboxes.json", "w") as f:
            json.dump(np.array(selected_bboxes).tolist(), f)
