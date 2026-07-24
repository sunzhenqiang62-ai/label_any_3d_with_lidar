import argparse
from omegaconf import OmegaConf
import sys
import os
from tqdm import tqdm
import torch
sys.path = [
    './',
    '../external/dreamgaussian',
    '../external/One-2-3-45',
] + sys.path
from dataset_model import get_scene
from pathlib import Path
from PIL import Image
from util import initialize_acompletion, complete_crop
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
        choices=["coco", "py123d", "bad_case_pkl"],
        default=None,
        help="coco / py123d / bad_case_pkl scene dirs under save_dir/split",
    )

    args, extras = parser.parse_known_args()
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    data_backend, split, loader, indices = setup_pipeline_loop(
        args, opt, require_files=("input.png",)
    )

    assert torch.cuda.is_available()
    device = f"cuda:{args.gpu_idx}"

    amodal_mode = opt.run.get("amodal_completion", "our")
    if amodal_mode in (None, "null", "none", ""):
        amodal_mode = None
    acompletion_p = initialize_acompletion(device) if amodal_mode == "our" else None

    for i in tqdm(indices):
        scene_entry = loader.get_scene_by_index(i)
        output_dir = scene_entry["scene_dir"]
        opt.scene.attributes.img_path = scene_entry["image_path"]
        scene = get_scene(opt.scene.type, opt.scene.attributes)

        out_dir = Path(output_dir)
        print(f"Saving to {out_dir}")
        out_dir.mkdir(exist_ok=True, parents=True)
        (out_dir / "crops").mkdir(exist_ok=True)
        (out_dir / "object_space").mkdir(exist_ok=True)
        (out_dir / "reconstruction").mkdir(exist_ok=True)

        crop_root = out_dir / "crops"
        crop_paths = list(crop_root.glob("*_reproj.png"))
        for crop_path in reversed(crop_paths):
            obj_id = crop_path.stem.replace("_reproj", "")
            label = obj_id.split("_", 1)[-1]

            full_crop_path = out_dir / "crops" / f"{obj_id}_rgba.png"
            if not full_crop_path.exists():
                crop = Image.open(crop_path)
                full_crop = complete_crop(crop, label, acompletion_p, opt.run.amodal_completion)
                full_crop.save(full_crop_path)
