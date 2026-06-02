import argparse
from omegaconf import OmegaConf
import sys
import os
from tqdm import tqdm
import torch
sys.path = [
    './',
    '../external/InvSR',
] + sys.path
from dataset_model import get_scene
from pathlib import Path
from inference_invsr_us import get_parser, get_configs, InvSamplerSR
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
        help="coco: COCONUT; py123d: scene dirs from depth.py --depth_source py123d",
    )

    args, extras = parser.parse_known_args()
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    data_backend, split, loader, indices = setup_pipeline_loop(
        args, opt, require_files=("input.png",)
    )

    assert torch.cuda.is_available()

    invsr_args = get_parser(description="My CLI tool")
    invsr_configs = get_configs(invsr_args)
    sampler = InvSamplerSR(invsr_configs)

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
        (out_dir / "enhanced").mkdir(exist_ok=True)

        if not (out_dir / "input.png").exists():
            scene.image_pil.save(out_dir / "input.png")

        enhance_path = out_dir / "enhanced" / "input.png"
        if enhance_path.exists():
            continue
        sampler.inference(f"{out_dir}/input.png", out_path=out_dir / "enhanced", bs=invsr_args.bs)
