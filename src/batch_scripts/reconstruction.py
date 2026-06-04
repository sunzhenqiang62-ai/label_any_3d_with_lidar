import argparse
from omegaconf import OmegaConf
import sys
import os
from tqdm import tqdm
import torch
sys.path = ["./"] + sys.path
from dataset_model import get_scene
from pathlib import Path
from model_wrappers import infer_with_trellis, infer_with_hunyuan
from batch_scripts.pipeline_loader import setup_pipeline_loop


def reconstruct_object(run_opt, out_dir, obj_id):
    if run_opt.obj_rec == "trellis":
        print("trellis is used for reconstruction")
        infer_with_trellis(out_dir, obj_id)
    elif run_opt.obj_rec == "hunyuan3d":
        print("hunyuan3d is used for reconstruction")
        infer_with_hunyuan(out_dir, obj_id)
    else:
        raise ValueError(
            f"Unknown reconstruction model: {run_opt.obj_rec}. Use 'trellis' or 'hunyuan3d'."
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to the yaml config file", default="configs/image.yaml", type=str)
    parser.add_argument("--gpu_idx", type=int, default=0, help="GPU index")
    parser.add_argument("--start_index", type=int, default=0, help="Object index to start processing")
    parser.add_argument("--end_index", type=int, default=-1, help="Object index to end processing (-1 = all)")
    parser.add_argument("--split", help="split", default="val", type=str)
    parser.add_argument("--save_dir", help="save directory", default="../experimental_results/COCO/", type=str)
    parser.add_argument(
        "--obj_rec",
        help="reconstruction model",
        default="trellis",
        choices=["trellis", "hunyuan3d"],
        type=str,
    )
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

    for i in tqdm(indices):
        scene_entry = loader.get_scene_by_index(i)
        output_dir = scene_entry["scene_dir"]
        opt.scene.attributes.img_path = scene_entry["image_path"]
        opt.run.obj_rec = args.obj_rec
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
            object_space_path = out_dir / "object_space" / f"{obj_id}.glb"
            if not object_space_path.exists():
                reconstruct_object(opt.run, out_dir, obj_id)
