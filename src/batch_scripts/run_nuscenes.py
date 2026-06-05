"""
One-command nuScenes (py123d) experiment runner.

Usage (from src/):
    export PY123D_DATA_ROOT=/path/to/py123d_data
    python batch_scripts/run_nuscenes.py --preset smoke
    python batch_scripts/run_nuscenes.py --preset full --steps depth,enhance,crops
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Run from src/
SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from batch_scripts.pipeline_loader import resolve_py123d_split

ALL_STEPS = [
    "depth",
    "enhance",
    "crops",
    "completion",
    "elevation",
    "reconstruction",
    "whole",
    "combine",
]

STEP_SCRIPTS = {
    "depth": ("batch_scripts/depth.py", []),
    "enhance": ("batch_scripts/enhance.py", []),
    "crops": ("batch_scripts/get_crops_enhanced.py", []),
    "completion": ("batch_scripts/completion.py", []),
    "elevation": ("batch_scripts/elevation.py", []),
    "reconstruction": ("batch_scripts/reconstruction.py", []),
    "whole": ("batch_scripts/whole.py", []),
    "combine": ("tools/combine_results.py", []),
}

SKIP_MARKERS = {
    "depth": ["depth_map.npy", "cam_params.json"],
    "enhance": ["enhanced/input.png"],
    "crops": ["crops"],
    "completion": [],  # per-object; always run
    "elevation": [],
    "reconstruction": [],
    "whole": ["3dbbox.json"],
    "combine": [],  # checked via output json below
}

VIZ_AFTER = {
    "after_depth": ["gt_2d", "depth"],
    "after_crops": ["gt_2d", "depth", "crops"],
    "after_whole": ["gt_2d", "depth", "crops", "bbox_3d", "compose"],
    "all": ["compose"],
}


def _scene_dirs(save_dir: str, split: str) -> List[Path]:
    root = Path(save_dir) / split
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)


def scene_skip_complete(scene_dir: Path, step: str) -> bool:
    markers = SKIP_MARKERS.get(step, [])
    if not markers:
        return False
    for m in markers:
        p = scene_dir / m
        if m == "crops":
            if not p.exists() or not any(p.glob("*_reproj.png")):
                return False
        elif not p.exists():
            return False
    return True


def should_skip_step(
    step: str,
    save_dir: str,
    split: str,
    start_index: int,
    end_index: int,
    skip_existing: bool,
) -> bool:
    if not skip_existing:
        return False
    if step == "combine":
        out_json = Path(save_dir) / f"nuScenes3D_{split}.json"
        return out_json.exists()
    scenes = _scene_dirs(save_dir, split)
    if end_index < 0:
        end_index = len(scenes)
    subset = scenes[start_index:end_index]
    if not subset:
        return False
    return all(scene_skip_complete(s, step) for s in subset)


def run_visualize(
    save_dir: str,
    split: str,
    when: str,
    viz_backend: str,
    start_index: int,
    end_index: int,
    dry_run: bool,
) -> None:
    modes = VIZ_AFTER.get(when, [])
    if not modes or when == "none":
        return
    mode_str = ",".join(modes)
    root = Path(save_dir) / split
    scenes = _scene_dirs(save_dir, split)
    if end_index < 0:
        end_index = len(scenes)
    for scene_dir in scenes[start_index:end_index]:
        cmd = [
            sys.executable,
            "tools/visualize_scene.py",
            "--scene_dir",
            str(scene_dir),
            "--mode",
            mode_str,
            "--backend",
            viz_backend,
        ]
        _run_cmd(cmd, dry_run)


def _run_cmd(cmd: List[str], dry_run: bool) -> int:
    print("$", " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.call(cmd, cwd=str(SRC_ROOT))


def build_step_cmd(
    step: str,
    args: argparse.Namespace,
    split: str,
) -> List[str]:
    script, extra = STEP_SCRIPTS[step]
    cmd = [
        sys.executable,
        script,
        "--config",
        args.config,
        "--save_dir",
        args.save_dir,
        "--start_index",
        str(args.start_index),
        "--end_index",
        str(args.end_index),
        "--gpu_idx",
        str(args.gpu_idx),
    ]

    if step == "depth":
        from omegaconf import OmegaConf

        opt = OmegaConf.load(str(SRC_ROOT / args.config))
        depth_cfg = opt.run.get("depth", {})
        depth_source = depth_cfg.get("source", "py123d")
        depth_fill = depth_cfg.get("fill", "nearest")
        cmd.extend(
            [
                "--depth_source",
                str(depth_source),
                "--depth_fill",
                str(depth_fill),
                "--split",
                split,
            ]
        )
        if args.py123d_data_root:
            cmd.extend(["--py123d_data_root", args.py123d_data_root])
        if args.py123d_dataset:
            cmd.extend(["--py123d_dataset", args.py123d_dataset])
        if args.py123d_split:
            cmd.extend(["--py123d_split", args.py123d_split])
        if args.py123d_max_scenes is not None:
            cmd.extend(["--py123d_max_scenes", str(args.py123d_max_scenes)])
    elif step == "combine":
        cmd = [
            sys.executable,
            script,
            "--split",
            split,
            "--results_dir",
            args.save_dir,
            "--output",
            str(Path(args.save_dir) / f"nuScenes3D_{split}.json"),
        ]
    else:
        cmd.extend(
            [
                "--data_backend",
                "py123d",
                "--split",
                split,
            ]
        )

    if args.extra_args and step != "combine":
        cmd.extend(args.extra_args.split())
    return cmd


def apply_preset(args: argparse.Namespace) -> None:
    if args.preset == "smoke":
        if args.py123d_max_scenes is None:
            args.py123d_max_scenes = 3
        if args.end_index == -1:
            args.end_index = 3
        if args.config == "configs/py123d_nuscenes.yaml":
            args.config = "configs/py123d_nuscenes_smoke.yaml"
    elif args.preset == "locateanything":
        if args.py123d_max_scenes is None:
            args.py123d_max_scenes = 1
        if args.end_index == -1:
            args.end_index = 1
        args.config = "configs/py123d_nuscenes_locateanything.yaml"
    elif args.preset == "dev":
        if args.py123d_max_scenes is None:
            args.py123d_max_scenes = 10
        if args.end_index == -1:
            args.end_index = 10


def parse_steps(steps_arg: str) -> List[str]:
    if steps_arg == "all":
        return ALL_STEPS.copy()
    steps = [s.strip() for s in steps_arg.split(",") if s.strip()]
    for s in steps:
        if s not in ALL_STEPS:
            raise ValueError(f"Unknown step: {s}. Choose from {ALL_STEPS}")
    return steps


def main():
    parser = argparse.ArgumentParser(description="Run nuScenes py123d LabelAny3D pipeline")
    parser.add_argument(
        "--preset",
        choices=["smoke", "dev", "full", "locateanything"],
        default="smoke",
        help="smoke/oneformer: py123d_nuscenes_smoke.yaml; locateanything: VLM detector preset",
    )
    parser.add_argument("--config", default="configs/py123d_nuscenes.yaml")
    parser.add_argument("--save_dir", default="../experimental_results/nuScenes/")
    parser.add_argument("--steps", default="all", help="Comma list or 'all'")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)
    parser.add_argument("--gpu_idx", type=int, default=0)
    parser.add_argument("--py123d_data_root", default=os.environ.get("PY123D_DATA_ROOT"))
    parser.add_argument("--py123d_dataset", default=None)
    parser.add_argument("--py123d_split", default=None)
    parser.add_argument("--py123d_max_scenes", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--visualize",
        choices=["none", "after_depth", "after_crops", "after_whole", "all"],
        default="none",
    )
    parser.add_argument(
        "--viz_backend",
        choices=["preview", "blender", "both"],
        default="preview",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--extra_args", default="", help="Extra args passed to each step script")
    args = parser.parse_args()

    apply_preset(args)

    try:
        from omegaconf import OmegaConf

        opt = OmegaConf.load(str(SRC_ROOT / args.config))
    except ImportError:
        import yaml

        with open(SRC_ROOT / args.config, "r") as f:
            opt = yaml.safe_load(f)

    split = resolve_py123d_split("val", opt)

    steps = parse_steps(args.steps)
    viz_triggers = {
        "depth": "after_depth",
        "crops": "after_crops",
        "whole": "after_whole",
    }

    for step in steps:
        if should_skip_step(
            step, args.save_dir, split, args.start_index, args.end_index, args.skip_existing
        ):
            print(f"Skipping {step} (outputs exist)")
            continue

        cmd = build_step_cmd(step, args, split)
        rc = _run_cmd(cmd, args.dry_run)
        if rc != 0:
            print(f"Step {step} failed with code {rc}")
            sys.exit(rc)

        if args.visualize != "none":
            trigger = viz_triggers.get(step)
            if trigger and (
                args.visualize == trigger
                or args.visualize == "all"
                or (args.visualize == "after_whole" and step in ("depth", "crops", "whole"))
            ):
                run_visualize(
                    args.save_dir,
                    split,
                    trigger,
                    args.viz_backend,
                    args.start_index,
                    args.end_index,
                    args.dry_run,
                )

    if args.visualize == "all" and not args.dry_run:
        run_visualize(
            args.save_dir,
            split,
            "all",
            args.viz_backend,
            args.start_index,
            args.end_index,
            args.dry_run,
        )

    print("Done.")


if __name__ == "__main__":
    main()
