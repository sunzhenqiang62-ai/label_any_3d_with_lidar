"""
Scan LabelAny3D output dirs produced by py123d depth step (for crops backend).
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


class Py123dOutputLoader:
    """List scene folders under save_dir/split with nuscenes_annotations.json."""

    def __init__(self, save_dir: str, split: str):
        self.root = Path(save_dir) / split
        if not self.root.exists():
            raise FileNotFoundError(f"Output root not found: {self.root}")
        self.scenes: List[Path] = sorted(
            [p for p in self.root.iterdir() if p.is_dir()],
            key=lambda p: p.name,
        )
        self.split = split

    def __len__(self) -> int:
        return len(self.scenes)

    def get_scene_by_index(self, index: int) -> Dict[str, Any]:
        scene_dir = self.scenes[index]
        ann_path = scene_dir / "nuscenes_annotations.json"
        img_path = scene_dir / "input.png"
        if not img_path.exists():
            raise FileNotFoundError(f"Missing input.png in {scene_dir}")
        if not ann_path.exists():
            raise FileNotFoundError(
                f"Missing nuscenes_annotations.json in {scene_dir}. "
                "Run depth.py with --depth_source py123d first."
            )
        with open(ann_path, "r") as f:
            annotations = json.load(f)
        return {
            "id": scene_dir.name,
            "scene_dir": str(scene_dir),
            "image_path": str(img_path),
            "annotations": annotations,
            "file_name": f"{scene_dir.name}.jpg",
        }

    def output_dir_name(self, scene: Dict[str, Any]) -> str:
        return Path(scene["scene_dir"]).name


def get_py123d_save_split(split_override: Optional[str], default_split: str) -> str:
    return split_override if split_override else default_split
