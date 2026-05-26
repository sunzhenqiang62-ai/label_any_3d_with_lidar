"""
Scan LabelAny3D output dirs produced by py123d depth step (for crops backend).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class Py123dOutputLoader:
    """List scene folders under save_dir/split."""

    def __init__(
        self,
        save_dir: str,
        split: str,
        require_files: Tuple[str, ...] = ("input.png",),
        require_annotations: bool = False,
    ):
        self.root = Path(save_dir) / split
        if not self.root.exists():
            raise FileNotFoundError(f"Output root not found: {self.root}")
        self.require_files = require_files
        self.require_annotations = require_annotations
        self.split = split
        self.scenes: List[Path] = []
        for p in sorted(self.root.iterdir(), key=lambda x: x.name):
            if not p.is_dir():
                continue
            if self._scene_valid(p):
                self.scenes.append(p)

    def _scene_valid(self, scene_dir: Path) -> bool:
        for name in self.require_files:
            if not (scene_dir / name).exists():
                return False
        if self.require_annotations and not (scene_dir / "nuscenes_annotations.json").exists():
            return False
        return True

    def __len__(self) -> int:
        return len(self.scenes)

    def get_scene_by_index(self, index: int) -> Dict[str, Any]:
        scene_dir = self.scenes[index]
        img_path = scene_dir / "input.png"
        if not img_path.exists():
            raise FileNotFoundError(f"Missing input.png in {scene_dir}")

        annotations = None
        ann_path = scene_dir / "nuscenes_annotations.json"
        if ann_path.exists():
            with open(ann_path, "r") as f:
                annotations = json.load(f)
        elif self.require_annotations:
            raise FileNotFoundError(
                f"Missing nuscenes_annotations.json in {scene_dir}. "
                "Run depth.py with --depth_source py123d first."
            )

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
