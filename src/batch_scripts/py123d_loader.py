"""
Scan LabelAny3D output dirs produced by py123d depth step (for crops backend).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from integrations.py123d.nuscenes_adapter import discover_camera_view_dirs


class Py123dOutputLoader:
    """List scene folders under save_dir/split (one entry per camera view)."""

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
        for scene_root in sorted(self.root.iterdir(), key=lambda x: x.name):
            if not scene_root.is_dir():
                continue
            for view_dir in discover_camera_view_dirs(scene_root):
                if self._view_valid(view_dir):
                    self.scenes.append(view_dir)

    def _view_valid(self, view_dir: Path) -> bool:
        for name in self.require_files:
            if not (view_dir / name).exists():
                return False
        if self.require_annotations and not (view_dir / "nuscenes_annotations.json").exists():
            return False
        return True

    def __len__(self) -> int:
        return len(self.scenes)

    def get_scene_by_index(self, index: int) -> Dict[str, Any]:
        view_dir = self.scenes[index]
        img_path = view_dir / "input.png"
        if not img_path.exists():
            raise FileNotFoundError(f"Missing input.png in {view_dir}")

        annotations = None
        ann_path = view_dir / "nuscenes_annotations.json"
        if ann_path.exists():
            with open(ann_path, "r") as f:
                annotations = json.load(f)
        elif self.require_annotations:
            raise FileNotFoundError(
                f"Missing nuscenes_annotations.json in {view_dir}. "
                "Run depth.py with --depth_source py123d first."
            )

        scene_root = view_dir.parent if view_dir.name.startswith("CAM_") else view_dir
        camera_key = view_dir.name if view_dir != scene_root else "CAM_FRONT"
        scene_id = scene_root.name
        return {
            "id": f"{scene_id}/{camera_key}" if view_dir != scene_root else scene_id,
            "scene_id": scene_id,
            "scene_root": str(scene_root),
            "camera_key": camera_key,
            "scene_dir": str(view_dir),
            "image_path": str(img_path),
            "annotations": annotations,
            "file_name": f"{scene_id}_{camera_key}.jpg",
        }

    def output_dir_name(self, scene: Dict[str, Any]) -> str:
        return Path(scene["scene_dir"]).name


def get_py123d_save_split(split_override: Optional[str], default_split: str) -> str:
    return split_override if split_override else default_split
