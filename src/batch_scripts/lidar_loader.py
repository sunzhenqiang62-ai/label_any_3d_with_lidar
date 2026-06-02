"""
Manifest-based loader for RGB + LiDAR + calibration scenes.

Usage:
    from batch_scripts.lidar_loader import LidarManifestLoader

    loader = LidarManifestLoader("../dataset/lidar/manifest.json")
    scene = loader.get_scene_by_index(0)
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


class LidarManifestLoader:
    """Load scenes from a JSON manifest (paths relative to manifest directory)."""

    def __init__(self, manifest_path: str):
        self.manifest_path = Path(manifest_path).resolve()
        if not self.manifest_path.exists():
            raise FileNotFoundError(self.manifest_path)

        with open(self.manifest_path, "r") as f:
            data = json.load(f)

        if "scenes" not in data:
            raise KeyError("manifest JSON must contain a 'scenes' array")

        self.root = self.manifest_path.parent
        self.scenes: List[Dict[str, Any]] = data["scenes"]
        self.split = data.get("split", "lidar")

    def _resolve(self, rel_path: str) -> str:
        return str((self.root / rel_path).resolve())

    def get_scene_by_index(self, index: int) -> Dict[str, Any]:
        """Return resolved paths and metadata for one scene."""
        entry = self.scenes[index]
        scene_id = entry.get("id", str(index))
        image_path = self._resolve(entry["image"])
        pointcloud_path = self._resolve(entry["pointcloud"])
        calib_path = self._resolve(entry["calib"])

        result = {
            "id": scene_id,
            "image_path": image_path,
            "pointcloud_path": pointcloud_path,
            "calib_path": calib_path,
            "file_name": Path(entry["image"]).name,
        }

        if "annotations" in entry:
            result["annotations_path"] = self._resolve(entry["annotations"])

        return result

    def output_dir_name(self, scene: Dict[str, Any]) -> str:
        """Filesystem-safe folder name under save_dir/split/."""
        stem = Path(scene["file_name"]).stem
        return stem.replace("/", "_").replace("-", "_")

    def __len__(self) -> int:
        return len(self.scenes)


def get_lidar_save_split(loader: LidarManifestLoader, override: Optional[str] = None) -> str:
    return override if override else loader.split
