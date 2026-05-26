"""Tests for unified pipeline scene loaders."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from batch_scripts.pipeline_loader import (
    resolve_py123d_split,
    setup_pipeline_loop,
)
from batch_scripts.py123d_loader import Py123dOutputLoader


def _make_scene(root: Path, name: str, with_ann: bool = True):
    scene = root / name
    scene.mkdir(parents=True)
    Image.fromarray(np.zeros((32, 48, 3), dtype=np.uint8)).save(scene / "input.png")
    if with_ann:
        with open(scene / "nuscenes_annotations.json", "w") as f:
            json.dump([], f)
    return scene


def test_resolve_py123d_split():
    opt = SimpleNamespace(run={"py123d": {"dataset": "nuscenes", "split_type": "val"}})
    assert resolve_py123d_split("val", opt) == "nuscenes_val"
    assert resolve_py123d_split("nuscenes_val", opt) == "nuscenes_val"


def test_py123d_loader_filters_scenes(tmp_path):
    split_dir = tmp_path / "nuscenes_val_a"
    split_dir.mkdir(parents=True, exist_ok=True)
    _make_scene(split_dir, "scene_a")
    _make_scene(split_dir, "scene_b", with_ann=False)

    loader = Py123dOutputLoader(
        str(tmp_path),
        "nuscenes_val_a",
        require_files=("input.png",),
        require_annotations=True,
    )
    assert len(loader) == 1
    entry = loader.get_scene_by_index(0)
    assert entry["id"] == "scene_a"
    assert (Path(entry["scene_dir"]) / "input.png").exists()


def test_setup_pipeline_loop_py123d(tmp_path):
    split_dir = tmp_path / "nuscenes_val"
    split_dir.mkdir(parents=True, exist_ok=True)
    _make_scene(split_dir, "s0")

    opt = SimpleNamespace(
        run={
            "data_backend": "py123d",
            "py123d": {"dataset": "nuscenes", "split_type": "val"},
        }
    )

    class Args:
        data_backend = "py123d"
        split = "val"
        save_dir = str(tmp_path)
        start_index = 0
        end_index = -1

    backend, split, loader, indices = setup_pipeline_loop(
        Args(),
        opt,
        require_files=("input.png",),
        require_annotations=False,
    )
    assert backend == "py123d"
    assert split == "nuscenes_val"
    assert len(loader) == 1
    assert list(indices) == [0]


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        test_resolve_py123d_split()
        test_py123d_loader_filters_scenes(root)
        test_setup_pipeline_loop_py123d(root)
    print("test_pipeline_loader: OK")
