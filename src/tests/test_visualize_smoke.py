"""Smoke tests for visualize_scene depth rendering (no full util stack)."""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.visualize_scene import render_depth, _viz_dir


def test_render_depth_only(tmp_path):
    scene = tmp_path / "scene0"
    scene.mkdir()
    Image.fromarray(np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8)).save(
        scene / "input.png"
    )
    depth = np.ones((64, 96), dtype=np.float32) * 5.0
    depth[0:5, :] = 0
    np.save(scene / "depth_map.npy", depth)

    out_dir = _viz_dir(scene)
    depth_paths = render_depth(scene, out_dir)
    assert len(depth_paths) >= 1
    assert (out_dir / "depth_colormap.png").exists()
    assert (out_dir / "rgb_depth.png").exists()


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        test_render_depth_only(Path(td))
    print("test_visualize_smoke: OK")
