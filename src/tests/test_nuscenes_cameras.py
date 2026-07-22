"""Unit tests for nuScenes surround camera helpers."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from integrations.py123d.nuscenes_adapter import (
    NUSCENES_SURROUND_CAMERAS,
    discover_camera_view_dirs,
    get_primary_camera_view,
    resolve_camera_keys,
    scene_camera_output_dir,
    write_cameras_manifest,
)
from batch_scripts.py123d_loader import Py123dOutputLoader


def test_resolve_camera_keys_all():
    keys = resolve_camera_keys(camera_keys="all")
    assert keys == NUSCENES_SURROUND_CAMERAS


def test_resolve_camera_keys_single():
    keys = resolve_camera_keys(camera_key="CAM_BACK")
    assert keys == ["CAM_BACK"]


def test_scene_camera_output_dir_multi():
    root = Path("/tmp/scene")
    assert scene_camera_output_dir(root, "CAM_FRONT", True) == root / "CAM_FRONT"
    assert scene_camera_output_dir(root, "CAM_FRONT", False) == root


def _write_camera_view(scene_root: Path, camera_key: str) -> None:
    view_dir = scene_root / camera_key
    view_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((32, 48, 3), dtype=np.uint8)).save(view_dir / "input.png")
    with open(view_dir / "nuscenes_annotations.json", "w") as f:
        json.dump([], f)


def test_discover_camera_view_dirs_multi(tmp_path):
    scene_root = tmp_path / "scene_a"
    write_cameras_manifest(scene_root, NUSCENES_SURROUND_CAMERAS[:2])
    _write_camera_view(scene_root, "CAM_FRONT_LEFT")
    _write_camera_view(scene_root, "CAM_FRONT")

    views = discover_camera_view_dirs(scene_root)
    assert [p.name for p in views] == ["CAM_FRONT_LEFT", "CAM_FRONT"]
    assert get_primary_camera_view(scene_root).name == "CAM_FRONT"


def test_discover_camera_view_dirs_legacy(tmp_path):
    scene_root = tmp_path / "scene_b"
    scene_root.mkdir()
    Image.fromarray(np.zeros((32, 48, 3), dtype=np.uint8)).save(scene_root / "input.png")
    views = discover_camera_view_dirs(scene_root)
    assert views == [scene_root]


def test_py123d_output_loader_multi_camera(tmp_path):
    split_root = tmp_path / "nuscenes_val"
    scene_root = split_root / "scene_a"
    write_cameras_manifest(scene_root, NUSCENES_SURROUND_CAMERAS[:3])
    for cam in NUSCENES_SURROUND_CAMERAS[:3]:
        _write_camera_view(scene_root, cam)

    loader = Py123dOutputLoader(str(tmp_path), "nuscenes_val")
    assert len(loader) == 3
    entry = loader.get_scene_by_index(0)
    assert entry["scene_id"] == "scene_a"
    assert entry["camera_key"] == "CAM_FRONT_LEFT"
    assert Path(entry["scene_dir"]).name == "CAM_FRONT_LEFT"


def test_compose_surround_spatial_layout():
    import importlib.util

    viz_path = Path(__file__).resolve().parents[1] / "tools" / "visualize_scene.py"
    spec = importlib.util.spec_from_file_location("visualize_scene", viz_path)
    viz = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(viz)
    SURROUND_SPATIAL_SLOTS = viz.SURROUND_SPATIAL_SLOTS
    _compose_surround_spatial_grid = viz._compose_surround_spatial_grid

    panels = {}
    for cam in NUSCENES_SURROUND_CAMERAS:
        color = {
            "CAM_FRONT": (0, 0, 255),
            "CAM_FRONT_LEFT": (0, 255, 0),
            "CAM_FRONT_RIGHT": (255, 0, 0),
            "CAM_BACK": (255, 255, 0),
            "CAM_BACK_LEFT": (255, 0, 255),
            "CAM_BACK_RIGHT": (0, 255, 255),
        }[cam]
        panels[cam] = np.full((90, 120, 3), color, dtype=np.uint8)

    bev = np.full((180, 180, 3), 128, dtype=np.uint8)
    grid = _compose_surround_spatial_grid(panels, bev_panel=bev, target_h=90, bev_height_scale=1.0)
    assert grid is not None
    assert grid.shape[0] == 90 * len(SURROUND_SPATIAL_SLOTS)

    cell_w = grid.shape[1] // 3
    front_px = grid[45, cell_w + cell_w // 2]
    bev_px = grid[90 + 45, cell_w + cell_w // 2]
    left_px = grid[45, cell_w // 2]
    assert tuple(front_px) == (0, 0, 255)
    assert tuple(bev_px) == (128, 128, 128)
    assert tuple(left_px) == (0, 255, 0)

    # Taller BEV middle row for readability in surround summaries.
    tall = _compose_surround_spatial_grid(panels, bev_panel=bev, target_h=90, bev_height_scale=2.0)
    assert tall is not None
    assert tall.shape[0] == 90 + 180 + 90
    tall_cell_w = tall.shape[1] // 3
    assert tuple(tall[90 + 90, tall_cell_w + tall_cell_w // 2]) == (128, 128, 128)
