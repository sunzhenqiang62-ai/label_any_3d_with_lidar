"""
Load nuScenes samples from py123d Arrow logs for LabelAny3D depth / crops pipeline.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from integrations.py123d.annotations import boxes_to_coco_annotations
from integrations.py123d.coord import (
    apply_extra_rotation,
    build_calib_dict,
    camera_global_to_world_to_cam,
    pinhole_intrinsics_to_K,
)


def _require_py123d():
    try:
        import py123d  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "py123d is required for --depth_source py123d. "
            "Install with: pip install -r requirements-py123d.txt"
        ) from e


def _resolve_camera_id(camera_key: str):
    from py123d.datatypes.sensors import CameraID

    key = camera_key.strip()
    try:
        return CameraID.from_arbitrary(key)
    except Exception:
        pass
    aliases = {
        "CAM_FRONT": CameraID.PCAM_F0,
        "cam_front": CameraID.PCAM_F0,
        "front": CameraID.PCAM_F0,
    }
    if key in aliases:
        return aliases[key]
    raise ValueError(
        f"Unknown camera_key '{camera_key}'. "
        f"Try PCAM_F0, CAM_FRONT, or one of: {[c.name for c in CameraID]}"
    )


def _resolve_lidar_id(lidar_key: str):
    from py123d.datatypes.sensors import LidarID

    key = lidar_key.strip().lower()
    if key in ("merged", "lidar_merged", "merge"):
        return LidarID.LIDAR_MERGED
    return LidarID.from_arbitrary(lidar_key)


def _scene_iteration(scene_api, frame_index: Optional[int]) -> int:
    meta = scene_api.get_scene_metadata()
    if frame_index is not None:
        it = meta.num_history_iterations + frame_index
        if it < 0 or it >= meta.total_iterations:
            raise IndexError(
                f"frame_index {frame_index} out of range for scene "
                f"(total_iterations={meta.total_iterations})"
            )
        return it
    return meta.num_history_iterations


def extract_frame_sample(
    scene_api,
    camera_key: str = "CAM_FRONT",
    lidar_key: str = "merged",
    frame_index: Optional[int] = None,
    extra_rot_4x4: Optional[np.ndarray] = None,
    category_map: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Extract one sync frame: RGB, global LiDAR points, calib, COCO annotations.

    Returns dict with keys: image_np, points_world, calib, annotations, scene_id, iteration
    """
    _require_py123d()
    from py123d.datatypes.sensors import PinholeCameraMetadata

    iteration = _scene_iteration(scene_api, frame_index)
    cam_id = _resolve_camera_id(camera_key)
    lidar_id = _resolve_lidar_id(lidar_key)

    camera = scene_api.get_camera_at_iteration(iteration, cam_id)
    if camera is None:
        raise RuntimeError(f"No camera {cam_id} at iteration {iteration}")

    lidar = scene_api.get_lidar_at_iteration(iteration, lidar_id)
    if lidar is None:
        raise RuntimeError(f"No lidar {lidar_id} at iteration {iteration}")

    meta = scene_api.get_scene_metadata()
    scene_id = meta.initial_uuid.replace("/", "_").replace("-", "_")

    image = camera.image
    if image.ndim == 2:
        image_np = np.stack([image, image, image], axis=-1)
    elif image.shape[-1] == 4:
        image_np = image[..., :3]
    else:
        image_np = image[..., :3] if image.shape[-1] > 3 else image

    if not isinstance(camera.metadata, PinholeCameraMetadata):
        raise TypeError(
            f"Camera {cam_id} is not pinhole (model={camera.metadata.camera_model}); "
            "only PinholeCameraMetadata is supported for K export."
        )

    K = pinhole_intrinsics_to_K(camera.metadata)
    world_to_cam = camera_global_to_world_to_cam(camera.camera_to_global_se3)
    world_to_cam = apply_extra_rotation(world_to_cam, extra_rot_4x4)

    calib = build_calib_dict(
        K, camera.metadata.width, camera.metadata.height, world_to_cam
    )

    points_world = np.array(lidar.point_cloud_3d, dtype=np.float64)

    boxes = scene_api.get_box_detections_se3_at_iteration(iteration)
    annotations = boxes_to_coco_annotations(
        boxes, camera, category_map=category_map
    )

    return {
        "image_np": image_np.astype(np.uint8),
        "points_world": points_world,
        "calib": calib,
        "annotations": annotations,
        "scene_id": scene_id,
        "iteration": iteration,
        "file_name": f"{scene_id}.jpg",
    }


class Py123dNuScenesLoader:
    """Enumerate py123d scenes for nuScenes and extract per-scene samples."""

    def __init__(
        self,
        data_root: Optional[str] = None,
        split_type: str = "val",
        dataset_name: str = "nuscenes",
        max_scenes: Optional[int] = None,
        camera_key: str = "CAM_FRONT",
        lidar_key: str = "merged",
        frame_index: Optional[int] = None,
        extra_rot_4x4: Optional[np.ndarray] = None,
    ):
        _require_py123d()
        from py123d.api import SceneFilter, get_filtered_scenes

        if data_root is None:
            data_root = os.environ.get("PY123D_DATA_ROOT")
        if not data_root:
            raise ValueError(
                "Set PY123D_DATA_ROOT or pass --py123d_data_root to converted Arrow logs."
            )

        self.data_root = Path(data_root)
        self.split_type = split_type
        self.split = f"{dataset_name}_{split_type}"
        self.camera_key = camera_key
        self.lidar_key = lidar_key
        self.frame_index = frame_index
        self.extra_rot_4x4 = extra_rot_4x4

        scene_filter = SceneFilter(
            datasets=[dataset_name],
            split_types=[split_type],
            max_num_scenes=max_scenes,
        )
        self.scenes = get_filtered_scenes(scene_filter, data_root=self.data_root)
        if len(self.scenes) == 0:
            raise RuntimeError(
                f"No py123d scenes found under {self.data_root}/logs for "
                f"dataset={dataset_name} split={split_type}"
            )

    def __len__(self) -> int:
        return len(self.scenes)

    def get_scene_by_index(self, index: int):
        return self.scenes[index]

    def extract_sample(self, index: int) -> Dict[str, Any]:
        return extract_frame_sample(
            self.scenes[index],
            camera_key=self.camera_key,
            lidar_key=self.lidar_key,
            frame_index=self.frame_index,
            extra_rot_4x4=self.extra_rot_4x4,
        )

    def output_dir_name(self, sample: Dict[str, Any]) -> str:
        return sample["scene_id"]
