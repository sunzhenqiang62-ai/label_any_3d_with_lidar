"""Coordinate helpers between py123d global frame and LabelAny3D OpenCV camera frame."""

from typing import Optional

import numpy as np


def pose_se3_to_matrix(pose) -> np.ndarray:
    """py123d PoseSE3 -> 4x4 homogeneous matrix."""
    return np.array(pose.transformation_matrix, dtype=np.float64)


def camera_global_to_world_to_cam(camera_to_global_se3) -> np.ndarray:
    """Invert camera-to-global pose to world/global-to-camera."""
    c2w = pose_se3_to_matrix(camera_to_global_se3)
    return np.linalg.inv(c2w)


def pinhole_intrinsics_to_K(metadata) -> np.ndarray:
    """Extract 3x3 K from py123d pinhole camera metadata."""
    from py123d.datatypes.sensors import PinholeCameraMetadata

    if not isinstance(metadata, PinholeCameraMetadata):
        raise TypeError(f"Expected PinholeCameraMetadata, got {type(metadata)}")
    if metadata.intrinsics is None:
        raise ValueError("Camera metadata has no pinhole intrinsics")
    return np.array(metadata.intrinsics.camera_matrix, dtype=np.float64)


def build_calib_dict(K: np.ndarray, width: int, height: int, world_to_cam: np.ndarray) -> dict:
    """LabelAny3D calib dict for lidar_depth.build_scene_outputs."""
    c2w = np.linalg.inv(world_to_cam)
    return {
        "K": K,
        "width": int(width),
        "height": int(height),
        "world_to_cam": world_to_cam,
        "c2w": c2w,
    }


def apply_extra_rotation(matrix_4x4: np.ndarray, extra_rot_4x4: Optional[np.ndarray]) -> np.ndarray:
    if extra_rot_4x4 is None:
        return matrix_4x4
    return extra_rot_4x4 @ matrix_4x4
