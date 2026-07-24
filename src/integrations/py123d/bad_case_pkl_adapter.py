"""
Adapter: data-pipeline-4d bad_case / loopify pickle → LabelAny3D sample dicts
compatible with py123d fuse depth (`run_fuse_py123d_depth`).

Pickle schema (see data-pipeline-4d/tools/generate_bad_case_pkl):
  {
    "metadata": {"calibration": {clip_tag: {lidar2ego_*, cams: {cam*: {...}}}}, ...},
    "infos": [{token, timestamp, lidar_path, cams:{cam*:{data_path}}, ann_infos, ...}, ...]
  }

ann_infos is typically [bbox_list, category_list] (may be empty for bad-case dumps).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from integrations.py123d.coord import build_calib_dict

# Map loopify cam ids → nuScenes-style keys used by surround summary layout.
DEFAULT_CAM_KEY_MAP = {
    "cam1": "CAM_FRONT",        # FW wide
    "cam4": "CAM_FRONT_LEFT",   # FL
    "cam8": "CAM_FRONT_RIGHT",  # FR
    "cam5": "CAM_BACK_LEFT",    # RL
    "cam6": "CAM_BACK",         # RN
    "cam7": "CAM_BACK_RIGHT",   # RR
    "cam0": "CAM_FRONT_NARROW", # FN
}

# Surround ring used by summary layout (excludes narrow).
DEFAULT_SURROUND_CAMS = [
    "cam4",
    "cam1",
    "cam8",
    "cam5",
    "cam6",
    "cam7",
]

# All calibrated cameras in the bad_case / loopify pickle.
DEFAULT_ALL_CAMS = [
    "cam0",
    "cam4",
    "cam1",
    "cam8",
    "cam5",
    "cam6",
    "cam7",
]


def quat_wxyz_to_rotation(q: Sequence[float]) -> np.ndarray:
    """Quaternion [w, x, y, z] → 3x3 rotation (pyquaternion / nuScenes convention)."""
    w, x, y, z = [float(v) for v in q]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_from_translation_quaternion(
    translation: Sequence[float], rotation_wxyz: Sequence[float]
) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = quat_wxyz_to_rotation(rotation_wxyz)
    mat[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return mat


def load_pcd_xyz(path: Union[str, Path]) -> np.ndarray:
    """Load Nx3 points from ASCII/binary PCD (open3d)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        import open3d as o3d
    except ImportError as e:
        raise ImportError(
            "open3d is required to read .pcd for bad_case pickle scenes"
        ) from e
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Unexpected point cloud shape from {path}: {pts.shape}")
    finite = np.isfinite(pts).all(axis=1)
    return pts[finite]


def _resolve_calibration_block(metadata: dict, info: dict) -> dict:
    calib_root = metadata.get("calibration") or {}
    if not isinstance(calib_root, dict) or not calib_root:
        raise KeyError("metadata.calibration missing")
    # Prefer explicit info.calibration key if present
    if isinstance(info.get("calibration"), str) and info["calibration"] in calib_root:
        return calib_root[info["calibration"]]
    # Else match by dataset_id / scene_token prefix
    for key in (info.get("dataset_id"), info.get("scene_token"), info.get("token")):
        if not key:
            continue
        for cal_key, block in calib_root.items():
            if str(key) in str(cal_key) or str(cal_key).startswith(str(key)[:16]):
                return block
    # Fallback: single entry
    if len(calib_root) == 1:
        return next(iter(calib_root.values()))
    raise KeyError(f"Cannot resolve calibration block from keys={list(calib_root.keys())}")


def _ann_infos_to_gt(
    ann_infos: Any,
    world_to_cam: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
) -> Tuple[List[dict], List[dict]]:
    """
    Convert [boxes, labels] (ego/lidar frame 3D boxes) to empty-safe annotations.
    Empty bad-case dumps return ([], []).
    """
    if not ann_infos:
        return [], []
    if isinstance(ann_infos, (list, tuple)) and len(ann_infos) >= 2:
        boxes, labels = ann_infos[0], ann_infos[1]
    else:
        return [], []
    if not boxes:
        return [], []

    # Boxes are typically [x,y,z,dx,dy,dz,yaw,...] in ego frame; project if present.
    annotations: List[dict] = []
    gt_3dbbox: List[dict] = []
    # For smoke / empty dumps we skip heavy box projection; keep hook for later.
    _ = (boxes, labels, world_to_cam, K, width, height)
    return annotations, gt_3dbbox


class BadCasePklLoader:
    """
    One pickle file = one clip / scene with multiple frames (infos).

    ``extract_samples(frame_index)`` returns a list of per-camera sample dicts
    matching ``extract_frame_sample`` output for fuse depth.
    """

    def __init__(
        self,
        pkl_path: Union[str, Path],
        camera_keys: Optional[Sequence[str]] = None,
        cam_key_map: Optional[Dict[str, str]] = None,
        frame_index: Optional[int] = None,
        split: str = "bad_case",
    ):
        self.pkl_path = Path(pkl_path)
        if not self.pkl_path.exists():
            raise FileNotFoundError(self.pkl_path)
        with open(self.pkl_path, "rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict) or "infos" not in payload:
            raise ValueError(f"{self.pkl_path} must be a dict with 'infos'")
        self.metadata = payload.get("metadata") or {}
        self.infos: List[dict] = list(payload["infos"])
        self.cam_key_map = dict(cam_key_map or DEFAULT_CAM_KEY_MAP)
        if camera_keys is None or (
            isinstance(camera_keys, str) and camera_keys.lower() in ("all",)
        ):
            self.raw_camera_keys = list(DEFAULT_ALL_CAMS)
        elif isinstance(camera_keys, str) and camera_keys.lower() in ("surround",):
            self.raw_camera_keys = list(DEFAULT_SURROUND_CAMS)
        elif isinstance(camera_keys, str):
            self.raw_camera_keys = [c.strip() for c in camera_keys.split(",") if c.strip()]
        else:
            self.raw_camera_keys = list(camera_keys)
        self.default_frame_index = frame_index
        self.split = split
        # Prefer dataset_id from first info
        self.scene_id = self._make_scene_id(self.infos[0] if self.infos else {})

    @staticmethod
    def _make_scene_id(info: dict) -> str:
        raw = str(info.get("dataset_id") or info.get("scene_token") or info.get("token") or "bad_case")
        return raw.replace("/", "_").replace("-", "_")

    def __len__(self) -> int:
        return len(self.infos)

    @property
    def camera_keys(self) -> List[str]:
        """Mapped LabelAny3D camera folder names."""
        return [self.cam_key_map.get(c, c.upper()) for c in self.raw_camera_keys]

    def output_dir_name(self, sample_or_info: Optional[dict] = None) -> str:
        if sample_or_info and sample_or_info.get("scene_id"):
            return str(sample_or_info["scene_id"])
        return self.scene_id

    def extract_samples(self, frame_index: int) -> List[dict]:
        if frame_index < 0 or frame_index >= len(self.infos):
            raise IndexError(f"frame_index {frame_index} out of range [0, {len(self.infos)})")
        info = self.infos[frame_index]
        calib_block = _resolve_calibration_block(self.metadata, info)
        lidar2ego = transform_from_translation_quaternion(
            calib_block["lidar2ego_translation"],
            calib_block["lidar2ego_rotation"],
        )
        ego2global = transform_from_translation_quaternion(
            info.get("ego2global_translation", calib_block.get("ego2global_translation", [0, 0, 0])),
            info.get("ego2global_rotation", calib_block.get("ego2global_rotation", [1, 0, 0, 0])),
        )

        points_lidar = load_pcd_xyz(info["lidar_path"])
        ones = np.ones((points_lidar.shape[0], 1), dtype=np.float64)
        pts_h = np.hstack([points_lidar, ones])
        points_ego = (lidar2ego @ pts_h.T).T[:, :3]
        points_world = (ego2global @ np.hstack([points_ego, ones]).T).T[:, :3]

        cam_calibs = calib_block.get("cams") or {}
        samples: List[dict] = []
        scene_id = self._make_scene_id(info)

        for raw_cam in self.raw_camera_keys:
            if raw_cam not in info.get("cams", {}):
                print(f"Warning: frame {frame_index} missing image for {raw_cam}, skip")
                continue
            if raw_cam not in cam_calibs:
                print(f"Warning: calibration missing for {raw_cam}, skip")
                continue
            cam_meta = cam_calibs[raw_cam]
            img_path = info["cams"][raw_cam]["data_path"]
            if not Path(img_path).exists():
                print(f"Warning: image not found {img_path}, skip")
                continue
            image_np = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
            h, w = image_np.shape[:2]
            # Prefer image size; fall back to calib metadata
            width = int(cam_meta.get("width", w))
            height = int(cam_meta.get("height", h))
            if (h, w) != (height, width):
                # Keep actual image size for K usage (assume K matches calib size)
                pass

            sensor2ego = transform_from_translation_quaternion(
                cam_meta["sensor2ego_translation"],
                cam_meta["sensor2ego_rotation"],
            )
            # camera_to_global = ego2global @ sensor2ego
            camera_to_global = ego2global @ sensor2ego
            world_to_cam = np.linalg.inv(camera_to_global)
            K = np.asarray(cam_meta["cam_intrinsic"], dtype=np.float64).reshape(3, 3)
            calib = build_calib_dict(K, width=w, height=h, world_to_cam=world_to_cam)

            annotations, gt_3dbbox = _ann_infos_to_gt(
                info.get("ann_infos"), world_to_cam, K, w, h
            )
            camera_key = self.cam_key_map.get(raw_cam, raw_cam.upper())
            samples.append(
                {
                    "image_np": image_np,
                    "points_world": points_world,
                    "calib": calib,
                    "annotations": annotations,
                    "gt_3dbbox": gt_3dbbox,
                    "scene_id": scene_id,
                    "camera_key": camera_key,
                    "raw_camera_key": raw_cam,
                    "iteration": int(frame_index),
                    "timestamp": info.get("timestamp"),
                    "token": info.get("token"),
                    "file_name": f"{scene_id}_{camera_key}.jpg",
                    "lidar_path": info.get("lidar_path"),
                    "image_path": img_path,
                }
            )
        if not samples:
            raise RuntimeError(f"No valid camera samples for frame {frame_index} in {self.pkl_path}")
        return samples
