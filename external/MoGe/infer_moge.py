"""
LabelAny3D MoGe inference wrapper.

Defaults to MoGe-2 (metric scale). MoGe-3 code/weights are not released yet
(microsoft/MoGe, 2026-07-21: "coming soon"); set MOGE_VERSION=v3 when available.

Env overrides:
  MOGE_VERSION     v1 | v2 | v3   (default: v2)
  MOGE_PRETRAINED  HF repo id or local checkpoint path
  MOGE_DEVICE      cuda / cuda:0 / cpu
  MOGE_FP16        1 to run half precision
"""
from __future__ import annotations

import math
import os
import sys
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

_ROOT = os.path.abspath(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from moge.model import import_model_class_by_version

DEFAULT_PRETRAINED = {
    "v1": "Ruicheng/moge-vitl",
    "v2": "Ruicheng/moge-2-vitl-normal",
    # Official IDs TBD when MoGe-3 ships; override via MOGE_PRETRAINED.
    "v3": os.environ.get("MOGE3_PRETRAINED", "Ruicheng/moge-3-vitl"),
}

_model = None
_model_meta = None


def _resolve_version(version: Optional[str] = None) -> str:
    ver = (version or os.environ.get("MOGE_VERSION", "v2")).lower().strip()
    if ver in ("moge3", "moge-3", "3"):
        ver = "v3"
    elif ver in ("moge2", "moge-2", "2"):
        ver = "v2"
    elif ver in ("moge1", "moge-1", "1"):
        ver = "v1"
    if ver not in ("v1", "v2", "v3"):
        raise ValueError(f"Unsupported MoGe version: {version!r}")
    return ver


def _focal_to_fov_x_deg(f_px: float, width: int) -> float:
    return float(2.0 * math.degrees(math.atan(0.5 * float(width) / float(f_px))))


def load_moge_model(version: Optional[str] = None, pretrained: Optional[str] = None):
    """Lazy-load MoGe on the configured device."""
    global _model, _model_meta
    ver = _resolve_version(version)
    pretrained = (
        pretrained
        or os.environ.get("MOGE_PRETRAINED")
        or DEFAULT_PRETRAINED[ver]
    )
    device_name = os.environ.get("MOGE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = os.environ.get("MOGE_FP16", "0") in ("1", "true", "True")
    meta = (ver, pretrained, device_name, use_fp16)
    if _model is not None and _model_meta == meta:
        return _model, ver, device_name

    if ver == "v3":
        try:
            cls = import_model_class_by_version("v3")
        except Exception as e:
            raise RuntimeError(
                "MoGe-3 is not available yet (code/weights marked 'coming soon' "
                "on microsoft/MoGe). Use MOGE_VERSION=v2 (MoGe-2 metric) for now, "
                "or set MOGE_PRETRAINED to a local MoGe-3 checkpoint once released."
            ) from e
    else:
        cls = import_model_class_by_version(ver)

    device = torch.device(device_name)
    print(f"Loading MoGe {ver} from {pretrained} on {device}...")
    model = cls.from_pretrained(pretrained).to(device).eval()
    if use_fp16:
        model.half()
    _model = model
    _model_meta = meta
    print(f"MoGe {ver} loaded.")
    return _model, ver, device_name


def infer_geometry_on_image(
    image_path,
    out_dir=None,
    *,
    version: Optional[str] = None,
    pretrained: Optional[str] = None,
    f_px: Optional[float] = None,
    fov_x: Optional[float] = None,
    resolution_level: int = 9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run MoGe and return (points, depth, mask, intrinsics_pixel).

    For MoGe-2/3 the depth/points are metric-scale. Optional ``f_px`` / ``fov_x``
    constrains camera FOV when known (e.g. LiDAR fuse).
    """
    model, ver, _ = load_moge_model(version=version, pretrained=pretrained)
    device = next(model.parameters()).device

    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    input_image = torch.tensor(
        image / 255.0, dtype=torch.float32, device=device
    ).permute(2, 0, 1)

    if fov_x is None and f_px is not None and f_px > 0:
        fov_x = _focal_to_fov_x_deg(float(f_px), w)

    infer_kwargs = {"resolution_level": int(resolution_level)}
    if fov_x is not None:
        infer_kwargs["fov_x"] = float(fov_x)

    with torch.inference_mode():
        output = model.infer(input_image, **infer_kwargs)

    points = output["points"].float().cpu().numpy()
    depth = output["depth"].float().cpu().numpy()
    mask = output["mask"].cpu().numpy().astype(bool)
    intrinsics = output["intrinsics"].float().cpu().numpy()
    # Normalized intrinsics -> pixel units
    intrinsics = intrinsics * np.array([[w, 1, w], [1, h, h], [1, 1, 1]], dtype=np.float64)
    return points, depth.astype(np.float32), mask, intrinsics.astype(np.float64)
