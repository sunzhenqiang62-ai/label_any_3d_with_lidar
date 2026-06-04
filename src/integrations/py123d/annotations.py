"""Export py123d 3D boxes to COCO-style annotations for read_bounding_boxes_segmentations."""

from typing import Dict, List, Optional

import cv2
import numpy as np

from integrations.py123d.coord import camera_global_to_world_to_cam

# DefaultBoxDetectionLabel name -> COCO category_id (subset used by LabelAny3D util)
DEFAULT_LABEL_TO_COCO = {
    "PERSON": 1,
    "VEHICLE": 3,
    "TRAIN": 6,
    "TWO_WHEELER": 4,
    "ANIMAL": 16,
    "TRAFFIC_SIGN": 13,
    "TRAFFIC_CONE": 9,
    "TRAFFIC_LIGHT": 10,
    "BARRIER": 62,
    "GENERIC_OBJECT": 90,
    "OTHER": 90,
    "EGO": 90,
}


DEFAULT_LABEL_TO_NAME = {
    "PERSON": "person",
    "VEHICLE": "car",
    "TRAIN": "train",
    "TWO_WHEELER": "motorcycle",
    "ANIMAL": "animal",
    "TRAFFIC_SIGN": "traffic_sign",
    "TRAFFIC_CONE": "traffic_cone",
    "TRAFFIC_LIGHT": "traffic_light",
    "BARRIER": "barrier",
    "GENERIC_OBJECT": "object",
    "OTHER": "object",
    "EGO": "ego",
}


def _label_to_category_name(label) -> str:
    try:
        default = label.to_default()
        name = default.name if hasattr(default, "name") else str(default)
    except Exception:
        name = str(label)
    return DEFAULT_LABEL_TO_NAME.get(name.upper(), name.lower())


def _label_to_category_id(label) -> int:
    try:
        default = label.to_default()
        name = default.name if hasattr(default, "name") else str(default)
    except Exception:
        name = str(label)
    return DEFAULT_LABEL_TO_COCO.get(name.upper(), 90)


def _convex_hull_polygon(points_uv: np.ndarray) -> List[List[float]]:
    if points_uv.shape[0] < 3:
        return []
    hull = cv2.convexHull(points_uv.astype(np.float32))
    poly = hull.reshape(-1, 2).tolist()
    return [poly]


def boxes_to_coco_annotations(
    box_detections,
    camera,
    category_map: Optional[Dict[str, int]] = None,
    min_visible_corners: int = 4,
) -> List[dict]:
    """
    Project py123d BoxDetectionSE3 list to COCO instance annotations.

    Args:
        box_detections: py123d BoxDetectionsSE3 or list of BoxDetectionSE3
        camera: py123d Camera with project_points_global
        category_map: optional override for label -> category_id
        min_visible_corners: min in-FOV corners to keep instance
    """
    if box_detections is None:
        return []

    detections = (
        box_detections.box_detections
        if hasattr(box_detections, "box_detections")
        else box_detections
    )
    W = camera.metadata.width
    H = camera.metadata.height
    anns = []

    for det in detections:
        corners = np.array(det.bounding_box_se3.corners_array, dtype=np.float64)
        pixel_coords, in_fov, _depth = camera.project_points_global(corners)
        if in_fov.sum() < min_visible_corners:
            continue

        uv = pixel_coords[in_fov]
        x_min, y_min = uv.min(axis=0)
        x_max, y_max = uv.max(axis=0)
        bw = float(x_max - x_min)
        bh = float(y_max - y_min)
        if bw < 2 or bh < 2:
            continue

        poly = _convex_hull_polygon(pixel_coords[in_fov])
        if len(poly) == 0:
            continue

        label = det.attributes.label
        if category_map is not None:
            try:
                cat_name = label.to_default().name
            except Exception:
                cat_name = str(label)
            cat_id = category_map.get(cat_name, category_map.get(cat_name.upper(), 90))
        else:
            cat_id = _label_to_category_id(label)

        anns.append(
            {
                "bbox": [float(x_min), float(y_min), bw, bh],
                "segmentation": poly,
                "category_id": int(cat_id),
                "iscrowd": 0,
            }
        )

    return anns


def _global_points_to_cam(points_global: np.ndarray, world_to_cam: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_global, dtype=np.float64)
    if pts.size == 0:
        return pts.reshape(0, 3)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    hom = np.hstack([pts, ones])
    return (world_to_cam @ hom.T).T[:, :3]


def boxes_to_gt_3dbbox_cam(
    box_detections,
    camera,
    world_to_cam: Optional[np.ndarray] = None,
    min_visible_corners: int = 4,
) -> List[dict]:
    """Export dataset GT 3D boxes in OpenCV camera frame (for BEV overlays)."""
    if box_detections is None:
        return []

    detections = (
        box_detections.box_detections
        if hasattr(box_detections, "box_detections")
        else box_detections
    )
    if world_to_cam is None:
        world_to_cam = camera_global_to_world_to_cam(camera.camera_to_global_se3)

    gt_boxes: List[dict] = []
    for idx, det in enumerate(detections):
        corners_global = np.array(det.bounding_box_se3.corners_array, dtype=np.float64)
        _pixel_coords, in_fov, _depth = camera.project_points_global(corners_global)
        if in_fov.sum() < min_visible_corners:
            continue

        corners_cam = _global_points_to_cam(corners_global, world_to_cam)
        if not np.isfinite(corners_cam).all() or np.min(corners_cam[:, 2]) <= 1e-6:
            continue

        label = det.attributes.label
        track = getattr(det.attributes, "track_token", None)
        gt_boxes.append(
            {
                "obj_id": str(track) if track else str(idx),
                "category_name": _label_to_category_name(label),
                "category_id": int(_label_to_category_id(label)),
                "center_cam": corners_cam.mean(axis=0).tolist(),
                "bbox3D_cam": corners_cam.tolist(),
                "source": "gt",
            }
        )
    return gt_boxes
