"""Export py123d 3D boxes to COCO-style annotations for read_bounding_boxes_segmentations."""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# DefaultBoxDetectionLabel name -> COCO category_id (subset used by LabelAny3D util)
DEFAULT_LABEL_TO_COCO = {
    "PERSON": 1,
    "VEHICLE": 2,
    "TRAIN": 6,
    "TWO_WHEELER": 3,
    "ANIMAL": 16,
    "TRAFFIC_SIGN": 13,
    "TRAFFIC_CONE": 9,
    "TRAFFIC_LIGHT": 10,
    "BARRIER": 62,
    "GENERIC_OBJECT": 90,
    "OTHER": 90,
    "EGO": 90,
}


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
