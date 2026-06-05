"""Run LocateAnything detection and convert boxes to instance masks."""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from integrations.locateanything.parser import parse_boxes_fallback, parse_prediction

_worker = None


def release_worker():
    """Unload LocateAnything model to free GPU memory before other models."""
    global _worker
    _worker = None
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _resolve_worker(model_path: str, device: str):
    global _worker
    if _worker is not None:
        return _worker

    # Prefer NVlabs/Eagle install if present
    eagle_embodied = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../external/Eagle/Embodied")
    )
    if os.path.isdir(eagle_embodied) and eagle_embodied not in sys.path:
        sys.path.insert(0, eagle_embodied)

    try:
        from locateanything_worker import LocateAnythingWorker
    except ImportError:
        from integrations.locateanything.locateanything_worker import LocateAnythingWorker

    _worker = LocateAnythingWorker(model_path, device=device)
    return _worker


def _normalize_label(label: str) -> str:
    return label.strip().lower().replace(" ", "_").replace("-", "_") or "object"


def _box_to_mask(box_xyxy: List[float], height: int, width: int) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    mask = np.zeros((height, width), dtype=bool)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = True
    return mask


def _iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms_boxes(
    boxes: List[List[float]],
    labels: List[str],
    iou_threshold: float = 0.5,
) -> Tuple[List[List[float]], List[str]]:
    if not boxes:
        return [], []
    order = sorted(
        range(len(boxes)),
        key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]),
        reverse=True,
    )
    keep_boxes, keep_labels = [], []
    while order:
        i = order.pop(0)
        keep_boxes.append(boxes[i])
        keep_labels.append(labels[i])
        order = [
            j for j in order
            if _iou(boxes[i], boxes[j]) < iou_threshold or labels[i] != labels[j]
        ]
    return keep_boxes, keep_labels


def detect_boxes(
    image_pil: Image.Image,
    categories: Optional[List[str]] = None,
    device: str = "cuda",
    model_path: Optional[str] = None,
    generation_mode: str = "hybrid",
    allowed_categories: Optional[List[str]] = None,
    min_box_area: int = 1600,
    nms_iou: float = 0.5,
) -> Tuple[List[str], List[List[float]]]:
    """LocateAnything 2D boxes only (no full-image masks)."""
    _, labels, boxes = detect_instances(
        image_pil,
        categories=categories,
        device=device,
        model_path=model_path,
        generation_mode=generation_mode,
        allowed_categories=allowed_categories,
        min_mask_area=min_box_area,
        nms_iou=nms_iou,
        return_masks=False,
    )
    return labels, boxes


def detect_instances(
    image_pil: Image.Image,
    categories: Optional[List[str]] = None,
    device: str = "cuda",
    model_path: Optional[str] = None,
    generation_mode: str = "hybrid",
    allowed_categories: Optional[List[str]] = None,
    min_mask_area: int = 1600,
    nms_iou: float = 0.5,
    return_masks: bool = True,
) -> Tuple[np.ndarray, List[str], List[List[float]]]:
    """
    Run LocateAnything object detection.

    Returns:
        masks: (N, H, W) bool array
        labels: length-N category strings
        boxes_xyxy: length-N pixel boxes
    """
    if model_path is None:
        model_path = os.environ.get("LOCATEANYTHING_MODEL", "nvidia/LocateAnything-3B")

    max_edge = int(os.environ.get("LOCATEANYTHING_MAX_EDGE", "1920"))
    scale = 1.0
    infer_image = image_pil
    w, h = image_pil.size
    if max(w, h) > max_edge:
        scale = max_edge / max(w, h)
        infer_image = image_pil.resize(
            (int(w * scale), int(h * scale)), Image.Resampling.BILINEAR
        )

    if categories is None:
        categories = [
            "person", "car", "truck", "bus", "motorcycle", "bicycle",
            "traffic_light", "traffic_sign",
        ]

    worker = _resolve_worker(model_path, device)
    result = worker.detect(
        infer_image.convert("RGB"),
        categories,
        generation_mode=generation_mode,
        verbose=False,
    )
    answer = result.get("answer", "")
    iw, ih = infer_image.size

    parsed = parse_prediction(answer, iw, ih)
    boxes_xyxy: List[List[float]] = []
    labels: List[str] = []

    allowed = None
    if allowed_categories:
        allowed = {_normalize_label(c) for c in allowed_categories}

    for category, box_list in parsed.items():
        label = _normalize_label(category)
        if allowed is not None and label not in allowed:
            continue
        for box in box_list:
            boxes_xyxy.append(box)
            labels.append(label)

    if not boxes_xyxy:
        fallback = parse_boxes_fallback(answer, iw, ih)
        default_label = _normalize_label(categories[0]) if len(categories) == 1 else "object"
        for box in fallback:
            label = default_label
            if allowed is not None and label not in allowed:
                continue
            boxes_xyxy.append(box)
            labels.append(label)

    boxes_xyxy, labels = _nms_boxes(boxes_xyxy, labels, iou_threshold=nms_iou)

    masks = []
    kept_boxes = []
    kept_labels = []
    orig_w, orig_h = image_pil.size
    inv_scale = 1.0 / scale if scale != 1.0 else 1.0
    for box, label in zip(boxes_xyxy, labels):
        if inv_scale != 1.0:
            box = [box[0] * inv_scale, box[1] * inv_scale, box[2] * inv_scale, box[3] * inv_scale]
        if return_masks:
            mask = _box_to_mask(box, orig_h, orig_w)
            if mask.sum() < min_mask_area:
                continue
            masks.append(mask)
        else:
            bw, bh = box[2] - box[0], box[3] - box[1]
            if bw * bh < min_mask_area:
                continue
        kept_boxes.append(box)
        kept_labels.append(label)

    if not kept_boxes:
        empty = np.zeros((0, orig_h, orig_w), dtype=bool)
        return empty, [], []

    if not return_masks:
        return np.zeros((0, orig_h, orig_w), dtype=bool), kept_labels, kept_boxes

    if not masks:
        return np.zeros((0, orig_h, orig_w), dtype=bool), [], []

    return np.stack(masks, axis=0), kept_labels, kept_boxes
