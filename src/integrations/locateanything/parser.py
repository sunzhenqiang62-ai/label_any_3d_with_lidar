"""Parse LocateAnything model outputs (<ref>label</ref><box>...</box>)."""

from __future__ import annotations

import re
from typing import Dict, List, Tuple


def parse_bbox_with_labels(text: str) -> List[Tuple[str, List[float]]]:
    """
    Parse <ref>category</ref><box><x1><y1><x2><y2></box> format.
    Returns [(category, [x1, y1, x2, y2]), ...] with coords in [0, 1000].
    """
    results: List[Tuple[str, List[float]]] = []
    ref_pattern = r"<ref>([^<]+)</ref>((?:<box>.*?</box>)+)"
    box_pattern = (
        r"<box>\s*<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*"
        r"<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*"
        r"<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*"
        r"<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*</box>"
    )
    for category, boxes_str in re.findall(ref_pattern, text):
        for match in re.findall(box_pattern, boxes_str):
            try:
                x1, y1, x2, y2 = map(float, match)
                if 0 <= x1 <= 10000 and 0 <= y1 <= 10000 and 0 <= x2 <= 10000 and 0 <= y2 <= 10000:
                    results.append((category.strip(), [x1, y1, x2, y2]))
            except (TypeError, ValueError):
                continue
    return results


def convert_normalized_bbox_to_absolute(nor_bbox: List[float], img_size: Tuple[int, int]) -> List[float]:
    """Convert normalized [0, 1000] xyxy to pixel xyxy."""
    w, h = img_size
    x1, y1, x2, y2 = nor_bbox
    x1 = max(0, min(x1 * w / 1000, w - 1))
    y1 = max(0, min(y1 * h / 1000, h - 1))
    x2 = max(0, min(x2 * w / 1000, w - 1))
    y2 = max(0, min(y2 * h / 1000, h - 1))
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def parse_prediction(text: str, width: int, height: int) -> Dict[str, List[List[float]]]:
    """Group parsed boxes by category name (pixel xyxy)."""
    result: Dict[str, List[List[float]]] = {}
    for category, nor_bbox in parse_bbox_with_labels(text):
        abs_bbox = convert_normalized_bbox_to_absolute(nor_bbox, (width, height))
        result.setdefault(category, []).append(abs_bbox)
    return result


def parse_boxes_fallback(text: str, width: int, height: int) -> List[List[float]]:
    """Parse bare <box> tokens when <ref> labels are missing."""
    boxes = []
    pattern = r"<box>\s*<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*</box>"
    for match in re.findall(pattern, text):
        try:
            nor = list(map(float, match))
            boxes.append(convert_normalized_bbox_to_absolute(nor, (width, height)))
        except (TypeError, ValueError):
            continue
    return boxes
