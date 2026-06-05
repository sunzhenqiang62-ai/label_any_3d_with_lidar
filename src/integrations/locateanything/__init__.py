"""LocateAnything (NVlabs/Eagle) open-vocabulary detection for LabelAny3D crops."""

from integrations.locateanything.detect import detect_boxes, detect_instances
from integrations.locateanything.parser import parse_prediction

__all__ = ["detect_boxes", "detect_instances", "parse_prediction"]
