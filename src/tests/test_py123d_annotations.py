"""Tests for COCO annotation export from synthetic boxes (no py123d)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _FakeLabel:
    def to_default(self):
        return self

    @property
    def name(self):
        return "PERSON"


class _FakeAttrs:
    label = _FakeLabel()
    track_token = "t1"


class _FakeBBox3d:
    def __init__(self):
        # unit cube centered at origin in global frame
        # Cube in front of camera (camera looks along +Z in this stub)
        z0 = 5.0
        self._corners = np.array(
            [
                [-0.5, -0.5, z0 - 0.5],
                [0.5, -0.5, z0 - 0.5],
                [0.5, 0.5, z0 - 0.5],
                [-0.5, 0.5, z0 - 0.5],
                [-0.5, -0.5, z0 + 0.5],
                [0.5, -0.5, z0 + 0.5],
                [0.5, 0.5, z0 + 0.5],
                [-0.5, 0.5, z0 + 0.5],
            ],
            dtype=np.float64,
        )

    @property
    def corners_array(self):
        return self._corners


class _FakeDet:
    attributes = _FakeAttrs()
    bounding_box_se3 = _FakeBBox3d()


class _FakePose:
    def __init__(self, m):
        self._m = m

    @property
    def transformation_matrix(self):
        return self._m


class _FakePinholeMeta:
    width = 200
    height = 200

    def project_to_image(self, points_cam):
        fx, fy, cx, cy = 100, 100, 100, 100
        z = points_cam[:, 2]
        u = fx * points_cam[:, 0] / z + cx
        v = fy * points_cam[:, 1] / z + cy
        pix = np.column_stack([u, v])
        mask = (z > 0) & (u >= 0) & (u < 200) & (v >= 0) & (v < 200)
        return pix, mask, z


class _FakeCamera:
    metadata = _FakePinholeMeta()
    camera_to_global_se3 = _FakePose(np.eye(4))

    def project_points_global(self, points_global):
        return self.metadata.project_to_image(points_global)


def test_boxes_to_coco():
    from integrations.py123d.annotations import boxes_to_coco_annotations

    cam = _FakeCamera()
    anns = boxes_to_coco_annotations([_FakeDet()], cam)
    assert len(anns) == 1
    assert "bbox" in anns[0]
    assert "segmentation" in anns[0]
    assert anns[0]["category_id"] == 1


if __name__ == "__main__":
    test_boxes_to_coco()
    print("test_py123d_annotations: OK")
