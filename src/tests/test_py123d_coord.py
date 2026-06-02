"""Tests for py123d coord helpers (no py123d install required)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from integrations.py123d.coord import build_calib_dict, camera_global_to_world_to_cam


class _FakePose:
    def __init__(self, matrix):
        self._m = matrix

    @property
    def transformation_matrix(self):
        return self._m


def test_world_to_cam_inverse():
    c2g = np.eye(4)
    c2g[:3, 3] = [1, 2, 3]
    w2c = camera_global_to_world_to_cam(_FakePose(c2g))
    assert np.allclose(w2c @ c2g, np.eye(4), atol=1e-6)


def test_build_calib_dict():
    K = np.eye(3)
    w2c = np.eye(4)
    calib = build_calib_dict(K, 100, 80, w2c)
    assert calib["width"] == 100
    assert np.allclose(calib["c2w"], np.eye(4))


if __name__ == "__main__":
    test_world_to_cam_inverse()
    test_build_calib_dict()
    print("test_py123d_coord: OK")
