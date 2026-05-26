"""
Optional integration test against real py123d nuScenes data.

Skipped when PY123D_DATA_ROOT is unset or py123d is not installed.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


pytestmark = pytest.mark.slow


def _has_py123d_data():
    root = os.environ.get("PY123D_DATA_ROOT")
    if not root:
        return False
    logs = Path(root) / "logs"
    return logs.exists()


@pytest.mark.skipif(not _has_py123d_data(), reason="PY123D_DATA_ROOT/logs not available")
def test_py123d_loader_one_scene():
    pytest.importorskip("py123d")
    from integrations.py123d.nuscenes_adapter import Py123dNuScenesLoader

    loader = Py123dNuScenesLoader(max_scenes=1)
    sample = loader.extract_sample(0)
    assert sample["image_np"].ndim == 3
    assert sample["points_world"].shape[1] == 3
    assert sample["calib"]["K"].shape == (3, 3)
    assert isinstance(sample["annotations"], list)
