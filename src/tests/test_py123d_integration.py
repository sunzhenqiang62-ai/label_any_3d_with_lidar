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


def _make_loader():
    """Build a loader for whichever nuScenes dataset exists in PY123D logs."""
    from integrations.py123d.nuscenes_adapter import Py123dNuScenesLoader

    # Optional explicit override for CI/local setups.
    preferred = os.environ.get("PY123D_DATASET")
    candidates = [preferred] if preferred else ["nuscenes", "nuscenes-mini"]

    last_error = None
    for dataset_name in candidates:
        try:
            return Py123dNuScenesLoader(dataset_name=dataset_name, max_scenes=1)
        except RuntimeError as exc:
            last_error = exc
            continue

    raise RuntimeError(
        f"No py123d scenes found for datasets={candidates}. "
        f"Set PY123D_DATASET to match your converted logs. Last error: {last_error}"
    )


@pytest.mark.skipif(not _has_py123d_data(), reason="PY123D_DATA_ROOT/logs not available")
def test_py123d_loader_one_scene():
    pytest.importorskip("py123d")

    loader = _make_loader()
    sample = loader.extract_sample(0)
    assert sample["image_np"].ndim == 3
    assert sample["points_world"].shape[1] == 3
    assert sample["calib"]["K"].shape == (3, 3)
    assert isinstance(sample["annotations"], list)
