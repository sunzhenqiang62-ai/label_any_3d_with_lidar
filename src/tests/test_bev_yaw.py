"""Unit tests for BEV yaw estimation / 180° disambiguation."""

import numpy as np

from util_3dbox import (
    disambiguate_yaw_180,
    estimate_bev_yaw,
    oriented_box_from_dims_yaw,
    refine_box_yaw_with_bev,
)


def _car_points(length=4.5, width=1.8, height=1.5, yaw=0.4, center=(1.0, 0.5, 12.0), n=800):
    rng = np.random.default_rng(0)
    # Local box samples: X=width, Y=height, Z=length
    local = rng.uniform(
        [-width / 2, -height / 2, -length / 2],
        [width / 2, height / 2, length / 2],
        size=(n, 3),
    )
    from util_3dbox import rotate_y

    world = (rotate_y(-yaw) @ local.T).T + np.asarray(center, dtype=np.float64)
    return world


def test_estimate_bev_yaw_recovers_long_axis():
    true_yaw = 0.55
    pts = _car_points(yaw=true_yaw)
    est = estimate_bev_yaw(pts, upright=None)
    # Long axis is defined mod π
    err = abs(((est - true_yaw + np.pi / 2) % np.pi) - np.pi / 2)
    assert err < 0.08, (est, true_yaw, err)


def test_disambiguate_prefers_away_from_camera():
    center = [2.0, 0.0, 10.0]
    yaw_toward = disambiguate_yaw_180(np.pi, center, upright=None)  # heading -Z initially
    from util_3dbox import rotate_y

    R = rotate_y(-yaw_toward)
    heading = R @ np.array([0.0, 0.0, 1.0])
    assert np.dot(heading[[0, 2]], np.array(center)[[0, 2]]) > 0


def test_refine_keeps_dimensions():
    pts = _car_points(yaw=-0.3)
    center = pts.mean(axis=0)
    dims = [4.5, 1.5, 1.8]
    corners, c2, dims2, R, yaw, src = refine_box_yaw_with_bev(
        center, dims, None, pts, category="car"
    )
    assert len(dims2) == 3
    assert corners.shape == (8, 3)
    assert R.shape == (3, 3)
    assert np.isfinite(yaw)
    assert src in ("bev_depth", "camera_forward", "ego_forward")


def test_filter_rejects_road_contamination():
    from util_3dbox import filter_object_points

    car = _car_points(yaw=0.2, center=(0.0, 0.5, 10.0), n=400)
    # Fake road / far plane contamination
    rng = np.random.default_rng(1)
    road = np.column_stack(
        [
            rng.uniform(-15, 15, 800),
            rng.uniform(1.0, 1.5, 800),
            rng.uniform(5, 35, 800),
        ]
    )
    mixed = np.vstack([car, road])
    filtered = filter_object_points(mixed, center_cam=(0.0, 0.5, 10.0))
    assert filtered is not None
    span = filtered[:, [0, 2]].max(0) - filtered[:, [0, 2]].min(0)
    assert span.max() < 8.0, span


def test_oriented_box_center_matches():
    center = np.array([1.0, 0.2, 8.0])
    dims = [4.0, 1.4, 1.7]
    corners, c_out, _, _ = oriented_box_from_dims_yaw(center, dims, yaw=0.2)
    assert np.allclose(c_out, center)
    assert np.allclose(corners.mean(axis=0), center, atol=1e-5)
