"""
Synthetic smoke test for LiDAR depth generation (no GPU).
Run from src/: python tests/test_lidar_depth_smoke.py
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geometry.lidar_depth import (
    build_scene_outputs,
    compute_lidar_depth,
    fuse_lidar_with_estimate,
    load_calib,
    load_pointcloud,
    rasterize_depth,
    transform_points,
)


def make_synthetic_scene(tmp: Path):
    H, W = 100, 100
    fx, fy, cx, cy = 80.0, 80.0, 50.0, 50.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    world_to_cam = np.eye(4, dtype=np.float64)
    c2w = np.eye(4, dtype=np.float64)

    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[30:70, 30:70] = [200, 100, 50]
    Image.fromarray(img).save(tmp / "image.png")

    # Plane at Z=5 in camera frame (also world frame with identity extrinsics)
    uu, vv = np.meshgrid(np.arange(35, 65), np.arange(35, 65))
    Z = np.full_like(uu, 5.0, dtype=np.float64)
    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy
    points_world = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    np.savez(tmp / "cloud.npz", points=points_world)

    calib = {
        "K": K.tolist(),
        "width": W,
        "height": H,
        "world_to_cam": world_to_cam.tolist(),
        "c2w": c2w.tolist(),
    }
    with open(tmp / "calib.json", "w") as f:
        json.dump(calib, f)

    manifest = {
        "split": "test",
        "scenes": [
            {
                "id": "syn",
                "image": "image.png",
                "pointcloud": "cloud.npz",
                "calib": "calib.json",
            }
        ],
    }
    with open(tmp / "manifest.json", "w") as f:
        json.dump(manifest, f)

    return img, K, H, W, points_world


def test_rasterize_roundtrip():
    tmp = Path(tempfile.mkdtemp())
    img, K, H, W, points_world = make_synthetic_scene(tmp)
    calib = load_calib(tmp / "calib.json")
    pts_cam = transform_points(points_world, calib["world_to_cam"])
    depth, mask = rasterize_depth(pts_cam, K, H, W)
    assert mask.sum() > 0, "expected projected lidar hits"
    valid_depth = depth[mask]
    assert np.allclose(valid_depth, 5.0, atol=0.05), f"depth should be ~5, got {valid_depth.mean()}"
    print("rasterize_roundtrip: OK")


def test_build_outputs():
    tmp = Path(tempfile.mkdtemp())
    img, K, H, W, points_world = make_synthetic_scene(tmp)
    calib = load_calib(tmp / "calib.json")
    points, _ = load_pointcloud(tmp / "cloud.npz")
    out = tmp / "out"
    build_scene_outputs(out, img, points, calib, depth_fill="nearest")
    assert (out / "depth_map.npy").exists()
    assert (out / "cam_params.json").exists()
    assert (out / "depth_scene.ply").exists()
    depth = np.load(out / "depth_map.npy")
    finite = np.isfinite(depth) & (depth < np.inf)
    assert finite.sum() > 0
    print("build_outputs: OK")


def test_fuse_lidar_estimate():
    tmp = Path(tempfile.mkdtemp())
    img, K, H, W, points_world = make_synthetic_scene(tmp)
    calib = load_calib(tmp / "calib.json")
    computed = compute_lidar_depth(
        img, points_world, calib, depth_fill="none", densify_radius=0, calib_refine=False
    )
    estimate = np.full((H, W), 8.0, dtype=np.float32)
    fused, fused_valid = fuse_lidar_with_estimate(
        computed["depth_map"],
        computed["valid_mask"],
        estimate,
        align=False,
        soft_blend=False,
        banded_align=False,
        semantic_guide=False,
        edge_fill=False,
    )
    assert computed["valid_mask"].sum() > 0
    assert np.allclose(fused[computed["valid_mask"]], 5.0, atol=0.05)
    assert np.allclose(fused[~computed["valid_mask"]], 8.0, atol=1e-3)
    assert fused_valid.all()
    print("fuse_lidar_estimate: OK")


def test_fuse_lidar_estimate_with_align():
    def scale_align(estimate, lidar, mask=None, apply_mask=None):
        fit = np.asarray(mask, dtype=bool)
        apply = (
            np.asarray(apply_mask, dtype=bool)
            if apply_mask is not None
            else fit
        )
        scale = np.median(lidar[fit] / estimate[fit])
        out = estimate.copy()
        out[apply] = estimate[apply] * scale
        return out.astype(np.float32)

    tmp = Path(tempfile.mkdtemp())
    img, K, H, W, points_world = make_synthetic_scene(tmp)
    calib = load_calib(tmp / "calib.json")
    computed = compute_lidar_depth(
        img, points_world, calib, depth_fill="none", densify_radius=0, calib_refine=False
    )
    estimate = np.full((H, W), 2.0, dtype=np.float32)
    fused, fused_valid = fuse_lidar_with_estimate(
        computed["depth_map"],
        computed["valid_mask"],
        estimate,
        align=True,
        align_fn=scale_align,
        soft_blend=False,
        banded_align=False,
        semantic_guide=False,
        edge_fill=False,
    )
    holes = ~computed["valid_mask"]
    assert np.allclose(fused[computed["valid_mask"]], 5.0, atol=0.05)
    assert not np.any(fused[holes] >= 9999.0)
    assert np.allclose(fused[holes], 5.0, atol=0.05)
    assert fused_valid.all()
    print("fuse_lidar_estimate_with_align: OK")


def test_robust_scale_ratio_rejects_outliers():
    from geometry.lidar_depth import robust_scale_ratio

    rng = np.random.default_rng(0)
    est = rng.uniform(5.0, 40.0, size=400).astype(np.float64)
    true_s = 1.7
    lidar = est * true_s
    # Inject gross outliers that would bias a plain median.
    lidar[:40] *= 8.0
    weights = np.ones(400, dtype=np.float64)
    weights[:40] = 0.05
    weights[40:] = 1.0
    s = robust_scale_ratio(lidar, est, weights, ransac_iters=80, inlier_rel_thresh=0.1)
    assert abs(s - true_s) < 0.08, (s, true_s)
    print("robust_scale_ratio: OK")


def test_fuse_optimized_pipeline():
    """Exercise soft blend + banded align + semantic + edge fill on synthetic data."""
    from geometry.lidar_depth import align_depth_banded, lidar_confidence_map
    from scipy.ndimage import binary_erosion

    tmp = Path(tempfile.mkdtemp())
    img, K, H, W, points_world = make_synthetic_scene(tmp)
    calib = load_calib(tmp / "calib.json")
    computed = compute_lidar_depth(
        img,
        points_world,
        calib,
        depth_fill="none",
        raster_mode="median",
        densify_radius=1,
        calib_refine=True,
        calib_max_shift=1,
    )
    estimate = np.full((H, W), 2.5, dtype=np.float32)
    # Contaminate a few overlap pixels so plain median would drift; RANSAC should hold.
    lidar = computed["depth_map"].copy()
    valid = computed["valid_mask"].copy()
    ys, xs = np.where(valid)
    if ys.size > 20:
        lidar[ys[:15], xs[:15]] *= 6.0
    conf = lidar_confidence_map(valid, computed.get("hit_count"))
    fused, fused_valid = fuse_lidar_with_estimate(
        lidar,
        valid,
        estimate,
        align=True,
        image_np=img,
        hit_count=computed.get("hit_count"),
        soft_blend=True,
        banded_align=True,
        semantic_guide=True,
        edge_fill=True,
    )
    core = binary_erosion(valid, iterations=2)
    if core.any():
        assert np.allclose(fused[core], 5.0, atol=0.25), fused[core].mean()
    assert fused_valid.sum() > valid.sum()
    assert conf.max() > 0.5
    banded = align_depth_banded(
        estimate,
        lidar,
        valid,
        apply_mask=np.ones_like(estimate, dtype=bool),
        weights=conf,
    )
    # After robust align, vision depth on LiDAR support should sit near true 5m.
    assert abs(float(np.median(banded[valid])) - 5.0) < 0.35
    print("fuse_optimized_pipeline: OK")


def test_manifest_loader_integration():
    from batch_scripts.lidar_loader import LidarManifestLoader

    tmp = Path(tempfile.mkdtemp())
    make_synthetic_scene(tmp)
    loader = LidarManifestLoader(tmp / "manifest.json")
    scene = loader.get_scene_by_index(0)
    assert Path(scene["image_path"]).exists()
    assert Path(scene["pointcloud_path"]).exists()
    print("manifest_loader: OK")


if __name__ == "__main__":
    test_rasterize_roundtrip()
    test_build_outputs()
    test_fuse_lidar_estimate()
    test_fuse_lidar_estimate_with_align()
    test_robust_scale_ratio_rejects_outliers()
    test_fuse_optimized_pipeline()
    test_manifest_loader_integration()
    print("All LiDAR depth smoke tests passed.")
