# LiDAR Point Cloud Input (Replace Depth Step)

Use calibrated world-frame LiDAR instead of MoGe + DepthPro in `depth.py`. Downstream steps (`enhance.py` … `whole.py`) consume the same artifacts: `depth_map.npy`, `cam_params.json`, and scene PLY files.

## Coordinate Conventions

- **OpenCV camera frame** (used by this repo): X right, Y down, Z forward (into the scene).
- **`world_to_cam`**: 4×4 matrix mapping homogeneous world (or map) points to the camera frame.
- **`c2w`**: camera-to-world; the code inverts this to obtain `world_to_cam`.
- **ROS LiDAR** often uses X forward, Y left, Z up. Bake the fixed rotation into `sensor_to_world` or into `world_to_cam`; do not assume axes match without checking.

## Dataset Layout

```
dataset/lidar/
├── manifest.json
├── images/
│   └── 001.png
├── lidar/
│   └── 001.ply          # or .npz with key "points"
└── calib/
    └── 001.json
```

### manifest.json

```json
{
  "split": "lidar",
  "scenes": [
    {
      "id": "scene_001",
      "image": "images/001.png",
      "pointcloud": "lidar/001.ply",
      "calib": "calib/001.json",
      "annotations": "annotations/001_bboxes.json"
    }
  ]
}
```

Paths are relative to the manifest file directory. Optional `annotations` is copied to `bboxes.json` in the output scene folder (for custom 2D boxes; COCONUT format is not required).

### calib/*.json

Required:

| Field | Description |
|-------|-------------|
| `K` | 3×3 pinhole intrinsics |
| `width`, `height` | Image size (must match RGB) |
| `world_to_cam` **or** `c2w` | 4×4 extrinsics |

Optional:

| Field | Description |
|-------|-------------|
| `sensor_to_world` | 4×4 if points are in sensor frame |
| `points_in_sensor_frame` | `true` to apply `sensor_to_world` before `world_to_cam` |

Example (`world_to_cam`):

```json
{
  "K": [[500, 0, 50], [0, 500, 50], [0, 0, 1]],
  "width": 100,
  "height": 100,
  "world_to_cam": [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ]
}
```

### Point cloud formats

- `.ply` / formats supported by `trimesh.load`
- `.npz` with array `points` (N×3); optional `colors` (N×3)

## Run

From the `src/` directory:

```bash
python batch_scripts/depth.py \
  --depth_source lidar \
  --manifest ../dataset/lidar/manifest.json \
  --config configs/lidar.yaml \
  --save_dir ../experimental_results/LiDAR/ \
  --start_index 0 \
  --end_index 1
```

Use `--end_index -1` to process all scenes in the manifest.

Sparse spinning LiDAR (many empty pixels):

```bash
python batch_scripts/depth.py ... --depth_fill nearest
```

Or set `run.depth.fill: nearest` in `configs/lidar.yaml`.

## Outputs (per scene)

Same layout as the COCO pipeline depth step:

```
experimental_results/LiDAR/lidar/<scene_id>/
├── input.png
├── depth_map.npy
├── cam_params.json      # K, c2w, W, H
├── depth_scene.ply
└── depth_scene_no_edge.ply
```

## Continue the Pipeline

After depth, run the remaining steps from `src/` if you have instance masks (`bboxes.json` or COCONUT-style annotations):

```bash
python batch_scripts/enhance.py --save_dir ../experimental_results/LiDAR/ --split lidar ...
# crops, completion, elevation, reconstruction, whole — use the same save_dir and split
```

Without masks, only the depth step applies; provide annotations via manifest `annotations` or integrate your own segmentation.

## Smoke Test

```bash
cd src
python tests/test_lidar_depth_smoke.py
```

## COCO Mode (unchanged)

```bash
python batch_scripts/depth.py --start_index 0 --end_index 100 --split val
```

Uses MoGe + DepthPro when `run.depth.source` is `estimate` (default in `configs/image.yaml`).
