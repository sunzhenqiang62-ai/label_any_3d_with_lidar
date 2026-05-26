# py123d nuScenes Integration

Run LabelAny3D depth (and crops) from [py123d](https://github.com/kesai-labs/py123d) converted nuScenes Arrow logs instead of MoGe/DepthPro and COCONUT.

## Install

```bash
pip install -r requirements-py123d.txt
```

Requires converted data under `PY123D_DATA_ROOT` (see [py123d documentation](https://kesai.eu/py123d/)).

```bash
export PY123D_DATA_ROOT=/path/to/py123d_data
# Example conversion (see py123d repo for exact Hydra config names):
# py123d-conversion dataset=nuscenes ...
```

## Depth step

From `src/`:

```bash
python batch_scripts/depth.py \
  --depth_source py123d \
  --config configs/py123d_nuscenes.yaml \
  --save_dir ../experimental_results/nuScenes/ \
  --start_index 0 \
  --end_index -1
```

CLI overrides:

| Flag | Default |
|------|---------|
| `--py123d_data_root` | `$PY123D_DATA_ROOT` |
| `--py123d_split` | `val` |
| `--camera_key` | `CAM_FRONT` (maps to py123d `PCAM_F0`) |
| `--lidar_key` | `merged` |
| `--depth_fill` | `nearest` (recommended for sparse LiDAR) |

**Per-scene outputs** (under `experimental_results/nuScenes/nuscenes_val/<scene_uuid>/`):

- `input.png`, `depth_map.npy`, `cam_params.json`, PLY files
- `nuscenes_annotations.json` — COCO-style 2D boxes/masks from 3D labels

## Crops step

```bash
python batch_scripts/get_crops_enhanced.py \
  --data_backend py123d \
  --config configs/py123d_nuscenes.yaml \
  --save_dir ../experimental_results/nuScenes/ \
  --split nuscenes_val \
  --start_index 0 \
  --end_index -1
```

Then continue with `enhance.py`, `completion.py`, `elevation.py`, `reconstruction.py`, `whole.py` using the same `--save_dir` and `--split nuscenes_val`.

## Coordinates

- py123d uses a unified **global** frame for LiDAR points and box centers.
- Camera pose is `camera_to_global_se3`; LabelAny3D uses `world_to_cam = inv(camera_to_global)`.
- Projections use py123d pinhole `project_points_global` (including distortion when applicable).

## Troubleshooting

- **No scenes found**: Check `$PY123D_DATA_ROOT/logs` contains converted nuScenes logs.
- **Unknown camera_key**: List cameras with py123d-viser or use `PCAM_F0` / `CAM_FRONT`.
- **Empty annotations**: Scene may have no boxes at the chosen iteration; adjust `run.py123d.frame_index` in config.
