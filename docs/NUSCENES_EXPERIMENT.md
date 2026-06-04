# nuScenes Experiment Guide (py123d)

End-to-end LabelAny3D on nuScenes converted with [py123d](https://github.com/kesai-labs/py123d).

## Prerequisites

```bash
pip install -r requirements-py123d.txt
export PY123D_DATA_ROOT=/path/to/py123d_data
```

Convert nuScenes to py123d Arrow logs (see [py123d docs](https://kesai.eu/py123d/)).

**Recommended:** Linux or WSL2 with NVIDIA GPU. py123d wheels may not install on all Windows Python versions.

## One-command pipeline

From `src/`:

```bash
# Smoke: 5 scenes, depth through whole (if models installed)
python batch_scripts/run_nuscenes.py --preset smoke --skip_existing --visualize after_depth

# Dev: 10 scenes
python batch_scripts/run_nuscenes.py --preset dev --steps all

# Full val split (no scene cap)
python batch_scripts/run_nuscenes.py --preset full --config configs/py123d_nuscenes.yaml
```

Or use the shell helper:

```bash
bash scripts/run_nuscenes.sh
```

### Pipeline order

1. `depth` — LiDAR depth + `nuscenes_annotations.json`
2. `enhance` — InvSR super-resolution
3. `crops` — object crops from GT masks
4. `completion` — amodal completion
5. `elevation` — Zero123++ elevation
6. `reconstruction` — TRELLIS per object
7. `whole` — scene alignment + `3dbbox.json`
8. `combine` — `nuScenes3D_<split>.json`

All steps after depth use `--data_backend py123d` and read `input.png` from each scene folder under `save_dir/nuscenes_val/<scene_id>/`.

## Step-by-step (debugging)

```bash
cd src
export PY123D_DATA_ROOT=/path/to/py123d_data

python batch_scripts/depth.py --depth_source py123d --config configs/py123d_nuscenes_smoke.yaml --save_dir ../experimental_results/nuScenes/ --end_index -1

python batch_scripts/enhance.py --data_backend py123d --config configs/py123d_nuscenes.yaml --save_dir ../experimental_results/nuScenes/ --split nuscenes_val --end_index -1

python batch_scripts/get_crops_enhanced.py --data_backend py123d --config configs/py123d_nuscenes.yaml --save_dir ../experimental_results/nuScenes/ --split nuscenes_val --end_index -1

# ... completion, elevation, reconstruction, whole with same flags

python tools/combine_results.py --split nuscenes_val --results_dir ../experimental_results/nuScenes --output ../experimental_results/nuScenes/nuScenes3D_nuscenes_val.json
```

## Visualization

Python preview (no Blender):

```bash
python tools/visualize_scene.py --scene_dir ../experimental_results/nuScenes/nuscenes_val/<scene_id> --mode compose
python tools/visualize_scene.py --root ../experimental_results/nuScenes/nuscenes_val --mode gt_2d,depth
```

Outputs under `<scene>/viz/`: `gt_overlay.png`, `depth_colormap.png`, `rgb_depth.png`, `crops_grid.png`, `bbox3d_overlay.png`, `summary.png`.

Blender video (optional):

```bash
export BLENDER_BIN=/path/to/blender
python tools/visualize_scene.py --scene_dir <scene_dir> --backend blender
```

Requires `depth_scene_no_edge.ply` and `3dbbox.json` (after `whole.py`).

Integrated with the runner:

```bash
python batch_scripts/run_nuscenes.py --preset smoke --visualize after_depth --viz_backend preview
python batch_scripts/run_nuscenes.py --preset smoke --visualize after_whole --viz_backend both
```

## Presets

| Preset | Config | Scenes |
|--------|--------|--------|
| `smoke` | `py123d_nuscenes_smoke.yaml` | max 5 |
| `dev` | `py123d_nuscenes.yaml` | max 10 |
| `full` | `py123d_nuscenes.yaml` | all val |

## Category mapping

nuScenes 3D labels are mapped to COCO-style `category_id` in `src/integrations/py123d/annotations.py`. Combined JSON follows Omni3D layout; evaluation may need label-aware metrics.

## Troubleshooting

- **No scenes in loader**: Run `depth.py` first; check `save_dir/nuscenes_val/` exists.
- **enhance fails on missing enhanced image**: Run steps in order (enhance before crops).
- **Empty crops**: Check `nuscenes_annotations.json` has valid instances for the frame.
- **py123d import error**: Use Linux/Python 3.9–3.12 and `pip install -r requirements-py123d.txt`.
