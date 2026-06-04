
## What changed in this fork

Based on [UVA LabelAny3D](https://github.com/UVA-Computer-Vision-Lab/LabelAny3D). **The original COCO + MoGe/DepthPro pipeline is unchanged**; we add optional ways to supply **metric depth from real LiDAR** and to run on **nuScenes via py123d**, without retraining the core 3D models.

### Highlights

| Change | What it does |
|--------|----------------|
| **Three depth backends** | `depth.py --depth_source estimate \| lidar \| py123d` — same outputs for downstream steps |
| **Custom LiDAR** | Manifest + calib JSON → `depth_map.npy`, PLY, `cam_params.json` ([guide](docs/LIDAR_INPUT.md)) |
| **py123d + nuScenes** | Arrow logs → depth + projected 3D-box masks (`nuscenes_annotations.json`) for crops ([guide](docs/PY123D_NUSCENES.md)) |
| **Skip finished scenes** | Resume when `depth_map.npy` + `cam_params.json` already exist |
| **Sparse LiDAR** | `--depth_fill nearest` fills empty depth pixels before alignment |
| **CPU smoke tests** | `test_lidar_depth_smoke.py`, `test_py123d_coord.py`, `test_py123d_annotations.py` |
| **Scene visualization** | `visualize_scene.py`: 3×N summary grid, high-res BEV with GT (color) + Pred (blue), mesh / point-cloud overlays |
| **GT 3D boxes (py123d)** | `nuscenes_gt_3dbbox.json` per scene; used in BEV without per-box GT text labels |
| **Model-based crops (nuScenes)** | OneFormer segmentation + tagging; `allowed_categories: ['car', 'person']` in smoke/full configs |

### New / modified files (fork)

```
src/geometry/lidar_depth.py          # LiDAR → depth_map / PLY / cam_params
src/batch_scripts/lidar_loader.py
src/batch_scripts/py123d_loader.py
src/integrations/py123d/              # nuScenes adapter (coord, annotations, loader)
src/configs/lidar.yaml
src/configs/py123d_nuscenes.yaml
src/configs/py123d_nuscenes_smoke.yaml
src/batch_scripts/pipeline_loader.py
src/batch_scripts/run_nuscenes.py
src/tools/visualize_scene.py
docs/LIDAR_INPUT.md
docs/PY123D_NUSCENES.md
docs/NUSCENES_EXPERIMENT.md
scripts/run_nuscenes.sh
requirements-py123d.txt                # optional: pip install -r requirements-py123d.txt
env.sh                                 # helper to activate la3d env + CUDA toolchain
```

Modified: `src/batch_scripts/depth.py`, `src/batch_scripts/get_crops_enhanced.py`, `src/dataset_model/BaseScene.py`, `PointCloudScene.py`

### Depth step at a glance

| Mode | When to use | Key command |
|------|-------------|-------------|
| `estimate` | Original COCO / in-the-wild (MoGe + DepthPro) | `python batch_scripts/depth.py --split val` |
| `lidar` | You have RGB + world-frame point cloud + calibration | `--depth_source lidar --manifest .../manifest.json` |
| `py123d` | nuScenes (or other sets) already converted with [py123d](https://github.com/kesai-labs/py123d) | `--depth_source py123d --config configs/py123d_nuscenes.yaml` |

All modes write the **same scene layout** (`input.png`, `depth_map.npy`, `cam_params.json`, …) so `enhance.py` → `whole.py` stay compatible.

---

## Quick start (fork features)

Run from the `src/` directory after [installation](docs/INSTALL.md).

**1. Custom LiDAR (manifest)**

```bash
python batch_scripts/depth.py \
  --depth_source lidar \
  --manifest ../dataset/lidar/manifest.json \
  --config configs/lidar.yaml \
  --save_dir ../experimental_results/LiDAR/ \
  --start_index 0 --end_index -1
```

**2. py123d nuScenes (one command)**

```bash
pip install -r requirements-py123d.txt
export PY123D_DATA_ROOT=/path/to/py123d_data

cd src
python batch_scripts/run_nuscenes.py --preset smoke --skip_existing --visualize after_depth
```

**3. Visualization**

```bash
python tools/visualize_scene.py --root ../experimental_results/nuScenes/nuscenes_val --mode compose
```

`compose` builds `viz/summary.png`: **3 panels per row**, full-width **BEV** on the last row. BEV draws dataset GT boxes (green / orange by class, no text on GT) and pipeline predictions (blue, `Pred:` labels). Requires `nuscenes_gt_3dbbox.json` (written at `depth` for py123d) or `PY123D_DATA_ROOT` for on-the-fly GT export.

Pipeline order: `depth` → `enhance` → `crops` → `completion` → `elevation` → `reconstruction` → `whole`. All nuScenes steps use `--data_backend py123d`.

Details: [nuScenes experiment](docs/NUSCENES_EXPERIMENT.md) · [COCO Pipeline](docs/COCO_PIPELINE.md) · [LiDAR](docs/LIDAR_INPUT.md) · [py123d nuScenes](docs/PY123D_NUSCENES.md)

### nuScenes smoke visualization

One-scene `nuscenes-mini` smoke run with **py123d LiDAR depth**, **OneFormer** instance masks (`car` / `person`), TRELLIS meshes, and **GT + predicted 3D boxes** in BEV.

![nuScenes OneFormer smoke summary](docs/assets/nuscenes_oneformer_smoke_summary.png)

The summary layout: row 1 — GT 2D, depth, crops; row 2 — 3D box overlay, mesh projection, LiDAR point projection; row 3 — full-width BEV (colored GT footprints + blue predictions).

| Panel | Description |
|-------|-------------|
| GT 2D / depth / crops | Input, metric depth colormap, detected instance crops |
| bbox / mesh / pc_proj | 3D boxes on image, reconstructed mesh overlay, LiDAR projected to camera |
| BEV | 1920×1920 bird's-eye (X–Z): gray point cloud, **GT** by category color, **Pred** in blue |

Older examples (LiDAR-only smoke): [summary](docs/assets/nuscenes_smoke_summary.png) · [mesh](docs/assets/nuscenes_mesh_overlay.png) · [BEV](docs/assets/nuscenes_bev_3d.png)

---

## Environment verification (2026-05-27)

The environment in this workspace has been validated end-to-end with the following setup:

- Conda env: `la3d` (`python=3.10`)
- PyTorch: `2.2.2+cu121`
- Optional acceleration packages: `flash-attn==2.7.4.post1`, `xformers==0.0.25.post1`, `kaolin==0.17.0`
- TRELLIS deps verified: `spconv`, `nvdiffrast`, `pytorch3d`, `detectron2`

### Activate the environment

```bash
source env.sh
cd src
```

### TRELLIS offline smoke test (local snapshot)

If your network is restricted, run TRELLIS from a local Hugging Face snapshot:

```bash
cd external/TRELLIS
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python - <<'PY'
import os
os.environ['ATTN_BACKEND'] = 'xformers'
os.environ['SPCONV_ALGO'] = 'native'
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline

model_path = '/root/.cache/huggingface/hub/models--JeffreyXiang--TRELLIS-image-large/snapshots/25e0d31ffbebe4b5a97464dd851910efc3002d96'
pipeline = TrellisImageTo3DPipeline.from_pretrained(model_path)
pipeline.cuda()
outputs = pipeline.run(
    Image.open('assets/example_image/T.png'),
    seed=1,
    sparse_structure_sampler_params={'steps': 1, 'cfg_strength': 1.0},
    slat_sampler_params={'steps': 1, 'cfg_strength': 1.0},
)
print(outputs.keys())
PY
```

Expected output keys include: `mesh`, `gaussian`, `radiance_field`.

---

## Documentation

| Guide | Content |
|-------|---------|
| [INSTALL.md](docs/INSTALL.md) | Environment, checkpoints, TRELLIS / external deps |
| [COCO_PIPELINE.md](docs/COCO_PIPELINE.md) | Original 8-step COCO pipeline |
| [LIDAR_INPUT.md](docs/LIDAR_INPUT.md) | Manifest format, calib JSON, coordinate frames |
| [PY123D_NUSCENES.md](docs/PY123D_NUSCENES.md) | `PY123D_DATA_ROOT`, conversion, py123d CLI |
| [NUSCENES_EXPERIMENT.md](docs/NUSCENES_EXPERIMENT.md) | `run_nuscenes.py`, presets, visualization |

---

## Deployment notes (optional)

For clusters or production:

- Centralize paths with env vars (`LA3D_ROOT`, `LA3D_CHECKPOINTS`, `HF_HOME`, `PY123D_DATA_ROOT`)
- Shard jobs with `--start_index` / `--end_index` (see SLURM example in [COCO_PIPELINE.md](docs/COCO_PIPELINE.md))
- Prefer **Linux + NVIDIA GPU**; use WSL2 on Windows
- Consider Docker + offline checkpoint bundles for air-gapped installs ([INSTALL.md](docs/INSTALL.md))

---

## Original project

### COCO3D dataset

The evaluation set of COCO3D and pseudo-labeled training set are available at [Hugging Face](https://huggingface.co/datasets/uva-cv-lab/COCO3D).

### 3D bbox human refinement interface

Source code: https://github.com/UVA-Computer-Vision-Lab/3d_annotator

### OVMono3D fine-tuning

https://github.com/UVA-Computer-Vision-Lab/LabelAny3D/tree/ovmono3d_finetune

---

## Citing

If you find this work useful for your research, please kindly cite:

```BibTeX
@inproceedings{yao2025labelany3d,
  title={LabelAny3D: Label Any Object 3D in the Wild},
  author={Jin Yao and Radowan Mahmud Redoy and Sebastian Elbaum and Matthew B. Dwyer and Zezhou Cheng},
  booktitle={Neural Information Processing Systems (NeurIPS)},
  year={2025}
}

@inproceedings{yao2025open,
  title={Open Vocabulary Monocular 3D Object Detection},
  author={Yao, Jin and Gu, Hao and Chen, Xuweiyi and Wang, Jiayun and Cheng, Zezhou},
  booktitle={Proceedings of the International Conference on 3D Vision (3DV)},
  year={2026}
}
```

## Acknowledgements

This work builds on many open-source projects:

- [Gen3DSR](https://github.com/AndreeaDogaru/Gen3DSR) — 3D reconstruction framework
- [TRELLIS](https://github.com/microsoft/TRELLIS) — 3D asset generation
- [MoGe](https://github.com/microsoft/MoGe) — Monocular geometry estimation
- [DepthPro](https://github.com/apple/ml-depth-pro) — Metric depth estimation
- [MASt3R](https://github.com/naver/mast3r) — Dense matching
- [InvSR](https://github.com/zsyOAOA/InvSR) — Image super-resolution
- [COCONUT](https://github.com/bytedance/coconut_cvpr2024) — COCO segmentation annotations
- [OVMono3D](https://github.com/UVA-Computer-Vision-Lab/ovmono3d) — Open vocabulary monocular 3D detection
- [py123d](https://github.com/kesai-labs/py123d) — Unified autonomous driving data API (optional adapter in this fork)

## License

This project is licensed under the [MIT License](LICENSE).

[jy]: https://yaojin17.github.io
[rr]: https://scholar.google.com/citations?user=066_RcMAAAAJ&hl=en
[se]: https://www.cs.virginia.edu/~se4ja/
[md]: https://matthewbdwyer.github.io/
[zc]: https://sites.google.com/site/zezhoucheng/
