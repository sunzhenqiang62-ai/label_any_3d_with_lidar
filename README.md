<div align="center">

# LabelAny3D: Label Any Object 3D in the Wild

[Jin Yao][jy], [Radowan Mahmud Redoy][rr], [Sebastian Elbaum][se], [Matthew B. Dwyer][md], [Zezhou Cheng][zc]


[![Website](https://img.shields.io/badge/Project-Page-b361ff
)](https://uva-computer-vision-lab.github.io/LabelAny3D/)
[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://openreview.net/pdf?id=Q2fU0JDHuW)


</div>

<table style="border-collapse: collapse; border: none;">
<tr>
    <td width="100%">
        <p align="center">
            Samples from COCO3D dataset
            <img src=".github/LA3D.png" alt="COCO3D samples"/ height="300">
        </p>
    </td>
</tr>
</table>

## COCO3D Dataset

The evaluation set of COCO3D and pseudo-labeled training set are available at [Hugging Face](https://huggingface.co/datasets/uva-cv-lab/COCO3D).

## 3D BBox Human Refinement Interface

We release the source code for the refinement interface at https://github.com/UVA-Computer-Vision-Lab/3d_annotator.

## Getting Started

📦 **[Installation Guide](docs/INSTALL.md)** - Setup instructions and external dependencies

📖 **[COCO Pipeline Guide](docs/COCO_PIPELINE.md)** - Run the pipeline on COCO dataset

📡 **[LiDAR Input Guide](docs/LIDAR_INPUT.md)** - World-frame LiDAR + calibration (manifest) instead of MoGe/DepthPro

🚗 **[py123d nuScenes Guide](docs/PY123D_NUSCENES.md)** - Autonomous-driving data via [py123d](https://github.com/kesai-labs/py123d) Arrow logs (nuScenes)

🔧 **[OVMono3D Fine-tuning](https://github.com/UVA-Computer-Vision-Lab/LabelAny3D/tree/ovmono3d_finetune)** - Code for fine-tuning OVMono3D on LabelAny3D pseudo annotations

## Extensions in This Fork

This fork extends [LabelAny3D](https://github.com/UVA-Computer-Vision-Lab/LabelAny3D) with **alternative depth inputs** and deployment notes. Downstream steps (`enhance.py` → `whole.py`) are unchanged once each scene folder contains `depth_map.npy`, `cam_params.json`, and (for crops) instance annotations.

### Depth backends (`batch_scripts/depth.py`)

| `--depth_source` | Input | Config | Docs |
|------------------|-------|--------|------|
| `estimate` (default) | MoGe + DepthPro | `configs/image.yaml` | [COCO Pipeline](docs/COCO_PIPELINE.md) |
| `lidar` | RGB + `.ply`/`.npz` + calib JSON manifest | `configs/lidar.yaml` | [LIDAR_INPUT](docs/LIDAR_INPUT.md) |
| `py123d` | py123d Arrow logs (nuScenes, etc.) | `configs/py123d_nuscenes.yaml` | [PY123D_NUSCENES](docs/PY123D_NUSCENES.md) |

Shared options: `--depth_fill nearest` (sparse LiDAR), `--start_index` / `--end_index` (`-1` = all), resume when `depth_map.npy` exists.

Optional dependency for py123d:

```bash
pip install -r requirements-py123d.txt
export PY123D_DATA_ROOT=/path/to/py123d_data
```

### Other features

| Feature | Description |
|---------|-------------|
| **Manifest loader** | `LidarManifestLoader` for custom RGB + LiDAR + calib datasets |
| **py123d adapter** | `src/integrations/py123d/` — nuScenes loader, 3D box → COCO `nuscenes_annotations.json` |
| **Crops backend** | `get_crops_enhanced.py --data_backend py123d` reads depth-step output folders |
| **Tests** | `tests/test_lidar_depth_smoke.py`, `tests/test_py123d_coord.py`, `tests/test_py123d_annotations.py` |

### Quick start (from `src/`)

**Custom LiDAR manifest:**

```bash
python batch_scripts/depth.py \
  --depth_source lidar \
  --manifest ../dataset/lidar/manifest.json \
  --config configs/lidar.yaml \
  --save_dir ../experimental_results/LiDAR/ \
  --start_index 0 --end_index -1
```

**py123d nuScenes** (after `py123d-conversion`; see [PY123D_NUSCENES.md](docs/PY123D_NUSCENES.md)):

```bash
python batch_scripts/depth.py \
  --depth_source py123d \
  --config configs/py123d_nuscenes.yaml \
  --save_dir ../experimental_results/nuScenes/ \
  --end_index -1

python batch_scripts/get_crops_enhanced.py \
  --data_backend py123d \
  --config configs/py123d_nuscenes.yaml \
  --split nuscenes_val \
  --end_index -1
```

Then run `enhance.py` → `completion.py` → `elevation.py` → `reconstruction.py` → `whole.py` with the same `--save_dir` and split. See [COCO Pipeline Guide](docs/COCO_PIPELINE.md).

### Deployment & Optimization Recommendations

The core pipeline still runs as eight batch scripts from `src/`. For production or cluster use, consider:

- **Environment variables** — Centralize paths (`LA3D_ROOT`, `LA3D_CHECKPOINTS`, `HF_HOME`) instead of hard-coded `../external` and `../dataset`.
- **Unified entrypoint** — Wrap all steps in one CLI (e.g. `pipeline.py run --steps depth,enhance,...`) with preflight checks for CUDA, weights, and data.
- **Docker** — Multi-stage image with CUDA 12.1, pinned PyTorch, and volume mounts for checkpoints, datasets, and `experimental_results/` ([install steps](docs/INSTALL.md)).
- **Offline weights** — Bundle `external/checkpoints` and Hugging Face caches for air-gapped clusters.
- **Stage services** — Split heavy steps (depth, enhance, TRELLIS recon) into separate GPU jobs with shared object storage between stages.
- **SLURM** — Shard by `--start_index` / `--end_index`; see array-job example in [COCO_PIPELINE.md](docs/COCO_PIPELINE.md).

Platform note: full install (TRELLIS, kaolin, flash-attn) is tested on **Linux + NVIDIA GPU**. Windows users should use WSL2 or a Linux server.

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
- [Gen3DSR](https://github.com/AndreeaDogaru/Gen3DSR) - 3D reconstruction framework
- [TRELLIS](https://github.com/microsoft/TRELLIS) - 3D asset generation
- [MoGe](https://github.com/microsoft/MoGe) - Monocular geometry estimation
- [DepthPro](https://github.com/apple/ml-depth-pro) - Metric depth estimation
- [MASt3R](https://github.com/naver/mast3r) - Dense matching
- [InvSR](https://github.com/zsyOAOA/InvSR) - Image super-resolution
- [COCONUT](https://github.com/bytedance/coconut_cvpr2024) - COCO segmentation annotations
- [OVMono3D](https://github.com/UVA-Computer-Vision-Lab/ovmono3d) - Open vocabulary monocular 3D detection
- [py123d](https://github.com/kesai-labs/py123d) - Unified autonomous driving dataset API (optional nuScenes adapter in this fork)


## License

This project is licensed under the [MIT License](LICENSE).

[jy]: https://yaojin17.github.io
[rr]: https://scholar.google.com/citations?user=066_RcMAAAAJ&hl=en
[se]: https://www.cs.virginia.edu/~se4ja/
[md]: https://matthewbdwyer.github.io/
[zc]: https://sites.google.com/site/zezhoucheng/
