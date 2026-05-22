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

📡 **[LiDAR Input Guide](docs/LIDAR_INPUT.md)** - Use world-frame LiDAR + calibration instead of MoGe/DepthPro for the depth step

🔧 **[OVMono3D Fine-tuning](https://github.com/UVA-Computer-Vision-Lab/LabelAny3D/tree/ovmono3d_finetune)** - Code for fine-tuning OVMono3D on LabelAny3D pseudo annotations

## Extensions in This Fork

This repository extends the upstream [LabelAny3D](https://github.com/UVA-Computer-Vision-Lab/LabelAny3D) pipeline for easier deployment and sensor fusion.

### Implemented

| Feature | Description |
|---------|-------------|
| **LiDAR depth step** | `--depth_source lidar` in `depth.py` skips MoGe/DepthPro; builds `depth_map.npy`, PLY, and `cam_params.json` from world-frame point clouds + `world_to_cam` / `c2w` calibration ([guide](docs/LIDAR_INPUT.md)) |
| **Manifest loader** | `LidarManifestLoader` + `configs/lidar.yaml` for RGB + `.ply`/`.npz` + per-scene calib JSON |
| **Sparse LiDAR holes** | `--depth_fill nearest` propagates depth into empty pixels before downstream alignment |
| **Resume** | Depth step skips scenes that already have `depth_map.npy` and `cam_params.json` |
| **Smoke tests** | `python tests/test_lidar_depth_smoke.py` (CPU, no GPU) |

Quick start (from `src/`):

```bash
python batch_scripts/depth.py \
  --depth_source lidar \
  --manifest ../dataset/lidar/manifest.json \
  --config configs/lidar.yaml \
  --save_dir ../experimental_results/LiDAR/ \
  --start_index 0 --end_index -1
```

Then continue the standard pipeline (`enhance.py` → `whole.py`) with the same `--save_dir` and split name. See [COCO Pipeline Guide](docs/COCO_PIPELINE.md) for step order.

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


## License

This project is licensed under the [MIT License](LICENSE).

[jy]: https://yaojin17.github.io
[rr]: https://scholar.google.com/citations?user=066_RcMAAAAJ&hl=en
[se]: https://www.cs.virginia.edu/~se4ja/
[md]: https://matthewbdwyer.github.io/
[zc]: https://sites.google.com/site/zezhoucheng/
