#!/usr/bin/env bash
# Example: run nuScenes smoke pipeline (Linux / WSL)
set -euo pipefail

: "${PY123D_DATA_ROOT:?Set PY123D_DATA_ROOT to converted py123d data}"

cd "$(dirname "$0")/../src"
python batch_scripts/run_nuscenes.py \
  --preset smoke \
  --save_dir ../experimental_results/nuScenes/ \
  --skip_existing \
  --visualize after_depth \
  "$@"
