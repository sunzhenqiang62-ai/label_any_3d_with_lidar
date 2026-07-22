#!/usr/bin/env bash
# Repair torch 2.2.2 + build pytorch3d after accidental conda pytorch3d install.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/scripts/mirrors_cn.sh"

echo "[1/4] Ensure GCC 11 toolchain..."
conda install -y -c conda-forge gxx_linux-64=11.4.0 gcc_linux-64=11.4.0
# shellcheck source=/dev/null
source "${ROOT}/scripts/mirrors_cn.sh"

echo "[2/4] Remove incompatible conda pytorch3d (if any)..."
conda remove -y pytorch3d --force 2>/dev/null || true

echo "[3/4] Restore torch 2.2.2+cu118 (Tsinghua + PyTorch cu118 index)..."
pip_cn torch==2.2.2+cu118 torchvision==0.17.2+cu118 xformers==0.0.25.post1 --force-reinstall
python - <<'PY'
import torch
assert torch.__version__.startswith("2.2.2"), torch.__version__
assert torch.cuda.is_available(), "CUDA unavailable"
print("torch OK", torch.__version__)
PY

echo "[4/4] Build pytorch3d from local source (do NOT use conda pytorch3d)..."
P3D="${ROOT}/../LabelAny3D/pytorch3d"
if [ ! -f "${P3D}/setup.py" ]; then
  fetch_github_repo "facebookresearch/pytorch3d" "${ROOT}/.deps_cn/pytorch3d"
  P3D="${ROOT}/.deps_cn/pytorch3d"
fi
rm -rf "${P3D}/build"
CC="${LA3D_BUILD_CC}" CXX="${LA3D_BUILD_CXX}" CUDA_HOME="${LA3D_CUDA_HOME}" \
  FORCE_CUDA=1 MAX_JOBS="$(nproc)" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}" \
  pip_cn "${P3D}" --no-build-isolation

python - <<'PY'
import torch, pytorch3d
from pytorch3d.io import IO
print("pytorch3d OK", pytorch3d.__version__, "torch", torch.__version__)
PY

echo "Done. Run whole/combine smoke:"
echo "  source env.sh && cd src && python batch_scripts/run_nuscenes.py --preset locateanything \\"
echo "    --save_dir ../experimental_results/nuScenes_smoke_locateanything_jun8 \\"
echo "    --steps whole,combine --skip_existing --end_index 1"
