#!/usr/bin/env bash
# Install/repair local LabelAny3D dependencies on base conda (no la3d env required).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "${ROOT}/scripts/mirrors_cn.sh"

echo "[1/7] Restore InvSR from LabelAny3D.tar.gz if sampler is missing..."
if [ ! -f external/InvSR/sampler_invsr.py ] && [ -f "${ROOT}/../LabelAny3D.tar.gz" ]; then
  tar -xzf "${ROOT}/../LabelAny3D.tar.gz" external/InvSR/
elif [ ! -f external/InvSR/sampler_invsr.py ] && [ -f "${ROOT}/LabelAny3D.tar.gz" ]; then
  tar -xzf "${ROOT}/LabelAny3D.tar.gz" external/InvSR/
fi

if [ ! -f external/TRELLIS/setup.sh ] && [ -f "${ROOT}/../LabelAny3D.tar.gz" ]; then
  tar -xOf "${ROOT}/../LabelAny3D.tar.gz" external/TRELLIS/setup.sh > external/TRELLIS/setup.sh
  chmod +x external/TRELLIS/setup.sh
fi

echo "[2/7] Core Python packages (Tsinghua PyPI + PyTorch cu118 extra index)..."
pip_cn -r requirements-py123d.txt
pip_cn "numpy>=1.24,<2" "protobuf==3.20.3"
pip_cn "git+https://gitclone.com/github.com/EasternJournalist/utils3d.git@3913c65d81e05e47b9f367250cf8c0f7462a0900" \
  || pip_cn "git+https://mirror.ghproxy.com/https://github.com/EasternJournalist/utils3d.git@3913c65d81e05e47b9f367250cf8c0f7462a0900" \
  || pip_cn "git+https://github.com/EasternJournalist/utils3d.git@3913c65d81e05e47b9f367250cf8c0f7462a0900"
pip_cn -r requirements-locateanything.txt
pip_cn torch==2.2.2+cu118 torchvision==0.17.2+cu118
pip_cn "git+https://gitclone.com/github.com/yaojin17/detectron2.git" --no-build-isolation \
  || pip_cn "git+https://mirror.ghproxy.com/https://github.com/yaojin17/detectron2.git" --no-build-isolation \
  || pip_cn "git+https://github.com/yaojin17/detectron2.git" --no-build-isolation
pip_cn -e external/ml-depth-pro --no-deps
pip_cn -e external/InvSR --no-deps
pip_cn loguru einops easydict opencv-python python-box sentencepiece albumentations fvcore iopath
pip_cn xformers==0.0.25.post1

echo "[3/7] Verify core imports..."
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable after torch install"
import detectron2
import py123d
print("torch", torch.__version__, "cuda ok")
print("detectron2", detectron2.__version__)
print("py123d ok")
PY

echo "[4/7] Optional: OneFormer natten..."
if pip_cn natten==0.17.1+torch220cu118 \
  -f https://shi-labs.com/natten/wheels --trusted-host shi-labs.com; then
  echo "natten installed (shi-labs wheel)"
else
  echo "natten install failed; crop refinement will fall back to box masks"
fi

echo "[5/7] TRELLIS extensions via domestic GitHub mirrors..."
EXT_ROOT="${ROOT}/.deps_cn"
mkdir -p "${EXT_ROOT}"

if ! python -c "import nvdiffrast" 2>/dev/null; then
  if fetch_github_repo "NVlabs/nvdiffrast" "${EXT_ROOT}/nvdiffrast"; then
    CC="${LA3D_BUILD_CC}" CXX="${LA3D_BUILD_CXX}" CUDA_HOME="${LA3D_CUDA_HOME}" \
      pip_cn "${EXT_ROOT}/nvdiffrast" --no-build-isolation || true
  fi
fi

if ! python -c "import diffoctreerast" 2>/dev/null; then
  if fetch_github_repo "JeffreyXiang/diffoctreerast" "${EXT_ROOT}/diffoctreerast"; then
    git -C "${EXT_ROOT}/diffoctreerast" submodule update --init --recursive 2>/dev/null || true
    CC="${LA3D_BUILD_CC}" CXX="${LA3D_BUILD_CXX}" CUDA_HOME="${LA3D_CUDA_HOME}" \
      pip_cn "${EXT_ROOT}/diffoctreerast" --no-build-isolation || true
  fi
fi

echo "[6/7] Build toolchain (conda GCC 11 for PyTorch 2.x extensions)..."
if ! "${LA3D_BUILD_CXX}" --version 2>/dev/null | grep -q "11\\."; then
  conda install -y -c conda-forge gxx_linux-64=11.4.0 gcc_linux-64=11.4.0 || true
  # shellcheck source=/dev/null
  source "${ROOT}/scripts/mirrors_cn.sh"
fi

echo "[7/7] Optional: pytorch3d (whole/combine step; do NOT use conda pytorch3d — it downgrades torch)..."
if ! python -c "import pytorch3d" 2>/dev/null; then
  P3D_SRC=""
  for candidate in \
    "${ROOT}/../LabelAny3D/pytorch3d" \
    "${ROOT}/external/pytorch3d"; do
    if [ -f "${candidate}/setup.py" ]; then
      P3D_SRC="${candidate}"
      break
    fi
  done
  if [ -n "${P3D_SRC}" ]; then
    echo "  building from ${P3D_SRC}"
    CC="${LA3D_BUILD_CC}" CXX="${LA3D_BUILD_CXX}" CUDA_HOME="${LA3D_CUDA_HOME}" \
      FORCE_CUDA=1 MAX_JOBS="$(nproc)" \
      pip_cn "${P3D_SRC}" --no-build-isolation || \
      echo "pytorch3d build failed; whole/combine will be unavailable"
  else
    echo "  pytorch3d source not found; clone with: github_clone facebookresearch/pytorch3d .deps_cn/pytorch3d"
  fi
fi

echo "[8/8] Final import check..."
python - <<'PY'
import importlib
for name in ("nvdiffrast", "pytorch3d", "natten", "xformers"):
    try:
        importlib.import_module(name)
        print(name, "OK")
    except Exception as exc:
        print(name, "MISSING:", exc)
PY

echo "Done. Source env and run:"
echo "  source env.sh"
echo "  cd src && python batch_scripts/run_nuscenes.py --preset locateanything \\"
echo "    --save_dir ../experimental_results/nuScenes_multicam_locateanything \\"
echo "    --py123d_dataset nuscenes-mini --skip_existing --visualize all --end_index 1"
