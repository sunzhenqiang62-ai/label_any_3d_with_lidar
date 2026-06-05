#!/usr/bin/env bash
# LabelAny3D environment activation helper
# Usage: source env.sh

if [ -n "${BASH_SOURCE[0]:-}" ]; then
    _LA3D_SCRIPT="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then
    _LA3D_SCRIPT="${(%):-%x}"
else
    _LA3D_SCRIPT="$0"
fi
LA3D_ROOT="$(cd "$(dirname "$_LA3D_SCRIPT")" && pwd)"
unset _LA3D_SCRIPT
export LA3D_ROOT
export EXT_DIR="${LA3D_ROOT}/external"
export LA3D_CHECKPOINTS="${EXT_DIR}/checkpoints"

# Activate conda env
if [ -f "${CONDA_PREFIX}/etc/profile.d/conda.sh" ] || [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    _conda_sh="${CONDA_PREFIX}/etc/profile.d/conda.sh"
    [ -f "$_conda_sh" ] || _conda_sh="/opt/conda/etc/profile.d/conda.sh"
    # shellcheck source=/dev/null
    source "$_conda_sh"
    conda activate la3d 2>/dev/null || true
fi

# Use conda CUDA 12.1 toolchain (matches PyTorch cu121)
export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"

# Optional: HuggingFace mirror for restricted networks
# export HF_ENDPOINT=https://hf-mirror.com

# LocateAnything: sdpa works on A100/L40; default config uses magi/flash which may fail
export LOCATEANYTHING_ATTN="${LOCATEANYTHING_ATTN:-sdpa}"
export LOCATEANYTHING_MAX_EDGE="${LOCATEANYTHING_MAX_EDGE:-1920}"

echo "LabelAny3D env ready: la3d (Python $(python --version 2>&1 | awk '{print $2}'))"
echo "  LA3D_ROOT=${LA3D_ROOT}"
_nvcc_ver="$(nvcc --version 2>/dev/null | grep release | awk '{print $6}' | tr -d ',')"
echo "  CUDA_HOME=${CUDA_HOME} (${_nvcc_ver})"
unset _nvcc_ver
