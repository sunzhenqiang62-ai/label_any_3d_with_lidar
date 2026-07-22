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

# Domestic mirrors (PyPI / HuggingFace / GitHub clone helpers)
# shellcheck source=/dev/null
source "${LA3D_ROOT}/scripts/mirrors_cn.sh"

# Activate conda env (prefer la3d; fall back to base)
_LA3D_CONDA_ENV="base"
if [ -f "${CONDA_PREFIX}/etc/profile.d/conda.sh" ] || [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    _conda_sh="${CONDA_PREFIX}/etc/profile.d/conda.sh"
    [ -f "$_conda_sh" ] || _conda_sh="/opt/conda/etc/profile.d/conda.sh"
    # shellcheck source=/dev/null
    source "$_conda_sh"
    if conda env list | awk '{print $1}' | grep -qx la3d; then
        conda activate la3d 2>/dev/null && _LA3D_CONDA_ENV="la3d"
    fi
fi
export LA3D_CONDA_ENV="${_LA3D_CONDA_ENV}"
unset _LA3D_CONDA_ENV

# Prefer local CUDA 12.1 toolkit assembled for PyTorch cu121 builds
if [ -z "${LA3D_CUDA_HOME:-}" ] && [ -x "${CONDA_PREFIX:-}/cuda-12.1/bin/nvcc" ]; then
    export LA3D_CUDA_HOME="${CONDA_PREFIX}/cuda-12.1"
elif [ -z "${LA3D_CUDA_HOME:-}" ] && [ -f "${LA3D_ROOT}/.cuda_home_local.sh" ]; then
    # shellcheck source=/dev/null
    source "${LA3D_ROOT}/.cuda_home_local.sh"
fi

# CUDA for runtime (conda) and extension builds (system nvcc when available)
if [ -x "${LA3D_CUDA_HOME}/bin/nvcc" ]; then
    export CUDA_HOME="${LA3D_CUDA_HOME}"
    export PATH="${LA3D_CUDA_HOME}/bin:${PATH}"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/nvcc" ]; then
    export CUDA_HOME="${CONDA_PREFIX}"
    export PATH="${CONDA_PREFIX}/bin:${PATH}"
else
    export CUDA_HOME="${CONDA_PREFIX:-/opt/conda}"
    export PATH="${CONDA_PREFIX:-/opt/conda}/bin:${PATH}"
fi
export CC="${LA3D_BUILD_CC}"
export CXX="${LA3D_BUILD_CXX}"

# HuggingFace mirror enabled via scripts/mirrors_cn.sh (HF_ENDPOINT)

# LocateAnything: sdpa works on A100/L40; default config uses magi/flash which may fail
export LOCATEANYTHING_ATTN="${LOCATEANYTHING_ATTN:-sdpa}"
export LOCATEANYTHING_MAX_EDGE="${LOCATEANYTHING_MAX_EDGE:-1920}"

# nuScenes / py123d defaults (override if your data lives elsewhere)
export PY123D_DATA_ROOT="${PY123D_DATA_ROOT:-${LA3D_ROOT}/dataset/py123d_smoke}"
export NUSCENES_DATA_ROOT="${NUSCENES_DATA_ROOT:-${LA3D_ROOT}/dataset/nuscenes_mini_raw}"

# Use local HF cache when hub is unreachable
if [ -d "${HOME}/.cache/huggingface/hub/models--nvidia--LocateAnything-3B" ]; then
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
fi

export PYTHONPATH="${LA3D_ROOT}/src:${PYTHONPATH:-}"

echo "LabelAny3D env ready: ${LA3D_CONDA_ENV} (Python $(python --version 2>&1 | awk '{print $2}'))"
echo "  LA3D_ROOT=${LA3D_ROOT}"
_nvcc_ver="$(nvcc --version 2>/dev/null | grep release | awk '{print $6}' | tr -d ',')"
echo "  CUDA_HOME=${CUDA_HOME} (${_nvcc_ver})"
unset _nvcc_ver
