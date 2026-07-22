#!/usr/bin/env bash
# Domestic mirror defaults for restricted networks (China).
# Usage: source scripts/mirrors_cn.sh

# PyPI (Tsinghua); PyTorch CUDA wheels use official index + Tsinghua as extra.
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-https://download.pytorch.org/whl/cu118}"

# HuggingFace mirror
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# GitHub mirrors
export GITHUB_MIRROR="${GITHUB_MIRROR:-https://gitclone.com/github.com}"
export GITHUB_MIRROR_ALT="${GITHUB_MIRROR_ALT:-https://mirror.ghproxy.com/https://github.com}"
export GITHUB_MIRROR_ALT2="${GITHUB_MIRROR_ALT2:-https://kgithub.com}"
export GITHUB_ZIP_MIRROR="${GITHUB_ZIP_MIRROR:-https://ghfast.top/https://github.com}"

# Build toolchain for CUDA extensions (prefer conda GCC 11+ for PyTorch 2.x)
export LA3D_BUILD_CC="${LA3D_BUILD_CC:-gcc}"
export LA3D_BUILD_CXX="${LA3D_BUILD_CXX:-g++}"
if [ -x "${CONDA_PREFIX:-/opt/conda}/bin/x86_64-conda-linux-gnu-g++" ]; then
  export LA3D_BUILD_CC="${CONDA_PREFIX:-/opt/conda}/bin/x86_64-conda-linux-gnu-gcc"
  export LA3D_BUILD_CXX="${CONDA_PREFIX:-/opt/conda}/bin/x86_64-conda-linux-gnu-g++"
fi
export LA3D_CUDA_HOME="${LA3D_CUDA_HOME:-/usr/local/cuda}"

pip_cn() {
  pip install -i "$PIP_INDEX_URL" --extra-index-url "$PIP_EXTRA_INDEX_URL" "$@"
}

github_clone() {
  local repo="$1"
  local dest="$2"
  rm -rf "$dest"
  for base in "$GITHUB_MIRROR" "$GITHUB_MIRROR_ALT" "$GITHUB_MIRROR_ALT2"; do
    if [ "$base" = "$GITHUB_MIRROR_ALT2" ]; then
      if git clone --depth 1 "${base}/${repo}.git" "$dest" 2>/dev/null; then
        return 0
      fi
    elif git clone --depth 1 "${base}/${repo}.git" "$dest" 2>/dev/null; then
      return 0
    fi
  done
  echo "github_clone failed for ${repo}" >&2
  return 1
}

# Download GitHub repo zip via domestic proxy (branch=main).
github_zip() {
  local repo="$1"
  local dest="$2"
  local branch="${3:-main}"
  local zip_url="${GITHUB_ZIP_MIRROR}/${repo}/archive/refs/heads/${branch}.zip"
  local tmp_zip
  tmp_zip="$(mktemp /tmp/github_zip.XXXXXX)"
  rm -rf "$dest"
  mkdir -p "$dest"
  if ! curl -fsSL --connect-timeout 30 --max-time 300 "$zip_url" -o "$tmp_zip"; then
    rm -f "$tmp_zip"
    echo "github_zip failed: ${zip_url}" >&2
    return 1
  fi
  python - <<PY
import zipfile, pathlib, shutil
zip_path = pathlib.Path("${tmp_zip}")
dest = pathlib.Path("${dest}")
with zipfile.ZipFile(zip_path) as zf:
    root = zf.namelist()[0].split("/")[0]
    zf.extractall(dest.parent)
src = dest.parent / root
if src.is_dir():
    if dest.exists():
        shutil.rmtree(dest)
    src.rename(dest)
zip_path.unlink(missing_ok=True)
print("extracted", dest)
PY
}

fetch_github_repo() {
  local repo="$1"
  local dest="$2"
  github_clone "$repo" "$dest" || github_zip "$repo" "$dest"
}
