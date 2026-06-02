#!/usr/bin/env bash
# Copy LabelAny3D to cluster workspace (run from Linux/WSL with SSH access).
set -euo pipefail

REMOTE_USER="${REMOTE_USER:-chengyue.sun}"
REMOTE_HOST="${REMOTE_HOST:-192.168.48.24}"
REMOTE_DIR="${REMOTE_DIR:-/mnt/cfs-baidu/public/chengyue.sun/workspace}"
PROJECT_NAME="${PROJECT_NAME:-LabelAny3D}"
SRC="$(cd "$(dirname "$0")/.." && pwd)"

echo "Source:      $SRC"
echo "Destination: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/${PROJECT_NAME}/"

ssh "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p '${REMOTE_DIR}/${PROJECT_NAME}'"

rsync -avz --progress \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.pytest_cache/' \
  --exclude 'experimental_results/' \
  --exclude 'dataset/coco/images/' \
  "${SRC}/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/${PROJECT_NAME}/"

echo "Done: ${REMOTE_DIR}/${PROJECT_NAME}/"
