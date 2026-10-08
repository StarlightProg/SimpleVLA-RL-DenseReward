#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-simplevla-rl:cu121-openvla}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "${REPO_ROOT}"
docker build \
    -f docker/simplevla/Dockerfile \
    -t "${IMAGE_NAME}" \
    .

echo "Built ${IMAGE_NAME}"
