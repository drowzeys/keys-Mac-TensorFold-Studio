#!/bin/bash
# Build and push the GHCR carrier image (TensorFold wheel at the pinned commit + dependency lock + render scripts).
#   bash scripts/build-carrier.sh           # build only
#   PUSH=1 bash scripts/build-carrier.sh    # build + push (needs `docker login ghcr.io` with write:packages)
# The wheel is pure Python, so the carrier can be built on any machine with Docker and python3.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-ghcr.io/drowzeys/keys-mac-tensorfold-studio}"
TAG="${TAG:-1.8}"
TF_REPO="https://github.com/drowzeys/TensorFold.git"
TF_COMMIT="a2068c031e08109a0ec14c26b1ca655cf50ac34c"
cd "$HERE"
mkdir -p payload
python3 -m pip wheel --no-deps -q -w payload "tensorfold @ git+$TF_REPO@$TF_COMMIT"
cp requirements.lock h3_generate.py qwen_image_generate.py payload/
( cd payload && shasum -a 256 tensorfold-*.whl requirements.lock h3_generate.py qwen_image_generate.py > SHA256SUMS )
cat payload/SHA256SUMS
if [ "${PUSH:-0}" = 1 ]; then
  docker buildx build --platform linux/arm64,linux/amd64 -f Dockerfile.ghcr -t "$IMAGE:$TAG" -t "$IMAGE:latest" --push .
  docker buildx imagetools inspect "$IMAGE:$TAG" | grep -E '^Digest' || true
else
  docker build -f Dockerfile.ghcr -t "$IMAGE:$TAG" -t "$IMAGE:latest" .
fi
