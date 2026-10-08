#!/bin/bash
# Build and push the GHCR carrier image (TensorFold wheel at the pinned commit + dependency lock + render scripts).
#   bash scripts/build-carrier.sh           # build only
#   PUSH=1 bash scripts/build-carrier.sh    # build + push (needs `docker login ghcr.io` with write:packages)
# The wheel is pure Python, so the carrier can be built on any machine with Docker and python3. The native engine
# is not: build it on a Mac with scripts/build-zig-engine.sh and copy payload/zig-engine here first; without it the
# carrier is built without the engine and oneshot-setup.sh builds from source or stays on the MLX engine.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-ghcr.io/drowzeys/keys-mac-tensorfold-studio}"
TAG="${TAG:-2.1}"
TF_REPO="https://github.com/drowzeys/TensorFold.git"
TF_COMMIT="a2068c031e08109a0ec14c26b1ca655cf50ac34c"
cd "$HERE"
mkdir -p payload
python3 -m pip wheel --no-deps -q -w payload "tensorfold @ git+$TF_REPO@$TF_COMMIT"
cp requirements.lock h3_generate.py qwen_image_generate.py zig_fasth3.py payload/
( cd payload && shasum -a 256 tensorfold-*.whl requirements.lock h3_generate.py qwen_image_generate.py zig_fasth3.py > SHA256SUMS )
if [ -x payload/zig-engine/tf-h3-dit ]; then ( cd payload/zig-engine && shasum -a 256 -c SHA256SUMS ) || { echo "payload/zig-engine does not match its checksums" >&2; exit 1; }
else echo "NOTE: no payload/zig-engine: this carrier will not hold the native engine" >&2; fi
cat payload/SHA256SUMS
if [ "${PUSH:-0}" = 1 ]; then
  docker buildx build --platform linux/arm64,linux/amd64 -f Dockerfile.ghcr -t "$IMAGE:$TAG" -t "$IMAGE:latest" --push .
  docker buildx imagetools inspect "$IMAGE:$TAG" | grep -E '^Digest' || true
else
  docker build -f Dockerfile.ghcr -t "$IMAGE:$TAG" -t "$IMAGE:latest" .
fi
