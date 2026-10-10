#!/bin/bash
# Build TensorFold's H3 engine for CUDA (libtf_h3.so) into $PREFIX/tf-h3. oneshot-setup-spark.sh runs this.
#   bash spark/build-engine.sh            # skips the build when the pinned commit is already built
# It needs the CUDA toolkit's nvcc (12.9 or newer; a Spark has 13 at /usr/local/cuda) and fetches Zig 0.17.0 and the
# engine's source (drowzeys/TensorFold at TF_COMMIT) under $PREFIX. About a minute.
# Environment: PREFIX, NVCC, TF_COMMIT, TF_SOURCE (an existing checkout to build instead of fetching one).
set -euo pipefail
PREFIX="${PREFIX:-$HOME/.local/opt/tensorfold-studio}"
TF_COMMIT="${TF_COMMIT:-3b08f592023ed036329fcc575c6d216795681bd2}"
ZIG_VERSION=0.17.0
OUT="$PREFIX/tf-h3"
if [ -s "$OUT/libtf_h3.so" ] && [ "$(cat "$OUT/commit" 2>/dev/null)" = "$TF_COMMIT" ] && [ -z "${TF_SOURCE:-}" ]; then
  echo "[tensorfold] engine already built at ${TF_COMMIT:0:8}"; exit 0
fi
NVCC="${NVCC:-}"
[ -n "$NVCC" ] || for candidate in /usr/local/cuda/bin/nvcc "$(command -v nvcc || true)"; do
  [ -x "$candidate" ] && { NVCC="$candidate"; break; }
done
[ -n "$NVCC" ] || { echo "no nvcc: install the CUDA toolkit (12.9 or newer) or set NVCC" >&2; exit 1; }
version="$("$NVCC" --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1 \2/p')"
read -r major minor <<<"$version"
[ "$major" -gt 12 ] || { [ "$major" = 12 ] && [ "$minor" -ge 9 ]; } || { echo "$NVCC is CUDA $major.$minor; the GB10 needs 12.9 or newer" >&2; exit 1; }
case "$(uname -m)" in aarch64) arch=aarch64;; x86_64) arch=x86_64;; *) echo "no Zig build for $(uname -m)" >&2; exit 1;; esac
ZIG="$PREFIX/zig-$ZIG_VERSION/zig"
if [ ! -x "$ZIG" ]; then
  echo "[tensorfold] fetching Zig $ZIG_VERSION"
  mkdir -p "$PREFIX/zig-$ZIG_VERSION"
  curl -sfL "https://ziglang.org/download/$ZIG_VERSION/zig-$arch-linux-$ZIG_VERSION.tar.xz" | tar -xJ --strip-components=1 -C "$PREFIX/zig-$ZIG_VERSION"
fi
SRC="${TF_SOURCE:-$PREFIX/tensorfold-src}"
if [ -z "${TF_SOURCE:-}" ]; then
  [ -d "$SRC/.git" ] || git init -q "$SRC"
  git -C "$SRC" fetch -q --depth 1 https://github.com/drowzeys/TensorFold.git "$TF_COMMIT"
  git -C "$SRC" checkout -q FETCH_HEAD
fi
echo "[tensorfold] building the engine with $NVCC"
# two compile jobs: the kernel image's compiler takes a few GB
(cd "$SRC" && nice -n 19 "$ZIG" build tf-h3 -j2 -Doptimize=ReleaseFast -Dnvcc="$NVCC")
mkdir -p "$OUT"
cp "$SRC/zig-out/lib/libtf_h3.so" "$OUT/libtf_h3.so"
[ -n "${TF_SOURCE:-}" ] && git -C "$SRC" rev-parse HEAD > "$OUT/commit" || echo "$TF_COMMIT" > "$OUT/commit"
echo "[tensorfold] engine -> $OUT/libtf_h3.so"
