#!/usr/bin/env bash
# =============================================================================
# One-shot: TensorFold Studio on Apple Silicon: Qwen-Image-2.1 (text to image) + MiniMax H3 (video + audio)
#
#   bash oneshot-setup.sh               # install, fetch both models, both turbo adapters and the 2x video decoder, render a test clip
#   bash oneshot-setup.sh --image-only  # the image model only (33 GB instead of 177 GB), render a test image
#   bash oneshot-setup.sh --no-render   # install and fetch only
#   bash oneshot-setup.sh --verify      # check an existing install, no downloads, no render
#
# Engine payload order: this clone's ./payload -> GHCR carrier image -> git at the pinned commit.
# Installs into its own venv ($PREFIX, default ~/.local/opt/tensorfold-studio). Touches nothing else.
# Weights: $H3_MODEL_DIR (default ~/h3-models/MiniMax-H3, FL2VA partition, 144 GB) and
#          $QWEN_MODEL_DIR (default ~/qwen-models/Qwen-Image-2.1, 33 GB).
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${PREFIX:-$HOME/.local/opt/tensorfold-studio}"
H3_MODEL_DIR="${H3_MODEL_DIR:-$HOME/h3-models/MiniMax-H3}"
QWEN_MODEL_DIR="${QWEN_MODEL_DIR:-$HOME/qwen-models/Qwen-Image-2.1}"
IMAGE="${IMAGE:-ghcr.io/drowzeys/keys-mac-tensorfold-studio:1.4}"
TF_REPO="https://github.com/drowzeys/TensorFold.git"
TF_COMMIT="55aa37c3c7f19a206b02aacf12946570ac50c069"
REF_REPO="https://github.com/mrbizarro/minimax-h3-mlx.git"
REF_COMMIT="79190205258454b43e6c9e50e577de234222419c"
MFLUX_REPO="https://github.com/mflux-community/mflux.git"
MFLUX_COMMIT="add5164e62c07cbcc9aec7a95c2e19c75605880c"
H3_ADAPTER_NAME="lightx2v_v1.0_768p_ourlayout.safetensors"
H3_ADAPTER_URL="https://github.com/mrbizarro/Phosphene/releases/download/weights-ltx25-v1/$H3_ADAPTER_NAME"
H3_ADAPTER_SHA256="d51d626fe0845da7e5845a47c323cf3f29086d44d24cb1a4b980882488746197"
QWEN_ADAPTER_REPO="Viggle/Qwen-Image-2.1-viggle-turbo"
QWEN_ADAPTER_NAME="Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r256.safetensors"
QWEN_ADAPTER_SHA256="f06c266e04438b5272bdfb99410421d52a65d7a37f6f42aabc3cb1faf0142644"
X2_REPO="speach1sdef178/MiniMax-H3-X2-Detail-VAE"
X2_NAME="MiniMax-H3-X2-Detail-v1.safetensors"
X2_SHA256="2296840f4acedcaa976688e7d7b97f7bf570b136e400385d3f46224011897aac"
MODE=""; VIDEO=1
for arg in "$@"; do
  case "$arg" in
    --image-only) VIDEO=0;;
    --verify|--no-render) MODE="$arg";;
    *) echo "unknown option $arg" >&2; exit 2;;
  esac
done

die() { echo "FATAL: $*" >&2; exit 1; }
ok()  { echo "  ✓ $*"; }
step(){ echo; echo "==> $*"; }

step "Preflight"
[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || die "Apple silicon macOS only"
CHIP=$(sysctl -n machdep.cpu.brand_string); RAM=$(( $(sysctl -n hw.memsize) / 1073741824 ))
ok "$CHIP, macOS $(sw_vers -productVersion), ${RAM} GB"
NEED=$([ "$VIDEO" = 1 ] && echo 128 || echo 48)
[ "$RAM" -ge "$NEED" ] || die "needs ${NEED} GB+ unified memory for this install (measured on 256 GB only)"
case "$CHIP" in *M5*) ok "M5: int8 kernels on the tensor units";;
  *) echo "  ! $CHIP is not M5: the int8 kernels need Metal 4 tensor operations; without them the engine runs bfloat16 and the README numbers do not apply";; esac
export PATH="/opt/homebrew/bin:$PATH"
command -v uv >/dev/null || die "uv required: brew install uv"
command -v python3.11 >/dev/null || die "python3.11 required: brew install python@3.11"
command -v git >/dev/null || die "git required"
[ "$VIDEO" = 0 ] || command -v ffmpeg >/dev/null || die "ffmpeg required: brew install ffmpeg"

fetch_ghcr() {
  command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1 || return 1
  echo "  payload from GHCR $IMAGE"
  docker pull -q "$IMAGE" >/dev/null || return 1
  docker run --rm -v "$HERE":/out "$IMAGE" cp -a /payload/. /out/payload/
}

step "TensorFold 0.6.5 + H3 and Qwen-Image families @ ${TF_COMMIT:0:8}, mflux @ ${MFLUX_COMMIT:0:8} (own venv at $PREFIX)"
HAS_ENGINE='import inspect, tensorfold.families.qwen_image.sampler, mflux.models.qwen21.qwen21_initializer; from tensorfold.families.h3.vae_video import load_video_decoder as d; assert "upscale_decoder" in inspect.signature(d).parameters; from tensorfold.families.h3.sampler import denoise as n; assert "audio_shift" in inspect.signature(n).parameters'
if [ ! -x "$PREFIX/venv/bin/python" ] || ! "$PREFIX/venv/bin/python" -c "$HAS_ENGINE" 2>/dev/null; then
  [ "$MODE" = "--verify" ] && die "TensorFold with the H3 and Qwen-Image families is not installed at $PREFIX"
  mkdir -p "$HERE/payload" "$PREFIX"
  ( cd "$HERE/payload" 2>/dev/null && shasum -a 256 -c SHA256SUMS >/dev/null 2>&1 ) \
    || { rm -f "$HERE"/payload/tensorfold-*.whl "$HERE"/payload/SHA256SUMS; fetch_ghcr || echo "  no carrier payload; installing from git"; }
  [ -x "$PREFIX/venv/bin/python" ] || uv venv -q -p "$(command -v python3.11)" "$PREFIX/venv"
  if ls "$HERE"/payload/tensorfold-*.whl >/dev/null 2>&1; then
    ( cd "$HERE/payload" && shasum -a 256 -c SHA256SUMS >/dev/null ) || die "carrier payload checksum mismatch"
    ok "carrier payload checksums verified"
    TF_SPEC=("$HERE"/payload/tensorfold-*.whl)
  else
    TF_SPEC=("tensorfold @ git+$TF_REPO@$TF_COMMIT")
  fi
  uv pip install -q --reinstall-package tensorfold -p "$PREFIX/venv/bin/python" -r "$HERE/requirements.lock" \
    "${TF_SPEC[@]}" "mflux @ git+$MFLUX_REPO@$MFLUX_COMMIT"
fi
cp "$HERE/h3_generate.py" "$HERE/qwen_image_generate.py" "$PREFIX/"
"$PREFIX/venv/bin/python" - <<'EOF' || die "the installed TensorFold lacks the H3 or Qwen-Image family"
import importlib.metadata as m
import mlx.core as mx
from tensorfold.families import families
from tensorfold.kernels.minimax.h3.v1 import mlp_int8
assert {"minimax_h3", "qwen_image_21"} <= set(families())
print(f"  ✓ tensorfold {m.version('tensorfold')} with the H3 and Qwen-Image families, mflux {m.version('mflux')}, mlx {mx.__version__}")
print("  ✓ int8 tensor-unit kernels available" if mlp_int8.available() else "  ! int8 kernels unavailable on this GPU: bfloat16 only")
EOF

step "Qwen-Image-2.1 weights (33 GB) -> $QWEN_MODEL_DIR"
if [ ! -f "$QWEN_MODEL_DIR/transformer/diffusion_pytorch_model-00002-of-00002.safetensors" ]; then
  [ "$MODE" = "--verify" ] && die "no Qwen-Image-2.1 weights at $QWEN_MODEL_DIR"
  echo "  Qwen-Image-2.1 is under the Qwen Research License Agreement: NON-COMMERCIAL use only. Read it first:"
  echo "  https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE"
  "$PREFIX/venv/bin/hf" download Qwen/Qwen-Image-2.1 --local-dir "$QWEN_MODEL_DIR"
fi
ok "Qwen-Image-2.1 at $QWEN_MODEL_DIR ($(du -sh "$QWEN_MODEL_DIR" | cut -f1))"

step "Image turbo adapter (Viggle turbo v0.3, rank 256, 1.36 GB)"
mkdir -p "$PREFIX/adapters"
QWEN_ADAPTER="$PREFIX/adapters/$QWEN_ADAPTER_NAME"
if [ ! -f "$QWEN_ADAPTER" ]; then
  [ "$MODE" = "--verify" ] && die "no image turbo adapter at $QWEN_ADAPTER"
  "$PREFIX/venv/bin/hf" download "$QWEN_ADAPTER_REPO" "$QWEN_ADAPTER_NAME" LICENSE NOTICE --local-dir "$PREFIX/adapters/viggle-turbo" >/dev/null
  echo "$QWEN_ADAPTER_SHA256  $PREFIX/adapters/viggle-turbo/$QWEN_ADAPTER_NAME" | shasum -a 256 -c - >/dev/null \
    || die "image turbo adapter checksum mismatch"
  ln -sf "viggle-turbo/$QWEN_ADAPTER_NAME" "$QWEN_ADAPTER"
fi
ok "$QWEN_ADAPTER_NAME"

if [ "$VIDEO" = 1 ]; then
  step "Text encoder, audio decoder and MP4 writer for H3: minimax-h3-mlx @ ${REF_COMMIT:0:8}"
  if [ ! -d "$PREFIX/minimax-h3-mlx/.git" ]; then
    [ "$MODE" = "--verify" ] && die "minimax-h3-mlx is not at $PREFIX/minimax-h3-mlx"
    git clone -q "$REF_REPO" "$PREFIX/minimax-h3-mlx"
  fi
  git -C "$PREFIX/minimax-h3-mlx" fetch -q origin "$REF_COMMIT" 2>/dev/null || true
  git -C "$PREFIX/minimax-h3-mlx" checkout -q "$REF_COMMIT" || die "cannot check out minimax-h3-mlx $REF_COMMIT"
  PYTHONPATH="$PREFIX/minimax-h3-mlx" "$PREFIX/venv/bin/python" -c "import minimax_h3_mlx.text_encoder, minimax_h3_mlx.media" \
    || die "minimax-h3-mlx does not import"
  ok "minimax-h3-mlx $(git -C "$PREFIX/minimax-h3-mlx" rev-parse --short HEAD)"

  step "MiniMax H3 weights (FL2VA partition, 144 GB) -> $H3_MODEL_DIR"
  if [ ! -f "$H3_MODEL_DIR/FL2VA/transformer/model.safetensors.index.json" ]; then
    [ "$MODE" = "--verify" ] && die "no H3 weights at $H3_MODEL_DIR"
    echo "  MiniMax H3 is under the MiniMax H3 Community License, which limits where it may be used. Read it first:"
    echo "  https://huggingface.co/MiniMaxAI/MiniMax-H3"
    "$PREFIX/venv/bin/hf" download MiniMaxAI/MiniMax-H3 --include "FL2VA/*" --include model_index.json \
      --include LICENSE --include README.md --local-dir "$H3_MODEL_DIR"
  fi
  ok "FL2VA at $H3_MODEL_DIR ($(du -sh "$H3_MODEL_DIR/FL2VA" | cut -f1))"

  step "Video Turbo adapter (lightx2v v1.0, runner layout, 1.96 GB)"
  H3_ADAPTER="$PREFIX/adapters/$H3_ADAPTER_NAME"
  if [ ! -f "$H3_ADAPTER" ]; then
    [ "$MODE" = "--verify" ] && die "no video Turbo adapter at $H3_ADAPTER"
    curl -L --fail --retry 3 -C - -o "$H3_ADAPTER.partial" "$H3_ADAPTER_URL"
    echo "$H3_ADAPTER_SHA256  $H3_ADAPTER.partial" | shasum -a 256 -c - >/dev/null || die "video Turbo adapter checksum mismatch"
    mv "$H3_ADAPTER.partial" "$H3_ADAPTER"
  fi
  ok "$H3_ADAPTER_NAME"

  step "2x video decoder (MiniMax-H3-X2-Detail-VAE, 5.2 GB)"
  X2_VAE="$PREFIX/adapters/$X2_NAME"
  if [ ! -f "$X2_VAE" ]; then
    [ "$MODE" = "--verify" ] && die "no 2x video decoder at $X2_VAE"
    "$PREFIX/venv/bin/hf" download "$X2_REPO" "$X2_NAME" LICENSE NOTICE --local-dir "$PREFIX/adapters/h3-x2" >/dev/null
    echo "$X2_SHA256  $PREFIX/adapters/h3-x2/$X2_NAME" | shasum -a 256 -c - >/dev/null || die "2x video decoder checksum mismatch"
    ln -sf "h3-x2/$X2_NAME" "$X2_VAE"
  fi
  ok "$X2_NAME"
fi

if [ "$MODE" = "--verify" ] || [ "$MODE" = "--no-render" ]; then
  echo; echo "DONE. Image: bash $HERE/scripts/image.sh \"a prompt\" out.png"
  [ "$VIDEO" = 0 ] || echo "      Studio: bash $HERE/scripts/studio.sh \"the picture\" \"what happens\" out.mp4"
  exit 0
fi

export PREFIX H3_MODEL_DIR QWEN_MODEL_DIR
if [ "$VIDEO" = 0 ]; then
  step "Test image: 1344x768, turbo adapter, int8 kernels"
  PROMPT_FILE="$HERE/prompts/baker-image.txt" SEED=42 bash "$HERE/scripts/image.sh" "" "$HERE/outputs/test.png" 2>&1 \
    | grep -E "tensorfold\] \{|rror|Trace" | sed 's/^/  /'
  [ -s "$HERE/outputs/test.png" ] || die "the test render wrote no image"
  echo; echo "DONE. $HERE/outputs/test.png"
  exit 0
fi
step "Test clip: text -> image -> 5 s video with sound, 864x480, both turbo adapters, int8 kernels"
WIDTH=864 HEIGHT=480 FRAMES=124 SEED=42 IMAGE_PROMPT_FILE="$HERE/prompts/baker-image.txt" \
  VIDEO_PROMPT_FILE="$HERE/prompts/baker-video.txt" bash "$HERE/scripts/studio.sh" "" "" "$HERE/outputs/test.mp4" 2>&1 \
  | grep -E "tensorfold\] \{|studio\]|rounded|rror|Trace" | sed 's/^/  /'
[ -s "$HERE/outputs/test.mp4" ] || die "the test render wrote no file"
echo
echo "DONE. $HERE/outputs/test.png and $HERE/outputs/test.mp4"
echo "Studio: bash $HERE/scripts/studio.sh \"the picture\" \"what happens in the clip\" out.mp4"
echo "Image:  bash $HERE/scripts/image.sh \"a prompt\" out.png"
echo "Video:  bash $HERE/scripts/video.sh \"a prompt\" out.mp4      (FIRST_FRAME=photo.jpg for image to video)"
