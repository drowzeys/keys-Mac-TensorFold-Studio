#!/bin/bash
# TensorFold Studio on an NVIDIA DGX Spark (GB10, 128 GB, CUDA 13): one-shot install.
#   bash oneshot-setup-spark.sh               # everything, then a 5 second test clip
#   bash oneshot-setup-spark.sh --image-only  # Qwen-Image-2.1 only (26 GB)
#   bash oneshot-setup-spark.sh --no-render   # skip the test clip
#   bash oneshot-setup-spark.sh --verify      # check an existing install, download nothing
# What it installs under $PREFIX (default ~/.local/opt/tensorfold-studio):
#   venv/       Python 3.12, PyTorch for CUDA 13, ComfyUI's requirements, the web app's
#   ComfyUI/    pinned; its own flash and block-sparse attention kernels are what the renders use
# and under $SPARK_MODELS (default ~/spark-studio-models), 103 GB:
#   FastH3 8-Step V2 for ComfyUI, bf16 transformer 44 GB, H3 text encoder (8-bit) 27 GB, video and audio decoders 6 GB
#   Qwen-Image-2.1 for ComfyUI, 14 GB, its text encoder (8-bit) 9 GB, decoder, and the Viggle turbo adapter 1.4 GB
# Licenses: Qwen-Image-2.1 and the Viggle adapter are non-commercial (Qwen Research License); FastH3 and the H3
# parts are under the MiniMax H3 Community License (territory limits).
# Credit: FastVideo (Hao AI Lab), MiniMax, the Qwen team, Viggle, ComfyUI, Ash Hart (TensorFold, whose Studio this is
# the CUDA build of).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${PREFIX:-$HOME/.local/opt/tensorfold-studio}"
MODELS="${SPARK_MODELS:-$HOME/spark-studio-models}"
COMFY_COMMIT=a4b5a045e56fc334903db8457b728b64e006119c
IMAGE_ONLY=0; RENDER=1; VERIFY=0
for arg in "$@"; do
  case "$arg" in
    --image-only) IMAGE_ONLY=1;; --no-render) RENDER=0;; --verify) VERIFY=1; RENDER=0;;
    *) echo "unknown option $arg" >&2; exit 2;;
  esac
done
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
say() { printf '==> %s\n' "$*"; }
ok() { printf '  \342\234\223 %s\n' "$*"; }
die() { printf '  \342\234\227 %s\n' "$*" >&2; exit 1; }

say "Preflight"
[ "$(uname -s)" = Linux ] || die "this is the DGX Spark installer; on a Mac run oneshot-setup.sh"
command -v nvidia-smi >/dev/null || die "no nvidia-smi: this needs an NVIDIA GPU with its driver"
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
MEM_GB=$(awk '/MemTotal/{print int($2/1048576)}' /proc/meminfo)
ok "$GPU, $(uname -m), ${MEM_GB} GB"
case "$GPU" in *GB10*) ;; *) echo "  ! measured only on a DGX Spark (GB10, 128 GB unified memory); other GPUs are untested";; esac
[ "$MEM_GB" -ge 100 ] || [ "$IMAGE_ONLY" = 1 ] || die "video needs about 90 GB of memory at 1344x768 (128 GB machine); this one has ${MEM_GB} GB. --image-only fits."
for tool in git ffmpeg; do command -v "$tool" >/dev/null || die "install $tool first (sudo apt install $tool)"; done
if ! command -v uv >/dev/null; then
  [ "$VERIFY" = 0 ] || die "no uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
fi
NEED=$([ "$IMAGE_ONLY" = 1 ] && echo 40 || echo 125)
if [ "$VERIFY" = 0 ]; then
  mkdir -p "$PREFIX" "$MODELS"
  FREE=$(df -BG --output=avail "$MODELS" | tail -1 | tr -dc 0-9)
  HAVE=$(du -sBG "$MODELS" 2>/dev/null | cut -f1 | tr -dc 0-9)
  [ $((FREE + HAVE)) -ge "$NEED" ] || die "$MODELS has ${FREE} GB free; the install needs about ${NEED} GB"
fi

say "Environment: Python 3.12 and PyTorch for CUDA 13 -> $PREFIX/venv"
if [ ! -x "$PREFIX/venv/bin/python" ]; then
  [ "$VERIFY" = 0 ] || die "no environment at $PREFIX/venv"
  uv venv "$PREFIX/venv" --python 3.12 --python-preference only-managed --seed -q
fi
PY="$PREFIX/venv/bin/python"
pipi() { UV_TORCH_BACKEND=cu130 uv pip install -q --python "$PY" "$@"; }
if ! "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  [ "$VERIFY" = 0 ] || die "PyTorch does not see the GPU"
  pipi torch torchvision torchaudio
fi
"$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || die "PyTorch does not see the GPU"
ok "torch $("$PY" -c 'import torch; print(torch.__version__, "on", torch.cuda.get_device_name(0))')"

say "ComfyUI @ ${COMFY_COMMIT:0:8} -> $PREFIX/ComfyUI"
if [ "$(git -C "$PREFIX/ComfyUI" rev-parse HEAD 2>/dev/null)" != "$COMFY_COMMIT" ]; then
  [ "$VERIFY" = 0 ] || die "ComfyUI is missing or not at the pinned commit"
  [ -d "$PREFIX/ComfyUI/.git" ] || git clone -q https://github.com/comfyanonymous/ComfyUI "$PREFIX/ComfyUI"
  git -C "$PREFIX/ComfyUI" fetch -q origin "$COMFY_COMMIT" 2>/dev/null || git -C "$PREFIX/ComfyUI" fetch -q origin
  git -C "$PREFIX/ComfyUI" checkout -q "$COMFY_COMMIT"
fi
if [ "$VERIFY" = 0 ]; then
  pipi -r "$PREFIX/ComfyUI/requirements.txt" huggingface_hub fastapi uvicorn python-multipart pillow
  rm -rf "$PREFIX/ComfyUI/custom_nodes/fasth3_sparse"
  cat > "$PREFIX/ComfyUI/extra_model_paths.yaml" <<YAML
studio:
  base_path: $MODELS
  diffusion_models: |
    FastH3-Comfy/diffusion_models
    Qwen-Image-2.1-Comfy/diffusion_models
  text_encoders: |
    FastH3-Comfy/text_encoders
    Qwen-Image-2.1-Comfy/text_encoders
  vae: |
    FastH3-Comfy/vae
    Qwen-Image-2.1-Comfy/vae
  loras: loras
YAML
fi
"$PY" -c "import comfy_kitchen, aiohttp, fastapi, uvicorn, PIL" || die "the environment is missing packages"
ok "ComfyUI and its requirements"

# fetch <repo> <folder under $MODELS> <file in the repo>...: download what is not there yet
fetch() {
  local repo="$1" folder="$2" missing=(); shift 2
  for file in "$@"; do [ -s "$MODELS/$folder/$file" ] || missing+=("$file"); done
  [ ${#missing[@]} = 0 ] && return 0
  [ "$VERIFY" = 0 ] || die "missing in $MODELS/$folder: ${missing[*]}"
  "$PREFIX/venv/bin/hf" download "$repo" "${missing[@]}" --local-dir "$MODELS/$folder" >/dev/null
  for file in "${missing[@]}"; do [ -s "$MODELS/$folder/$file" ] || die "$file did not download"; done
}

say "Qwen-Image-2.1 for ComfyUI (26 GB) -> $MODELS"
fetch Comfy-Org/Qwen-Image-2.1 Qwen-Image-2.1-Comfy diffusion_models/qwen_image_2.1_bf16.safetensors \
  text_encoders/qwen3vl_8b_int8_convrot.safetensors vae/qwen_image_2.1_vae_bf16.safetensors
fetch Viggle/Qwen-Image-2.1-viggle-turbo loras Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r256.safetensors
ok "Qwen-Image-2.1, its text encoder and decoder, Viggle turbo adapter"

if [ "$IMAGE_ONLY" = 0 ]; then
  say "FastH3 8-Step V2 for ComfyUI (77 GB) -> $MODELS"
  fetch FastVideo/FastVideo-FastH3-Comfy FastH3-Comfy diffusion_models/fastvideo_fasth3_8step_v2_pruned_bf16.safetensors \
    text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors vae/minimax_h3_video_vae_fp16.safetensors \
    vae/minimax_h3_audio_vae_fp32.safetensors
  ok "FastH3 transformer (bf16), H3 text encoder (8-bit), video and audio decoders"
fi

if [ "$IMAGE_ONLY" = 0 ]; then
  say "TensorFold's H3 engine for CUDA -> $PREFIX/tf-h3"
  if [ "$VERIFY" = 1 ]; then
    [ -s "$PREFIX/tf-h3/libtf_h3.so" ] && ok "libtf_h3.so" || echo "  not built: clips render on ComfyUI's own blocks (slower)"
  elif PREFIX="$PREFIX" bash "$HERE/spark/build-engine.sh"; then
    ok "libtf_h3.so (FastH3's blocks in int8 on the tensor cores)"
  else
    echo "  the engine did not build: clips will render on ComfyUI's own blocks (slower). Fix and rerun spark/build-engine.sh"
  fi
fi

if [ "$RENDER" = 1 ]; then
  say "Test render"
  mkdir -p "$HERE/outputs"
  PREFIX="$PREFIX" bash "$HERE/scripts/image.sh" "A lighthouse on a rocky coast at sunrise, waves breaking, warm light." "$HERE/outputs/test.png"
  ok "outputs/test.png"
  if [ "$IMAGE_ONLY" = 0 ]; then
    PREFIX="$PREFIX" bash "$HERE/scripts/video.sh" 'A woman in a yellow raincoat stands on a pier, looks at the camera and says in a clear voice: "The tide is coming in." Gulls and waves.' "$HERE/outputs/test.mp4"
    ok "outputs/test.mp4 (864x480, 124 frames, 8 passes)"
  fi
fi
echo "DONE. Web app: bash $HERE/scripts/app.sh   (http://127.0.0.1:7870)"
echo "      Image:   bash $HERE/scripts/image.sh \"a prompt\" out.png"
echo "      Video:   bash $HERE/scripts/video.sh \"a prompt\" out.mp4"
echo "      Free the models' memory: bash $HERE/spark/comfy.sh stop"
