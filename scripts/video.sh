#!/bin/bash
# Render one clip with MiniMax H3 on TensorFold (Turbo adapter and int8 kernels by default).
#   bash scripts/video.sh "a prompt" out.mp4
#   PROMPT_FILE=prompts/black-mirror-scene.txt bash scripts/video.sh "" out.mp4
#   ADAPTER=none POINTS=21 bash scripts/video.sh "a prompt" out.mp4     # 20 steps, no adapter
#   FIRST_FRAME=photo.jpg bash scripts/video.sh "what happens next" out.mp4   # image to video
#   X2=1 WIDTH=1344 HEIGHT=768 bash scripts/video.sh "a prompt" out.mp4   # generate at 672x384, decode at 2x
# Environment: PREFIX, H3_MODEL_DIR, ADAPTER (a file, or none), FIRST_FRAME (an image the clip starts from; it is
# stretched onto the canvas, so match its aspect ratio), WIDTH, HEIGHT, FRAMES (17n+5), SEED,
# POINTS (sigma points: one more than the forwards), EXTRA (extra flags for h3_generate.py).
# X2=1 makes WIDTH x HEIGHT the size of the finished clip: the model generates at half of each and the 2x decoder
# ($X2_VAE) doubles it, so both must be multiples of 64 (default 1344x768). CROP=WxH centre-crops the frames before the clip is written.
set -euo pipefail
PREFIX="${PREFIX:-$HOME/.local/opt/tensorfold-studio}"
H3_MODEL_DIR="${H3_MODEL_DIR:-$HOME/h3-models/MiniMax-H3}"
ADAPTER="${ADAPTER:-$PREFIX/adapters/lightx2v_v1.0_768p_ourlayout.safetensors}"
X2_VAE="${X2_VAE:-$PREFIX/adapters/MiniMax-H3-X2-Detail-v1.safetensors}"
PROMPT="${1:-}"; OUT="${2:-outputs/h3.mp4}"
mkdir -p "$(dirname "$OUT")"
export PATH="/opt/homebrew/bin:$PATH" PYTHONPATH="$PREFIX/minimax-h3-mlx"
if [ "${X2:-0}" = 1 ]; then
  W="${WIDTH:-1344}"; H="${HEIGHT:-768}"
  [ $((W % 64)) = 0 ] && [ $((H % 64)) = 0 ] || { echo "with X2=1, WIDTH and HEIGHT must be multiples of 64, got ${W}x${H}" >&2; exit 2; }
  [ -f "$X2_VAE" ] || { echo "no 2x decoder at $X2_VAE (run oneshot-setup.sh, or set X2_VAE)" >&2; exit 2; }
  W=$((W / 2)); H=$((H / 2))
else
  W="${WIDTH:-864}"; H="${HEIGHT:-480}"
fi
ARGS=("$H3_MODEL_DIR" -o "$OUT" --width "$W" --height "$H" --frames "${FRAMES:-124}"
      --seed "${SEED:-0}" --points "${POINTS:-4}" --int8-mlp --int8-qkv --int8-out)
[ "${X2:-0}" != 1 ] || ARGS+=(--upscale-vae "$X2_VAE")
[ "$ADAPTER" = none ] || ARGS+=(--lora "$ADAPTER")
[ -z "${FIRST_FRAME:-}" ] || ARGS+=(--first-frame "$FIRST_FRAME")
if [ -n "${PROMPT_FILE:-}" ]; then ARGS+=(--prompt-file "$PROMPT_FILE"); else ARGS+=(--prompt "$PROMPT"); fi
[ -z "${CROP:-}" ] || ARGS+=(--crop "$CROP")
# shellcheck disable=SC2086
exec "$PREFIX/venv/bin/python" "$PREFIX/h3_generate.py" "${ARGS[@]}" ${EXTRA:-}
