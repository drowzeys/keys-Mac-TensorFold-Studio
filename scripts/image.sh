#!/bin/bash
# Make one image with Qwen-Image-2.1 on TensorFold.
#   bash scripts/image.sh "a prompt" out.png
#   TURBO=0 bash scripts/image.sh "a prompt" out.png        # 40 steps, no adapter
#   PROMPT_FILE=prompts/baker-image.txt bash scripts/image.sh "" out.png
# Environment: PREFIX, QWEN_MODEL_DIR, TURBO (1: Viggle turbo adapter on its six nodes; 0: base model),
# STEPS (base model only, default 40), WIDTH, HEIGHT (multiples of 16), SEED, EXTRA (extra flags).
set -euo pipefail
PREFIX="${PREFIX:-$HOME/.local/opt/tensorfold-studio}"
QWEN_MODEL_DIR="${QWEN_MODEL_DIR:-$HOME/qwen-models/Qwen-Image-2.1}"
TURBO_ADAPTER="${TURBO_ADAPTER:-$PREFIX/adapters/Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r256.safetensors}"
PROMPT="${1:-}"; OUT="${2:-outputs/image.png}"
mkdir -p "$(dirname "$OUT")"
export PATH="/opt/homebrew/bin:$PATH"
ARGS=("$QWEN_MODEL_DIR" -o "$OUT" --width "${WIDTH:-1344}" --height "${HEIGHT:-768}" --seed "${SEED:-0}")
if [ "${TURBO:-1}" = 1 ]; then
  ARGS+=(--lora "$TURBO_ADAPTER" --nodes 1.0,0.9375,0.875,0.75,0.5,0.25)
else
  ARGS+=(--steps "${STEPS:-40}")
fi
if [ -n "${PROMPT_FILE:-}" ]; then ARGS+=(--prompt-file "$PROMPT_FILE"); else ARGS+=(--prompt "$PROMPT"); fi
# shellcheck disable=SC2086
exec "$PREFIX/venv/bin/python" "$PREFIX/qwen_image_generate.py" "${ARGS[@]}" ${EXTRA:-}
