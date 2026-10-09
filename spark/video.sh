#!/bin/bash
# One clip with FastH3 on a DGX Spark (CUDA). scripts/video.sh hands over to this on Linux; same interface:
#   bash scripts/video.sh "a prompt" out.mp4
#   FIRST_FRAME=photo.jpg bash scripts/video.sh "what happens next" out.mp4
#   WIDTH=1344 HEIGHT=768 STEPS=8 FRAMES=124 SEED=3 bash scripts/video.sh "a prompt" out.mp4
# Environment: PREFIX, WIDTH, HEIGHT (multiples of 32, default 864x480), FRAMES (17n+5), SEED, STEPS (4, 8 or 20
# passes), FIRST_FRAME, PROMPT_FILE, CROP=WxH, SPARK_WEIGHTS (bf16, the default, or int8: no faster on a Spark, 22 GB
# less memory), SPARK_SPARSITY (0 is dense attention; unset picks by size).
# Not on the Spark: MiniMax H3 Turbo (ENGINE=h3, QUALITY) and the 2x decoder (X2, TWOK, QHD).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${PREFIX:-$HOME/.local/opt/tensorfold-studio}"
[ "${ENGINE:-fasth3}" = fasth3 ] || { echo "the Spark build renders with FastH3 only (ENGINE=fasth3)" >&2; exit 2; }
[ "${X2:-0}" != 1 ] || { echo "the 2x decoder is not part of the Spark build: use WIDTH and HEIGHT up to 1344x768" >&2; exit 2; }
[ -x "$PREFIX/venv/bin/python" ] || { echo "run oneshot-setup-spark.sh first: no environment at $PREFIX" >&2; exit 1; }
PROMPT="${1:-}"; OUT="${2:-outputs/h3.mp4}"
ARGS=(video -o "$OUT" --width "${WIDTH:-864}" --height "${HEIGHT:-480}" --frames "${FRAMES:-124}"
      --seed "${SEED:-0}" --steps "${STEPS:-8}")
[ -z "${FIRST_FRAME:-}" ] || ARGS+=(--first-frame "$FIRST_FRAME")
[ -z "${CROP:-}" ] || ARGS+=(--crop "$CROP")
[ -z "${SPARK_SPARSITY:-}" ] || ARGS+=(--sparsity "$SPARK_SPARSITY")
if [ -n "${PROMPT_FILE:-}" ]; then ARGS+=(--prompt-file "$PROMPT_FILE"); else ARGS+=(--prompt "$PROMPT"); fi
START=$(date +%s)
PREFIX="$PREFIX" "$PREFIX/venv/bin/python" "$HERE/spark_generate.py" "${ARGS[@]}"
echo "[studio] FastH3 ${STEPS:-8} passes, ${WIDTH:-864}x${HEIGHT:-480}, total $(( $(date +%s) - START )) s -> $OUT"
