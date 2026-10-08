#!/bin/bash
# Text to video with FastH3 (FastVideo's distilled MiniMax H3). On an M5 with the native engine installed the passes
# run on TensorFold 1.0's Zig + Metal runtime; otherwise on the MLX engine's tile-sparse attention kernel.
#   bash scripts/fast.sh "a prompt" out.mp4                      # 480p, 8 passes
#   STEPS=4 bash scripts/fast.sh "a prompt" out.mp4              # 4, 8 or 20 passes
#   RES=720p bash scripts/fast.sh "a prompt" out.mp4             # 1280x720
#   UPSCALE=1 bash scripts/fast.sh "a prompt" out.mp4            # 480p generated, 2x decoder -> 1728x960
#   RES=720p UPSCALE=1 bash scripts/fast.sh "a prompt" out.mp4   # 720p generated, 2x decoder -> 2560x1440 (2K)
# Environment: STEPS (4, 8 or 20; default 8: what the checkpoint was trained for; 20 for prompts with several
# actions), FASTH3_ENGINE (zig or mlx, to force one), RES (480p or 720p), UPSCALE (1 for
# the 2x decoder), FRAMES (17n+5; 124 is 5 s), SEED, PROMPT_FILE, FASTH3_DIR, and what scripts/video.sh takes.
# FIRST_FRAME=image.png starts the clip from an image; for Qwen image -> FastH3 video in one go use
#   ENGINE=fasth3 STEPS=8 WIDTH=864 HEIGHT=480 bash scripts/studio.sh "the picture" "what happens" out.mp4
# Needs `bash oneshot-setup.sh --fasth3` once (70 GB).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "${STEPS:=8}" in 4|8|20) ;; *) echo "STEPS is 4, 8 or 20, got $STEPS" >&2; exit 2;; esac
# 720 is not a multiple of 32, which the model needs: 720p is generated 736 rows tall and cropped by 8 top and bottom
case "${RES:-480p}:${UPSCALE:-0}" in
  480p:0) export WIDTH=864 HEIGHT=480;;
  480p:1) export X2=1 WIDTH=1728 HEIGHT=960;;
  720p:0) export WIDTH=1280 HEIGHT=736 CROP=1280x720;;
  720p:1) export X2=1 WIDTH=2560 HEIGHT=1472 CROP=2560x1440;;
  *) echo "RES is 480p or 720p and UPSCALE 0 or 1, got ${RES:-} and ${UPSCALE:-}" >&2; exit 2;;
esac
START=$(date +%s)
ENGINE=fasth3 STEPS="$STEPS" bash "$HERE/video.sh" "${1:-}" "${2:-outputs/fast.mp4}"
echo "[studio] FastH3 $STEPS passes, ${RES:-480p}$([ "${UPSCALE:-0}" = 1 ] && echo " + 2x decoder"), total $(( $(date +%s) - START )) s -> ${2:-outputs/fast.mp4}"
