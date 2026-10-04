#!/bin/bash
# Text to image to video: Qwen-Image-2.1 makes the first frame, MiniMax H3 animates it with sound.
#   bash scripts/studio.sh "what the picture shows" "what happens in the clip" out.mp4
#   IMAGE_PROMPT_FILE=prompts/baker-image.txt VIDEO_PROMPT_FILE=prompts/baker-video.txt bash scripts/studio.sh "" "" out.mp4
# The image is written beside the clip as out.png. Environment: everything scripts/image.sh and scripts/video.sh
# take (PREFIX, QWEN_MODEL_DIR, H3_MODEL_DIR, TURBO, ADAPTER, WIDTH, HEIGHT, FRAMES, SEED, POINTS), plus
# IMAGE_SEED and VIDEO_SEED to set the two seeds apart, and RAW_PROMPT=1 to pass the video prompt untouched.
# WIDTH and HEIGHT must suit both models: multiples of 32, at most 768x1344 pixels in total (default 1344x768).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_PROMPT="${1:-}"; VIDEO_PROMPT="${2:-}"; OUT="${3:-outputs/studio.mp4}"
IMAGE="${OUT%.*}.png"
export WIDTH="${WIDTH:-1344}" HEIGHT="${HEIGHT:-768}"
[ -z "${VIDEO_PROMPT_FILE:-}" ] || VIDEO_PROMPT="$(cat "$VIDEO_PROMPT_FILE")"
[ -n "$VIDEO_PROMPT" ] || { echo "give a video prompt or VIDEO_PROMPT_FILE" >&2; exit 2; }
# MiniMax's own image-to-video prompts open by saying where the picture is used
LEAD="For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
case "$VIDEO_PROMPT" in "For the target video"*) ;; *) [ "${RAW_PROMPT:-0}" = 1 ] || VIDEO_PROMPT="$LEAD

$VIDEO_PROMPT";; esac
START=$(date +%s)
SEED="${IMAGE_SEED:-${SEED:-0}}" PROMPT_FILE="${IMAGE_PROMPT_FILE:-}" bash "$HERE/image.sh" "$IMAGE_PROMPT" "$IMAGE"
[ -s "$IMAGE" ] || { echo "no image was written" >&2; exit 1; }
MID=$(date +%s)
SEED="${VIDEO_SEED:-${SEED:-0}}" PROMPT_FILE="" FIRST_FRAME="$IMAGE" bash "$HERE/video.sh" "$VIDEO_PROMPT" "$OUT"
END=$(date +%s)
echo "[studio] image $((MID - START)) s, video $((END - MID)) s, total $((END - START)) s -> $IMAGE, $OUT"
