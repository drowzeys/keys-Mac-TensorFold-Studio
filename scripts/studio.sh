#!/bin/bash
# Text to image to video: Qwen-Image-2.1 makes the first frame, MiniMax H3 animates it with sound.
#   bash scripts/studio.sh "what the picture shows" "what happens in the clip" out.mp4
#   IMAGE_PROMPT_FILE=prompts/baker-image.txt VIDEO_PROMPT_FILE=prompts/baker-video.txt bash scripts/studio.sh "" "" out.mp4
# The image is written beside the clip as out.png. Environment: everything scripts/image.sh and scripts/video.sh
# take (PREFIX, QWEN_MODEL_DIR, H3_MODEL_DIR, TURBO, ADAPTER, WIDTH, HEIGHT, FRAMES, SEED, POINTS), plus
# IMAGE_SEED and VIDEO_SEED to set the two seeds apart, RAW_PROMPT=1 to pass the video prompt untouched, and
# IMAGE_FILE to animate an image you already have (a scout from scripts/scout.sh) instead of making one.
# WIDTH and HEIGHT must suit both models: multiples of 32, at most 768x1344 pixels in total (default 1344x768).
# X2=1 generates the video at half of WIDTH and HEIGHT and decodes it at 2x (multiples of 64, up to 2688x1536); the
# image is made at full size unless IMAGE_WIDTH and IMAGE_HEIGHT say otherwise. QHD=1 is the 2560x1440 preset and
# TWOK=1 the faster 2048x1152 one.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_PROMPT="${1:-}"; VIDEO_PROMPT="${2:-}"; OUT="${3:-outputs/studio.mp4}"
IMAGE="${OUT%.*}.png"
# QHD=1: a 2560x1440 clip. The video is generated at 1280x736, decoded at 2x and centre-cropped by 16 rows top and
# bottom (720 is not a multiple of 32, which the video model needs). TWOK=1: a 2048x1152 clip, generated at 1024x576.
# With either, the image is made at the size the video model starts from, since a larger one would only be shrunk.
if [ "${QHD:-0}" = 1 ]; then export WIDTH=2560 HEIGHT=1472 X2=1 CROP=2560x1440 IMAGE_WIDTH="${IMAGE_WIDTH:-1280}" IMAGE_HEIGHT="${IMAGE_HEIGHT:-736}"; fi
if [ "${TWOK:-0}" = 1 ]; then export WIDTH=2048 HEIGHT=1152 X2=1 IMAGE_WIDTH="${IMAGE_WIDTH:-1024}" IMAGE_HEIGHT="${IMAGE_HEIGHT:-576}"; fi
export WIDTH="${WIDTH:-1344}" HEIGHT="${HEIGHT:-768}"
[ -z "${VIDEO_PROMPT_FILE:-}" ] || VIDEO_PROMPT="$(cat "$VIDEO_PROMPT_FILE")"
[ -n "$VIDEO_PROMPT" ] || { echo "give a video prompt or VIDEO_PROMPT_FILE" >&2; exit 2; }
# MiniMax's own image-to-video prompts open by saying where the picture is used
LEAD="For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
case "$VIDEO_PROMPT" in "For the target video"*) ;; *) [ "${RAW_PROMPT:-0}" = 1 ] || VIDEO_PROMPT="$LEAD

$VIDEO_PROMPT";; esac
START=$(date +%s)
if [ -n "${IMAGE_FILE:-}" ]; then
  # an image chosen earlier (a scout, or any picture) takes the place of the image step
  [ -s "$IMAGE_FILE" ] || { echo "no image at $IMAGE_FILE" >&2; exit 2; }
  IMAGE="$IMAGE_FILE"
else
  SEED="${IMAGE_SEED:-${SEED:-0}}" PROMPT_FILE="${IMAGE_PROMPT_FILE:-}" WIDTH="${IMAGE_WIDTH:-$WIDTH}" \
    HEIGHT="${IMAGE_HEIGHT:-$HEIGHT}" bash "$HERE/image.sh" "$IMAGE_PROMPT" "$IMAGE"
fi
[ -s "$IMAGE" ] || { echo "no image was written" >&2; exit 1; }
MID=$(date +%s)
SEED="${VIDEO_SEED:-${SEED:-0}}" PROMPT_FILE="" FIRST_FRAME="$IMAGE" bash "$HERE/video.sh" "$VIDEO_PROMPT" "$OUT"
END=$(date +%s)
echo "[studio] image $((MID - START)) s, video $((END - MID)) s, total $((END - START)) s -> $IMAGE, $OUT"
