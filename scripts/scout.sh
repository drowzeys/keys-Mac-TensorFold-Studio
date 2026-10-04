#!/bin/bash
# Scout images: several Qwen-Image-2.1 pictures of one prompt from one model load, and a contact sheet to choose from.
#   bash scripts/scout.sh "what the picture shows" [count] [folder]
#   PROMPT_FILE=prompts/baker-image.txt bash scripts/scout.sh "" 6 scouts/baker
# Writes <folder>/scout_s<seed>.png and <folder>/sheet.jpg. Then animate the one you like:
#   IMAGE_FILE=<folder>/scout_s3.png QHD=1 bash scripts/studio.sh "" "what happens in the clip" out.mp4
# Environment: everything scripts/image.sh takes. Seeds run from SEED (default 1) upwards. The default size, 1280x736,
# is what the video model starts from for a 2560x1440 clip, so a scout is exactly the frame the clip will open on.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${PREFIX:-$HOME/.local/opt/tensorfold-studio}"
PROMPT="${1:-}"; COUNT="${2:-4}"; DIR="${3:-outputs/scout}"
FIRST="${SEED:-1}"
SEEDS=$(seq -s, "$FIRST" $((FIRST + COUNT - 1)))
export WIDTH="${WIDTH:-1280}" HEIGHT="${HEIGHT:-736}"
mkdir -p "$DIR"
EXTRA="${EXTRA:-} --seeds ${SEEDS%,}" bash "$HERE/image.sh" "$PROMPT" "$DIR/scout.png"
"$PREFIX/venv/bin/python" - "$DIR" "${SEEDS%,}" <<'PY'
import sys
from pathlib import Path
from PIL import Image, ImageDraw
folder, seeds = Path(sys.argv[1]), sys.argv[2].split(",")
images = [(seed, Image.open(folder / f"scout_s{seed}.png").convert("RGB")) for seed in seeds]
columns = 2 if len(images) <= 4 else 3
width = 640
height = images[0][1].height * width // images[0][1].width
rows = -(-len(images) // columns)
sheet = Image.new("RGB", (columns * width, rows * height), "black")
for index, (seed, image) in enumerate(images):
    tile = image.resize((width, height))
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, 110, 26), fill="black")
    draw.text((8, 6), f"seed {seed}", fill="white")
    sheet.paste(tile, ((index % columns) * width, (index // columns) * height))
sheet.save(folder / "sheet.jpg", quality=90)
print(f"[studio] {len(images)} scouts in {folder}; contact sheet {folder / 'sheet.jpg'}")
PY
