#!/bin/bash
# Render one clip with MiniMax H3 on TensorFold.
# Standard: the Turbo adapter makes the picture in 5 passes, then the model without the adapter makes the sound again
# against the finished picture (20 audio steps). QUALITY=high: 20 steps without the adapter, fast recipe. QUALITY=full: plain 20 steps.
#   bash scripts/video.sh "a prompt" out.mp4
#   PROMPT_FILE=prompts/black-mirror-scene.txt bash scripts/video.sh "" out.mp4
#   QUALITY=high bash scripts/video.sh "a prompt" out.mp4               # 20 steps, no adapter, fast recipe (QUALITY=full for plain)
#   FIRST_FRAME=photo.jpg bash scripts/video.sh "what happens next" out.mp4   # image to video
#   X2=1 WIDTH=1344 HEIGHT=768 bash scripts/video.sh "a prompt" out.mp4   # generate at 672x384, decode at 2x
#   ENGINE=fasth3 STEPS=8 bash scripts/video.sh "a prompt" out.mp4       # FastH3; see scripts/fast.sh
# Environment: PREFIX, H3_MODEL_DIR, ADAPTER (a file, or none), FIRST_FRAME (an image the clip starts from; it is
# stretched onto the canvas, so match its aspect ratio), WIDTH, HEIGHT, FRAMES (17n+5), SEED,
# QUALITY (standard, high or full), POINTS (sigma points, one more than the passes: default 6 with the adapter, 21
# without), REVOICE (audio steps made again without the adapter after an adapter render: default 20, 0 keeps the
# adapter's own sound), AUDIO_EQ / AUDIO_BASS / AUDIO_BASS_TARGET (see below), AUDIO_SHIFT (experimental: the audio schedule's shift,
# 3 in the model; lower values add bass but roughen speech), EXTRA (extra flags for h3_generate.py).
# ENGINE=fasth3 swaps the transformer for FastVideo's distilled FastH3 ($FASTH3_DIR, fetched by
# `oneshot-setup.sh`) with its sparse attention on TensorFold's tile kernel. STEPS is the number of passes:
# 8 is what the checkpoint was trained for, 4 is faster and softer, 20 slower with more texture. FIRST_FRAME works
# too, though FastVideo trained FastH3 on text to video only.
# FastH3's passes run on TensorFold 1.0's native Zig + Metal runtime ($PREFIX/zig-engine) when it is installed and
# the chip has tensor units (M5 or later), with or without FIRST_FRAME; a machine without the engine uses the MLX
# engine as before. FASTH3_ENGINE=zig or mlx forces one.
# X2=1 makes WIDTH x HEIGHT the size of the finished clip: the model generates at half of each and the 2x decoder
# ($X2_VAE) doubles it, so both must be multiples of 64 (default 1344x768). CROP=WxH centre-crops the frames before the clip is written.
set -euo pipefail
PREFIX="${PREFIX:-$HOME/.local/opt/tensorfold-studio}"
H3_MODEL_DIR="${H3_MODEL_DIR:-$HOME/h3-models/MiniMax-H3}"
# QUALITY=high: 20 steps without the adapter, with the fast recipe (the first four and last two steps in full, a
# velocity cache and attention reuse in between: about twice as fast as plain). QUALITY=full: plain 20 steps.
case "${QUALITY:-standard}" in high) ADAPTER=none; RECIPE="--step-cache 0.05 --attention-every 2";; full) ADAPTER=none; RECIPE="";; *) RECIPE="";; esac
FASTH3_DIR="${FASTH3_DIR:-$HOME/h3-models/FastH3-8-Step-V2}"
# MiniMax H3's own transformer (the Turbo adapter and the 20-step qualities run on it) is an optional part of the
# install: oneshot-setup.sh --turbo. Without it the engine is FastH3, which needs only H3's text encoder and decoders.
TURBO_OK=0; ls "$H3_MODEL_DIR"/FL2VA/transformer/*.safetensors >/dev/null 2>&1 && TURBO_OK=1
ENGINE="${ENGINE:-$([ "$TURBO_OK" = 1 ] && echo h3 || echo fasth3)}"; STEPS="${STEPS:-8}"
[ "$ENGINE" = fasth3 ] || [ "$TURBO_OK" = 1 ] || { echo "MiniMax H3's transformer is not installed (oneshot-setup.sh --turbo adds it, 128 GB of memory): use ENGINE=fasth3" >&2; exit 2; }
if [ "$ENGINE" = fasth3 ]; then
  ADAPTER=none; RECIPE=""
  [ -f "$FASTH3_DIR/fastvideo_inference.json" ] || { echo "no FastH3 checkpoint at $FASTH3_DIR (run oneshot-setup.sh, or set FASTH3_DIR)" >&2; exit 2; }
  case "$STEPS" in ''|*[!0-9]*|0) echo "STEPS must be a positive number of passes, got $STEPS" >&2; exit 2;; esac
fi
ADAPTER="${ADAPTER:-$PREFIX/adapters/lightx2v_v1.0_768p_ourlayout.safetensors}"
# sigma points are one more than the passes: 5 passes with the adapter, 20 without
if [ "$ADAPTER" = none ]; then POINTS="${POINTS:-21}"; REVOICE="${REVOICE:-0}"; else POINTS="${POINTS:-6}"; REVOICE="${REVOICE:-20}"; fi
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
      --seed "${SEED:-0}" --points "$POINTS" --int8-mlp --int8-qkv --int8-out)
[ "$ENGINE" != fasth3 ] || ARGS+=(--fasth3 "$FASTH3_DIR" --fasth3-steps "$STEPS")
[ "$REVOICE" = 0 ] || ARGS+=(--revoice "$REVOICE")
[ "${X2:-0}" != 1 ] || ARGS+=(--upscale-vae "$X2_VAE")
if [ "$ENGINE" = fasth3 ]; then
  # which engine runs the passes, and why
  ZIG_ENGINE="${ZIG_ENGINE:-$PREFIX/zig-engine}"; WHY=""
  [ -x "$ZIG_ENGINE/tf-h3-dit" ] && [ -f "$PREFIX/zig_fasth3.py" ] || WHY="the native engine is not installed"
  [ -n "$WHY" ] || sysctl -n machdep.cpu.brand_string 2>/dev/null | grep -Eq 'Apple M([5-9]|[1-9][0-9])' || WHY="this chip has no tensor units"
  [ -n "$WHY" ] || [ -z "${AUDIO_SHIFT:-}${EXTRA:-}" ] || WHY="AUDIO_SHIFT and EXTRA are MLX options"
  case "${FASTH3_ENGINE:-auto}" in
    mlx) WHY="FASTH3_ENGINE=mlx";;
    zig) [ -z "$WHY" ] || { echo "FASTH3_ENGINE=zig, but $WHY" >&2; exit 2; };;
    auto) ;;
    *) echo "FASTH3_ENGINE is zig or mlx, got $FASTH3_ENGINE" >&2; exit 2;;
  esac
  if [ -z "$WHY" ]; then
    ZARGS=("$H3_MODEL_DIR" --fasth3 "$FASTH3_DIR" --engine "$ZIG_ENGINE" -o "$OUT" --width "$W" --height "$H"
           --frames "${FRAMES:-124}" --seed "${SEED:-0}" --steps "$STEPS")
    [ "${X2:-0}" != 1 ] || ZARGS+=(--upscale-vae "$X2_VAE")
    [ -z "${CROP:-}" ] || ZARGS+=(--crop "$CROP")
    [ -z "${FIRST_FRAME:-}" ] || ZARGS+=(--first-frame "$FIRST_FRAME")
    if [ -n "${PROMPT_FILE:-}" ]; then ZARGS+=(--prompt-file "$PROMPT_FILE"); else ZARGS+=(--prompt "$PROMPT"); fi
    exec "$PREFIX/venv/bin/python" "$PREFIX/zig_fasth3.py" "${ZARGS[@]}"
  fi
  echo "[tensorfold] engine: mlx ($WHY)"
fi
[ "$ADAPTER" = none ] || ARGS+=(--lora "$ADAPTER")
[ -z "${AUDIO_SHIFT:-}" ] || ARGS+=(--audio-shift "$AUDIO_SHIFT")
[ -z "${FIRST_FRAME:-}" ] || ARGS+=(--first-frame "$FIRST_FRAME")
if [ -n "${PROMPT_FILE:-}" ]; then ARGS+=(--prompt-file "$PROMPT_FILE"); else ARGS+=(--prompt "$PROMPT"); fi
[ -z "${CROP:-}" ] || ARGS+=(--crop "$CROP")
# shellcheck disable=SC2086
"$PREFIX/venv/bin/python" "$PREFIX/h3_generate.py" "${ARGS[@]}" $RECIPE ${EXTRA:-}
# Few-step audio varies a lot from take to take, and some takes come out thin (little 120-300 Hz). With an adapter,
# the uncompressed track the render leaves beside the clip is measured: when the bass band is under the level of a
# full take, a low shelf lifts it by what is missing (6 dB at most) and a limiter guards the peaks. A take that is
# already full is left exactly as rendered. The picture is never re-encoded. AUDIO_EQ=off skips this;
# AUDIO_BASS=<dB> forces a gain; AUDIO_BASS_TARGET sets the bass to mid-band energy ratio aimed for (default 0.5).
WAV="${OUT%.*}.wav"
if [ "$ADAPTER" != none ] && [ "${AUDIO_EQ:-on}" != off ] && [ -s "$WAV" ]; then
  GAIN="${AUDIO_BASS:-$("$PREFIX/venv/bin/python" - "$WAV" "${AUDIO_BASS_TARGET:-0.5}" <<'PY'
import sys, wave
import numpy as np
with wave.open(sys.argv[1]) as handle:
    rate = handle.getframerate()
    audio = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16).astype(np.float32)
    audio = audio.reshape(-1, handle.getnchannels()).mean(axis=1)
power = np.abs(np.fft.rfft(audio * np.hanning(len(audio)))) ** 2
freq = np.fft.rfftfreq(len(audio), 1 / rate)
bass = power[(freq >= 120) & (freq < 300)].sum()
mid = power[(freq >= 300) & (freq < 4000)].sum()
ratio = bass / max(mid, 1e-20)
gain = float(np.clip(10 * np.log10(float(sys.argv[2]) / max(ratio, 1e-6)), 0.0, 6.0))
print(f"{gain:.1f}")
PY
)}"
  if [ "$(printf '%.0f' "$(echo "$GAIN * 10" | bc)")" -ge 10 ]; then
    FILTER="bass=g=$GAIN:f=250:w=0.6,alimiter=limit=0.95:level=disabled"
    ffmpeg -v error -y -i "$OUT" -i "$WAV" -map 0:v -map 1:a -c:v copy -af "$FILTER" -c:a aac -b:a 256k "${OUT%.*}.eq.mp4"
    mv "${OUT%.*}.eq.mp4" "$OUT"
    echo "[tensorfold] thin take: bass lifted by $GAIN dB"
  else
    echo "[tensorfold] audio left as rendered (bass is full)"
  fi
fi
