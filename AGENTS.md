# Agent one-shot — TensorFold Studio: Qwen-Image-2.1 + MiniMax H3 (MLX, Apple Silicon)

**Repo:** https://github.com/drowzeys/keys-Mac-TensorFold-Studio  
**Carrier image:** `ghcr.io/drowzeys/keys-mac-tensorfold-studio:1.1` (not a runtime; TensorFold wheel + lock + render scripts)  
**Engine:** TensorFold 0.6.5 + H3 and Qwen-Image families @ `drowzeys/TensorFold` `52960a1c`, own venv, int8 tensor-unit kernels

```bash
git clone https://github.com/drowzeys/keys-Mac-TensorFold-Studio.git
cd keys-Mac-TensorFold-Studio
brew install python@3.11 uv ffmpeg
bash oneshot-setup.sh               # both models (177 GB of weights) and a test clip; --image-only for 33 GB
bash scripts/studio.sh "the picture" "what happens in the clip" out.mp4   # text -> image -> video with sound
bash scripts/scout.sh "the picture" 6 scouts/x      # six candidates + scouts/x/sheet.jpg, 27 s
IMAGE_FILE=scouts/x/scout_s3.png QHD=1 FRAMES=192 bash scripts/studio.sh "" "what happens" final.mp4   # 2560x1440
bash scripts/image.sh "a prompt" out.png
FIRST_FRAME=photo.jpg bash scripts/video.sh "what happens next" out.mp4
```

Rules:

- This is not a server. `tensorfold serve` routes neither model; render with the scripts.
- Qwen-Image-2.1 and the Viggle adapter are non-commercial (Qwen Research License). MiniMax H3 has territory limits.
  Do not redistribute weights, and say so when asked to use the outputs commercially.
- Image sizes are multiples of 16. For the studio chain both models must accept the canvas: multiples of 32, at most
  768x1344 pixels in total. Video frames must be `17n + 5` (56, 73, 90, 124, 192, 243, 362).
- Preferred flow: scout at 1280x736, let the person pick, then `IMAGE_FILE=... TWOK=1` (2048x1152, 2.5 min) or `QHD=1` (2560x1440, 6 min) for the 2K final; `X2=1` for a
  one-minute 1344x768 draft. `X2=1` needs WIDTH and HEIGHT in multiples of 64. 720-row canvases are not valid for the
  video model: use the QHD preset, which generates 736 rows and crops.
- The 2x decoder is an upscale, softer than a native generation at the same size. Do not describe it as added detail.
- The image turbo adapter runs on six fixed nodes (`--nodes 1.0,0.9375,0.875,0.75,0.5,0.25`); do not change the step
  count with it. `TURBO=0` runs the base model at 40 steps.
- Merge adapters with the tools (`--lora`), never by hand into bfloat16.
- The fast path is M5-only (Metal 4 tensor operations).
- The two models never load together; `studio.sh` runs them one after the other.
- mflux and minimax-h3-mlx are pinned by commit and used at run time. Do not unpin.

**Credits:** keep CREDITS.md in sync. Qwen team (Qwen-Image-2.1), MiniMax (H3), Ash Hart (TensorFold), antirez (h3.c),
RobZombAI (H3MLX), mrbizarro (minimax-h3-mlx / Phosphene), Filip Strand and mflux contributors, Viggle, speach1sdef178, LightX2V,
NVIDIA Research, FastVideo, Apple MLX. Authors only — no credit hyperlinks.
