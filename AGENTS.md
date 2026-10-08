# Agent one-shot — TensorFold Studio: Qwen-Image-2.1 + MiniMax H3 (MLX, Apple Silicon)

**Repo:** https://github.com/drowzeys/keys-Mac-TensorFold-Studio  
**Carrier image:** `ghcr.io/drowzeys/keys-mac-tensorfold-studio:2.0` (not a runtime; TensorFold wheel + lock + render scripts + the native FastH3 engine)  
**Engine:** TensorFold 0.6.5 + H3 and Qwen-Image families @ `drowzeys/TensorFold` `a2068c03`, own venv, int8 tensor-unit kernels

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

## Install for a person and build the one-click app

Do these in order on the person's Mac, and report each result plainly.

1. Check the machine: `uname -m` is `arm64`, `sysctl -n machdep.cpu.brand_string` names an M5-family chip (others run
   without the int8 kernels, far slower: say so and ask before continuing), memory is 128 GB or more, and
   `df -h ~` shows 180 GB free (250 GB with FastH3). Stop and report if memory or disk falls short.
2. `brew install python@3.11 uv ffmpeg` (install Homebrew first only with the person's agreement).
3. Tell the person the download sizes and licenses before starting: Qwen-Image-2.1 33 GB (Qwen Research License,
   non-commercial), MiniMax H3 144 GB (MiniMax H3 Community License, territory limits), FastH3 70 GB (same license).
4. `bash oneshot-setup.sh --fasth3 --app` (drop `--fasth3` if they do not want it; `--app` does step 5 with a Desktop shortcut). It resumes if interrupted. It ends by
   rendering `outputs/test.mp4`; confirm that file exists and plays.
5. `bash scripts/make-app.sh` builds `~/Applications/TensorFold Studio.app`. If the models are not in the default
   folders, export `QWEN_MODEL_DIR`, `H3_MODEL_DIR`, `FASTH3_DIR` first: the app records them. Do not set
   `HOST=0.0.0.0` unless the person asks to reach it from other machines; there is no login.
6. `open "$HOME/Applications/TensorFold Studio.app"`, then check `curl -s http://127.0.0.1:7870/api/state` answers.
7. Tell the person: double-click TensorFold Studio in Applications; renders land in `~/TensorFoldStudio`; the log is
   `~/Library/Logs/TensorFoldStudio.log`; to stop it run the app's executable with `stop`.

The app is a launcher around this clone, unsigned and built locally, so Gatekeeper does not block it. Do not move
the clone afterwards without rebuilding the app. Do not sign, notarize or redistribute the built app.

Native engine (2.0): FastH3 text to video runs its passes in `$PREFIX/zig-engine/tf-h3-dit` (TensorFold 1.0 Zig +
Metal runtime, fork branch `h3-speed`), driven by `zig_fasth3.py`; `scripts/video.sh` chooses it on an M5 when the
clip has no first frame and prints `[tensorfold] engine: zig|mlx (why)`. `FASTH3_ENGINE=mlx|zig` forces one. Clips
from an image, image generation and the Turbo path are MLX. Do not apply the base-model audio step to FastH3 clips:
the owner compared it by ear and chose FastH3's own sound. Use 20 passes when a prompt has several actions.
`scripts/build-zig-engine.sh` builds the payload on a Mac (zig 0.17); never commit the binary.

FastH3 and long videos: `scripts/fast.sh` (STEPS=4|8|20, RES=480p|720p, UPSCALE=1); `ENGINE=fasth3` works with
`studio.sh` and `FIRST_FRAME` too, though FastVideo trained it on text to video only. Videos over 15 s are chains of
clips made by the web app's `long` job (each opens on the last frame of the one before); only an 18 s chain has been
rendered, so do not promise the look of a 30 minute one.

Web app: `bash scripts/app.sh` (http://127.0.0.1:7870; `HOST=0.0.0.0` to open it to the LAN, no login). It runs the
same scripts through a one-at-a-time queue and stores everything under `~/TensorFoldStudio`.

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
- Standard video = Turbo adapter, 5 passes, then the sound made again by the base model (`REVOICE=20`). Do not go back
  to the adapter's own sound (`REVOICE=0`) unless asked: it was judged poor by ear. `QUALITY=high` = 20 steps, no
  adapter, about three times slower. `POINTS` is passes plus one.
- Levels are handled after the render (no hard clip; bass lifted on thin takes; `AUDIO_EQ=off` to skip). Leave
  `AUDIO_SHIFT` alone.
- The image turbo adapter runs on six fixed nodes (`--nodes 1.0,0.9375,0.875,0.75,0.5,0.25`); do not change the step
  count with it. `TURBO=0` runs the base model at 40 steps.
- Merge adapters with the tools (`--lora`), never by hand into bfloat16.
- The fast path is M5-only (Metal 4 tensor operations).
- The two models never load together; `studio.sh` runs them one after the other.
- mflux and minimax-h3-mlx are pinned by commit and used at run time. Do not unpin.

**Credits:** keep CREDITS.md in sync. Qwen team (Qwen-Image-2.1), MiniMax (H3), Ash Hart (TensorFold), antirez (h3.c),
RobZombAI (H3MLX), mrbizarro (minimax-h3-mlx / Phosphene), Filip Strand and mflux contributors, Viggle, speach1sdef178, LightX2V,
NVIDIA Research, FastVideo, Apple MLX. Authors only — no credit hyperlinks.
