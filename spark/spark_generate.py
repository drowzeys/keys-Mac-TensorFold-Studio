"""TensorFold Studio on a DGX Spark (CUDA): FastH3 video with sound and Qwen-Image-2.1 pictures through ComfyUI.

    spark_generate.py video --prompt "..." -o clip.mp4 [--first-frame photo.jpg] [--steps 8] [--frames 124]
    spark_generate.py image --prompt "..." -o picture.png [--seeds 1,2,3] [--base --steps 40]

It talks to a ComfyUI server that it starts on first use (spark/comfy.sh) and leaves running with the models
loaded, so the next render skips the load. `bash spark/comfy.sh stop` frees the memory.

Credit: FastVideo (Hao AI Lab) for FastH3 and its sparse attention, MiniMax for H3, the Qwen team for
Qwen-Image-2.1-Turbo, ComfyUI for the runtime.
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREFIX = Path(os.environ.get("PREFIX", Path.home() / ".local/opt/tensorfold-studio"))
URL = os.environ.get("SPARK_COMFY_URL", "http://127.0.0.1:8190")
STATE = PREFIX / "comfy.kind"
ENGINE_LIB = PREFIX / "tf-h3" / "libtf_h3.so"

FASTH3 = {"bf16": "fastvideo_fasth3_8step_v2_pruned_bf16.safetensors",
          "int8": "fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors"}
H3_TEXT = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
# ComfyUI's 8-bit video decoder: half the decode time of the fp16 one on a GB10 (14 s against 26 s for 124 frames of
# 864x480), the same picture to 42 to 44 dB. SPARK_VAE=minimax_h3_video_vae_fp16.safetensors selects the other.
H3_VAE = os.environ.get("SPARK_VAE", "minimax_h3_video_vae_int8_convrot.safetensors")
H3_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
QWEN = "qwen_image_2.1_turbo_bf16.safetensors"   # Qwen-Image-2.1-Turbo, the 8-step checkpoint
QWEN_BASE = "qwen_image_2.1_bf16.safetensors"    # the 40-step base model, only when it has been downloaded
QWEN_TEXT = "qwen3vl_8b_int8_convrot.safetensors"
QWEN_VAE = "qwen_image_2.1_vae_bf16.safetensors"
# the schedule saved with the Turbo checkpoint (model_index.json, sample_sigmas), then zero
TURBO_SIGMAS = "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0"


def call(path, data=None):
    request = urllib.request.Request(URL + path, data=json.dumps(data).encode() if data is not None else None,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as reply:
        body = reply.read()
    return json.loads(body) if body else {}


def up() -> bool:
    try:
        call("/system_stats")
        return True
    except OSError:
        return False


def ensure_server(kind: str) -> None:
    """Start ComfyUI if it is not running; unload the other family's models when the job kind changes."""

    if not up():
        print("[tensorfold] starting ComfyUI (first render after a start loads the models)", flush=True)
        subprocess.run(["bash", str(HERE / "comfy.sh"), "start"], check=True)
        for _ in range(180):
            if up():
                break
            time.sleep(2)
        else:
            sys.exit(f"ComfyUI did not answer at {URL}: see {PREFIX}/comfy.log")
    last = STATE.read_text().strip() if STATE.exists() else ""
    if last and last != kind:
        # the picture and video models together would not leave room for the render itself in 128 GB
        call("/free", {"unload_models": True, "free_memory": True})
    STATE.write_text(kind)


def upload(image: str) -> str:
    """Put an image in ComfyUI's input folder; returns the name LoadImage wants."""

    path = Path(image)
    name = f"studio_{uuid.uuid4().hex[:10]}{path.suffix.lower() or '.png'}"
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n").encode() + path.read_bytes() + (
            f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{boundary}--\r\n").encode()
    request = urllib.request.Request(URL + "/upload/image", data=body,
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(request, timeout=120) as reply:
        return json.loads(reply.read())["name"]


async def run(workflow: dict, sampler: str) -> str:
    """Queue a workflow and print the sampler's progress as `[tensorfold] step i/n` until it ends; returns its id.

    The progress socket is opened before the job is queued, so a job that ends at once (ComfyUI had it cached) is
    not missed.
    """

    import aiohttp

    client = uuid.uuid4().hex
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(URL.replace("http", "ws", 1) + f"/ws?clientId={client}", max_msg_size=0) as ws:
            prompt_id = call("/prompt", {"prompt": workflow, "client_id": client})["prompt_id"]
            stages, running, since = [], None, time.time()
            async for message in ws:
                if message.type != aiohttp.WSMsgType.TEXT:
                    continue
                event = json.loads(message.data)
                data = event.get("data", {})
                if data.get("prompt_id") not in (None, prompt_id):
                    continue
                if event["type"] == "progress" and str(data.get("node")) == sampler:
                    print(f"[tensorfold] step {data['value']}/{data['max']}", flush=True)
                elif event["type"] == "executing":
                    # SPARK_TIMING=1: how long each node of the workflow ran
                    if running is not None:
                        stages.append((workflow[running]["class_type"], time.time() - since))
                    running, since = (str(data["node"]) if data.get("node") is not None else None), time.time()
                    if running is None:
                        break
                elif event["type"] in ("execution_error", "execution_interrupted", "execution_success"):
                    break
    if os.environ.get("SPARK_TIMING") == "1":
        print("[tensorfold] stages: " + ", ".join(f"{name} {took:.1f} s" for name, took in stages if took >= 0.05), flush=True)
    return prompt_id


def render(workflow: dict, sampler: str) -> tuple[dict, float]:
    """Run a workflow; returns (the saved file's record, seconds)."""

    started = time.time()
    try:
        prompt_id = asyncio.run(run(workflow, sampler))
    except KeyboardInterrupt:
        call("/interrupt", {})
        raise
    while True:
        history = call(f"/history/{prompt_id}")
        if prompt_id in history and history[prompt_id].get("status", {}).get("completed") is not None:
            break
        time.sleep(1)
    record = history[prompt_id]
    if record["status"].get("status_str") != "success":
        errors = [m[1] for m in record["status"].get("messages", []) if m[0] == "execution_error"]
        detail = errors[0].get("exception_message", "").strip() if errors else json.dumps(record["status"])[:600]
        sys.exit(f"ComfyUI failed: {detail} (log: {PREFIX}/comfy.log)")
    for output in record["outputs"].values():
        for items in output.values():
            for item in items if isinstance(items, list) else []:
                if isinstance(item, dict) and item.get("filename"):
                    return item, time.time() - started
    sys.exit("ComfyUI finished without writing a file")


def fetch(item: dict, target: Path) -> None:
    query = urllib.parse.urlencode({"filename": item["filename"], "subfolder": item.get("subfolder", ""),
                                    "type": item.get("type", "output")})
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL + "/view?" + query, timeout=600) as reply:
        target.write_bytes(reply.read())


def prompt_of(args) -> str:
    text = Path(args.prompt_file).read_text() if args.prompt_file else args.prompt
    if not text or not text.strip():
        sys.exit("give --prompt or --prompt-file")
    return text.strip()


def video(args) -> None:
    for name, value, step in (("width", args.width, 32), ("height", args.height, 32)):
        if value % step:
            sys.exit(f"--{name} must be a multiple of {step}, got {value}")
    if (args.frames - 5) % 17:
        sys.exit(f"--frames must be 17n + 5 (124, 192, 243, 362), got {args.frames}")
    sparsity = args.sparsity
    if sparsity is None and args.engine == "tensorfold":
        sparsity = 0.8    # what FastH3 was trained with; the engine's sparse kernel is the faster one at every size
    if sparsity is None:
        # sparse attention pays off once the video rows dominate. Measured on a GB10, a pass at 864x480 takes
        # 10.5 s dense and 11.1 s sparse; at 1344x768, 33.4 s dense and 29.4 s sparse
        sparsity = 0.8 if args.width * args.height >= 700_000 else 0.0
    if args.engine == "tensorfold" and not ENGINE_LIB.exists():
        sys.exit(f"no TensorFold engine at {ENGINE_LIB}: run spark/build-engine.sh, or use --engine comfy")
    ensure_server("video")
    model = ["2", 0]
    w = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": FASTH3[args.weights], "weight_dtype": "default"}},
        # FastH3's schedule: video shift 10, audio shift 3 (fastvideo_inference.json)
        "2": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["1", 0], "shift_video": 10.0, "shift_audio": 3.0}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": H3_TEXT, "type": "minimax", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": H3_VAE}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": H3_AUDIO_VAE}},
        "6": {"class_type": "MiniMaxH3ImageToVideo",
              "inputs": {"clip": ["3", 0], "vae": ["4", 0], "prompt": prompt_of(args), "width": args.width,
                         "height": args.height, "length": args.frames}},
        "7": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["6", 0]}},
        "8": {"class_type": "RandomNoise", "inputs": {"noise_seed": args.seed}},
        "9": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "12": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["8", 0], "guider": ["11", 0], "sampler": ["9", 0], "sigmas": ["10", 0],
                          "latent_image": ["6", 1]}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["4", 0]}},
        "14": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["12", 0], "vae": ["5", 0]}},
        "15": {"class_type": "CreateVideo", "inputs": {"images": ["13", 0], "audio": ["14", 0], "fps": 24}},
        "16": {"class_type": "SaveVideo", "inputs": {"video": ["15", 0], "filename_prefix": f"studio/clip_{uuid.uuid4().hex[:8]}",
                                                     "format": "auto", "codec": "auto"}},
    }
    if args.first_frame:
        w["20"] = {"class_type": "LoadImage", "inputs": {"image": upload(args.first_frame)}}
        w["6"]["inputs"]["first_frame"] = ["20", 0]
    if args.engine == "tensorfold":
        # the 50 blocks of every pass on TensorFold's CUDA family, int8 on the tensor cores, with FastH3's own sparse
        # attention; ComfyUI keeps the text encoder, the sampler and the decoders
        w["22"] = {"class_type": "TensorFoldH3Blocks",
                   "inputs": {"model": model, "unet_name": FASTH3["bf16"], "sparsity": sparsity}}
        model = ["22", 0]
    elif sparsity > 0:
        # ComfyUI's compiled block-sparse kernel with FastVideo's selection: 3D video cubes, the top share kept per
        # query cube, FastH3's own gate weights for the coarse branch; text and audio rows stay exact
        w["21"] = {"class_type": "BlockSparseAttention",
                   "inputs": {"model": model, "selection": "vsa", "selection.keep_percent": round(100 * (1 - sparsity), 1),
                              "start_percent": 0.0, "end_percent": 1.0, "dense_blocks": "", "min_tokens": 0,
                              "extra_tokens": 0, "sink_conditioning": "exact_kv_and_rows", "verbose": False}}
        model = ["21", 0]
    w["10"] = {"class_type": "BasicScheduler", "inputs": {"model": model, "scheduler": "simple", "steps": args.steps,
                                                          "denoise": 1.0}}
    w["11"] = {"class_type": "CFGGuider", "inputs": {"model": model, "positive": ["6", 0], "negative": ["7", 0],
                                                     "cfg": 1.0}}
    attention = f"sparse attention {sparsity}" if sparsity > 0 else "dense attention"
    runs = "TensorFold int8 blocks in ComfyUI" if args.engine == "tensorfold" else f"ComfyUI, FastH3 {args.weights}"
    print(f"[tensorfold] engine: cuda ({runs}, {attention}), generating "
          f"{args.width}x{args.height}, {args.frames} frames", flush=True)
    item, seconds = render(w, "12")
    target = Path(args.output)
    fetch(item, target)
    if args.crop:
        width, height = args.crop.lower().split("x")
        cropped = target.with_suffix(".crop.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(target), "-vf", f"crop={width}:{height}", "-c:v",
                        "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "copy",
                        str(cropped)], check=True)
        cropped.replace(target)
    print("[tensorfold] " + json.dumps({"output": str(target), "engine": "cuda", "forwards": args.steps,
                                        "total_s": round(seconds, 1), "frames": args.frames,
                                        "size": [args.width, args.height], "sparsity": sparsity}), flush=True)


def image(args) -> None:
    for name, value in (("width", args.width), ("height", args.height)):
        if value % 16:
            sys.exit(f"--{name} must be a multiple of 16, got {value}")
    ensure_server("image")
    text = prompt_of(args)
    target = Path(args.output)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    for seed in seeds:
        w = {
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": QWEN_BASE if args.base else QWEN,
                                                         "weight_dtype": "default"}},
            "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": QWEN_TEXT, "type": "qwen_image", "device": "default"}},
            "4": {"class_type": "VAELoader", "inputs": {"vae_name": QWEN_VAE}},
            "5": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["3", 0], "prompt": text,
                                                                    "negative_prompt": args.negative, "resolution": 1024}},
            "6": {"class_type": "EmptyLatentImage", "inputs": {"width": args.width, "height": args.height, "batch_size": 1}},
            "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["4", 0]}},
            "16": {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": f"studio/image_{uuid.uuid4().hex[:8]}"}},
        }
        if args.base:
            w["12"] = {"class_type": "KSampler",
                       "inputs": {"model": ["1", 0], "seed": seed, "steps": args.steps, "cfg": args.cfg,
                                  "sampler_name": "euler", "scheduler": "simple", "positive": ["5", 0],
                                  "negative": ["5", 1], "latent_image": ["6", 0], "denoise": 1.0}}
        else:
            w["8"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
            w["9"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
            w["10"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": TURBO_SIGMAS}}
            w["11"] = {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": ["5", 0],
                                                             "negative": ["5", 1], "cfg": 1.0}}
            w["12"] = {"class_type": "SamplerCustomAdvanced",
                       "inputs": {"noise": ["8", 0], "guider": ["11", 0], "sampler": ["9", 0], "sigmas": ["10", 0],
                                  "latent_image": ["6", 0]}}
        item, seconds = render(w, "12")
        out = target.with_name(f"{target.stem}_s{seed}{target.suffix}") if args.seeds else target
        fetch(item, out)
        print("[tensorfold] " + json.dumps({"output": str(out), "engine": "cuda", "width": args.width,
                                            "height": args.height, "seed": seed, "turbo": not args.base,
                                            "total_s": round(seconds, 1)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    kinds = parser.add_subparsers(dest="kind", required=True)
    v = kinds.add_parser("video")
    v.add_argument("--prompt", default=""); v.add_argument("--prompt-file", default="")
    v.add_argument("-o", "--output", required=True)
    v.add_argument("--width", type=int, default=864); v.add_argument("--height", type=int, default=480)
    v.add_argument("--frames", type=int, default=124); v.add_argument("--seed", type=int, default=0)
    v.add_argument("--steps", type=int, default=8, help="passes: 8 is what FastH3 was trained for")
    v.add_argument("--first-frame", default="", help="an image the clip starts from")
    v.add_argument("--crop", default="", help="WxH centre crop of the finished clip")
    v.add_argument("--weights", choices=sorted(FASTH3), default=os.environ.get("SPARK_WEIGHTS", "bf16"))
    v.add_argument("--engine", choices=("tensorfold", "comfy"),
                   default=os.environ.get("SPARK_ENGINE") or ("tensorfold" if ENGINE_LIB.exists() else "comfy"),
                   help="what runs the transformer blocks: TensorFold's CUDA family (the default once "
                        "spark/build-engine.sh has built it) or ComfyUI's own")
    v.add_argument("--sparsity", type=float, default=None, help="sparse attention (0 is dense); default by size")
    v.set_defaults(run=video)
    i = kinds.add_parser("image")
    i.add_argument("--prompt", default=""); i.add_argument("--prompt-file", default="")
    i.add_argument("-o", "--output", required=True)
    i.add_argument("--width", type=int, default=1344); i.add_argument("--height", type=int, default=768)
    i.add_argument("--seed", type=int, default=0); i.add_argument("--seeds", default="", help="several pictures: 1,2,3")
    i.add_argument("--base", action="store_true", help="the base model without the turbo adapter")
    i.add_argument("--steps", type=int, default=40, help="base model only")
    i.add_argument("--cfg", type=float, default=4.0, help="base model only")
    i.add_argument("--negative", default="")
    i.set_defaults(run=image)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
