#!/usr/bin/env python3
"""TensorFold Studio web app: a create wizard, a production board and a clip editor over the pack's scripts.

    bash scripts/app.sh            # http://127.0.0.1:7870
    HOST=0.0.0.0 bash scripts/app.sh   # reachable from other machines on the network (no login: trusted LANs only)

One job runs at a time, in the order queued; every job is one of this pack's scripts with environment variables.
Projects, uploads and renders live under $STUDIO_HOME (default ~/TensorFoldStudio).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).resolve().parent
PACK = HERE.parent
HOME = Path(os.environ.get("STUDIO_HOME", Path.home() / "TensorFoldStudio"))
for name in ("projects", "uploads", "renders", "scouts", "exports"):
    (HOME / name).mkdir(parents=True, exist_ok=True)

FPS = 24
DURATIONS = {5: 124, 8: 192, 10: 243, 15: 362}  # seconds -> frames (17n + 5)
# name -> (label, final size, environment). Generation happens at half size where X2 is set.
PRESETS = {
    "draft": ("Draft 1344x768 (generated at 672x384, 2x decoder)", (1344, 768), {"X2": "1"}),
    "2k": ("2K 2048x1152 (generated at 1024x576, 2x decoder)", (2048, 1152), {"TWOK": "1"}),
    "qhd": ("2K 2560x1440 (generated at 1280x736, 2x decoder)", (2560, 1440), {"QHD": "1"}),
    "small": ("Small 864x480 (native)", (864, 480), {"WIDTH": "864", "HEIGHT": "480"}),
    "native": ("Native 1344x768 (no 2x decoder)", (1344, 768), {"WIDTH": "1344", "HEIGHT": "768"}),
}
QUALITIES = {
    "standard": ("Standard: Turbo 5 passes + base-model sound", {}),
    "high": ("High: 20 steps, no adapter, fast recipe (about 1.5x slower)", {"QUALITY": "high"}),
    "full": ("Full: plain 20 steps, no adapter (about 3x slower)", {"QUALITY": "full"}),
    "fast": ("Fast: Turbo 3 passes, adapter sound", {"POINTS": "4", "REVOICE": "0"}),
}
SCOUT_SIZES = {"draft": (1344, 768), "2k": (1024, 576), "qhd": (1280, 736), "small": (864, 480),
               "native": (1344, 768)}
# measured on a Mac Studio M5 Ultra, 8 second clips, standard quality, image step included (README)
MEASURED_8S = {"draft": 108, "2k": 299, "qhd": 635, "small": 174, "native": 709}
LEAD = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."

app = FastAPI(title="TensorFold Studio")
jobs: dict[str, dict] = {}
order: list[str] = []
lock = threading.Lock()
wake = threading.Event()
running: dict = {"process": None, "id": None}


def inside(path: str | Path) -> Path:
    """A path under the studio folder, or 400."""

    resolved = (HOME / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if HOME.resolve() not in (resolved, *resolved.parents):
        raise HTTPException(400, "path is outside the studio folder")
    return resolved


def relative(path: Path) -> str:
    return str(Path(path).resolve().relative_to(HOME.resolve()))


def compose(fields: dict, first_frame: bool) -> str:
    """MiniMax H3's three-field prompt from the wizard's parts."""

    shot = " ".join(part.strip() for part in (fields.get("scene", ""), fields.get("camera", ""),
                                               fields.get("action", "")) if part and part.strip())
    spoken = []
    for index, line in enumerate(fields.get("dialogue") or [], start=1):
        text = (line.get("text") or "").strip()
        if not text:
            continue
        voice = (line.get("voice") or "a natural voice").strip()
        verb = "sings" if line.get("sung") else "says"
        language = (line.get("language") or "English").strip()
        spoken.append(f"The speaker, in {voice} (S{index}), {verb}: <d>[{language}] {text}</d>")
    if spoken:
        shot += " " + " ".join(spoken) + " Mouth movements follow the words."
    else:
        shot += " Nobody speaks."
    sound = (fields.get("soundscape") or "Natural ambient sound for the scene.").strip()
    if not spoken and "no speech" not in sound.lower():
        sound += " No voices, no speech."
    music = (fields.get("music") or "N/A").strip()
    body = (f"integrated_multimodal_description: [Shot 1] {shot.strip()}\n\n"
            f"overall_soundscape: {sound}\n\nnon_diegetic_music: {music}")
    return f"{LEAD}\n\n{body}" if first_frame else body


def estimate(preset: str, quality: str, seconds: int) -> int | None:
    base = MEASURED_8S.get(preset)
    if base is None:
        return None
    scale = {"standard": 1.0, "high": 1.45, "full": 3.0, "fast": 0.58}[quality] * (seconds / 8.0) ** 1.6
    return int(base * scale)


def environment(extra: dict) -> dict:
    env = dict(os.environ)
    env.update({key: str(value) for key, value in extra.items() if value not in (None, "")})
    env["PATH"] = "/opt/homebrew/bin:" + env.get("PATH", "")
    return env


def build(kind: str, params: dict, job_id: str) -> tuple[list[str], dict, list[str]]:
    """(command, environment, output paths) for one job."""

    scripts = PACK / "scripts"
    seed = str(int(params.get("seed", 0)))
    if kind == "scout":
        folder = HOME / "scouts" / job_id
        prompt = params["prompt"]
        count = max(1, min(int(params.get("count", 4)), 12))
        # a scout is made at the size the video model will start from (the draft keeps a full 1344x768 still)
        width, height = SCOUT_SIZES.get(params.get("preset", "2k"), (1024, 576))
        env = {"SEED": seed, "WIDTH": width, "HEIGHT": height}
        first = int(seed)
        outputs = [str(folder / f"scout_s{first + i}.png") for i in range(count)] + [str(folder / "sheet.jpg")]
        return ["bash", str(scripts / "scout.sh"), prompt, str(count), str(folder)], env, outputs
    if kind == "export":
        return ["python3", "-c", "pass"], {}, []
    preset = params.get("preset", "draft")
    quality = params.get("quality", "standard")
    seconds = int(params.get("seconds", 5))
    if seconds not in DURATIONS or preset not in PRESETS or quality not in QUALITIES:
        raise HTTPException(400, "unknown duration, preset or quality")
    target = Path(params["output"]) if params.get("output") else HOME / "renders" / f"{job_id}.mp4"
    target = inside(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    env = {"SEED": seed, "FRAMES": DURATIONS[seconds], **PRESETS[preset][2], **QUALITIES[quality][1]}
    prompt = params["prompt"]
    image = params.get("image")
    if kind == "video":  # text to video, or from an image the user already has
        if image:
            return (["bash", str(scripts / "studio.sh"), "", prompt, str(target)],
                    {**env, "IMAGE_FILE": str(inside(image)), "RAW_PROMPT": "1"}, [str(target)])
        size = PRESETS[preset][1]
        plain = {k: v for k, v in env.items() if k not in ("TWOK", "QHD")}
        if "TWOK" in env:
            plain.update(X2="1", WIDTH="2048", HEIGHT="1152")
        if "QHD" in env:
            plain.update(X2="1", WIDTH="2560", HEIGHT="1472", CROP="2560x1440")
        if "X2" in env and "WIDTH" not in plain:
            plain.update(WIDTH=str(size[0]), HEIGHT=str(size[1]))
        return ["bash", str(scripts / "video.sh"), prompt, str(target)], plain, [str(target)]
    if kind == "studio":  # text to image to video in one go
        return (["bash", str(scripts / "studio.sh"), params["image_prompt"], prompt, str(target)],
                {**env, "RAW_PROMPT": "1"}, [str(target), str(target.with_suffix(".png"))])
    raise HTTPException(400, f"unknown job kind {kind}")


def last_frame(video: Path, image: Path) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-sseof", "-0.2", "-i", str(video), "-vsync", "0", "-update", "1",
                    "-q:v", "1", str(image)], check=True, env=environment({}))


def export(job: dict) -> None:
    """Cut each timeline clip to its in and out points and join them, with a short audio fade at each join."""

    params = job["params"]
    clips = params["clips"]
    if not clips:
        raise RuntimeError("the timeline is empty")
    width, height = int(params.get("width", 1920)), int(params.get("height", 1080))
    target = HOME / "exports" / f"{job['id']}.mp4"
    inputs, filters, labels = [], [], []
    for index, clip in enumerate(clips):
        source = inside(clip["path"])
        start = max(0.0, float(clip.get("in", 0.0)))
        end = float(clip.get("out", 0.0))
        inputs += ["-i", str(source)]
        span = f"trim=start={start}:end={end},setpts=PTS-STARTPTS," if end > start else ""
        aspan = f"atrim=start={start}:end={end},asetpts=PTS-STARTPTS," if end > start else ""
        length = (end - start) if end > start else float(clip.get("duration", 5.0))
        fade = min(0.08, length / 4)
        filters.append(f"[{index}:v]{span}scale={width}:{height}:force_original_aspect_ratio=decrease,"
                       f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={FPS}[v{index}]")
        filters.append(f"[{index}:a]{aspan}afade=t=in:d={fade},afade=t=out:st={max(length - fade, 0)}:d={fade},"
                       f"aresample=48000[a{index}]")
        labels.append(f"[v{index}][a{index}]")
    filters.append(f"{''.join(labels)}concat=n={len(clips)}:v=1:a=1[v][a]")
    command = ["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", ";".join(filters), "-map", "[v]", "-map",
               "[a]", "-c:v", "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac",
               "-b:a", "256k", str(target)]
    done = subprocess.run(command, capture_output=True, text=True, env=environment({}))
    job["log"] += done.stderr
    if done.returncode:
        raise RuntimeError("ffmpeg failed")
    job["outputs"] = [relative(target)]


STEP = re.compile(r"\[tensorfold\] (step|voice) (\d+)/(\d+)")


def work() -> None:
    while True:
        wake.wait()
        with lock:
            job = next((jobs[i] for i in order if jobs[i]["status"] == "queued"), None)
            if job is None:
                wake.clear()
                continue
            job["status"], job["started"] = "running", time.time()
        try:
            params = job["params"]
            if params.get("first_frame_from"):  # continuity: open on the last frame of an earlier clip
                source = inside(params["first_frame_from"])
                if not source.is_file():
                    raise RuntimeError("the clip this one continues from has not been rendered")
                still = HOME / "uploads" / f"{job['id']}_first.jpg"
                last_frame(source, still)
                params["image"] = relative(still)
            if job["kind"] == "export":
                export(job)
            else:
                command, env, outputs = build(job["kind"], params, job["id"])
                process = subprocess.Popen(command, cwd=PACK, env=environment(env), stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, start_new_session=True)
                running.update(process=process, id=job["id"])
                for line in process.stdout:
                    job["log"] = (job["log"] + line)[-20000:]
                    found = STEP.search(line)
                    if found:
                        kind, at, total = found.group(1), int(found.group(2)), int(found.group(3))
                        job["progress"] = {"stage": "picture" if kind == "step" else "sound", "at": at, "of": total}
                code = process.wait()
                running.update(process=None, id=None)
                if job["status"] == "cancelled":
                    continue
                if code:
                    raise RuntimeError(f"the script exited with status {code}")
                job["outputs"] = [relative(Path(p)) for p in outputs if Path(p).is_file()]
                if not job["outputs"]:
                    raise RuntimeError("the script finished without writing its output")
            job["status"] = "done"
        except Exception as error:  # noqa: BLE001 - a failed job must not stop the queue
            if job["status"] != "cancelled":
                job["status"], job["error"] = "failed", str(error)
        finally:
            job["finished"] = time.time()


threading.Thread(target=work, daemon=True).start()


def public(job: dict) -> dict:
    out = {k: job.get(k) for k in ("id", "kind", "status", "title", "outputs", "progress", "error", "created",
                                    "started", "finished", "estimate")}
    out["log"] = job["log"][-3000:]
    out["params"] = {k: v for k, v in job["params"].items() if k != "clips"}
    return out


@app.get("/api/state")
def state() -> dict:
    return {"presets": {k: {"label": v[0], "size": v[1]} for k, v in PRESETS.items()},
            "qualities": {k: v[0] for k, v in QUALITIES.items()}, "durations": DURATIONS,
            "home": str(HOME), "pack": str(PACK), "version": (PACK / "VERSION").read_text().strip()}


@app.post("/api/compose")
def compose_prompt(body: dict) -> dict:
    return {"prompt": compose(body.get("fields", {}), bool(body.get("first_frame")))}


@app.post("/api/estimate")
def estimate_time(body: dict) -> dict:
    return {"seconds": estimate(body.get("preset", "draft"), body.get("quality", "standard"),
                                int(body.get("seconds", 5)))}


@app.post("/api/jobs")
def submit(body: dict) -> dict:
    kind = body.get("kind")
    params = dict(body.get("params") or {})
    if kind not in ("scout", "video", "studio", "export"):
        raise HTTPException(400, "unknown job kind")
    if kind != "export" and not (params.get("prompt") or "").strip():
        raise HTTPException(400, "a prompt is needed")
    job_id = time.strftime("%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    if kind not in ("export", "scout"):
        build(kind, params, job_id)  # reject bad settings now, not when the job's turn comes
    job = {"id": job_id, "kind": kind, "status": "queued", "title": body.get("title") or kind, "params": params,
           "outputs": [], "progress": None, "error": None, "log": "", "created": time.time(), "started": None,
           "finished": None, "estimate": estimate(params.get("preset", ""), params.get("quality", "standard"),
                                                  int(params.get("seconds", 5))) if kind in ("video", "studio")
           else None}
    with lock:
        jobs[job_id] = job
        order.append(job_id)
    wake.set()
    return public(job)


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    return [public(jobs[i]) for i in reversed(order)]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    if job_id not in jobs:
        raise HTTPException(404, "no such job")
    return public(jobs[job_id])


@app.post("/api/jobs/{job_id}/cancel")
def cancel(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    if job["status"] in ("queued", "running"):
        job["status"] = "cancelled"
        process = running["process"] if running["id"] == job_id else None
        if process is not None:
            os.killpg(process.pid, signal.SIGTERM)
    return public(job)


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "image.png").suffix.lower()
    if suffix not in (".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mov"):
        raise HTTPException(400, "images (png, jpg, webp) and videos (mp4, mov) only")
    target = HOME / "uploads" / f"{uuid.uuid4().hex[:10]}{suffix}"
    with open(target, "wb") as handle:
        shutil.copyfileobj(file.file, handle)
    return {"path": relative(target)}


@app.get("/api/library")
def library() -> list[dict]:
    """Every clip and still under the studio folder, newest first."""

    found = []
    for folder, kinds in (("renders", (".mp4",)), ("exports", (".mp4",)), ("scouts", (".png",)),
                          ("uploads", (".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mov")), ("projects", (".mp4",))):
        for path in (HOME / folder).rglob("*"):
            if path.suffix.lower() in kinds and path.is_file():
                found.append({"path": relative(path), "kind": "video" if path.suffix.lower() in (".mp4", ".mov")
                              else "image", "folder": folder, "modified": path.stat().st_mtime,
                              "size": path.stat().st_size})
    return sorted(found, key=lambda item: -item["modified"])[:400]


@app.get("/api/probe")
def probe(path: str) -> dict:
    target = inside(path)
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,nb_frames:format=duration", "-of", "json", str(target)],
                         capture_output=True, text=True, env=environment({}))
    data = json.loads(out.stdout or "{}")
    stream = (data.get("streams") or [{}])[0]
    return {"width": stream.get("width"), "height": stream.get("height"), "frames": stream.get("nb_frames"),
            "duration": float((data.get("format") or {}).get("duration") or 0)}


def project_file(project_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", project_id):
        raise HTTPException(400, "bad project id")
    return HOME / "projects" / project_id / "project.json"


@app.get("/api/projects")
def projects() -> list[dict]:
    out = []
    for file in sorted((HOME / "projects").glob("*/project.json"), key=lambda f: -f.stat().st_mtime):
        data = json.loads(file.read_text())
        out.append({"id": data["id"], "name": data.get("name", data["id"]), "segments": len(data.get("segments", [])),
                    "modified": file.stat().st_mtime})
    return out


@app.post("/api/projects")
def new_project(body: dict) -> dict:
    project_id = time.strftime("p%m%d-%H%M%S")
    data = {"id": project_id, "name": body.get("name") or "Untitled", "preset": "draft", "quality": "standard",
            "segments": [], "timeline": []}
    file = project_file(project_id)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(data, indent=1))
    return data


@app.get("/api/projects/{project_id}")
def get_project(project_id: str) -> dict:
    file = project_file(project_id)
    if not file.is_file():
        raise HTTPException(404, "no such project")
    return json.loads(file.read_text())


@app.put("/api/projects/{project_id}")
def save_project(project_id: str, body: dict) -> dict:
    file = project_file(project_id)
    if not file.is_file():
        raise HTTPException(404, "no such project")
    body["id"] = project_id
    file.write_text(json.dumps(body, indent=1))
    return body


@app.get("/api/wait")
def wait(ms: int = 1000):
    """Holds a response open; a page asked for with ?shot=1 loads an image from here so that a headless
    screenshot is taken after the app has drawn."""

    time.sleep(min(max(ms, 0), 10000) / 1000)
    return JSONResponse({"waited": ms})


@app.get("/files/{path:path}")
def files(path: str):
    target = inside(path)
    if not target.is_file():
        raise HTTPException(404, "no such file")
    return FileResponse(target)


@app.exception_handler(HTTPException)
async def errors(_, error: HTTPException):
    return JSONResponse({"error": error.detail}, status_code=error.status_code)


app.mount("/", StaticFiles(directory=HERE / "static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "7870")),
                log_level="warning")
