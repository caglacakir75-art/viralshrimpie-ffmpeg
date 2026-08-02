
import asyncio
import json
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import List

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, HttpUrl

app = FastAPI(title="ViralShrimpie FFmpeg Renderer", version="1.1.0")

BASE_DIR = Path(os.getenv("JOB_DIR", "/tmp/viralshrimpie_jobs"))
BASE_DIR.mkdir(parents=True, exist_ok=True)
JOBS: dict[str, dict] = {}


class RenderRequest(BaseModel):
    video_urls: List[HttpUrl] = Field(min_length=4, max_length=4)
    audio_url: HttpUrl
    scene_texts: List[str] = Field(min_length=4, max_length=4)
    width: int = 1080
    height: int = 1920
    fps: int = 30
    font_size: int = 58
    text_margin: int = 170


def status_path(job_id: str) -> Path:
    return BASE_DIR / job_id / "status.json"


def set_job(job_id: str, data: dict) -> None:
    JOBS[job_id] = data
    path = status_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def get_job(job_id: str):
    if job_id in JOBS:
        return JOBS[job_id]
    path = status_path(job_id)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            JOBS[job_id] = data
            return data
        except Exception:
            return None
    return None


def run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-4000:])


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return float(result.stdout.strip())


def escape_drawtext(text: str) -> str:
    for old, new in [
        ("\\", r"\\"),
        (":", r"\:"),
        ("'", r"\'"),
        ("%", r"\%"),
        (",", r"\,"),
        ("[", r"\["),
        ("]", r"\]"),
    ]:
        text = text.replace(old, new)
    return text


async def download_file(client: httpx.AsyncClient, url: str, target: Path) -> None:
    async with client.stream("GET", url, follow_redirects=True, timeout=240) as response:
        response.raise_for_status()
        with target.open("wb") as f:
            async for chunk in response.aiter_bytes():
                f.write(chunk)


def weighted_durations(texts: list[str], total: float) -> list[float]:
    weights = [max(1, len(re.findall(r"\b[\w'-]+\b", t))) for t in texts]
    durations = [total * w / sum(weights) for w in weights]
    durations[-1] += total - sum(durations)
    return durations


async def render_job(job_id: str, payload: RenderRequest) -> None:
    job_dir = BASE_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    set_job(job_id, {"status": "downloading", "progress": 5})

    try:
        video_paths = [job_dir / f"video_{i+1}.mp4" for i in range(4)]
        audio_path = job_dir / "voiceover.mp3"

        # Sequential downloads reduce memory/network pressure on Render Free.
        async with httpx.AsyncClient() as client:
            for url, path in zip(payload.video_urls, video_paths):
                await download_file(client, str(url), path)
            await download_file(client, str(payload.audio_url), audio_path)

        set_job(job_id, {"status": "rendering", "progress": 25})

        audio_duration = probe_duration(audio_path)
        durations = weighted_durations(payload.scene_texts, audio_duration)
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        rendered = []

        for i, (source, duration, text) in enumerate(
            zip(video_paths, durations, payload.scene_texts), start=1
        ):
            target = job_dir / f"scene_{i}.mp4"
            rendered.append(target)

            vf = (
                f"scale={payload.width}:{payload.height}:force_original_aspect_ratio=increase,"
                f"crop={payload.width}:{payload.height},fps={payload.fps},"
                f"drawtext=fontfile={font_path}:text='{escape_drawtext(text)}':"
                f"fontcolor=white:fontsize={payload.font_size}:"
                f"borderw=4:bordercolor=black:"
                f"box=1:boxcolor=black@0.28:boxborderw=24:"
                f"x=(w-text_w)/2:y=h-text_h-{payload.text_margin}"
            )

            run([
                "ffmpeg", "-y",
                "-stream_loop", "-1",
                "-i", str(source),
                "-t", f"{duration:.3f}",
                "-vf", vf,
                "-an",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "22",
                "-threads", "1",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(target),
            ])
            set_job(job_id, {"status": "rendering", "progress": 25 + i * 12})

        concat = job_dir / "concat.txt"
        concat.write_text(
            "\n".join(f"file '{p.as_posix()}'" for p in rendered),
            encoding="utf-8",
        )

        silent = job_dir / "silent.mp4"
        run([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat),
            "-c", "copy",
            str(silent),
        ])

        final = job_dir / "final.mp4"
        run([
            "ffmpeg", "-y",
            "-i", str(silent),
            "-i", str(audio_path),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            str(final),
        ])

        set_job(job_id, {
            "status": "succeeded",
            "progress": 100,
            "duration": round(audio_duration, 3),
            "scene_durations": [round(x, 3) for x in durations],
            "download_url": f"/download/{job_id}",
        })

    except Exception as exc:
        set_job(job_id, {
            "status": "failed",
            "progress": 100,
            "error": str(exc),
        })


@app.get("/")
def root():
    return {"ok": True, "service": "ViralShrimpie FFmpeg Renderer", "version": "1.1.0"}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/render")
async def create_render(payload: RenderRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    set_job(job_id, {"status": "queued", "progress": 0})
    background_tasks.add_task(render_job, job_id, payload)
    return {
        "job_id": job_id,
        "status": "queued",
        "status_url": f"/status/{job_id}",
        "download_url": f"/download/{job_id}",
    }


@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, **job}


@app.get("/download/{job_id}")
def download(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("status") != "succeeded":
        raise HTTPException(status_code=409, detail=f"Job status: {job.get('status')}")

    final = BASE_DIR / job_id / "final.mp4"
    if not final.exists():
        raise HTTPException(status_code=404, detail="Rendered file not found")

    return FileResponse(final, media_type="video/mp4", filename=f"viralshrimpie-{job_id}.mp4")
