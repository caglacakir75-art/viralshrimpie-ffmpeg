
import asyncio
import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import List

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, HttpUrl

app = FastAPI(title="ViralShrimpie FFmpeg Renderer", version="1.0.0")

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


def run(cmd: list[str]) -> None:
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed:\n{' '.join(cmd)}\n\nSTDERR:\n{completed.stderr[-4000:]}"
        )


def probe_duration(path: Path) -> float:
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {completed.stderr}")
    return float(completed.stdout.strip())


def escape_drawtext(text: str) -> str:
    # FFmpeg drawtext escaping.
    text = text.replace("\\", r"\\")
    text = text.replace(":", r"\:")
    text = text.replace("'", r"\'")
    text = text.replace("%", r"\%")
    text = text.replace(",", r"\,")
    text = text.replace("[", r"\[")
    text = text.replace("]", r"\]")
    return text


async def download_file(client: httpx.AsyncClient, url: str, target: Path) -> None:
    async with client.stream("GET", url, follow_redirects=True, timeout=120) as response:
        response.raise_for_status()
        with target.open("wb") as file:
            async for chunk in response.aiter_bytes():
                file.write(chunk)


def weighted_durations(texts: list[str], total: float) -> list[float]:
    weights = [max(1, len(re.findall(r"\b[\w'-]+\b", text))) for text in texts]
    total_weight = sum(weights)
    raw = [total * weight / total_weight for weight in weights]

    # Keep every scene visible for at least 3 seconds.
    minimum = 3.0
    if total >= minimum * 4:
        raw = [max(minimum, value) for value in raw]
        scale = total / sum(raw)
        raw = [value * scale for value in raw]

    raw[-1] += total - sum(raw)
    return raw


async def render_job(job_id: str, payload: RenderRequest) -> None:
    job_dir = BASE_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    JOBS[job_id] = {"status": "downloading", "progress": 5}

    try:
        video_paths = [job_dir / f"video_{index + 1}.mp4" for index in range(4)]
        audio_path = job_dir / "voiceover.mp3"

        async with httpx.AsyncClient() as client:
            await asyncio.gather(
                *[
                    download_file(client, str(url), path)
                    for url, path in zip(payload.video_urls, video_paths)
                ],
                download_file(client, str(payload.audio_url), audio_path),
            )

        JOBS[job_id] = {"status": "rendering", "progress": 25}

        audio_duration = probe_duration(audio_path)
        durations = weighted_durations(payload.scene_texts, audio_duration)

        normalized_paths: list[Path] = []
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

        for index, (source, duration, text) in enumerate(
            zip(video_paths, durations, payload.scene_texts), start=1
        ):
            target = job_dir / f"scene_{index}.mp4"
            normalized_paths.append(target)
            safe_text = escape_drawtext(text)

            video_filter = (
                f"scale={payload.width}:{payload.height}:force_original_aspect_ratio=increase,"
                f"crop={payload.width}:{payload.height},"
                f"fps={payload.fps},"
                f"drawtext=fontfile={font_path}:"
                f"text='{safe_text}':"
                f"fontcolor=white:fontsize={payload.font_size}:"
                f"borderw=4:bordercolor=black:"
                f"box=1:boxcolor=black@0.28:boxborderw=24:"
                f"x=(w-text_w)/2:"
                f"y=h-text_h-{payload.text_margin}"
            )

            run(
                [
                    "ffmpeg", "-y",
                    "-stream_loop", "-1",
                    "-i", str(source),
                    "-t", f"{duration:.3f}",
                    "-vf", video_filter,
                    "-an",
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", "18",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    str(target),
                ]
            )
            JOBS[job_id] = {
                "status": "rendering",
                "progress": 25 + index * 12,
            }

        concat_file = job_dir / "concat.txt"
        concat_file.write_text(
            "\n".join(f"file '{path.as_posix()}'" for path in normalized_paths),
            encoding="utf-8",
        )

        silent_video = job_dir / "silent.mp4"
        run(
            [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                "-c", "copy",
                str(silent_video),
            ]
        )

        output_path = job_dir / "final.mp4"
        run(
            [
                "ffmpeg", "-y",
                "-i", str(silent_video),
                "-i", str(audio_path),
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                "-movflags", "+faststart",
                str(output_path),
            ]
        )

        JOBS[job_id] = {
            "status": "succeeded",
            "progress": 100,
            "duration": round(audio_duration, 3),
            "scene_durations": [round(value, 3) for value in durations],
            "download_url": f"/download/{job_id}",
        }

    except Exception as exc:
        JOBS[job_id] = {
            "status": "failed",
            "progress": 100,
            "error": str(exc),
        }


@app.get("/")
def root() -> dict:
    return {"ok": True, "service": "ViralShrimpie FFmpeg Renderer"}


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/render")
async def create_render(payload: RenderRequest, background_tasks: BackgroundTasks) -> dict:
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "queued", "progress": 0}
    background_tasks.add_task(render_job, job_id, payload)
    return {
        "job_id": job_id,
        "status": "queued",
        "status_url": f"/status/{job_id}",
        "download_url": f"/download/{job_id}",
    }


@app.get("/status/{job_id}")
def get_status(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, **job}


@app.get("/download/{job_id}")
def download(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("status") != "succeeded":
        raise HTTPException(status_code=409, detail=f"Job status: {job.get('status')}")

    output_path = BASE_DIR / job_id / "final.mp4"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Rendered file not found")

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"viralshrimpie-{job_id}.mp4",
    )
