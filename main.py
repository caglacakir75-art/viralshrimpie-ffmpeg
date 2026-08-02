import asyncio
import json
import os
import re
import subprocess
import textwrap
import uuid
from pathlib import Path
from typing import List

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, HttpUrl

app = FastAPI(title="ViralShrimpie FFmpeg Renderer", version="1.4.0")

BASE_DIR = Path(os.getenv("JOB_DIR", "/tmp/viralshrimpie_jobs"))
BASE_DIR.mkdir(parents=True, exist_ok=True)
JOBS: dict[str, dict] = {}

HOOK_FONT_SIZE = 72
BODY_FONT_SIZE = 66
MIN_FONT_SIZE = 40
BOTTOM_MARGIN = 230
TEXT_BOX_ALPHA = 0.35
TEXT_BOX_PADDING = 34
TEXT_BORDER_WIDTH = 5
TEXT_LINE_SPACING = 12
TEXT_FADE_SECONDS = 0.25

ENABLE_KEN_BURNS = True
KEN_BURNS_MAX_ZOOM = 1.045


class RenderRequest(BaseModel):
    video_urls: List[HttpUrl] = Field(min_length=4, max_length=4)
    audio_url: HttpUrl
    scene_texts: List[str] = Field(min_length=4, max_length=4)
    width: int = 1080
    height: int = 1920
    fps: int = 30
    font_size: int = BODY_FONT_SIZE
    text_margin: int = BOTTOM_MARGIN


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
        raise RuntimeError(result.stderr[-5000:])


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return float(result.stdout.strip())


async def download_file(client: httpx.AsyncClient, url: str, target: Path) -> None:
    async with client.stream("GET", url, follow_redirects=True, timeout=240) as response:
        response.raise_for_status()
        with target.open("wb") as file:
            async for chunk in response.aiter_bytes():
                file.write(chunk)


def weighted_durations(texts: list[str], total: float) -> list[float]:
    weights = [max(1, len(re.findall(r"\b[\w'-]+\b", text))) for text in texts]
    durations = [total * weight / sum(weights) for weight in weights]
    durations[-1] += total - sum(durations)
    return durations


def wrap_for_limit(text: str, max_chars: int) -> list[str]:
    cleaned = " ".join(text.split())
    return textwrap.wrap(
        cleaned,
        width=max_chars,
        break_long_words=False,
        break_on_hyphens=False,
    )


def choose_caption_layout(
    text: str,
    is_hook: bool,
    video_width: int,
) -> tuple[str, int]:
    """
    Selects the largest font that keeps the caption within the safe width
    and at no more than three lines.
    """
    starting_size = HOOK_FONT_SIZE if is_hook else BODY_FONT_SIZE
    font_sizes = [
        size
        for size in (starting_size, 66, 62, 58, 54, 50, 46, 42, MIN_FONT_SIZE)
        if size <= starting_size
    ]

    # DejaVu Sans Bold averages roughly 0.62 x font size per character.
    # Reserve 10% safe space on each side and account for box padding.
    usable_width = (video_width * 0.90) - (TEXT_BOX_PADDING * 2)

    for font_size in font_sizes:
        max_chars = max(
            16,
            int(usable_width / (font_size * 0.62)),
        )
        lines = wrap_for_limit(text, max_chars)

        if len(lines) <= 3:
            return "\n".join(lines), font_size

    # Emergency fallback for unusually long model output.
    font_size = MIN_FONT_SIZE
    max_chars = max(
        18,
        int(usable_width / (font_size * 0.62)),
    )
    lines = wrap_for_limit(text, max_chars)

    # Rebalance across exactly three lines without deleting words.
    words = " ".join(text.split()).split()
    target = max(1, (len(words) + 2) // 3)
    balanced = [
        " ".join(words[index:index + target])
        for index in range(0, len(words), target)
    ]

    if len(balanced) <= 3:
        lines = balanced

    return "\n".join(lines), font_size


def build_ken_burns_filter(
    scene_index: int,
    duration: float,
    fps: int,
    width: int,
    height: int,
) -> str:
    total_frames = max(1, int(duration * fps))
    zoom = (
        f"min(1+({KEN_BURNS_MAX_ZOOM - 1:.6f})"
        f"*on/{total_frames},"
        f"{KEN_BURNS_MAX_ZOOM:.6f})"
    )

    direction = (scene_index - 1) % 4

    if direction == 0:
        x = "iw/2-(iw/zoom/2)"
        y = "ih/2-(ih/zoom/2)"
    elif direction == 1:
        x = f"(iw-iw/zoom)*on/{total_frames}"
        y = "ih/2-(ih/zoom/2)"
    elif direction == 2:
        x = f"(iw-iw/zoom)*(1-on/{total_frames})"
        y = "ih/2-(ih/zoom/2)"
    else:
        x = "iw/2-(iw/zoom/2)"
        y = f"(ih-ih/zoom)*(1-on/{total_frames})"

    return (
        f"zoompan=z='{zoom}':"
        f"x='{x}':"
        f"y='{y}':"
        f"d=1:"
        f"s={width}x{height}:"
        f"fps={fps}"
    )


async def render_job(job_id: str, payload: RenderRequest) -> None:
    job_dir = BASE_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    set_job(job_id, {"status": "downloading", "progress": 5})

    try:
        video_paths = [job_dir / f"video_{index + 1}.mp4" for index in range(4)]
        audio_path = job_dir / "voiceover.mp3"

        async with httpx.AsyncClient() as client:
            for url, path in zip(payload.video_urls, video_paths):
                await download_file(client, str(url), path)
            await download_file(client, str(payload.audio_url), audio_path)

        set_job(job_id, {"status": "rendering", "progress": 25})

        audio_duration = probe_duration(audio_path)
        durations = weighted_durations(payload.scene_texts, audio_duration)
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        rendered_paths = []

        for index, (source, duration, text) in enumerate(
            zip(video_paths, durations, payload.scene_texts), start=1
        ):
            target = job_dir / f"scene_{index}.mp4"
            rendered_paths.append(target)

            caption, current_font_size = choose_caption_layout(
                text=text,
                is_hook=index == 1,
                video_width=payload.width,
            )
            text_path = job_dir / f"scene_{index}.txt"
            text_path.write_text(caption, encoding="utf-8")

            fade = TEXT_FADE_SECONDS
            alpha_expression = (
                f"if(lt(t,{fade}),t/{fade},"
                f"if(gt(t,{duration - fade}),"
                f"({duration}-t)/{fade},1))"
            )

            filters = [
                (
                    f"scale={payload.width}:{payload.height}:"
                    f"force_original_aspect_ratio=increase"
                ),
                f"crop={payload.width}:{payload.height}",
            ]

            if ENABLE_KEN_BURNS:
                filters.append(
                    build_ken_burns_filter(
                        scene_index=index,
                        duration=duration,
                        fps=payload.fps,
                        width=payload.width,
                        height=payload.height,
                    )
                )
            else:
                filters.append(f"fps={payload.fps}")

            filters.append(
                (
                    f"drawtext=fontfile={font_path}:"
                    f"textfile={text_path.as_posix()}:"
                    f"fontcolor=white:"
                    f"fontsize={current_font_size}:"
                    f"line_spacing={TEXT_LINE_SPACING}:"
                    f"borderw={TEXT_BORDER_WIDTH}:"
                    f"bordercolor=black:"
                    f"box=1:"
                    f"boxcolor=black@{TEXT_BOX_ALPHA}:"
                    f"boxborderw={TEXT_BOX_PADDING}:"
                    f"alpha='{alpha_expression}':"
                    f"x=(w-text_w)/2:"
                    f"y=h-text_h-{payload.text_margin}"
                )
            )

            video_filter = ",".join(filters)

            run([
                "ffmpeg", "-y", "-stream_loop", "-1", "-i", str(source),
                "-t", f"{duration:.3f}", "-vf", video_filter, "-an",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
                "-threads", "1", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(target)
            ])

            set_job(job_id, {"status": "rendering", "progress": 25 + index * 12})

        concat_path = job_dir / "concat.txt"
        concat_path.write_text(
            "\n".join(f"file '{path.as_posix()}'" for path in rendered_paths),
            encoding="utf-8"
        )

        silent_video = job_dir / "silent.mp4"
        run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i",
             str(concat_path), "-c", "copy", str(silent_video)])

        final_video = job_dir / "final.mp4"
        run([
            "ffmpeg", "-y", "-i", str(silent_video), "-i", str(audio_path),
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k", "-shortest",
            "-movflags", "+faststart", str(final_video)
        ])

        set_job(job_id, {
            "status": "succeeded",
            "progress": 100,
            "duration": round(audio_duration, 3),
            "scene_durations": [round(value, 3) for value in durations],
            "download_url": f"/download/{job_id}"
        })

    except Exception as exc:
        set_job(job_id, {"status": "failed", "progress": 100, "error": str(exc)})


@app.get("/")
def root():
    return {"ok": True, "service": "ViralShrimpie FFmpeg Renderer", "version": "1.4.0"}


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
        "download_url": f"/download/{job_id}"
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

    final_video = BASE_DIR / job_id / "final.mp4"
    if not final_video.exists():
        raise HTTPException(status_code=404, detail="Rendered file not found")

    return FileResponse(
        final_video,
        media_type="video/mp4",
        filename=f"viralshrimpie-{job_id}.mp4"
    )
