import asyncio
import json
import os
import random
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

app = FastAPI(title="ViralShrimpie FFmpeg Renderer", version="2.0.1")

BASE_DIR = Path(os.getenv("JOB_DIR", "/tmp/viralshrimpie_jobs"))
BASE_DIR.mkdir(parents=True, exist_ok=True)
JOBS: dict[str, dict] = {}

HOOK_FONT_SIZE = 60
BODY_FONT_SIZE = 66
MIN_FONT_SIZE = 40
BOTTOM_MARGIN = 230
TEXT_BOX_ALPHA = 0.35
TEXT_BOX_PADDING = 34
TEXT_BORDER_WIDTH = 5
TEXT_LINE_SPACING = 12
TEXT_FADE_SECONDS = 0.25

ENABLE_KEN_BURNS = False
KEN_BURNS_MAX_ZOOM = 1.045

ENABLE_SCENE_FADE = False
SCENE_FADE_SECONDS = 0.20

ENABLE_COLOR_SOFTENING = True
VIDEO_CONTRAST = 0.97
VIDEO_SATURATION = 0.90

ENABLE_AUDIO_NORMALIZATION = True
VOICE_TARGET_LUFS = -16
VOICE_TRUE_PEAK = -1.5
VOICE_LRA = 11

DEFAULT_MUSIC_VOLUME = 0.08
MUSIC_FADE_SECONDS = 1.0

ENABLE_CAPTION_SLIDE = False
CAPTION_SLIDE_SECONDS = 0.22
CAPTION_SLIDE_DISTANCE = 42

THUMBNAIL_WIDTH = 1280
THUMBNAIL_HEIGHT = 720
THUMBNAIL_FONT_SIZE = 82
THUMBNAIL_MAX_LINE_CHARS = 18
THUMBNAIL_TEXT_X = 70
THUMBNAIL_TEXT_PANEL_WIDTH = 740


class RenderRequest(BaseModel):
    video_urls: List[HttpUrl] = Field(min_length=12, max_length=12)
    hook_audio_url: HttpUrl
    story_audio_url: HttpUrl
    follow_audio_url: HttpUrl
    hook_pause: float = Field(default=1.5, ge=0.0, le=5.0)
    outro_pause: float = Field(default=1.8, ge=0.0, le=5.0)
    music_url: HttpUrl | None = None
    music_urls: List[HttpUrl] | None = None
    music_volume: float = DEFAULT_MUSIC_VOLUME
    thumbnail_text: str | None = None
    title: str | None = None
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
    requested_font_size: int,
) -> tuple[str, int]:
    """
    Selects the largest font that keeps the caption within the safe width
    and at no more than three lines.
    """
    configured_size = max(MIN_FONT_SIZE, requested_font_size)
    hook_size = HOOK_FONT_SIZE
    starting_size = hook_size if is_hook else configured_size

    candidates = [
        starting_size,
        46,
        44,
        42,
        40,
        38,
        36,
    ]
    font_sizes = []
    for size in candidates:
        if size <= starting_size and size not in font_sizes:
            font_sizes.append(size)

    # Use a conservative safe area. The former 0.62 width estimate was too
    # optimistic for bold capitals and caused captions to leave the frame.
    usable_width = (video_width * 0.82) - (TEXT_BOX_PADDING * 2)

    for font_size in font_sizes:
        max_chars = max(
            14,
            int(usable_width / (font_size * 0.78)),
        )
        lines = wrap_for_limit(text, max_chars)

        if len(lines) <= 4:
            return "\n".join(lines), font_size

    font_size = 36
    max_chars = max(
        14,
        int(usable_width / (font_size * 0.78)),
    )
    lines = wrap_for_limit(text, max_chars)
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


def wrap_thumbnail_text(text: str) -> str:
    cleaned = " ".join(text.split()).upper()
    lines = textwrap.wrap(
        cleaned,
        width=THUMBNAIL_MAX_LINE_CHARS,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return "\n".join(lines[:4])


def create_thumbnail(
    source_video: Path,
    output_path: Path,
    text: str,
    job_dir: Path,
) -> None:
    font_path = (
        "/usr/share/fonts/truetype/dejavu/"
        "DejaVuSans-Bold.ttf"
    )
    thumbnail_text_path = job_dir / "thumbnail_text.txt"
    thumbnail_text_path.write_text(
        wrap_thumbnail_text(text),
        encoding="utf-8",
    )

    filter_complex = (
        f"[0:v]split=2[bg][fg];"
        f"[bg]"
        f"scale={THUMBNAIL_WIDTH}:{THUMBNAIL_HEIGHT}:"
        f"force_original_aspect_ratio=increase,"
        f"crop={THUMBNAIL_WIDTH}:{THUMBNAIL_HEIGHT},"
        f"boxblur=22:10,"
        f"eq=brightness=-0.40:saturation=0.85[bg2];"
        f"[fg]"
        f"scale=500:-2,"
        f"crop=500:{THUMBNAIL_HEIGHT}[fg2];"
        f"[bg2][fg2]"
        f"overlay=x=W-w-30:y=0,"
        f"drawbox=x=0:y=0:"
        f"w={THUMBNAIL_TEXT_PANEL_WIDTH}:"
        f"h=ih:"
        f"color=black@0.28:t=fill,"
        f"drawtext=fontfile={font_path}:"
        f"textfile={thumbnail_text_path.as_posix()}:"
        f"fontcolor=white:"
        f"fontsize={THUMBNAIL_FONT_SIZE}:"
        f"line_spacing=8:"
        f"borderw=5:"
        f"bordercolor=black:"
        f"x={THUMBNAIL_TEXT_X}:"
        f"y=(h-text_h)/2"
    )

    run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            "1.0",
            "-i",
            str(source_video),
            "-frames:v",
            "1",
            "-filter_complex",
            filter_complex,
            "-q:v",
            "3",
            str(output_path),
        ]
    )


async def render_job(job_id: str, payload: RenderRequest) -> None:
    job_dir = BASE_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    set_job(job_id, {"status": "downloading", "progress": 5})

    try:
        video_paths = [job_dir / f"video_{index + 1}.mp4" for index in range(12)]
        hook_audio_path = job_dir / "hook_voice.mp3"
        story_audio_path = job_dir / "story_voice.mp3"
        follow_audio_path = job_dir / "follow_voice.mp3"
        audio_path = job_dir / "combined_voice.wav"
        music_path = job_dir / "background_music.mp3"

        selected_music_url = None
        available_music_urls = []

        if payload.music_urls:
            available_music_urls.extend(
                str(url) for url in payload.music_urls
            )

        if payload.music_url:
            available_music_urls.append(str(payload.music_url))

        if available_music_urls:
            selected_music_url = random.choice(available_music_urls)

        async with httpx.AsyncClient() as client:
            for url, path in zip(payload.video_urls, video_paths):
                await download_file(client, str(url), path)
            await download_file(client, str(payload.hook_audio_url), hook_audio_path)
            await download_file(client, str(payload.story_audio_url), story_audio_path)
            await download_file(client, str(payload.follow_audio_url), follow_audio_path)

            if selected_music_url:
                await download_file(
                    client,
                    selected_music_url,
                    music_path,
                )

        set_job(job_id, {"status": "rendering", "progress": 25})

        hook_audio_duration = probe_duration(hook_audio_path)
        story_audio_duration = probe_duration(story_audio_path)
        follow_audio_duration = probe_duration(follow_audio_path)

        # Build one narration track with intentional pauses:
        # hook -> pause -> story -> pause -> follow CTA.
        run([
            "ffmpeg", "-y",
            "-i", str(hook_audio_path),
            "-f", "lavfi", "-t", f"{payload.hook_pause:.3f}",
            "-i", "anullsrc=r=48000:cl=stereo",
            "-i", str(story_audio_path),
            "-f", "lavfi", "-t", f"{payload.outro_pause:.3f}",
            "-i", "anullsrc=r=48000:cl=stereo",
            "-i", str(follow_audio_path),
            "-filter_complex",
            "[0:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[a0];"
            "[1:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[a1];"
            "[2:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[a2];"
            "[3:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[a3];"
            "[4:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[a4];"
            "[a0][a1][a2][a3][a4]concat=n=5:v=0:a=1[aout]",
            "-map", "[aout]",
            "-c:a", "pcm_s16le",
            str(audio_path),
        ])

        audio_duration = probe_duration(audio_path)

        # Scene 1 owns the hook and its pause. Scenes 2-4 share story audio
        # according to spoken word count. The final scene also holds the
        # outro pause and the separate Follow for more audio.
        story_scene_durations = weighted_durations(
            payload.scene_texts[1:],
            story_audio_duration,
        )
        durations = [
            hook_audio_duration + payload.hook_pause,
            story_scene_durations[0],
            story_scene_durations[1],
            story_scene_durations[2] + payload.outro_pause + follow_audio_duration,
        ]
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        rendered_paths = []
        clip_counter = 0

        for scene_index, (scene_duration, scene_text) in enumerate(
            zip(durations, payload.scene_texts),
            start=1
        ):
            base_clip_duration = scene_duration / 3.0
            clip_durations = [
                base_clip_duration,
                base_clip_duration,
                scene_duration - (base_clip_duration * 2),
            ]

            caption, current_font_size = choose_caption_layout(
                text=scene_text,
                is_hook=scene_index == 1,
                video_width=payload.width,
                requested_font_size=payload.font_size,
            )

            text_path = job_dir / f"scene_{scene_index}.txt"
            text_path.write_text(caption, encoding="utf-8")

            for clip_index in range(3):
                source = video_paths[clip_counter]
                duration = clip_durations[clip_index]
                clip_counter += 1

                target = job_dir / (
                    f"scene_{scene_index}_clip_{clip_index + 1}.mp4"
                )
                rendered_paths.append(target)

                fade = TEXT_FADE_SECONDS

                # Caption belongs to the whole spoken scene, not each clip.
                # Clip 1: fade/slide in once.
                # Clip 2: remain fully visible and still.
                # Clip 3: remain visible, then fade out once.
                if clip_index == 0:
                    alpha_expression = (
                        f"if(lt(t,{fade}),t/{fade},1)"
                    )
                elif clip_index == 2:
                    alpha_expression = (
                        f"if(gt(t,{duration - fade}),"
                        f"({duration}-t)/{fade},1)"
                    )
                else:
                    alpha_expression = "1"

                base_y = f"h-text_h-{payload.text_margin}"

                if ENABLE_CAPTION_SLIDE and clip_index == 0:
                    slide = CAPTION_SLIDE_SECONDS
                    caption_y = (
                        f"if(lt(t,{slide}),"
                        f"({base_y})+"
                        f"{CAPTION_SLIDE_DISTANCE}*"
                        f"(1-t/{slide}),"
                        f"({base_y}))"
                    )
                else:
                    caption_y = base_y

                filters = [
                    (
                        f"scale={payload.width}:{payload.height}:"
                        f"force_original_aspect_ratio=increase"
                    ),
                    f"crop={payload.width}:{payload.height}",
                ]

                if ENABLE_COLOR_SOFTENING:
                    filters.append(
                        f"eq=contrast={VIDEO_CONTRAST}:"
                        f"saturation={VIDEO_SATURATION}"
                    )

                if ENABLE_KEN_BURNS:
                    filters.append(
                        build_ken_burns_filter(
                            scene_index=clip_counter,
                            duration=duration,
                            fps=payload.fps,
                            width=payload.width,
                            height=payload.height,
                        )
                    )
                else:
                    filters.append(f"fps={payload.fps}")

                if ENABLE_SCENE_FADE:
                    fade_out_start = max(
                        0.0,
                        duration - SCENE_FADE_SECONDS,
                    )
                    filters.extend(
                        [
                            (
                                f"fade=t=in:st=0:"
                                f"d={SCENE_FADE_SECONDS}"
                            ),
                            (
                                f"fade=t=out:"
                                f"st={fade_out_start:.3f}:"
                                f"d={SCENE_FADE_SECONDS}"
                            ),
                        ]
                    )

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
                        f"y='{caption_y}'"
                    )
                )

                video_filter = ",".join(filters)

                run([
                    "ffmpeg", "-y", "-stream_loop", "-1", "-i", str(source),
                    "-t", f"{duration:.3f}", "-vf", video_filter, "-an",
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
                    "-threads", "1", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(target)
                ])

                progress = 25 + int((clip_counter / 12) * 48)
                set_job(
                    job_id,
                    {
                        "status": "rendering",
                        "progress": progress,
                    },
                )

        concat_path = job_dir / "concat.txt"
        concat_path.write_text(
            "\n".join(f"file '{path.as_posix()}'" for path in rendered_paths),
            encoding="utf-8"
        )

        silent_video = job_dir / "silent.mp4"
        run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i",
             str(concat_path), "-c", "copy", str(silent_video)])

        final_video = job_dir / "final.mp4"

        voice_filter = (
            f"loudnorm=I={VOICE_TARGET_LUFS}:"
            f"TP={VOICE_TRUE_PEAK}:"
            f"LRA={VOICE_LRA}"
            if ENABLE_AUDIO_NORMALIZATION
            else "anull"
        )

        if selected_music_url and music_path.exists():
            music_duration = probe_duration(music_path)
            max_start = max(
                0.0,
                music_duration - audio_duration - 1.0,
            )
            music_start = (
                random.uniform(0.0, max_start)
                if max_start > 0
                else 0.0
            )

            music_fade_out_start = max(
                0.0,
                audio_duration - MUSIC_FADE_SECONDS,
            )

            # Keep the mix intentionally simple and memory-safe on Render Free.
            # -stream_loop loops the music input, so aloop is unnecessary.
            # At 0.08 volume, music stays behind narration without sidechain.
            filter_complex = (
                f"[1:a]{voice_filter},"
                f"aresample=48000[voice];"
                f"[2:a]"
                f"atrim=start={music_start:.3f}:"
                f"duration={audio_duration:.3f},"
                f"asetpts=N/SR/TB,"
                f"aresample=48000,"
                f"afade=t=in:st=0:d={MUSIC_FADE_SECONDS},"
                f"afade=t=out:"
                f"st={music_fade_out_start:.3f}:"
                f"d={MUSIC_FADE_SECONDS},"
                f"volume={payload.music_volume}[music];"
                f"[voice][music]"
                f"amix=inputs=2:"
                f"duration=first:"
                f"dropout_transition=0:"
                f"normalize=0[aout]"
            )

            run([
                "ffmpeg", "-y",
                "-i", str(silent_video),
                "-i", str(audio_path),
                "-stream_loop", "-1",
                "-i", str(music_path),
                "-filter_complex", filter_complex,
                "-map", "0:v:0",
                "-map", "[aout]",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                "-movflags", "+faststart",
                str(final_video)
            ])
        else:
            run([
                "ffmpeg", "-y",
                "-i", str(silent_video),
                "-i", str(audio_path),
                "-filter:a", voice_filter,
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                "-movflags", "+faststart",
                str(final_video)
            ])

        thumbnail_path = job_dir / "thumbnail.jpg"
        thumbnail_label = (
            payload.thumbnail_text
            or payload.title
            or payload.scene_texts[0]
        )
        create_thumbnail(
            source_video=video_paths[0],
            output_path=thumbnail_path,
            text=thumbnail_label,
            job_dir=job_dir,
        )

        set_job(job_id, {
            "status": "succeeded",
            "progress": 100,
            "duration": round(audio_duration, 3),
            "hook_audio_duration": round(hook_audio_duration, 3),
            "story_audio_duration": round(story_audio_duration, 3),
            "follow_audio_duration": round(follow_audio_duration, 3),
            "hook_pause": round(payload.hook_pause, 3),
            "outro_pause": round(payload.outro_pause, 3),
            "scene_durations": [round(value, 3) for value in durations],
            "download_url": f"/download/{job_id}",
            "thumbnail_url": f"/thumbnail/{job_id}",
            "clips_per_scene": 3,
            "total_clips": 12
        })

    except Exception as exc:
        set_job(job_id, {"status": "failed", "progress": 100, "error": str(exc)})


@app.get("/")
def root():
    return {"ok": True, "service": "ViralShrimpie FFmpeg Renderer", "version": "2.0.1"}


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


@app.get("/thumbnail/{job_id}")
def thumbnail(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("status") != "succeeded":
        raise HTTPException(
            status_code=409,
            detail=f"Job status: {job.get('status')}",
        )

    thumbnail_path = BASE_DIR / job_id / "thumbnail.jpg"
    if not thumbnail_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Thumbnail not found",
        )

    return FileResponse(
        thumbnail_path,
        media_type="image/jpeg",
        filename=f"viralshrimpie-{job_id}-thumbnail.jpg",
    )


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
