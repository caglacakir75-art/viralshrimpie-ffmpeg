import asyncio
import json
import os
import random
import re
import subprocess
import textwrap
import uuid
from pathlib import Path
from typing import Any, List, Literal

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, HttpUrl

app = FastAPI(title="Channel Factory FFmpeg Renderer", version="3.3.1")

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
VIDEO_CONTRAST = 1.02
VIDEO_SATURATION = 1.00

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


class StorySegment(BaseModel):
    scene_index: int = Field(ge=1)
    text: str = Field(min_length=1, max_length=1000)
    audio_url: HttpUrl


class DocumentaryRenderRequest(BaseModel):
    video_urls: List[HttpUrl] = Field(min_length=1, max_length=60)
    hook_audio_url: HttpUrl
    hook_text: str = Field(min_length=1, max_length=300)
    story_segments: List[StorySegment] = Field(min_length=1, max_length=20)
    follow_audio_url: HttpUrl
    hook_pause: float = Field(default=1.5, ge=0.0, le=5.0)
    outro_pause: float = Field(default=1.8, ge=0.0, le=5.0)
    follow_text: str = Field(default="Follow for more.", min_length=1, max_length=100)
    music_url: HttpUrl | None = None
    music_urls: List[HttpUrl] | None = None
    music_volume: float = DEFAULT_MUSIC_VOLUME
    thumbnail_text: str | None = None
    title: str | None = None
    width: int = 1080
    height: int = 1920
    fps: int = 30
    font_size: int = BODY_FONT_SIZE
    text_margin: int = BOTTOM_MARGIN


class PhoneIntro(BaseModel):
    text: List[str] = Field(min_length=1, max_length=8)
    full_text: str = Field(min_length=1, max_length=500)
    audio_url: HttpUrl
    duration_seconds: float = Field(default=0.0, ge=0.0, le=30.0)


class PhoneAlignment(BaseModel):
    characters: List[str]
    character_start_times_seconds: List[float]
    character_end_times_seconds: List[float]


class PhoneMessageSegment(BaseModel):
    segment_index: int = Field(ge=0, le=20)
    segment_number: int = Field(ge=1, le=20)
    purpose: str = Field(min_length=1, max_length=100)
    text: str = Field(min_length=1, max_length=1500)
    audio_url: HttpUrl
    duration_seconds: float = Field(default=0.0, ge=0.0, le=120.0)
    alignment: PhoneAlignment | None = None


class PhoneMessage(BaseModel):
    speaker_label: str = Field(default="Someone who cares", max_length=120)
    title: str = Field(default="You Have A Call", max_length=200)
    full_speech: str = Field(min_length=1, max_length=8000)
    word_count: int = Field(default=0, ge=0, le=2000)
    segments: List[PhoneMessageSegment] = Field(min_length=1, max_length=20)


class PhoneTimeline(BaseModel):
    intro_start_seconds: float = Field(default=0.0, ge=0.0)
    post_intro_pause_seconds: float = Field(default=3.5, ge=0.0, le=10.0)
    message_start_seconds: float = Field(default=0.0, ge=0.0)
    estimated_total_duration_seconds: float = Field(default=0.0, ge=0.0)


class PhoneVisual(BaseModel):
    intro_scene: str = "incoming_phone_call"
    intro_action: str = "person_raises_phone_to_ear"
    message_scene: str = "phone_held_to_ear"
    background_style: str = "cinematic_emotional_minimal"
    subtitle_style: str = "centered_clean"


class PhoneSafety(BaseModel):
    fictional_emotional_experience: bool = True
    supernatural_claim: bool = False
    call_to_action: bool = False


class PhoneCallRenderRequest(BaseModel):
    renderer: Literal["phone_call_static_v1"]
    intro: PhoneIntro
    message: PhoneMessage
    timeline: PhoneTimeline | None = None
    visual: PhoneVisual | None = None
    safety: PhoneSafety | None = None
    width: int = Field(default=1080, ge=360, le=2160)
    height: int = Field(default=1920, ge=640, le=3840)
    fps: int = Field(default=30, ge=15, le=60)
    font_size: int = Field(default=46, ge=24, le=100)
    text_margin: int = Field(default=240, ge=40, le=700)


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


async def render_documentary_job(job_id: str, payload: DocumentaryRenderRequest) -> None:
    job_dir = BASE_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    set_job(job_id, {"status": "downloading", "progress": 5})

    try:
        story_segments = sorted(
            payload.story_segments,
            key=lambda segment: segment.scene_index,
        )
        expected_indexes = list(range(1, len(story_segments) + 1))
        actual_indexes = [segment.scene_index for segment in story_segments]
        if actual_indexes != expected_indexes:
            raise ValueError(
                "story_segments scene_index values must be consecutive and start at 1"
            )

        narrative_scene_count = 1 + len(story_segments)
        if len(payload.video_urls) < narrative_scene_count:
            raise ValueError(
                f"At least {narrative_scene_count} video URLs are required for "
                f"{narrative_scene_count} narrative scenes"
            )

        video_paths = [
            job_dir / f"video_{index + 1}.mp4"
            for index in range(len(payload.video_urls))
        ]
        hook_audio_path = job_dir / "hook_voice.mp3"
        story_audio_paths = [
            job_dir / f"story_voice_{segment.scene_index}.mp3"
            for segment in story_segments
        ]
        follow_audio_path = job_dir / "follow_voice.mp3"
        audio_path = job_dir / "combined_voice.wav"
        music_path = job_dir / "background_music.mp3"

        selected_music_url = None
        available_music_urls: list[str] = []
        if payload.music_urls:
            available_music_urls.extend(str(url) for url in payload.music_urls)
        if payload.music_url:
            available_music_urls.append(str(payload.music_url))
        if available_music_urls:
            selected_music_url = random.choice(available_music_urls)

        async with httpx.AsyncClient() as client:
            await asyncio.gather(*[
                download_file(client, str(url), path)
                for url, path in zip(payload.video_urls, video_paths)
            ])
            await download_file(client, str(payload.hook_audio_url), hook_audio_path)
            await asyncio.gather(*[
                download_file(client, str(segment.audio_url), path)
                for segment, path in zip(story_segments, story_audio_paths)
            ])
            await download_file(client, str(payload.follow_audio_url), follow_audio_path)
            if selected_music_url:
                await download_file(client, selected_music_url, music_path)

        set_job(job_id, {"status": "rendering", "progress": 25})

        hook_audio_duration = probe_duration(hook_audio_path)
        story_audio_durations = [probe_duration(path) for path in story_audio_paths]
        follow_audio_duration = probe_duration(follow_audio_path)

        # Build exact narration timing:
        # hook -> hook pause -> each story segment -> outro pause -> follow CTA.
        audio_inputs: list[str] = ["-i", str(hook_audio_path)]
        filter_parts = [
            "[0:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[a0]"
        ]
        concat_labels = ["[a0]"]
        input_index = 1

        if payload.hook_pause > 0:
            audio_inputs.extend([
                "-f", "lavfi", "-t", f"{payload.hook_pause:.3f}",
                "-i", "anullsrc=r=48000:cl=stereo",
            ])
            filter_parts.append(
                f"[{input_index}:a]aresample=48000,"
                f"aformat=sample_fmts=fltp:channel_layouts=stereo[a{input_index}]"
            )
            concat_labels.append(f"[a{input_index}]")
            input_index += 1

        for path in story_audio_paths:
            audio_inputs.extend(["-i", str(path)])
            filter_parts.append(
                f"[{input_index}:a]aresample=48000,"
                f"aformat=sample_fmts=fltp:channel_layouts=stereo[a{input_index}]"
            )
            concat_labels.append(f"[a{input_index}]")
            input_index += 1

        if payload.outro_pause > 0:
            audio_inputs.extend([
                "-f", "lavfi", "-t", f"{payload.outro_pause:.3f}",
                "-i", "anullsrc=r=48000:cl=stereo",
            ])
            filter_parts.append(
                f"[{input_index}:a]aresample=48000,"
                f"aformat=sample_fmts=fltp:channel_layouts=stereo[a{input_index}]"
            )
            concat_labels.append(f"[a{input_index}]")
            input_index += 1

        audio_inputs.extend(["-i", str(follow_audio_path)])
        filter_parts.append(
            f"[{input_index}:a]aresample=48000,"
            f"aformat=sample_fmts=fltp:channel_layouts=stereo[a{input_index}]"
        )
        concat_labels.append(f"[a{input_index}]")
        filter_parts.append(
            "".join(concat_labels)
            + f"concat=n={len(concat_labels)}:v=0:a=1[aout]"
        )

        run([
            "ffmpeg", "-y",
            *audio_inputs,
            "-filter_complex", ";".join(filter_parts),
            "-map", "[aout]",
            "-c:a", "pcm_s16le",
            str(audio_path),
        ])

        audio_duration = probe_duration(audio_path)

        # Narrative scenes get source clips. Pause and CTA reuse the final source.
        base_count, remainder = divmod(len(video_paths), narrative_scene_count)
        clip_counts = [
            base_count + (1 if index < remainder else 0)
            for index in range(narrative_scene_count)
        ]
        scene_sources: list[list[Path]] = []
        cursor = 0
        for count in clip_counts:
            scene_sources.append(video_paths[cursor:cursor + count])
            cursor += count

        timeline_segments = [
            {
                "kind": "hook",
                "duration": hook_audio_duration + payload.hook_pause,
                "text": payload.story_segments[0].text if False else None,
                "sources": scene_sources[0],
            }
        ]
        # Hook text is not part of story_segments, so obtain it from the prepared
        # scene data sent by n8n through the first visual caption fallback below.
        # The render request intentionally keeps hook audio separate; caption text
        # is supplied via title only if no explicit hook_text is present.
        hook_caption = getattr(payload, "hook_text", None) or ""
        timeline_segments[0]["text"] = hook_caption

        for index, (segment, duration) in enumerate(
            zip(story_segments, story_audio_durations),
            start=1,
        ):
            timeline_segments.append({
                "kind": "story",
                "duration": duration,
                "text": segment.text,
                "sources": scene_sources[index],
            })

        if payload.outro_pause > 0:
            timeline_segments.append({
                "kind": "pause",
                "duration": payload.outro_pause,
                "text": "",
                "sources": [video_paths[-1]],
            })

        timeline_segments.append({
            "kind": "follow",
            "duration": follow_audio_duration,
            "text": payload.follow_text,
            "sources": [video_paths[-1]],
        })

        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        rendered_paths: list[Path] = []
        rendered_clip_counter = 0
        total_render_clips = sum(len(segment["sources"]) for segment in timeline_segments)

        for timeline_index, segment in enumerate(timeline_segments, start=1):
            segment_duration = float(segment["duration"])
            if segment_duration <= 0:
                continue
            sources = segment["sources"]
            clip_count = max(1, len(sources))
            base_clip_duration = segment_duration / clip_count
            clip_durations = [base_clip_duration] * clip_count
            clip_durations[-1] += segment_duration - sum(clip_durations)

            segment_text = str(segment["text"] or "").strip()
            show_caption = bool(segment_text)
            caption = ""
            current_font_size = payload.font_size
            text_path = job_dir / f"timeline_{timeline_index}.txt"
            if show_caption:
                caption, current_font_size = choose_caption_layout(
                    text=segment_text,
                    is_hook=segment["kind"] == "hook",
                    video_width=payload.width,
                    requested_font_size=payload.font_size,
                )
                text_path.write_text(caption, encoding="utf-8")

            for clip_index, (source, duration) in enumerate(zip(sources, clip_durations)):
                rendered_clip_counter += 1
                target = job_dir / (
                    f"timeline_{timeline_index}_clip_{clip_index + 1}.mp4"
                )
                rendered_paths.append(target)

                filters = [
                    (
                        f"scale={payload.width}:{payload.height}:"
                        f"force_original_aspect_ratio=increase"
                    ),
                    f"crop={payload.width}:{payload.height}",
                ]

                if ENABLE_COLOR_SOFTENING:
                    filters.append(
                        f"eq=contrast={VIDEO_CONTRAST}:saturation={VIDEO_SATURATION}"
                    )

                if ENABLE_KEN_BURNS:
                    filters.append(
                        build_ken_burns_filter(
                            scene_index=rendered_clip_counter,
                            duration=duration,
                            fps=payload.fps,
                            width=payload.width,
                            height=payload.height,
                        )
                    )
                else:
                    filters.append(f"fps={payload.fps}")

                if ENABLE_SCENE_FADE:
                    fade_out_start = max(0.0, duration - SCENE_FADE_SECONDS)
                    filters.extend([
                        f"fade=t=in:st=0:d={SCENE_FADE_SECONDS}",
                        (
                            f"fade=t=out:st={fade_out_start:.3f}:"
                            f"d={SCENE_FADE_SECONDS}"
                        ),
                    ])

                if show_caption:
                    fade = min(TEXT_FADE_SECONDS, max(0.05, duration / 3))
                    if clip_index == 0:
                        alpha_expression = f"if(lt(t,{fade}),t/{fade},1)"
                    elif clip_index == clip_count - 1:
                        alpha_expression = (
                            f"if(gt(t,{max(0.0, duration - fade):.3f}),"
                            f"({duration:.3f}-t)/{fade},1)"
                        )
                    else:
                        alpha_expression = "1"

                    base_y = f"h-text_h-{payload.text_margin}"
                    filters.append(
                        f"drawtext=fontfile={font_path}:"
                        f"textfile={text_path.as_posix()}:"
                        f"fontcolor=white:fontsize={current_font_size}:"
                        f"line_spacing={TEXT_LINE_SPACING}:"
                        f"borderw={TEXT_BORDER_WIDTH}:bordercolor=black:"
                        f"box=1:boxcolor=black@{TEXT_BOX_ALPHA}:"
                        f"boxborderw={TEXT_BOX_PADDING}:"
                        f"alpha='{alpha_expression}':"
                        f"x=(w-text_w)/2:y='{base_y}'"
                    )

                video_filter = ",".join(filters)
                run([
                    "ffmpeg", "-y", "-stream_loop", "-1", "-i", str(source),
                    "-t", f"{duration:.3f}", "-vf", video_filter, "-an",
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
                    "-threads", "1", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(target),
                ])

                progress = 25 + int((rendered_clip_counter / total_render_clips) * 48)
                set_job(job_id, {"status": "rendering", "progress": progress})

        concat_path = job_dir / "concat.txt"
        concat_path.write_text(
            "\n".join(f"file '{path.as_posix()}'" for path in rendered_paths),
            encoding="utf-8",
        )
        silent_video = job_dir / "silent.mp4"
        run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i",
            str(concat_path), "-c", "copy", str(silent_video),
        ])

        final_video = job_dir / "final.mp4"
        voice_filter = (
            f"loudnorm=I={VOICE_TARGET_LUFS}:TP={VOICE_TRUE_PEAK}:LRA={VOICE_LRA}"
            if ENABLE_AUDIO_NORMALIZATION
            else "anull"
        )

        if selected_music_url and music_path.exists():
            music_duration = probe_duration(music_path)
            max_start = max(0.0, music_duration - audio_duration - 1.0)
            music_start = random.uniform(0.0, max_start) if max_start > 0 else 0.0
            music_fade_out_start = max(0.0, audio_duration - MUSIC_FADE_SECONDS)
            filter_complex = (
                f"[1:a]{voice_filter},aresample=48000[voice];"
                f"[2:a]atrim=start={music_start:.3f}:duration={audio_duration:.3f},"
                f"asetpts=N/SR/TB,aresample=48000,"
                f"afade=t=in:st=0:d={MUSIC_FADE_SECONDS},"
                f"afade=t=out:st={music_fade_out_start:.3f}:d={MUSIC_FADE_SECONDS},"
                f"volume={payload.music_volume}[music];"
                f"[voice][music]amix=inputs=2:duration=first:"
                f"dropout_transition=0:normalize=0[aout]"
            )
            run([
                "ffmpeg", "-y", "-i", str(silent_video), "-i", str(audio_path),
                "-stream_loop", "-1", "-i", str(music_path),
                "-filter_complex", filter_complex,
                "-map", "0:v:0", "-map", "[aout]",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-shortest", "-movflags", "+faststart", str(final_video),
            ])
        else:
            run([
                "ffmpeg", "-y", "-i", str(silent_video), "-i", str(audio_path),
                "-filter:a", voice_filter,
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-shortest", "-movflags", "+faststart", str(final_video),
            ])

        thumbnail_path = job_dir / "thumbnail.jpg"
        thumbnail_label = payload.thumbnail_text or payload.title or payload.follow_text
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
            "story_audio_durations": [round(value, 3) for value in story_audio_durations],
            "follow_audio_duration": round(follow_audio_duration, 3),
            "hook_pause": round(payload.hook_pause, 3),
            "outro_pause": round(payload.outro_pause, 3),
            "story_segment_count": len(story_segments),
            "download_url": f"/download/{job_id}",
            "thumbnail_url": f"/thumbnail/{job_id}",
            "total_source_videos": len(video_paths),
            "total_rendered_clips": len(rendered_paths),
        })

    except Exception as exc:
        set_job(job_id, {"status": "failed", "progress": 100, "error": str(exc)})


def _escape_drawtext(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace("%", "\\%")
    )


def split_caption_chunks(text: str, max_words: int = 8) -> list[str]:
    """Split spoken text into short subtitle phrases.

    This is sentence/phrase-level timing, not true word alignment. It keeps
    punctuation when possible and limits each on-screen caption to a compact
    phrase so the subtitle follows the voice much more closely.
    """
    cleaned = " ".join(str(text or "").split()).strip()
    if not cleaned:
        return []

    # First split on sentence endings and deliberate ellipsis pauses.
    sentences = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+|\s*…\s*|\s*\.\.\.\s*", cleaned)
        if part and part.strip()
    ]

    chunks: list[str] = []
    for sentence in sentences:
        words = sentence.split()
        if len(words) <= max_words:
            chunks.append(sentence)
            continue

        # Prefer natural comma/semicolon boundaries before hard word limits.
        clauses = [
            clause.strip()
            for clause in re.split(r"(?<=[,;:])\s+", sentence)
            if clause and clause.strip()
        ]

        current: list[str] = []
        for clause in clauses:
            clause_words = clause.split()
            if current and len(current) + len(clause_words) > max_words:
                chunks.append(" ".join(current).strip())
                current = []

            if len(clause_words) <= max_words:
                current.extend(clause_words)
                continue

            if current:
                chunks.append(" ".join(current).strip())
                current = []

            for start in range(0, len(clause_words), max_words):
                part = " ".join(clause_words[start:start + max_words]).strip()
                if part:
                    chunks.append(part)

        if current:
            chunks.append(" ".join(current).strip())

    return [chunk for chunk in chunks if chunk]



def normalize_speaker_label(value: str) -> str:
    """Return a short caller label that is safe for one-line display."""
    raw = " ".join(str(value or "").split()).strip()
    if not raw:
        return "Someone Who Cares"

    lowered = raw.lower()
    if any(token in lowered for token in ("future self", "older self", "healed self")):
        return "Your Future Self"
    if any(token in lowered for token in ("younger self", "childhood self", "inner child")):
        return "Your Younger Self"
    if any(token in lowered for token in ("mother", "mom", "maternal")):
        return "Mom"
    if any(token in lowered for token in ("father", "dad", "paternal")):
        return "Dad"
    if "best friend" in lowered or "close friend" in lowered:
        return "Your Best Friend"
    if any(token in lowered for token in ("grandmother", "grandma")):
        return "Grandma"
    if any(token in lowered for token in ("grandfather", "grandpa")):
        return "Grandpa"

    # Prefer the first concise phrase when the model returned a description.
    candidate = re.split(r"[,;–—|]", raw, maxsplit=1)[0].strip()
    if len(candidate) <= 24:
        return candidate

    words = candidate.split()
    kept: list[str] = []
    for word in words:
        proposed = " ".join([*kept, word])
        if len(proposed) > 24:
            break
        kept.append(word)
    return " ".join(kept).strip() or "Someone Who Cares"


def aligned_caption_timeline(
    alignment: PhoneAlignment | None,
    fallback_text: str,
    segment_duration: float,
    max_words: int = 7,
) -> list[tuple[str, float]]:
    """Build short caption clips from ElevenLabs character timestamps.

    Returns (caption_text, clip_duration) entries whose durations add up to the
    complete audio segment. Empty text entries intentionally preserve pauses.
    """
    if alignment is None:
        chunks = split_caption_chunks(fallback_text, max_words=max_words) or [fallback_text]
        return list(zip(chunks, weighted_durations(chunks, segment_duration)))

    chars = alignment.characters
    starts = alignment.character_start_times_seconds
    ends = alignment.character_end_times_seconds
    if not chars or len(chars) != len(starts) or len(chars) != len(ends):
        chunks = split_caption_chunks(fallback_text, max_words=max_words) or [fallback_text]
        return list(zip(chunks, weighted_durations(chunks, segment_duration)))

    # Convert character alignment into timed words.
    words: list[dict[str, Any]] = []
    current_chars: list[str] = []
    word_start: float | None = None
    word_end: float | None = None

    def flush_word() -> None:
        nonlocal current_chars, word_start, word_end
        text = "".join(current_chars).strip()
        if text and word_start is not None and word_end is not None:
            words.append({"text": text, "start": word_start, "end": word_end})
        current_chars = []
        word_start = None
        word_end = None

    for char, start, end_time in zip(chars, starts, ends):
        if str(char).isspace():
            flush_word()
            continue
        if word_start is None:
            word_start = float(start)
        current_chars.append(str(char))
        word_end = float(end_time)
    flush_word()

    if not words:
        chunks = split_caption_chunks(fallback_text, max_words=max_words) or [fallback_text]
        return list(zip(chunks, weighted_durations(chunks, segment_duration)))

    groups: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for word in words:
        current.append(word)
        ends_phrase = bool(re.search(r"[.!?…,:;]$", word["text"]))
        if len(current) >= max_words or (ends_phrase and len(current) >= 2):
            groups.append({
                "text": " ".join(item["text"] for item in current),
                "start": current[0]["start"],
                "end": current[-1]["end"],
            })
            current = []
    if current:
        groups.append({
            "text": " ".join(item["text"] for item in current),
            "start": current[0]["start"],
            "end": current[-1]["end"],
        })

    timeline: list[tuple[str, float]] = []
    cursor = 0.0
    for index, group in enumerate(groups):
        start = max(cursor, float(group["start"]))
        if start - cursor > 0.06:
            timeline.append(("", start - cursor))
        next_start = (
            float(groups[index + 1]["start"])
            if index + 1 < len(groups)
            else segment_duration
        )
        end = max(float(group["end"]), next_start)
        end = min(max(end, start + 0.08), segment_duration)
        timeline.append((str(group["text"]).strip(), end - start))
        cursor = end

    if cursor < segment_duration - 0.03:
        timeline.append(("", segment_duration - cursor))

    # Avoid zero-length clips and correct small floating-point drift.
    timeline = [(text, duration) for text, duration in timeline if duration > 0.03]
    total = sum(duration for _, duration in timeline)
    if timeline and abs(total - segment_duration) > 0.01:
        text, duration = timeline[-1]
        timeline[-1] = (text, max(0.04, duration + segment_duration - total))
    return timeline

def create_phone_visual_clip(
    output_path: Path,
    duration: float,
    width: int,
    height: int,
    fps: int,
    speaker_label: str,
    caption_text: str,
    job_dir: Path,
    clip_index: int,
    is_intro: bool,
    requested_font_size: int,
) -> None:
    """Create an ethereal, softly blurred coastal-style background.

    The visual is generated procedurally, so the renderer does not depend on
    external stock footage. It suggests sky, horizon and sea without making a
    literal supernatural claim.
    """
    font_regular = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font_bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

    caption, caption_size = choose_caption_layout(
        text=caption_text,
        is_hook=is_intro,
        video_width=width,
        requested_font_size=requested_font_size,
    )
    caption_path = job_dir / f"phone_caption_{clip_index}.txt"
    caption_path.write_text(caption, encoding="utf-8")

    speaker_path = job_dir / f"phone_speaker_{clip_index}.txt"
    speaker_path.write_text(normalize_speaker_label(speaker_label), encoding="utf-8")

    top_label = "INCOMING CALL" if is_intro else "ON THE LINE"
    top_label = _escape_drawtext(top_label)

    # Build a soft sky / horizon / sea composition, then blur and vignette it.
    # The subtle zoom creates life without looking like a stock-video loop.
    background_filters = [
        "drawbox=x=0:y=0:w=iw:h=ih*0.55:color=0x9FB7C8:t=fill",
        "drawbox=x=0:y=ih*0.55:w=iw:h=ih*0.45:color=0x49677A:t=fill",
        "drawbox=x=0:y=ih*0.49:w=iw:h=ih*0.12:color=0xD8C6A2@0.48:t=fill",
        "drawbox=x=0:y=ih*0.70:w=iw:h=ih*0.30:color=0x243D4A@0.26:t=fill",
        "gblur=sigma=42:steps=3",
        "eq=brightness=-0.06:contrast=0.92:saturation=0.78",
        "noise=alls=3:allf=t+u",
        "vignette=PI/5",
        (
            f"zoompan=z='min(1.0+0.018*on/max(1,{max(1, int(duration * fps))}),1.018)':"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d=1:s={width}x{height}:fps={fps}"
        ),
    ]

    overlay_filters = [
        "drawbox=x=55:y=95:w=iw-110:h=ih-190:color=black@0.10:t=fill",
        "drawbox=x=95:y=165:w=iw-190:h=245:color=black@0.18:t=fill",
        (
            f"drawtext=fontfile={font_bold}:text='{top_label}':"
            f"fontcolor=white@0.68:fontsize=32:x=(w-text_w)/2:y=215"
        ),
        (
            f"drawtext=fontfile={font_bold}:textfile={speaker_path.as_posix()}:"
            f"fontcolor=white:fontsize=56:x=(w-text_w)/2:y=285"
        ),
        (
            f"drawtext=fontfile={font_regular}:textfile={caption_path.as_posix()}:"
            f"fontcolor=white:fontsize={caption_size}:line_spacing={TEXT_LINE_SPACING}:"
            f"borderw=3:bordercolor=black@0.68:box=1:boxcolor=black@0.16:"
            f"boxborderw=30:x=(w-text_w)/2:y=h-text_h-285"
        ),
    ]

    run([
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c=0x9FB7C8:s={width}x{height}:r={fps}",
        "-t", f"{duration:.3f}",
        "-vf", ",".join([*background_filters, *overlay_filters]),
        "-an",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path),
    ])


async def render_phone_call_static_job(
    job_id: str, payload: PhoneCallRenderRequest
) -> None:
    job_dir = BASE_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    set_job(job_id, {"status": "downloading", "progress": 5, "renderer": payload.renderer})

    try:
        segments = sorted(payload.message.segments, key=lambda item: item.segment_index)
        expected_indexes = list(range(len(segments)))
        actual_indexes = [item.segment_index for item in segments]
        if actual_indexes != expected_indexes:
            raise ValueError(
                "message.segments segment_index values must be consecutive and start at 0"
            )

        intro_audio_path = job_dir / "intro_voice.mp3"
        segment_audio_paths = [
            job_dir / f"message_voice_{segment.segment_index + 1}.mp3"
            for segment in segments
        ]

        async with httpx.AsyncClient() as client:
            await download_file(client, str(payload.intro.audio_url), intro_audio_path)
            await asyncio.gather(*[
                download_file(client, str(segment.audio_url), path)
                for segment, path in zip(segments, segment_audio_paths)
            ])

        set_job(job_id, {"status": "rendering", "progress": 20, "renderer": payload.renderer})

        intro_duration = probe_duration(intro_audio_path)
        segment_durations = [probe_duration(path) for path in segment_audio_paths]

        # Concatenate intro -> intentional post-intro silence -> message segments.
        # The pause gives the viewer time to raise the phone after
        # "Put it on your ear." before the emotional message begins.
        post_intro_pause = float(payload.timeline.post_intro_pause_seconds or 0.0)

        audio_inputs: list[str] = []
        filter_parts: list[str] = []
        labels: list[str] = []
        input_index = 0

        audio_inputs.extend(["-i", str(intro_audio_path)])
        filter_parts.append(
            f"[{input_index}:a]aresample=48000,"
            f"aformat=sample_fmts=fltp:channel_layouts=stereo[a{input_index}]"
        )
        labels.append(f"[a{input_index}]")
        input_index += 1

        if post_intro_pause > 0:
            audio_inputs.extend([
                "-f", "lavfi", "-t", f"{post_intro_pause:.3f}",
                "-i", "anullsrc=r=48000:cl=stereo",
            ])
            filter_parts.append(
                f"[{input_index}:a]aresample=48000,"
                f"aformat=sample_fmts=fltp:channel_layouts=stereo[a{input_index}]"
            )
            labels.append(f"[a{input_index}]")
            input_index += 1

        for path in segment_audio_paths:
            audio_inputs.extend(["-i", str(path)])
            filter_parts.append(
                f"[{input_index}:a]aresample=48000,"
                f"aformat=sample_fmts=fltp:channel_layouts=stereo[a{input_index}]"
            )
            labels.append(f"[a{input_index}]")
            input_index += 1

        filter_parts.append(
            "".join(labels) + f"concat=n={len(labels)}:v=0:a=1[aout]"
        )

        combined_audio = job_dir / "combined_voice.wav"
        run([
            "ffmpeg", "-y", *audio_inputs,
            "-filter_complex", ";".join(filter_parts),
            "-map", "[aout]",
            "-c:a", "pcm_s16le",
            str(combined_audio),
        ])
        total_duration = probe_duration(combined_audio)

        visual_clips: list[Path] = []
        intro_clip = job_dir / "phone_visual_intro.mp4"
        create_phone_visual_clip(
            output_path=intro_clip,
            duration=intro_duration,
            width=payload.width,
            height=payload.height,
            fps=payload.fps,
            speaker_label=payload.message.speaker_label or "You Have A Call",
            caption_text=payload.intro.full_text,
            job_dir=job_dir,
            clip_index=0,
            is_intro=True,
            requested_font_size=payload.font_size,
        )
        visual_clips.append(intro_clip)

        if post_intro_pause > 0:
            pause_clip = job_dir / "phone_visual_post_intro_pause.mp4"
            create_phone_visual_clip(
                output_path=pause_clip,
                duration=post_intro_pause,
                width=payload.width,
                height=payload.height,
                fps=payload.fps,
                speaker_label=payload.message.speaker_label or "You Have A Call",
                caption_text="",
                job_dir=job_dir,
                clip_index=1,
                is_intro=False,
                requested_font_size=payload.font_size,
            )
            visual_clips.append(pause_clip)

        phone_clip_counter = 2 if post_intro_pause > 0 else 1
        rendered_caption_chunks = 0

        for segment_position, (segment, duration) in enumerate(
            zip(segments, segment_durations),
            start=1,
        ):
            caption_timeline = aligned_caption_timeline(
                alignment=segment.alignment,
                fallback_text=segment.text,
                segment_duration=duration,
                max_words=7,
            )

            for chunk_position, (caption_chunk, chunk_duration) in enumerate(
                caption_timeline,
                start=1,
            ):
                target = job_dir / (
                    f"phone_visual_message_{segment_position}_"
                    f"caption_{chunk_position}.mp4"
                )
                create_phone_visual_clip(
                    output_path=target,
                    duration=chunk_duration,
                    width=payload.width,
                    height=payload.height,
                    fps=payload.fps,
                    speaker_label=(
                        payload.message.speaker_label or "Someone who cares"
                    ),
                    caption_text=caption_chunk,
                    job_dir=job_dir,
                    clip_index=phone_clip_counter,
                    is_intro=False,
                    requested_font_size=payload.font_size,
                )
                visual_clips.append(target)
                phone_clip_counter += 1
                rendered_caption_chunks += 1

            progress = 25 + int(
                (segment_position / max(1, len(segments))) * 50
            )
            set_job(
                job_id,
                {
                    "status": "rendering",
                    "progress": progress,
                    "renderer": payload.renderer,
                },
            )

        concat_path = job_dir / "phone_concat.txt"
        concat_path.write_text(
            "\n".join(f"file '{path.as_posix()}'" for path in visual_clips),
            encoding="utf-8",
        )
        silent_video = job_dir / "silent.mp4"
        run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_path), "-c", "copy", str(silent_video),
        ])

        final_video = job_dir / "final.mp4"
        voice_filter = (
            f"loudnorm=I={VOICE_TARGET_LUFS}:TP={VOICE_TRUE_PEAK}:LRA={VOICE_LRA}"
            if ENABLE_AUDIO_NORMALIZATION else "anull"
        )

        # Procedural ambient layer: very soft, filtered pink noise with a slow
        # swell. It behaves like distant ocean air without requiring an
        # external audio asset and remains far below the narration.
        ambient_source = (
            "anoisesrc=color=pink:amplitude=0.10:r=48000,"
            "highpass=f=70,lowpass=f=850,"
            "tremolo=f=0.12:d=0.62,"
            "aecho=0.8:0.75:90:0.12"
        )
        ambient_fade_out = max(0.0, total_duration - 1.4)
        filter_complex = (
            f"[1:a]{voice_filter},aformat=channel_layouts=stereo[voice];"
            f"[2:a]volume=0.032,"
            f"afade=t=in:st=0:d=1.2,"
            f"afade=t=out:st={ambient_fade_out:.3f}:d=1.4,"
            f"aformat=channel_layouts=stereo[ambient];"
            f"[voice][ambient]amix=inputs=2:duration=first:"
            f"dropout_transition=0:normalize=0[aout]"
        )

        run([
            "ffmpeg", "-y",
            "-i", str(silent_video),
            "-i", str(combined_audio),
            "-f", "lavfi", "-t", f"{total_duration:.3f}",
            "-i", ambient_source,
            "-filter_complex", filter_complex,
            "-map", "0:v:0",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            str(final_video),
        ])

        thumbnail_path = job_dir / "thumbnail.jpg"
        run([
            "ffmpeg", "-y", "-ss", "0.2", "-i", str(final_video),
            "-frames:v", "1", "-q:v", "3", str(thumbnail_path),
        ])

        set_job(job_id, {
            "status": "succeeded",
            "progress": 100,
            "renderer": payload.renderer,
            "duration": round(total_duration, 3),
            "intro_audio_duration": round(intro_duration, 3),
            "post_intro_pause_seconds": round(post_intro_pause, 3),
            "message_audio_durations": [round(value, 3) for value in segment_durations],
            "message_segment_count": len(segments),
            "caption_chunk_count": rendered_caption_chunks,
            "caption_timing_mode": "elevenlabs_alignment_v1",
            "visual_style": "procedural_blurred_coast_v1",
            "ambient_style": "procedural_soft_ocean_v1",
            "ambient_volume": 0.032,
            "download_url": f"/download/{job_id}",
            "thumbnail_url": f"/thumbnail/{job_id}",
        })

    except Exception as exc:
        set_job(job_id, {
            "status": "failed",
            "progress": 100,
            "renderer": payload.renderer,
            "error": str(exc),
        })


@app.get("/")
def root():
    return {"ok": True, "service": "Channel Factory FFmpeg Renderer", "version": "3.3.1", "renderers": ["documentary_v1", "phone_call_static_v1"]}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/render")
async def create_render(payload: dict[str, Any], background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    set_job(job_id, {"status": "queued", "progress": 0})
    renderer_name = str(payload.get("renderer") or "documentary_v1")
    try:
        if renderer_name == "phone_call_static_v1":
            validated_payload = PhoneCallRenderRequest.model_validate(payload)
            task = render_phone_call_static_job
        elif renderer_name in {"documentary_v1", "documentary_stock_v1"}:
            validated_payload = DocumentaryRenderRequest.model_validate(payload)
            task = render_documentary_job
        else:
            raise ValueError(f"Unknown renderer: {renderer_name}")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    background_tasks.add_task(task, job_id, validated_payload)
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
