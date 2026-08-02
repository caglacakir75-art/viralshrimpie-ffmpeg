# ViralShrimpie FFmpeg Renderer

A small FastAPI + FFmpeg service for rendering four portrait scenes, four text overlays,
and one voiceover MP3 into a 1080x1920 YouTube Short.

## API

### POST /render

```json
{
  "video_urls": [
    "https://example.com/scene1.mp4",
    "https://example.com/scene2.mp4",
    "https://example.com/scene3.mp4",
    "https://example.com/scene4.mp4"
  ],
  "audio_url": "https://example.com/voiceover.mp3",
  "scene_texts": [
    "Scene one text.",
    "Scene two text.",
    "Scene three text.",
    "Scene four text."
  ]
}
```

Returns a `job_id`.

### GET /status/{job_id}

Poll until `status` is `succeeded`.

### GET /download/{job_id}

Downloads the final MP4.

## Render deployment

1. Create a GitHub repository.
2. Upload these files to the repository root.
3. In Render, create a **New Web Service** from the repository.
4. Render detects the Dockerfile.
5. Select the free plan.
6. Deploy.

The free Render service can sleep when inactive, so the first request may take longer.
