# faster-whisper-api

OpenAI Whisper API-compatible server backed by [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

## Endpoints

- `POST /v1/audio/transcriptions` — mirrors OpenAI's transcription endpoint
- `POST /v1/audio/translations` — mirrors OpenAI's translation endpoint (translates to English)
- `GET  /v1/models`
- `GET  /health`

### Supported form fields

`file`, `model`, `language`, `prompt`, `response_format` (`json` | `text` | `srt` | `verbose_json` | `vtt`), `temperature`, `timestamp_granularities[]` (`segment` | `word`).

## Environment

| Var | Default | Notes |
| --- | --- | --- |
| `WHISPER_MODEL` | `base` | `tiny`, `base`, `small`, `medium`, `large-v3`, etc. or HF id |
| `WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` |
| `WHISPER_COMPUTE_TYPE` | `int8` | `int8`, `int8_float16`, `float16`, `float32` |
| `WHISPER_DOWNLOAD_ROOT` | `/models` | Persisted model cache |
| `WHISPER_VAD` | `false` | Enable Silero VAD filter |
| `API_KEY` | _(unset)_ | If set, requests must send `Authorization: Bearer <key>` |

## Run locally

```bash
docker compose up --build
```

## Use with the OpenAI SDK

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="anything")
client.audio.transcriptions.create(model="whisper-1", file=open("a.mp3", "rb"))
```

## Dokploy

Deploy as a Docker Compose or Dockerfile application. Mount a persistent volume at `/models` so model weights survive redeploys. Set `WHISPER_MODEL`, `API_KEY`, and (for GPU hosts) `WHISPER_DEVICE=cuda` with an appropriate `WHISPER_COMPUTE_TYPE`.
