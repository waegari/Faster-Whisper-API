<div align="center">

# **Faster-Whisper-API**

[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3100/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)

A lean **FastAPI wrapper around `faster-whisper`** for local transcription on **GPU or CPU**.
Designed for **inference-only** on an internal server: upload an audio/video file → get full text + timestamped segments.

> Backend: [`faster-whisper`](https://github.com/guillaumekln/faster-whisper) (CT2).
> Default: `large-v3`, `device="cuda"` if available (otherwise `cpu`), `compute_type="float16"` on GPU / `float32` on CPU.
</div>

---

## Features

* **Single process / single model instance** (warm & fast)
* **Audio or video input**
* **VAD filter** to suppress non-speech
* **Segment timestamps** (+ optional word timestamps)
* **Configurable language** (default `ko`)
* **File-size guardrails** (`MAX_AUDIO_BYTES`)
* Works on **CUDA GPUs** *and* **pure CPU** (no NVIDIA required)

---

## Project Structure

```
Faster-Whisper-API/
├─ app/
│  ├─ main.py
│  ├─ dependencies.py         # WhisperModel singleton
│  ├─ jobs.py
│  ├─ middleware.py
│  ├─ schemas.py              # Pydantic I/O models
│  ├─ routers/
│  │   └─ transcribe_async.py # /transcribe_async, /health
│  ├─ services/
│  │   ├─ transcriber.py
│  │   └─ audio_processor.py
│  └─ config/
│      └─ settings.py
├─ requirements.txt
└─ Dockerfile
```

---

## Requirements

* **Python 3.10+**
* **ffmpeg** installed on the system
* **GPU (optional)**: recent NVIDIA driver & CUDA for best performance
  → If no GPU is present, the API runs on **CPU** automatically when configured.

---

## Quick Start

### 1) Install

```bash
pip install -r requirements.txt
# or
pip install fastapi uvicorn[standard] pydub ffmpeg-python python-dotenv faster-whisper
```

Make sure `ffmpeg` is available on PATH.

### 2) Run API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Optional temp-file settings:

```bash
TEMP_DIR=D:\Temp
TEMP_FILE_TTL_HOURS=24
```

### 3) Health Check

```bash
curl http://localhost:8000/health
```

---

## API

### POST `/transcribe_async`

**Form data**

* `file` *(required)*: audio/video file (`.wav`, `.mp3`, `.mp4`, `.mkv`, `.mov`, `.m4a`, `.ogg`, …)
* `query` *(JSON string, optional)*:

  * `request_id`: string
  * `language`: default `"ko"`
  * `is_video`: default `false`
  * `start`: int (seconds)
  * `end`: int (seconds)
  * `vad`: bool, default `true`
  * `word_timestamps`: bool, default `false`


**Response (202 + Location header)**

* **Status**: `202 Accepted`
* **Headers**:

  * `Location: /jobs/<job_id>`
  * `X-Request-ID: <same-as-request-or-job_id>`
* **Body**:

  ```json
  {
    "job_id": "12345678-a1b2-c3d4-e5f6-g7h8i9j0k1l2",
    "status_url": "/jobs/12345678-a1b2-c3d4-e5f6-g7h8i9j0k1l2"
  }
  ```


**cURL**

```bash
curl -X POST "http://localhost:8000/transcribe_async" \
  -F "file=@/path/to/sample.mp3" \
  -F 'query={"language":"ko","vad":true,"word_timestamps":false}' -i
# HTTP/1.1 202 Accepted
# Location: /jobs/<job_id>
# ...
# {"job_id":"<job_id>","status_url":"/jobs/<job_id>"}
```

---


### GET `/jobs/{job_id}`

Fetch the job status. While processing, `progress` updates; when finished, the transcription result is included.

**Responses (examples)**

* **queued**

  ```json
  {
    "job_id": "<job_id>",
    "status": "queued",
    "started_at": null,
    "ended_at": null,
    "progress": 0.0,
    "message": "",
    "result": null,
    "request_id": null
  }
  ```

* **processing**

  ```json
  {
    "job_id": "<job_id>",
    "status": "processing",
    "started_at": 1762409945.60871,
    "ended_at": null,
    "progress": 0.2874,
    "message": "transcribing",
    "result": null,
    "request_id": null
  }
  ```

* **done**

  ```json
  {
    "job_id": "<job_id>",
    "status": "done",
    "started_at": 1762409945.60871,
    "ended_at": 1762410345.10293,
    "progress": 1.0,
    "message": "done",
    "result": {
      "language": "ko",
      "duration": 2926.59,
      "result": {
        "text": "full transcription text ...",
        "segments": [
          { "index": 0, "start": 0.50, "end": 2.10, "content": "..." }
        ]
      }
    },
    "request_id": null
  }
  ```

* **error**

  ```json
  {
    "job_id": "<job_id>",
    "status": "error",
    "started_at": 1762409945.60871,
    "ended_at": 1762410001.00412,
    "progress": 0.0,
    "message": "error details ...",
    "result": null,
    "request_id": null
  }
  ```

**cURL**

```bash
curl "http://localhost:8000/jobs/<job_id>"
```

> **Note:** To throttle GPU load and request bursts, tune the queue/concurrency via environment variables:
> `FW_MAX_CONCURRENCY` (per-GPU concurrent jobs), `FW_QUEUE_MAXSIZE` (pending queue size), `FW_RETRY_AFTER` (seconds for 503 Retry-After).

---


## Configuration

`app/config/settings.py` (defaults)

* `MAX_AUDIO_BYTES`: e.g., `25 * 1024 * 1024`
* `DEFAULT_SR`: 16000
* `DEFAULT_BR`: `'96k'`
* `DEFAULT_CH`: 1
* `MAX_CHUNK_DURATION_MS`: e.g., 2h05m

`app/dependencies.py` (model init)

* `model_name`: `"large-v3"`, `"medium"`, etc.
* `device`: `"cuda"` **or** `"cpu"`
* `compute_type` (recommendations):

  * GPU (CUDA): `"float16"` *(fast & accurate)*, or `"int8_float16"` *(lower VRAM)*
  * CPU: `"float32"` *(default & stable)*, or `"int8"` / `"int8_float32"` *(lower RAM, faster on some CPUs)*

> **Tip (CPU)**: Set environment variable `OMP_NUM_THREADS` or `CT2_NUM_THREADS` to tune CPU threads.

---

## Selecting GPU vs CPU

You can force the backend via environment variables (read by `dependencies.py`) or by editing `WhisperFactory`:

```bash
# CPU mode (no CUDA required)
export FW_DEVICE=cpu
export FW_COMPUTE=float32   # or int8 / int8_float32
uvicorn app.main:app --host 0.0.0.0 --port 8000

# GPU mode (if CUDA available)
export FW_DEVICE=cuda
export FW_COMPUTE=float16   # or int8_float16
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Example `dependencies.py` snippet:

```python
import os
from functools import lru_cache
from faster_whisper import WhisperModel

DEVICE = os.getenv("FW_DEVICE", "cuda")   # "cuda" or "cpu"
COMPUTE = os.getenv("FW_COMPUTE", "float16" if DEVICE == "cuda" else "float32")
MODEL = os.getenv("FW_MODEL", "large-v3")

@lru_cache(maxsize=1)
def get_model() -> WhisperModel:
    return WhisperModel(MODEL, device=DEVICE, compute_type=COMPUTE)
```

---

## Accuracy & Speed Notes

* Keep `temperature=0.0` to reduce repetition/hallucination.
* For long programs with music, `condition_on_previous_text=False` stabilizes first utterances after silence.
* **VAD** helps suppress long non-speech stretches.
* **Word timestamps** add latency—use only when needed.
* **CPU mode**: prefer `"float32"` first; try `"int8"` variants if you need to save RAM or speed up on large files.

---

## Docker

### GPU (CUDA)

**Dockerfile (GPU)**

```dockerfile
# CUDA + cuDNN runtime (Ubuntu 24.04)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies (Python, ffmpeg, build tools for wheels like PyAV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip python3-dev \
    ffmpeg \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# App
WORKDIR /srv/app
COPY requirements.txt .
RUN python3 -m venv /opt/venv && . /opt/venv/bin/activate \
 && pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY app ./app
EXPOSE 8000

# One model instance, one worker
ENV PATH="/opt/venv/bin:$PATH"
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000","--workers","1"]
```

**Build**

```bash
docker build -t faster-whisper-api:gpu .
```

**Run**

```bash
docker run --gpus all -p 8000:8000 \
  -e FW_DEVICE=cuda \
  -e FW_COMPUTE=float16 \
  -e FW_MODEL=large-v3 \
  -v /opt/whisper-cache:/root/.cache \
  faster-whisper-api:gpu
```

* `--gpus all` requires the **NVIDIA Container Toolkit**.
* The cache volume (`/root/.cache`) speeds up model reuse.
* If you hit cuDNN load errors, make sure the **host NVIDIA driver** is recent enough for CUDA 12.4 and that Docker sees the GPU (`docker run --gpus all nvidia/cuda:12.4.1-base-ubuntu24.04 nvidia-smi`).

**Quick health check**

```bash
curl http://localhost:8000/health
```

---

### CPU-only

**Dockerfile (CPU)**

```dockerfile
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/app
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY app ./app
EXPOSE 8000

# Force CPU path
ENV FW_DEVICE=cpu FW_COMPUTE=float32
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000","--workers","1"]
```

**Build & Run (CPU)**

```bash
docker build -t faster-whisper-api:cpu -f Dockerfile.cpu .
docker run -p 8000:8000 faster-whisper-api:cpu
```

---

### Notes & Troubleshooting

* **WSL2 (Windows):** enable *Docker Desktop → Settings → Resources → WSL Integration* for your distro. From inside WSL, `docker` must be on `PATH`.
* **PyAV build error (“pkg-config is required for building PyAV”):** keep `pkg-config` installed (already included above). If needed, retry the build (network hiccups can make pip fall back to sdist).
* **Slow first request:** the first call downloads the model and warms up the kernels. Use the cache volume and hit `/health` once on startup.
* **Throughput control:** tune queue/concurrency with env vars:

  * `FW_MAX_CONCURRENCY` — concurrent GPU jobs per container (default 1)
  * `FW_QUEUE_MAXSIZE` — pending queue size
  * `FW_RETRY_AFTER` — seconds for `503 Retry-After`
* **Model/config via env:**

  * `FW_MODEL` (e.g., `large-v3`, `medium`)
  * `FW_DEVICE` (`cuda` or `cpu`)
  * `FW_COMPUTE` (`float16` on GPU; `float32` or `int8_*` on CPU)

---

## Attribution

This project wraps **`faster-whisper`** (CT2).
See its license and the model licenses you deploy.

---

## License

MIT — applies to this repository’s source code.
Upstream libraries and any pretrained models retain their own licenses.