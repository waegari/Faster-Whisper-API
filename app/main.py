import os
import logging
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import transcribe_async
from .middleware import timing_middleware


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


def _mb(env_key: str, default_mb: int) -> int:
    return int(os.getenv(env_key, str(default_mb))) * 1024 * 1024

if sys.platform.startswith("win"):
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    BIN_DIR = os.path.join(ROOT_DIR, "bin")

    if os.path.exists(BIN_DIR):
        os.environ["PATH"] += os.pathsep + BIN_DIR
        print(f"[INFO] (Windows) Added local bin to PATH: {BIN_DIR}")
    else:
        print(f"[WARNING] (Windows) Could not find bin directory at {BIN_DIR}")

MAX_FORM_MB = 10*1024*1024*1024

app = FastAPI(
    title="FasterWhisperAPI",
    version="0.1.0",
    max_form_memory_size=MAX_FORM_MB,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

app.middleware("http")(timing_middleware)

app.include_router(transcribe_async.router, prefix="")

# uvicorn app.main:app --host 0.0.0.0 --port 8000
