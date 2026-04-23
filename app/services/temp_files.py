from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

from app.config.settings import settings

logger = logging.getLogger("app.temp")

APP_TEMP_PREFIXES = ("in_", "stt_")


def ensure_temp_dir() -> Path:
    temp_dir = settings.TEMP_DIR
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def create_named_temp_file(*, prefix: str, suffix: str):
    return tempfile.NamedTemporaryFile(
        prefix=prefix,
        suffix=suffix,
        dir=ensure_temp_dir(),
        delete=False,
    )


def cleanup_path(path: Path | None) -> None:
    if not path:
        return
    try:
        path.unlink(missing_ok=True)
    except Exception:
        logger.exception("Failed to remove temp file: %s", path)


def cleanup_stale_temp_files() -> int:
    temp_dir = ensure_temp_dir()
    ttl_seconds = max(0, settings.TEMP_FILE_TTL_HOURS) * 3600
    if ttl_seconds <= 0:
        return 0

    cutoff = time.time() - ttl_seconds
    removed = 0

    for path in temp_dir.iterdir():
        if not path.is_file():
            continue
        if not path.name.startswith(APP_TEMP_PREFIXES):
            continue
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
                removed += 1
        except FileNotFoundError:
            continue
        except Exception:
            logger.exception("Failed to remove stale temp file: %s", path)

    return removed
