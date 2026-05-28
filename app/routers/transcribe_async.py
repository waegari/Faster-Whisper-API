from datetime import datetime
import math
from fastapi import APIRouter, Request, UploadFile, Query, File, BackgroundTasks, Depends, HTTPException, Form
from fastapi.responses import JSONResponse
from pathlib import Path
import asyncio, time, logging
from urllib.request import urlopen
from ..dependencies import get_model, get_senko
from ..services.transcriber import TranscriptionService
from faster_whisper.audio import decode_audio
from ..services.audio_processor import AudioProcessor
from ..services.temp_files import create_named_temp_file, cleanup_path
from ..config.settings import settings
from ..jobs import create_job, update_job, get_job, JobStatus
from ..schemas import TranscribeQuery

logger = logging.getLogger("app.timing")

router = APIRouter()

cancellation_flags: dict[str, bool] = {}

def parse_query(
    q: str = Form('{"task":"transcribe","language":"ko","vad":true,"is_video":false,"word_timestamps":false}'),
) -> TranscribeQuery:
    return TranscribeQuery.model_validate_json(q)

def to_prob_int(avg_logprob) -> int:
    # exp(-0.1) ≒ 0.904 -> 90
    # 0~100 사이 int로 변환
    try:
        p = math.exp(avg_logprob) * 100
        return int(min(100, max(0, round(p))))
    except (ValueError, OverflowError):
        return 0


def _download_to_temp(media_url: str) -> Path:
    """media_url을 GET으로 받아 임시 파일에 저장. 경로 반환."""
    suffix = Path(media_url).suffix or ".bin"
    if "?" in suffix:
        suffix = ".bin"
    req = urlopen(media_url)
    with create_named_temp_file(prefix="in_", suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        while chunk := req.read(1024 * 1024):
            tmp.write(chunk)
    return tmp_path


@router.post("/transcribe_async", status_code=202)
async def transcribe_async(
    request: Request,
    background_tasks: BackgroundTasks,
    media_url: str = Form(..., description="미디어 파일 URL (스케줄러가 서빙하는 결과 URL 등)"),
    query: TranscribeQuery = Depends(parse_query),
    request_id: str = Query(None),
):
    """파일 업로드 없이 media_url만 받아 202 즉시 반환. worker에서 URL 다운로드 후 전사."""
    final_req_id = request_id or request.headers.get("X-Request-ID") or getattr(query, "request_id", None)

    job = create_job(final_req_id)
    cancellation_flags[job.job_id] = False

    background_tasks.add_task(_worker, job.job_id, media_url, query)

    status_path = f"/jobs/{job.job_id}"
    body = {
        "job_id": job.job_id,
        "status_url": status_path,
    }
    headers = {
        "Location": status_path,  # 202 Location 헤더
        "X-Request-ID": final_req_id or job.job_id,
    }
    return JSONResponse(content=body, headers=headers, status_code=202)


@router.get("/jobs/{job_id}")
def get_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    if job_id in cancellation_flags:
        cancellation_flags[job_id] = True
        logger.info(f"🚩 Received cancel request for {job_id}")
        return {"status": "cancelled"}
    return {"status": "job not found or already finished"}
