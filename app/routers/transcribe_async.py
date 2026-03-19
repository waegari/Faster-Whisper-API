from datetime import datetime
import math
from fastapi import APIRouter, Request, UploadFile, Query, File, BackgroundTasks, Depends, HTTPException, Form
from fastapi.responses import JSONResponse
from pathlib import Path
import tempfile, os, asyncio, time, logging
from urllib.request import urlopen
from ..dependencies import get_model
from ..services.transcriber import TranscriptionService
from ..services.audio_processor import AudioProcessor
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
    with tempfile.NamedTemporaryFile(prefix="in_", suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        size = 0
        while chunk := req.read(1024 * 1024):
            size += len(chunk)
            if size > settings.MAX_AUDIO_BYTES:
                tmp.close()
                os.unlink(tmp_path)
                req.close()
                raise HTTPException(413, detail="File too large")
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


async def _worker(job_id: str, media_url: str, query: TranscribeQuery):
    update_job(job_id, status=JobStatus.processing, started_at=time.time(), message="downloading")
    tmp_path = None
    try:
        tmp_path = _download_to_temp(media_url)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Task {job_id} download failed: {e}")
        update_job(job_id, status=JobStatus.error, ended_at=time.time(), message=str(e))
        return

    try:
        update_job(job_id, message="received")
        model = get_model()
        ap = AudioProcessor(
            path=tmp_path,
            sr=settings.DEFAULT_SR,
            br=settings.DEFAULT_BR,
            channels=settings.DEFAULT_CH,
            max_bytes=settings.MAX_AUDIO_BYTES,
        )

        update_job(job_id, message="converting")
        media = (
            ap.demux(format="wav", start=query.start, end=query.end, export_to_disk=False)
            if query.is_video
            else ap.convert(start=query.start, end=query.end, export_to_disk=False)
        )

        # 1. TranscriptionService 인스턴스화 (모델 및 오디오 포맷 설정)
        svc = TranscriptionService(source=media, model=model, sr=settings.DEFAULT_SR, ch=settings.DEFAULT_CH)

        update_job(job_id, message="transcribing", progress=0.0)
        
        # 2. 제너레이터(segments) 받아오기
        segments, info = svc.model.transcribe(
            str(svc._ensure_wav_path(media)),
            task=query.task,
            language=query.language,
            vad_filter=query.vad,
            vad_parameters=dict(min_silence_duration_ms=300),
            temperature=0.0,
            beam_size=6,
            best_of=1,
            patience=1.0,
            word_timestamps=query.word_timestamps,
            condition_on_previous_text=False,
        )

        raw_segments = []
        duration = float(getattr(info, "duration", 0.0) or 0.0)

        # 3. 취소 여부 체크 및 진행률 업데이트
        for seg in segments:
            if cancellation_flags.get(job_id) is True:
                logger.warning(f"🛑 [Process Killer] Task {job_id} cancelled by Node server. Stopping immediately.")
                return
            
            txt = (seg.text or "").strip()
            if txt:
                # logprob가 None인 경우를 대비해 0.0으로 기본값 처리
                avg_logprob = getattr(seg, 'avg_logprob', None)
                if avg_logprob is None:
                    avg_logprob = 0.0
                    
                raw_segments.append({
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "content": txt,
                    "avg_logprob": avg_logprob
                })
            
            if duration > 0:
                update_job(job_id, progress=min(0.99, seg.end / duration))
            await asyncio.sleep(0)

        # 4. hallucination 대응
        processed_segments = svc._post_process_segments(raw_segments)
        
        # 5. 텍스트 병합
        all_text = " ".join([seg["content"] for seg in processed_segments])

        now = datetime.now()
        result = {
            "language": query.language,
            "duration": duration,
            "created_at": now.strftime("%Y-%m-%d %H:%M:%S.") + str(now.microsecond)[-3:],
            "result": {"text": all_text.strip(), "segments": processed_segments},
        }
        
        update_job(job_id, status=JobStatus.done, ended_at=time.time(), progress=1.0, message="done", result=result)
        
    except Exception as e:
        error_message = str(e)
        if hasattr(e, 'stderr') and e.stderr:
            try:
                decoded_stderr = e.stderr.decode('utf-8', errors='ignore') if isinstance(e.stderr, bytes) else str(e.stderr)
                error_message += f" | Details: {decoded_stderr}"
            except Exception:
                pass
        logger.error(f"Task {job_id} failed: {error_message}")
        update_job(job_id, status=JobStatus.error, ended_at=time.time(), message=error_message)
    finally:
        try:
            if job_id in cancellation_flags:
                del cancellation_flags[job_id]
            if tmp_path and tmp_path.exists():
                os.unlink(tmp_path)
        except Exception:
            pass