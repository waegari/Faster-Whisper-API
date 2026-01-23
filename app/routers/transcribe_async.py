from datetime import datetime
import math
from fastapi import APIRouter, Request, UploadFile, Query, File, BackgroundTasks, Depends, HTTPException, Form
from fastapi.responses import JSONResponse
from pathlib import Path
import tempfile, os, asyncio, time, logging
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
    q: str = Form('{"language":"ko","vad":true,"is_video":false,"word_timestamps":false}'),
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


@router.post("/transcribe_async", status_code=202)
async def transcribe_async(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    query: TranscribeQuery = Depends(parse_query),
    request_id: str = Query(None),
):
    suffix = Path(file.filename).suffix or ".bin"
    with tempfile.NamedTemporaryFile(prefix="in_", suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        size = 0
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > settings.MAX_AUDIO_BYTES:
                tmp.close()
                os.unlink(tmp_path)
                raise HTTPException(413, detail="File too large")
            tmp.write(chunk)

    final_req_id = request_id or request.headers.get("X-Request-ID") or getattr(query, "request_id", None)

    job = create_job(final_req_id)
    cancellation_flags[job.job_id] = False

    background_tasks.add_task(_worker, job.job_id, tmp_path, query)

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

async def _worker(job_id: str, tmp_path: Path, query: TranscribeQuery):
    update_job(job_id, status=JobStatus.processing, started_at=time.time(), message="received")
    try:
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

        svc = TranscriptionService(source=media, model=model, sr=settings.DEFAULT_SR, ch=settings.DEFAULT_CH)

        update_job(job_id, message="transcribing", progress=0.0)
        segments, info = svc.model.transcribe(
            str(svc._ensure_wav_path(media)),
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

        all_text, all_segments = [], []
        duration = float(getattr(info, "duration", 0.0) or 0.0)
        idx = 0
        try:
            for seg in segments:
                if cancellation_flags.get(job_id) is True:
                    logger.warning(f"🛑 [Zombie Killer] Task {job_id} cancelled by Node server. Stopping immediately.")
                    return
                txt = (seg.text or "").strip()
                if txt:
                    all_text.append(txt)
                    all_segments.append(
                        {"index": idx, "start": float(seg.start), "end": float(seg.end), "content": txt, "avg_logprob": seg.avg_logprob, "prob": to_prob_int(seg.avg_logprob),}
                    )
                    idx += 1
                    if duration > 0:
                        update_job(job_id, progress=min(0.99, seg.end / duration))
                await asyncio.sleep(0)
        except TypeError:
            for seg in segments:
                txt = (seg.text or "").strip()
                if txt:
                    all_text.append(txt)
                    all_segments.append(
                        {"index": idx, "start": float(seg.start), "end": float(seg.end), "content": txt}
                    )
                    idx += 1
                    if duration > 0:
                        update_job(job_id, progress=min(0.99, seg.end / duration))

        now = datetime.now()
        result = {
            "language": query.language,
            "duration": duration,
            "created_at": now.strftime("%Y-%m-%d %H:%M:%S.") + str(now.microsecond)[-3:],
            "result": {"text": " ".join(all_text).strip(), "segments": all_segments},
        }
        update_job(job_id, status=JobStatus.done, ended_at=time.time(), progress=1.0, message="done", result=result)
    except Exception as e:
        error_message = str(e)
        
        # ✅ [추가] FFmpeg/Subprocess 에러라면 stderr(진짜 원인)를 끄집어냄
        if hasattr(e, 'stderr') and e.stderr:
            try:
                # bytes를 string으로 변환
                decoded_stderr = e.stderr.decode('utf-8', errors='ignore') if isinstance(e.stderr, bytes) else str(e.stderr)
                error_message += f" | Details: {decoded_stderr}"
            except:
                pass
        
        logger.error(f"Task {job_id} failed: {error_message}") # Python 로그에도 남김
        
        update_job(job_id, status=JobStatus.error, ended_at=time.time(), message=error_message)
        pass
    finally:
        try:
            if job_id in cancellation_flags:
                del cancellation_flags[job_id]
            os.unlink(tmp_path)
        except:
            pass
