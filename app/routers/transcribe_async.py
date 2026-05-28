from datetime import datetime
import math
from fastapi import APIRouter, Request, UploadFile, Query, File, BackgroundTasks, Depends, HTTPException, Form
from fastapi.responses import JSONResponse
from pathlib import Path
import asyncio, time, logging
from urllib.request import urlopen
from urllib.request import Request as URLRequest
from ..dependencies import get_model, get_senko
from ..services.transcriber import TranscriptionService
from faster_whisper.audio import decode_audio
from ..services.audio_processor import AudioProcessor
from ..services.temp_files import create_named_temp_file, cleanup_path
from ..config.settings import settings
from ..jobs import create_job, update_job, get_job, JobStatus, get_active_job_count
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
    # Remove any accidental quotes or whitespaces
    clean_url = media_url.strip(' "\'\n\r')
    
    logger.info(f"Attempting to download from: {clean_url}")
    
    suffix = Path(clean_url).suffix or ".bin"
    if "?" in suffix:
        suffix = ".bin"
        
    # Add User-Agent to prevent 403 Forbidden errors from some servers
    req = URLRequest(
        clean_url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    )
    
    # Bypass proxy settings just in case a dead local proxy is configured
    from urllib.request import build_opener, ProxyHandler
    opener = build_opener(ProxyHandler({}))
    
    with opener.open(req) as response:
        with create_named_temp_file(prefix="in_", suffix=suffix) as tmp:
            tmp_path = Path(tmp.name)
            while chunk := response.read(1024 * 1024):
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

@router.get("/status")
def server_status():
    """
    STT 서버의 현재 상태를 반환합니다. 
    Node.js 스케줄러가 서버가 대기 중(idle)인지 작업 중(busy)인지 확인하여 작업을 할당할 수 있도록 합니다.
    """
    active_jobs = get_active_job_count()
    return {
        "status": "busy" if active_jobs > 0 else "idle",
        "active_jobs": active_jobs,
        "message": "작업 처리 중" if active_jobs > 0 else "대기 중"
    }

async def _worker(job_id: str, media_url: str, query: TranscribeQuery):
    update_job(job_id, status=JobStatus.processing, started_at=time.time(), message="downloading")
    tmp_path = None
    full_wav_path = None
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
        diarizer = get_senko()
        
        ap = AudioProcessor(
            path=tmp_path,
            sr=settings.DEFAULT_SR,
            br=settings.DEFAULT_BR,
            channels=settings.DEFAULT_CH,
            max_bytes=settings.MAX_AUDIO_BYTES,
        )

        update_job(job_id, message="extracting full audio")
        
        # 1. 전체 오디오를 WAV로 추출
        full_wav_path = ap.extract_wav(start=query.start, end=query.end)

        update_job(job_id, message="diarizing (senko)")
        
        # 2. Senko 화자 분리
        senko_result = diarizer.diarize(str(full_wav_path), generate_colors=False)
        merged_segments = senko_result.get("merged_segments", [])
        
        update_job(job_id, message="loading audio for slicing")
        
        # 3. 오디오 배열 메모리 로드
        audio_array = decode_audio(str(full_wav_path), sampling_rate=settings.DEFAULT_SR)
        
        # 4. TranscriptionService 인스턴스화
        svc = TranscriptionService(source=None, model=model, sr=settings.DEFAULT_SR, ch=settings.DEFAULT_CH)

        update_job(job_id, message="transcribing", progress=0.0)
        
        raw_segments = []
        total_duration = float(audio_array.shape[0]) / settings.DEFAULT_SR
        
        for i, spk_seg in enumerate(merged_segments):
            if cancellation_flags.get(job_id) is True:
                logger.warning(f"🛑 [Process Killer] Task {job_id} cancelled by Node server. Stopping immediately.")
                return
            
            s_time = float(spk_seg["start"])
            e_time = float(spk_seg["end"])
            speaker = spk_seg.get("speaker", "SPEAKER_00")
            
            s_sample = int(s_time * settings.DEFAULT_SR)
            e_sample = int(e_time * settings.DEFAULT_SR)
            
            if e_sample <= s_sample:
                continue
                
            seg_audio = audio_array[s_sample:e_sample]
            
            # Whisper Transcribe
            segments, info = svc.model.transcribe(
                seg_audio,
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

            # Whisper 세그먼트 시간 보정 및 병합
            for w_seg in segments:
                if cancellation_flags.get(job_id) is True:
                    logger.warning(f"🛑 [Process Killer] Task {job_id} cancelled by Node server. Stopping immediately.")
                    return
                
                txt = (w_seg.text or "").strip()
                if txt:
                    avg_logprob = getattr(w_seg, 'avg_logprob', None)
                    if avg_logprob is None:
                        avg_logprob = 0.0
                        
                    raw_segments.append({
                        "start": float(w_seg.start) + s_time + query.start,
                        "end": float(w_seg.end) + s_time + query.start,
                        "content": txt,
                        "avg_logprob": avg_logprob,
                        "speaker": speaker
                    })
                
                # 진행률 업데이트
                if e_time > s_time:
                    current_progress = (i + (w_seg.end / (e_time - s_time))) / max(1, len(merged_segments))
                    update_job(job_id, progress=min(0.99, current_progress))
                await asyncio.sleep(0)

        # 5. hallucination 대응 및 병합
        raw_segments.sort(key=lambda x: x["start"])
        processed_segments = svc._post_process_segments(raw_segments)
        
        # 6. 텍스트 병합
        all_text = " ".join([seg["content"] for seg in processed_segments])

        now = datetime.now()
        result = {
            "language": query.language,
            "duration": total_duration,
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
            cleanup_path(tmp_path)
            if full_wav_path:
                cleanup_path(full_wav_path)
        except Exception:
            pass