from datetime import datetime
import math
import uuid
from fastapi import APIRouter, Request, Query, BackgroundTasks, Depends, HTTPException, Form
from fastapi.responses import JSONResponse
from pathlib import Path
import time, logging
from urllib.request import urlopen
from ..dependencies import get_model, get_senko
from ..services.transcriber import TranscriptionService
from faster_whisper.audio import decode_audio
from ..services.audio_processor import AudioProcessor
from ..services.temp_files import create_named_temp_file, cleanup_path
from ..config.settings import settings
from ..jobs import create_job, update_job, get_job, finish_job_if_active, JobStatus, TERMINAL_STATUSES
from ..job_control import (
    JobControl,
    JobCancelled,
    try_acquire_slot,
    release_slot,
    get_control,
    current_job_id,
)
from ..schemas import TranscribeQuery

logger = logging.getLogger("app.timing")

router = APIRouter()

DOWNLOAD_TIMEOUT_SEC = 60


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


def _download_to_temp(media_url: str, ctrl: JobControl) -> Path:
    """media_url을 GET으로 받아 임시 파일에 저장. 경로 반환. 취소 시 부분 파일 정리 후 중단."""
    suffix = Path(media_url).suffix or ".bin"
    if "?" in suffix:
        suffix = ".bin"
    req = urlopen(media_url, timeout=DOWNLOAD_TIMEOUT_SEC)
    tmp_path = None
    try:
        with create_named_temp_file(prefix="in_", suffix=suffix) as tmp:
            tmp_path = Path(tmp.name)
            while chunk := req.read(1024 * 1024):
                ctrl.check()
                tmp.write(chunk)
        return tmp_path
    except BaseException:
        cleanup_path(tmp_path)
        raise


@router.post("/transcribe_async", status_code=202)
async def transcribe_async(
    request: Request,
    background_tasks: BackgroundTasks,
    media_url: str = Form(..., description="미디어 파일 URL (스케줄러가 서빙하는 결과 URL 등)"),
    query: TranscribeQuery = Depends(parse_query),
    request_id: str = Query(None),
):
    """파일 업로드 없이 media_url만 받아 202 즉시 반환. worker에서 URL 다운로드 후 전사.

    서버당 1잡 불변식: 이미 작업 중이면 409를 반환한다.
    """
    final_req_id = request_id or request.headers.get("X-Request-ID") or getattr(query, "request_id", None)
    job_id = final_req_id or str(uuid.uuid4())

    max_duration_sec = settings.JOB_MAX_MS / 1000 if settings.JOB_MAX_MS > 0 else None
    ctrl = JobControl(job_id, max_duration_sec=max_duration_sec)

    if not try_acquire_slot(ctrl):
        ctrl.close()
        raise HTTPException(
            status_code=409,
            detail={"error": "server busy", "job_id": current_job_id()},
        )

    job = create_job(request_id=final_req_id, job_id=job_id)
    background_tasks.add_task(_worker, job.job_id, media_url, query, ctrl)

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
    """잡 취소. 실행 중인 워커/하위 프로세스를 실제로 중단시키고 잡을 cancelled로 종결한다. 멱등."""
    job = get_job(job_id)
    ctrl = get_control(job_id)
    if job is None and ctrl is None:
        raise HTTPException(404, "job not found")

    if ctrl is not None:
        logger.info(f"🚩 Received cancel request for {job_id}")
        ctrl.cancel()

    if job is not None and job.status in TERMINAL_STATUSES:
        # 이미 끝난 잡: 멱등 응답
        return {"job_id": job_id, "status": job.status.value}

    finish_job_if_active(job_id, JobStatus.cancelled, ended_at=time.time(), message="cancelled by request")
    return {"job_id": job_id, "status": "cancelled"}


@router.get("/status")
def server_status():
    """
    STT 서버의 현재 상태를 반환합니다.
    스케줄러가 서버가 대기 중(idle)인지 작업 중(busy)인지 확인하여 작업을 할당할 수 있도록 합니다.
    busy일 때는 현재 실행 중인 job_id를 함께 노출한다 (게이트웨이 orphan cleanup용).
    """
    running_job_id = current_job_id()
    if running_job_id is None:
        return {"status": "idle", "active_jobs": 0, "message": "대기 중"}

    body = {
        "status": "busy",
        "job_id": running_job_id,
        "active_jobs": 1,
        "message": "작업 처리 중",
    }
    job = get_job(running_job_id)
    if job is not None:
        body["request_id"] = job.request_id
        body["progress"] = job.progress
        if job.message:
            body["message"] = job.message
    return body


def _worker(job_id: str, media_url: str, query: TranscribeQuery, ctrl: JobControl):
    """전사 워커. 동기 함수라 스레드풀에서 실행된다 (이벤트 루프를 막지 않아 /status, /cancel이 즉시 응답).

    - 취소/TTL은 ctrl.check() 체크포인트 + ffmpeg/ffprobe kill로 반영
    - 어떤 경로로 끝나든 finally에서 잡을 종결 상태로 만들고 슬롯을 해제
    """
    tmp_path = None
    full_wav_path = None
    try:
        ctrl.check()
        update_job(job_id, status=JobStatus.processing, started_at=time.time(), message="downloading")
        tmp_path = _download_to_temp(media_url, ctrl)

        ctrl.check()
        update_job(job_id, message="received")
        model = get_model()
        diarizer = get_senko()

        ap = AudioProcessor(
            path=tmp_path,
            sr=settings.DEFAULT_SR,
            br=settings.DEFAULT_BR,
            channels=settings.DEFAULT_CH,
            max_bytes=settings.MAX_AUDIO_BYTES,
            job_control=ctrl,
        )

        update_job(job_id, message="extracting full audio")

        # 1. 전체 오디오를 WAV로 추출
        try:
            full_wav_path = ap.extract_wav(start=query.start, end=query.end)
        except RuntimeError as e:
            if "No audio stream found" in str(e):
                logger.info(f"Task {job_id}: No audio stream found. Returning empty result.")
                now = datetime.now()
                empty_result = {
                    "language": query.language,
                    "duration": 0.0,
                    "created_at": now.strftime("%Y-%m-%d %H:%M:%S.") + str(now.microsecond)[-3:],
                    "result": {"text": "", "segments": []},
                }
                finish_job_if_active(
                    job_id,
                    JobStatus.done,
                    ended_at=time.time(),
                    progress=1.0,
                    message="done (no audio)",
                    result=empty_result,
                )
                return
            else:
                raise e

        ctrl.check()
        update_job(job_id, message="diarizing (senko)")

        # 2. Senko 화자 분리
        senko_result = diarizer.diarize(str(full_wav_path), generate_colors=False)
        merged_segments = senko_result.get("merged_segments", []) if senko_result else []

        ctrl.check()
        update_job(job_id, message="loading audio for slicing")

        # 3. 오디오 배열 메모리 로드
        audio_array = decode_audio(str(full_wav_path), sampling_rate=settings.DEFAULT_SR)

        # 4. TranscriptionService 인스턴스화
        svc = TranscriptionService(source=None, model=model, sr=settings.DEFAULT_SR, ch=settings.DEFAULT_CH)

        update_job(job_id, message="transcribing", progress=0.0)

        raw_segments = []
        total_duration = float(audio_array.shape[0]) / settings.DEFAULT_SR

        if not merged_segments:
            merged_segments = [{"start": 0.0, "end": total_duration, "speaker": "SPEAKER_00"}]

        for i, spk_seg in enumerate(merged_segments):
            ctrl.check()

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
            # (segments는 lazy generator라 매 반복이 곧 추론 진행 → 취소 체크가 실제로 추론을 멈춘다)
            for w_seg in segments:
                ctrl.check()

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

        ctrl.check()

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

        finish_job_if_active(
            job_id, JobStatus.done, ended_at=time.time(), progress=1.0, message="done", result=result
        )

    except JobCancelled as e:
        logger.warning(f"🛑 Task {job_id} cancelled ({e.reason}). Worker stopped.")
        finish_job_if_active(job_id, JobStatus.cancelled, ended_at=time.time(), message=e.reason)
    except Exception as e:
        if ctrl.cancel_event.is_set():
            # 취소로 인해 하위 프로세스가 죽는 등 파생 예외 → cancelled로 종결
            logger.warning(f"🛑 Task {job_id} cancelled (derived error: {e}). Worker stopped.")
            finish_job_if_active(
                job_id, JobStatus.cancelled, ended_at=time.time(), message=ctrl.cancel_reason or "cancelled"
            )
        else:
            error_message = str(e)
            if hasattr(e, 'stderr') and e.stderr:
                try:
                    decoded_stderr = e.stderr.decode('utf-8', errors='ignore') if isinstance(e.stderr, bytes) else str(e.stderr)
                    error_message += f" | Details: {decoded_stderr}"
                except Exception:
                    pass
            logger.error(f"Task {job_id} failed: {error_message}")
            finish_job_if_active(job_id, JobStatus.error, ended_at=time.time(), message=error_message)
    finally:
        # 어떤 경로로 끝났든 잡이 queued/processing으로 남지 않게 보장
        finish_job_if_active(job_id, JobStatus.error, ended_at=time.time(), message="worker exited unexpectedly")
        try:
            cleanup_path(tmp_path)
            if full_wav_path:
                cleanup_path(full_wav_path)
        except Exception:
            pass
        # 슬롯 해제 → /status가 idle로 복귀
        release_slot(ctrl)
