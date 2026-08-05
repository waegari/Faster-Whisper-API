"""잡 취소 제어 + 서버 단일 슬롯 관리.

- JobControl: 잡 하나에 대한 취소 이벤트 / TTL 타이머 / 하위 프로세스(ffmpeg 등) kill
- 서버당 1잡 불변식: try_acquire_slot / release_slot / current_job_id
"""
from __future__ import annotations

import logging
import subprocess
import threading
import time
from typing import Dict, Optional

logger = logging.getLogger("app.job_control")


class JobCancelled(Exception):
    """워커 내부에서 잡이 취소(또는 TTL 초과)되었을 때 발생."""

    def __init__(self, reason: str = "cancelled"):
        super().__init__(reason)
        self.reason = reason


class JobControl:
    def __init__(self, job_id: str, max_duration_sec: Optional[float] = None):
        self.job_id = job_id
        self.cancel_event = threading.Event()
        self.cancel_reason: Optional[str] = None
        self._proc_lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._deadline: Optional[float] = None
        self._timer: Optional[threading.Timer] = None

        if max_duration_sec and max_duration_sec > 0:
            self._deadline = time.monotonic() + max_duration_sec
            # 워커가 블로킹 구간(다운로드/ffmpeg)에 있어도 TTL 초과 시 취소가 걸리도록 타이머 사용
            self._timer = threading.Timer(
                max_duration_sec, self.cancel, kwargs={"reason": "job ttl exceeded"}
            )
            self._timer.daemon = True
            self._timer.start()

    def cancel(self, reason: str = "cancelled by request") -> None:
        """취소 표시 + 실행 중인 하위 프로세스 즉시 kill. 멱등."""
        if not self.cancel_event.is_set():
            self.cancel_reason = reason
            logger.warning("Job %s cancel requested: %s", self.job_id, reason)
        self.cancel_event.set()
        with self._proc_lock:
            proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                logger.exception("Failed to kill subprocess for job %s", self.job_id)

    def attach_process(self, proc: subprocess.Popen) -> None:
        """실행 중인 하위 프로세스를 등록. 이미 취소되었으면 즉시 kill."""
        with self._proc_lock:
            self._proc = proc
        if self.cancel_event.is_set() and proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass

    def detach_process(self) -> None:
        with self._proc_lock:
            self._proc = None

    def check(self) -> None:
        """취소/TTL 체크포인트. 취소 상태면 JobCancelled 발생."""
        if self._deadline is not None and time.monotonic() > self._deadline:
            self.cancel(reason="job ttl exceeded")
        if self.cancel_event.is_set():
            raise JobCancelled(self.cancel_reason or "cancelled")

    def close(self) -> None:
        if self._timer is not None:
            self._timer.cancel()


_LOCK = threading.Lock()
_CONTROLS: Dict[str, JobControl] = {}
_CURRENT: Optional[JobControl] = None


def try_acquire_slot(ctrl: JobControl) -> bool:
    """서버 슬롯(서버당 1잡)을 원자적으로 점유. 실패 시 False."""
    global _CURRENT
    with _LOCK:
        if _CURRENT is not None:
            return False
        _CURRENT = ctrl
        _CONTROLS[ctrl.job_id] = ctrl
        return True


def release_slot(ctrl: JobControl) -> None:
    """워커 종료 시(성공/실패/취소 무관) 반드시 호출해 슬롯을 해제."""
    global _CURRENT
    with _LOCK:
        if _CURRENT is ctrl:
            _CURRENT = None
        _CONTROLS.pop(ctrl.job_id, None)
    ctrl.close()


def get_control(job_id: str) -> Optional[JobControl]:
    with _LOCK:
        return _CONTROLS.get(job_id)


def current_job_id() -> Optional[str]:
    with _LOCK:
        return _CURRENT.job_id if _CURRENT is not None else None
