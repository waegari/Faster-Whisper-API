"""STT 서버 취소/슬롯 수락 테스트 (수정 지시서 5절 T1/T2 + 추가 검증).

- T1: cancel 후 3초 이내 /status idle + /jobs/{id} cancelled
- T2: cancel 멱등 (재호출 200, 없는 잡 404)
- T3: 서버당 1잡 불변식 (busy 중 신규 submit → 409)
- T4: 정상 완료 후 idle 복귀 + 결과 반환
- T5: STT_JOB_MAX_MS 초과 시 자체 취소 → idle

사용법: <senko_env python> tests/acceptance_cancel.py
가짜 모델 서버(tests.fake_stt_server)를 스크립트가 직접 띄우고 내린다.
"""
import os
import subprocess
import sys
import time
import wave
from pathlib import Path

import httpx

PORT = 18123
BASE = f"http://127.0.0.1:{PORT}"
ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

FAILURES = []


def check(name: str, cond: bool, detail: str = ""):
    mark = "PASS" if cond else "FAIL"
    print(f"[{mark}] {name}" + (f" -- {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def make_wav(path: Path, seconds: int = 30):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000 * seconds)


def start_server(extra_env: dict | None = None, log_name: str = "server.log") -> subprocess.Popen:
    env = os.environ.copy()
    env.update({
        "FW_FFMPEG": "/usr/bin/ffmpeg",
        "FW_FFPROBE": "/usr/bin/ffprobe",
        "TEMP_DIR": str(ROOT / "Temp"),
    })
    if extra_env:
        env.update(extra_env)
    log = open(ROOT / "tests" / log_name, "w")
    proc = subprocess.Popen(
        [PY, "-m", "uvicorn", "tests.fake_stt_server:app", "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            if httpx.get(f"{BASE}/status", timeout=1).status_code == 200:
                return proc
        except Exception:
            time.sleep(0.2)
    proc.terminate()
    raise RuntimeError(f"server did not start (see tests/{log_name})")


def stop_server(proc: subprocess.Popen):
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def submit(client: httpx.Client, media_url: str, request_id: str) -> httpx.Response:
    return client.post(
        f"{BASE}/transcribe_async",
        params={"request_id": request_id},
        data={"media_url": media_url},
    )


def wait_status(client: httpx.Client, want: str, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if client.get(f"{BASE}/status").json().get("status") == want:
            return True
        time.sleep(0.2)
    return False


def main():
    wav = ROOT / "tests" / "sample_30s.wav"
    make_wav(wav)
    media_url = wav.resolve().as_uri()

    client = httpx.Client(timeout=10)

    # ---- 기본 서버 (잡 실행 약 9초: 30세그 x 0.3초) ----
    proc = start_server(log_name="server_main.log")
    try:
        r = client.get(f"{BASE}/status").json()
        check("초기 /status idle", r.get("status") == "idle", str(r))
        check("idle 시 job_id 미노출", "job_id" not in r or r.get("job_id") is None, str(r))

        # ---- T1: cancel 후 idle ----
        r = submit(client, media_url, "test-cancel-1")
        check("T1 submit 202", r.status_code == 202, f"{r.status_code} {r.text}")
        job_id = r.json()["job_id"]
        check("T1 job_id == request_id", job_id == "test-cancel-1", job_id)

        time.sleep(1.5)  # transcribing 구간 진입 대기
        st = client.get(f"{BASE}/status").json()
        check("T1 busy + job_id 노출", st.get("status") == "busy" and st.get("job_id") == job_id, str(st))

        r = client.post(f"{BASE}/jobs/{job_id}/cancel")
        check("T1 cancel 200", r.status_code == 200, f"{r.status_code} {r.text}")

        t0 = time.time()
        idle = wait_status(client, "idle", 3.0)
        check("T1 3초 이내 idle 복귀", idle, f"{time.time() - t0:.2f}s")

        job = client.get(f"{BASE}/jobs/{job_id}").json()
        check("T1 job status cancelled", job.get("status") == "cancelled", str(job.get("status")))

        # ---- T2: cancel 멱등 ----
        r = client.post(f"{BASE}/jobs/{job_id}/cancel")
        check("T2 재cancel 200", r.status_code == 200, f"{r.status_code} {r.text}")
        check("T2 재cancel body cancelled", r.json().get("status") == "cancelled", r.text)
        r = client.post(f"{BASE}/jobs/no-such-job/cancel")
        check("T2 없는 잡 cancel 404", r.status_code == 404, str(r.status_code))

        # ---- T3: 서버당 1잡 (busy 중 submit → 409) ----
        r = submit(client, media_url, "test-slot-a")
        check("T3 첫 submit 202", r.status_code == 202, str(r.status_code))
        r2 = submit(client, media_url, "test-slot-b")
        check("T3 busy 중 submit 409", r2.status_code == 409, f"{r2.status_code} {r2.text}")
        detail = r2.json().get("detail", {})
        check("T3 409 응답에 현재 job_id", detail.get("job_id") == "test-slot-a", str(detail))
        client.post(f"{BASE}/jobs/test-slot-a/cancel")
        check("T3 cancel 후 idle", wait_status(client, "idle", 3.0))
        job_b = client.get(f"{BASE}/jobs/test-slot-b")
        check("T3 거부된 잡은 미등록(404)", job_b.status_code == 404, str(job_b.status_code))

        # ---- T4: 정상 완료 ----
        r = submit(client, media_url, "test-done-1")
        check("T4 submit 202", r.status_code == 202, str(r.status_code))
        deadline = time.time() + 30
        job = {}
        while time.time() < deadline:
            job = client.get(f"{BASE}/jobs/test-done-1").json()
            if job.get("status") in ("done", "error", "cancelled"):
                break
            time.sleep(0.5)
        check("T4 완료 status done", job.get("status") == "done", str(job.get("status")))
        text = ((job.get("result") or {}).get("result") or {}).get("text", "")
        check("T4 결과 텍스트 존재", bool(text), text[:40])
        check("T4 완료 후 idle", wait_status(client, "idle", 3.0))
    finally:
        stop_server(proc)

    # ---- T5: 자체 TTL (STT_JOB_MAX_MS=3000) ----
    proc = start_server(extra_env={"STT_JOB_MAX_MS": "3000"}, log_name="server_ttl.log")
    try:
        r = submit(client, media_url, "test-ttl-1")
        check("T5 submit 202", r.status_code == 202, str(r.status_code))
        idle = wait_status(client, "idle", 8.0)
        check("T5 TTL 초과 후 idle", idle)
        job = client.get(f"{BASE}/jobs/test-ttl-1").json()
        check(
            "T5 job cancelled (ttl)",
            job.get("status") == "cancelled" and "ttl" in (job.get("message") or ""),
            f"{job.get('status')} / {job.get('message')}",
        )
    finally:
        stop_server(proc)

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} checks -> {FAILURES}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
