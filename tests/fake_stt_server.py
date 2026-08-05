"""수락 테스트용 STT 서버: 무거운 Whisper/Senko 대신 가짜 모델을 주입한다.

실행: uvicorn tests.fake_stt_server:app
환경변수:
  FAKE_SEG_COUNT (기본 30)  : 가짜 Whisper 세그먼트 개수
  FAKE_SEG_DELAY (기본 0.3) : 세그먼트당 지연(초) → 잡 실행 시간 = COUNT * DELAY
"""
import os
import time
from types import SimpleNamespace

import app.routers.transcribe_async as router_mod

SEG_COUNT = int(os.getenv("FAKE_SEG_COUNT", "30"))
SEG_DELAY = float(os.getenv("FAKE_SEG_DELAY", "0.3"))


class FakeModel:
    def transcribe(self, audio, **kwargs):
        duration = float(len(audio)) / 16000.0

        def gen():
            step = duration / max(1, SEG_COUNT)
            for i in range(SEG_COUNT):
                time.sleep(SEG_DELAY)
                yield SimpleNamespace(
                    text=f"seg {i}",
                    start=i * step,
                    end=(i + 1) * step,
                    avg_logprob=-0.1,
                )

        info = SimpleNamespace(language="ko", duration=duration)
        return gen(), info


class FakeDiarizer:
    def diarize(self, path, generate_colors=False):
        # 빈 리스트 → worker가 전체 구간 단일 세그먼트로 대체
        return {"merged_segments": []}


router_mod.get_model = lambda: FakeModel()
router_mod.get_senko = lambda: FakeDiarizer()

from app.main import app  # noqa: E402, F401
