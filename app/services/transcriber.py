from __future__ import annotations

import io
import math
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Union, List, Optional

from pydub import AudioSegment
from faster_whisper import WhisperModel

from app.services.audio_processor import AudioProcessor

try:
    from app.dependencies import get_model as _get_model
except Exception:
    _get_model = None


class TranscriptionService:
    """
    STT(faster-whisper) 기반 전사 서비스.
    - 입력: 파일 경로(Path) 또는 바이트(bytes)
    - 내부 출력 포맷: WAV(PCM16, 16kHz, mono)
    - 결과: 전체 텍스트 + 세그먼트(start/end/content)
    """

    def __init__(
        self,
        source: Union[bytes, Path, str],
        *,
        model: Optional[WhisperModel] = None,
        sr: int = 16000,
        ch: int = 1,
        max_bytes: int = 25 * 1024 * 1024,
    ):
        if model is not None:
            self.model: WhisperModel = model
        elif _get_model is not None:
            self.model = _get_model()
        else:
            raise RuntimeError("WhisperModel is not provided and app.deps.get_model() not found.")

        self.source_path: Optional[Path] = None
        if isinstance(source, (str, Path)):
            self.source_path = Path(source)

        self.target_sr = int(sr)
        self.target_ch = int(ch)
        self.max_bytes = int(max_bytes)


    def _bytes_to_tmp_wav(self, data: bytes) -> Path:
        """bytes → 임시 WAV 파일로 저장"""
        tmp = tempfile.NamedTemporaryFile(prefix="stt_", suffix=".wav", delete=False)
        audio = AudioSegment.from_file(io.BytesIO(data))
        audio = audio.set_frame_rate(self.target_sr).set_channels(self.target_ch).set_sample_width(2)
        audio.export(tmp.name, format="wav")
        return Path(tmp.name)

    def _ensure_wav_path(self, payload: Union[bytes, Path]) -> Path:
        """
        Whisper 입력을 항상 파일 경로로.
        - Path면 그대로(확장자 관계없이 pydub/ffmpeg가 읽음)
        - bytes면 임시 WAV를 만들어 경로 반환
        """
        if isinstance(payload, Path):
            return payload
        return self._bytes_to_tmp_wav(payload)
    
    def to_prob_int(avg_logprob) -> int:
        # exp(-0.1) ≒ 0.904 -> 90
        # 0~100 사이 int로 변환
        try:
            p = math.exp(avg_logprob) * 100
            return int(min(100, max(0, round(p))))
        except (ValueError, OverflowError):
            return 0

    def transcribe(
        self,
        *,
        language: str = "ko",
        is_video: bool = False,
        start: int = 0,
        end: int = 0,
        vad: bool = True,
        word_timestamps: bool = False,
        export_audio_to_disk: bool = False,
        export_json_to_disk: bool = False,
    ) -> dict:
        """
        오디오/비디오 파일을 전사.
        - is_video=True → demux()로 오디오 추출
        - is_video=False → convert()로 WAV 변환
        """
        if not self.source_path:
            raise ValueError(
                "For convert/demux, `source` must be a file path. Got bytes—use router to write temp file first."
            )

        # 1) 오디오 준비 (WAV bytes or Path)
        ap = AudioProcessor(
            path=self.source_path,
            sr=self.target_sr,
            channels=self.target_ch,
            max_bytes=self.max_bytes,
        )
        media = (
            ap.demux(start=start, end=end, export_to_disk=export_audio_to_disk)
            if is_video
            else ap.convert(start=start, end=end, export_to_disk=export_audio_to_disk)
        )

        # 2) Whisper가 읽을 수 있도록 파일 경로 확보
        wav_path = self._ensure_wav_path(media)

        # 3) transcribe
        segments, info = self.model.transcribe(
            str(wav_path),
            language=language,
            vad_filter=vad,
            vad_parameters=dict(min_silence_duration_ms=300),
            temperature=0.0,
            beam_size=6,
            best_of=1,
            patience=1.0,
            word_timestamps=word_timestamps,
            condition_on_previous_text=False,
        )

        # 4) 결과 조립
        all_text: List[str] = []
        all_segments: List[dict] = []
        for i, seg in enumerate(segments):
            txt = (seg.text or "").strip()
            if not txt:
                continue
            all_text.append(txt)
            all_segments.append({"index": i, "avg_logprob": seg.avg_logprob, "prob": self.to_prob_int(seg.avg_logprob), "start": float(seg.start), "end": float(seg.end), "content": txt})

        now = datetime.now()
        result = {
            "language": language,
            "duration": float(getattr(info, "duration", 0.0) or 0.0),
            "created_at": now.strftime("%Y-%m-%d %H:%M:%S.") + str(now.microsecond)[-3:],
            "result": {"text": " ".join(all_text).strip(), "segments": all_segments},
        }

        # 5) 임시 파일 정리
        # - 호출자가 Path를 넘긴 경우는 삭제하지 않음
        try:
            if isinstance(media, bytes):
                wav_path.unlink(missing_ok=True)
        except Exception:
            pass

        return result
