from __future__ import annotations

import io
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Union, List, Optional

from pydub import AudioSegment
from faster_whisper import WhisperModel

from app.services.audio_processor import AudioProcessor
from app.services.temp_files import create_named_temp_file, cleanup_path

try:
    from app.dependencies import get_model as _get_model
except Exception:
    _get_model = None


class TranscriptionService:
    """
    STT(faster-whisper) 기반 transcribe 서비스
    - 입력: 파일 경로(Path) 또는 바이트(bytes)
    - 내부 출력 포맷: WAV(PCM16, 16kHz, mono)
    - 결과: 전체 텍스트 + 세그먼트(start/end/content)
    """

    def __init__(
        self,
        source: Union[bytes, Path, str, None] = None,
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
        if source is not None and isinstance(source, (str, Path)):
            self.source_path = Path(source)

        self.target_sr = int(sr)
        self.target_ch = int(ch)
        self.max_bytes = int(max_bytes)

    def _bytes_to_tmp_wav(self, data: bytes) -> Path:
        """bytes → 임시 WAV 파일로 저장"""
        with create_named_temp_file(prefix="stt_", suffix=".wav") as tmp:
            audio = AudioSegment.from_file(io.BytesIO(data))
            audio = audio.set_frame_rate(self.target_sr).set_channels(self.target_ch).set_sample_width(2)
            audio.export(tmp.name, format="wav")
            return Path(tmp.name)

    def _ensure_wav_path(self, payload: Union[bytes, Path]) -> Path:
        """
        Whisper 입력을 항상 파일 경로로
        - Path면 그대로(확장자 관계없이 pydub/ffmpeg가 읽음)
        - bytes면 임시 WAV를 만들어 경로 반환
        """
        if isinstance(payload, Path):
            return payload
        return self._bytes_to_tmp_wav(payload)
    
    @staticmethod
    def to_prob_int(avg_logprob) -> int:
        # exp(-0.1) ≒ 0.904 -> 90
        # 0~100 사이 int로 변환
        try:
            p = math.exp(avg_logprob) * 100
            return int(min(100, max(0, round(p))))
        except (ValueError, OverflowError):
            return 0
        
    @staticmethod
    def _clean_hallucination(text: str) -> tuple[str, float]:
        """
        동일 단어/구문 반복을 압축하고, 환각 정도에 따른 페널티 비율(0.0~1.0)을 반환
        """
        cleaned_text = re.sub(r'(\b.+?\b)( \1){4,}', r'\1', text)
        penalty = 1.0
        
        # 1. 반복 패턴 압축으로 인한 텍스트 손실 페널티
        if len(cleaned_text) < len(text):
            # 압축된 비율만큼 신뢰도 깎기 (예: 100자 -> 20자면 0.2 곱함)
            penalty *= (len(cleaned_text) / len(text))
            
        # 2. "오오오 오..." 같은 단일/소수 문자 무한 반복 페널티
        text_no_space = text.replace(" ", "")
        total_chars = len(text_no_space)
        if total_chars > 10:
            unique_chars = len(set(text_no_space))
            # 고유 글자 비율이 10% 미만이면 극심한 환각으로 간주하고 확률 폭락시킴
            if (unique_chars / total_chars) < 0.1:
                penalty *= 0.2
                
        return cleaned_text, max(0.01, penalty) # 최소 1%의 확률은 보존

    def _post_process_segments(self, raw_segments: List[dict]) -> List[dict]:
        if not raw_segments:
            return []

        # 1. 개별 세그먼트 단위로 환각 압축 및 기본 prob 계산
        processed_raw = []
        for seg in raw_segments:
            raw_text = seg["content"].strip()
            if not raw_text:
                continue

            cleaned_text, penalty = self._clean_hallucination(raw_text)
            base_prob = self.to_prob_int(seg["avg_logprob"])
            final_prob = int(base_prob * penalty)

            seg_dict = {
                "start": seg["start"],
                "end": seg["end"],
                "content": cleaned_text,
                "avg_logprob": seg["avg_logprob"],
                "prob": final_prob
            }
            if "speaker" in seg:
                seg_dict["speaker"] = seg["speaker"]
                
            processed_raw.append(seg_dict)

        # 2. 동일 문장 다중 세그먼트 반복 페널티 적용
        deduped_raw = []
        i = 0
        while i < len(processed_raw):
            current_seg = processed_raw[i]
            count = 1
            j = i + 1
            while j < len(processed_raw) and processed_raw[j]["content"] == current_seg["content"]:
                count += 1
                j += 1

            if count >= 3:
                # 3번 이상 반복되면 첫 번째 세그먼트의 확률을 1/n로 깎음
                current_seg["prob"] = max(1, int(current_seg["prob"] / count))
                deduped_raw.append(current_seg)
            else:
                for k in range(i, j):
                    deduped_raw.append(processed_raw[k])
            i = j

        # 3. 신뢰도가 극단적으로 낮은 세그먼트(10 이하) 삭제
        survived_segments = [seg for seg in deduped_raw if seg["prob"] > 10]

        # 4. 인덱스 재부여
        for idx, seg in enumerate(survived_segments):
            seg["index"] = idx

        return survived_segments

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
        오디오/비디오 파일을 전사 (OOM 방지를 위해 내부적으로 청킹 처리)
        """
        if not self.source_path:
            raise ValueError(
                "For convert/demux, `source` must be a file path. Got bytes—use router to write temp file first."
            )

        ap = AudioProcessor(
            path=self.source_path,
            sr=self.target_sr,
            channels=self.target_ch,
            max_bytes=self.max_bytes,
        )
        
        segment_time = 3600
        chunk_paths = ap.segment(segment_time=segment_time, start=start, end=end)

        all_raw_segments: List[dict] = []
        total_duration = 0.0

        for i, chunk_path in enumerate(chunk_paths):
            segments, info = self.model.transcribe(
                str(chunk_path),
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

            chunk_duration = float(getattr(info, "duration", 0.0) or 0.0)
            time_offset = total_duration + start

            for seg in segments:
                txt = (seg.text or "").strip()
                if not txt:
                    continue
                avg_logprob = getattr(seg, 'avg_logprob', 0.0)
                if avg_logprob is None:
                    avg_logprob = 0.0
                all_raw_segments.append({
                    "start": float(seg.start) + time_offset,
                    "end": float(seg.end) + time_offset,
                    "content": txt,
                    "avg_logprob": avg_logprob
                })

            total_duration += chunk_duration
            cleanup_path(chunk_path)

        # 후처리 적용 (문장 병합 및 환각 제거)
        processed_segments = self._post_process_segments(all_raw_segments)
        
        # 전체 텍스트 추출
        all_text = " ".join([seg["content"] for seg in processed_segments])

        now = datetime.now()
        result = {
            "language": language,
            "duration": total_duration,
            "created_at": now.strftime("%Y-%m-%d %H:%M:%S.") + str(now.microsecond)[-3:],
            "result": {"text": all_text.strip(), "segments": processed_segments},
        }

        return result
