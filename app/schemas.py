from typing import List, Optional
from pydantic import BaseModel, Field


class Word(BaseModel):
    word: str
    start: float
    end: float
    probability: float


class Segment(BaseModel):
    index: int
    start: float
    end: float
    content: str
    avg_logprob: float
    prob: int
    speaker: Optional[str] = None
    words: Optional[List[Word]] = None


class TranscribeResult(BaseModel):
    request_id: Optional[str] = None
    task: str = "transcribe"
    language: str
    duration: float
    created_at: str
    result: dict

class TranscribeQuery(BaseModel):
    request_id: Optional[str] = None
    task: str = "transcribe"
    language: str = "ko"
    is_video: bool = False
    start: int = 0
    end: int = 0
    vad: bool = True
    word_timestamps: bool = Field(
        default=False,
        description="단어 단위 타임스탬프. true면 각 세그먼트에 words[]를 포함.",
    )
    max_speech_duration_s: Optional[float] = Field(
        default=None,
        gt=0,
        description="VAD 최장 발화 길이(초). 값이 있으면 해당 길이의 조각 단위로 전사·반환. 생략 또는 null이면 무제한(통짜 전사).",
    )
