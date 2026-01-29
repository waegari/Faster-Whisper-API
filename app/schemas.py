from typing import List, Optional
from pydantic import BaseModel


class Segment(BaseModel):
    index: int
    start: float
    end: float
    content: str
    avg_logprob: float
    prob: int


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
    word_timestamps: bool = False
