from pathlib import Path
from dotenv import load_dotenv
import os
from dataclasses import dataclass

BASE_DIR = Path(__file__).parent
load_dotenv(dotenv_path=BASE_DIR / "env" / ".env")


@dataclass
class Settings:
    openai_api_key: str = ""
    MAX_AUDIO_BYTES: int = 10 * 1024 * 1024 * 1024  # 한 번에 처리하는 오디오 용량
    DEFAULT_SR: int = 16000  # 샘플링 레이트 기본값(Hz)
    DEFAULT_BR: str = "96k"  # 비트 레이트 기본값
    DEFAULT_CH: int = 1  # 채널 수 기본값(1, mono)
    MAX_CHUNK_DURATION_MS: int = (2 * 60 + 5) * 60 * 1000  # 오디오 처리 단위의 상한(2시간 5분)

    def __post_init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY", self.openai_api_key)


settings = Settings()
