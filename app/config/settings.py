from pathlib import Path
from dotenv import load_dotenv
import os
from dataclasses import dataclass

BASE_DIR = Path(__file__).parent
load_dotenv(dotenv_path=BASE_DIR / "env" / ".env")


@dataclass
class Settings:
    openai_api_key: str = ""
    MAX_AUDIO_BYTES: int = 10 * 1024 * 1024 * 1024
    DEFAULT_SR: int = 16000  # sampling rate
    DEFAULT_BR: str = "96k"  # bit rate
    DEFAULT_CH: int = 1  # number of channel(s)
    MAX_CHUNK_DURATION_MS: int = (2 * 60 + 5) * 60 * 1000

    def __post_init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY", self.openai_api_key)


settings = Settings()
