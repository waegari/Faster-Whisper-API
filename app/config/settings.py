from pathlib import Path
from dotenv import load_dotenv
import os
from dataclasses import dataclass

BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent.parent
load_dotenv(dotenv_path=BASE_DIR / "env" / ".env")


@dataclass
class Settings:
    openai_api_key: str = ""
    MAX_AUDIO_BYTES: int = 10 * 1024 * 1024 * 1024
    DEFAULT_SR: int = 16000  # sampling rate
    DEFAULT_BR: str = "96k"  # bit rate
    DEFAULT_CH: int = 1  # number of channel(s)
    MAX_CHUNK_DURATION_MS: int = (2 * 60 + 5) * 60 * 1000
    TEMP_DIR: Path = Path(r"D:\Temp")
    TEMP_FILE_TTL_HOURS: int = 24

    def __post_init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY", self.openai_api_key)
        self.TEMP_DIR = Path(os.getenv("TEMP_DIR", str(self.TEMP_DIR))).resolve()
        self.TEMP_FILE_TTL_HOURS = int(os.getenv("TEMP_FILE_TTL_HOURS", str(self.TEMP_FILE_TTL_HOURS)))


settings = Settings()
