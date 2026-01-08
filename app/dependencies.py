from functools import lru_cache
from faster_whisper import WhisperModel


class WhisperFactory:
    def __init__(self, model_name="large-v3", device="cuda", compute_type="float16", device_index=0, cpu_threads=4):
        self.kw = dict(model_name=model_name, device=device, compute_type=compute_type)
        self.device_index = device_index
        self.cpu_threads = cpu_threads

    @lru_cache(maxsize=1)
    def get(self) -> WhisperModel:
        return WhisperModel(self.kw["model_name"], device=self.kw["device"], compute_type=self.kw["compute_type"])


factory = WhisperFactory()
get_model = factory.get
