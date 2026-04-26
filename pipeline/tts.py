import time
from dataclasses import dataclass

import numpy as np

from .config import PipelineConfig


@dataclass
class TTSResult:
    audio: np.ndarray   # float32, normalized to [-1, 1]
    sample_rate: int
    latency_s: float
    rtf: float


class TTSEngine:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._voice = None
        self._sample_rate: int = 22050

    def load(self) -> None:
        from piper import PiperVoice

        print(f"[TTS] Loading Piper from {self.config.tts_model_path} ...")
        self._voice = PiperVoice.load(self.config.tts_model_path)
        # Piper config exposes the output sample rate
        self._sample_rate = self._voice.config.sample_rate
        print(f"[TTS] Piper loaded (sample_rate={self._sample_rate}).")

    def synthesize(self, text: str) -> TTSResult:
        assert self._voice is not None, "Call load() first"

        t0 = time.perf_counter()
        raw = b"".join(self._voice.synthesize_stream_raw(text))
        latency_s = time.perf_counter() - t0

        if not raw:
            return TTSResult(audio=np.zeros(0, dtype=np.float32),
                             sample_rate=self._sample_rate, latency_s=latency_s, rtf=0.0)

        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        audio_duration_s = len(audio) / self._sample_rate
        rtf = latency_s / audio_duration_s if audio_duration_s > 0 else 0.0

        return TTSResult(audio=audio, sample_rate=self._sample_rate, latency_s=latency_s, rtf=rtf)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def is_loaded(self) -> bool:
        return self._voice is not None
