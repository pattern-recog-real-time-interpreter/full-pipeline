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
        self._kokoro = None
        self._sample_rate: int = 24000

    def load(self) -> None:
        import onnxruntime as ort
        from kokoro_onnx import Kokoro

        print(f"[TTS] Loading Kokoro from {self.config.tts_model_path} ...")
        session = ort.InferenceSession(
            self.config.tts_model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        print(f"[TTS] ORT providers in use: {session.get_providers()}")
        self._kokoro = Kokoro.from_session(session, self.config.tts_voices_path)
        print(f"[TTS] Warming up Kokoro ...")
        # Warm up to get the actual sample rate
        samples, sr = self._kokoro.create("hi", voice=self.config.tts_voice,
                                           speed=1.0, lang="en-us")
        self._sample_rate = int(sr)
        print(f"[TTS] Kokoro loaded (sample_rate={self._sample_rate}).")

    def synthesize(self, text: str) -> TTSResult:
        assert self._kokoro is not None, "Call load() first"

        t0 = time.perf_counter()
        samples, sr = self._kokoro.create(
            text,
            voice=self.config.tts_voice,
            speed=1.0,
            lang="en-us",
        )
        latency_s = time.perf_counter() - t0

        audio = np.array(samples, dtype=np.float32)
        audio_duration_s = len(audio) / sr if sr > 0 else 0.0
        rtf = latency_s / audio_duration_s if audio_duration_s > 0 else 0.0

        return TTSResult(audio=audio, sample_rate=int(sr),
                         latency_s=latency_s, rtf=rtf)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def is_loaded(self) -> bool:
        return self._kokoro is not None
