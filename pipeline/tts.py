import io
import time
import wave
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

        buf = io.BytesIO()
        t0 = time.perf_counter()
        with wave.open(buf, "wb") as wav_file:
            self._voice.synthesize(text, wav_file)
        latency_s = time.perf_counter() - t0

        buf.seek(0)
        with wave.open(buf) as wav_file:
            n_frames = wav_file.getnframes()
            sr = wav_file.getframerate()
            raw = wav_file.readframes(n_frames)

        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        audio_duration_s = len(audio) / sr
        rtf = latency_s / audio_duration_s if audio_duration_s > 0 else 0.0

        return TTSResult(audio=audio, sample_rate=sr, latency_s=latency_s, rtf=rtf)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def is_loaded(self) -> bool:
        return self._voice is not None
