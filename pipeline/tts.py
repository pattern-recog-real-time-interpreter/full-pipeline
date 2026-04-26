import io
import time
import traceback
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
        self._sample_rate = self._voice.config.sample_rate
        print(f"[TTS] Piper loaded (sample_rate={self._sample_rate}).")

    def synthesize(self, text: str) -> TTSResult:
        assert self._voice is not None, "Call load() first"

        buf = io.BytesIO()
        t0 = time.perf_counter()
        with wave.open(buf, "wb") as wav_file:
            # Pre-set headers so wav_file.close() never raises if synthesize() fails
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self._sample_rate)
            try:
                self._voice.synthesize(text, wav_file)
            except Exception:
                traceback.print_exc()
        latency_s = time.perf_counter() - t0

        buf.seek(0)
        with wave.open(buf) as wav_file:
            raw = wav_file.readframes(wav_file.getnframes())

        if not raw:
            print(f"[TTS] synthesize() produced 0 bytes for text: {repr(text[:80])}")
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
