import io
import os
import sys
import time
import traceback
import wave
from dataclasses import dataclass

import numpy as np

from .config import PipelineConfig


def _fix_piper_phonemize_windows() -> None:
    # Must be called at module import time — before piper or piper_phonemize is imported.
    """Register piper-phonemize's espeak-ng DLL directory BEFORE the module is imported.

    piper-phonemize ships its own espeak-ng DLL but doesn't add its own directory
    to the DLL search path (required since Python 3.8 on Windows). The fix must run
    before any import of piper_phonemize, so we use find_spec() which locates the
    package without executing it.
    """
    if sys.platform != "win32":
        return
    try:
        import importlib.util
        spec = importlib.util.find_spec("piper_phonemize")
        if spec and spec.origin:
            phonemize_dir = os.path.dirname(spec.origin)
            os.add_dll_directory(phonemize_dir)
            if "ESPEAK_DATA_PATH" not in os.environ:
                data_path = os.path.join(phonemize_dir, "espeak-ng-data")
                if os.path.isdir(data_path):
                    os.environ["ESPEAK_DATA_PATH"] = data_path
    except Exception:
        pass


_fix_piper_phonemize_windows()  # run at import time, before piper is ever imported


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
