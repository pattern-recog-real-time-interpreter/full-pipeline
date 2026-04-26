import os
import time
import tempfile
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from .config import PipelineConfig


@dataclass
class ASRResult:
    thai_text: str
    latency_s: float
    audio_duration_s: float
    rtf: float


class ASREngine:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._model = None

    def load(self) -> None:
        import nemo.collections.asr as nemo_asr

        model_path = self.config.asr_model_path
        if os.path.exists(model_path):
            print(f"[ASR] Loading Typhoon from {model_path}")
            self._model = nemo_asr.models.ASRModel.restore_from(model_path)
        else:
            print(f"[ASR] Downloading Typhoon from {self.config.asr_model_name} ...")
            self._model = nemo_asr.models.ASRModel.from_pretrained(self.config.asr_model_name)

        self._model.eval()
        if self.config.device == "cuda":
            self._model = self._model.cuda()
        print("[ASR] Typhoon loaded.")

    def transcribe(self, audio: np.ndarray) -> ASRResult:
        assert self._model is not None, "Call load() first"
        audio_duration_s = len(audio) / self.config.sample_rate

        # NeMo transcribe accepts file paths — write to a temp WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        try:
            sf.write(tmp_path, audio, self.config.sample_rate)
            t0 = time.perf_counter()
            results = self._model.transcribe([tmp_path])
            latency_s = time.perf_counter() - t0
        finally:
            os.unlink(tmp_path)

        # NeMo returns list; each item may be a string or a Hypothesis object
        raw = results[0]
        thai_text = raw.text if hasattr(raw, "text") else str(raw)
        thai_text = thai_text.strip()

        rtf = latency_s / audio_duration_s if audio_duration_s > 0 else 0.0
        return ASRResult(thai_text=thai_text, latency_s=latency_s,
                         audio_duration_s=audio_duration_s, rtf=rtf)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
