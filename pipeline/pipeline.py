import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import PipelineConfig
from .asr import ASREngine
from .nmt import NMTEngine
from .tts import TTSEngine
from .vad import VADSegmenter


@dataclass
class PipelineResult:
    audio_duration_s: float
    thai_text: str
    english_text: str
    audio_output: Optional[np.ndarray]  # float32, sr = tts_sample_rate
    tts_sample_rate: int
    vad_latency_s: float
    asr_latency_s: float
    nmt_latency_s: float
    tts_latency_s: float
    total_latency_s: float
    end_to_end_rtf: float  # total_latency_s / audio_duration_s


class ThaiToEnglishPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.vad = VADSegmenter(self.config)
        self.asr = ASREngine(self.config)
        self.nmt = NMTEngine(self.config)
        self.tts = TTSEngine(self.config)

    def load(self) -> None:
        """Load all models. VAD loads in __init__ (torch.hub)."""
        self.asr.load()
        self.nmt.load()
        self.tts.load()

    def process_segment(self, audio: np.ndarray) -> PipelineResult:
        """Run VAD trim → ASR → NMT → TTS on a pre-captured audio segment."""
        t_total_start = time.perf_counter()

        # VAD: trim the segment
        t0 = time.perf_counter()
        segments = self.vad.segment_file(audio)
        vad_latency_s = time.perf_counter() - t0
        audio = np.concatenate(segments) if segments else audio
        audio_duration_s = len(audio) / self.config.sample_rate

        # ASR
        asr_result = self.asr.transcribe(audio)

        # NMT
        nmt_result = self.nmt.translate(asr_result.thai_text)

        # TTS
        english_text = nmt_result.english_text
        tts_result = self.tts.synthesize(english_text) if english_text.strip() else None

        total_latency_s = time.perf_counter() - t_total_start
        e2e_rtf = total_latency_s / audio_duration_s if audio_duration_s > 0 else 0.0

        return PipelineResult(
            audio_duration_s=audio_duration_s,
            thai_text=asr_result.thai_text,
            english_text=english_text,
            audio_output=tts_result.audio if tts_result else None,
            tts_sample_rate=tts_result.sample_rate if tts_result else self.tts.sample_rate,
            vad_latency_s=vad_latency_s,
            asr_latency_s=asr_result.latency_s,
            nmt_latency_s=nmt_result.latency_s,
            tts_latency_s=tts_result.latency_s if tts_result else 0.0,
            total_latency_s=total_latency_s,
            end_to_end_rtf=e2e_rtf,
        )

    def process_file(self, wav_path: str) -> list[PipelineResult]:
        """Load audio file, split by VAD, process each speech segment."""
        import soundfile as sf
        import librosa

        audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != self.config.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.config.sample_rate)

        segments = self.vad.segment_file(audio)
        if not segments:
            print("[Pipeline] No speech segments detected.")
            return []

        results = []
        for i, seg in enumerate(segments, 1):
            print(f"[Pipeline] Processing segment {i}/{len(segments)} ({len(seg)/self.config.sample_rate:.1f}s) ...")
            result = self.process_segment(seg)
            results.append(result)
        return results
