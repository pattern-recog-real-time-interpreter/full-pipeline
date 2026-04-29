from dataclasses import dataclass, field
import os


@dataclass
class PipelineConfig:
    device: str = "cpu"  # "cpu" | "cuda"

    # NMT
    nmt_model_dir: str = field(default_factory=lambda: os.path.join("models", "nllb-600m-int8"))
    nmt_tokenizer_name: str = "facebook/nllb-200-distilled-600M"

    # TTS — Kokoro-82M ONNX
    tts_model_path: str = field(default_factory=lambda: os.path.join("models", "kokoro", "kokoro-v1.0.onnx"))
    tts_voices_path: str = field(default_factory=lambda: os.path.join("models", "kokoro", "voices-v1.0.bin"))
    tts_voice: str = "af_heart"  # American English female; see kokoro-onnx docs for options

    # ASR
    asr_model_path: str = field(default_factory=lambda: os.path.join("models", "typhoon-asr.nemo"))
    asr_model_name: str = "typhoon-ai/typhoon-asr-realtime"  # HF model id for download

    # VAD
    sample_rate: int = 16000
    vad_min_silence_ms: int = 500
    vad_min_speech_ms: int = 250
    max_segment_seconds: float = 15.0
    rms_threshold: float = 0.01  # energy gate for streaming VAD layer 1
