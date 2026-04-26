import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline import PipelineConfig, ThaiToEnglishPipeline

MODELS_READY = (
    os.path.exists(os.path.join("models", "typhoon-asr.nemo"))
    and os.path.exists(os.path.join("models", "nllb-600m-int8"))
    and os.path.exists(os.path.join("models", "piper", "en_US-lessac-medium.onnx"))
)


@pytest.fixture(scope="module")
def pipeline():
    if not MODELS_READY:
        pytest.skip("Not all models present — run python setup.py first")
    p = ThaiToEnglishPipeline(PipelineConfig())
    p.load()
    return p


def _make_audio(duration_s: float = 3.0) -> np.ndarray:
    sr = 16000
    t = np.linspace(0, duration_s, int(sr * duration_s), dtype=np.float32)
    return (0.05 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)


def test_process_segment_returns_result(pipeline):
    audio = _make_audio(3.0)
    result = pipeline.process_segment(audio)
    assert isinstance(result.thai_text, str)
    assert isinstance(result.english_text, str)
    assert result.total_latency_s > 0
    assert result.audio_duration_s > 0


def test_e2e_rtf_logged(pipeline):
    audio = _make_audio(5.0)
    result = pipeline.process_segment(audio)
    assert result.end_to_end_rtf >= 0


def test_audio_output_when_text_present(pipeline):
    audio = _make_audio(3.0)
    result = pipeline.process_segment(audio)
    # If English text was produced, audio output should be present
    if result.english_text.strip():
        assert result.audio_output is not None
        assert result.audio_output.size > 0
