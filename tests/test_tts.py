import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.config import PipelineConfig
from pipeline.tts import TTSEngine

PIPER_MODEL = os.path.join("models", "piper", "en_US-lessac-medium.onnx")


@pytest.fixture(scope="module")
def tts():
    if not os.path.exists(PIPER_MODEL):
        pytest.skip("Piper model not found — run python setup.py --tts-only first")
    config = PipelineConfig()
    engine = TTSEngine(config)
    engine.load()
    return engine


def test_synthesizes_audio(tts):
    result = tts.synthesize("Hello, world!")
    assert result.audio.size > 0
    assert result.sample_rate > 0
    assert result.latency_s > 0


def test_audio_dtype(tts):
    result = tts.synthesize("Testing audio output.")
    assert result.audio.dtype == np.float32
    assert result.audio.max() <= 1.0
    assert result.audio.min() >= -1.0


def test_rtf_realtime(tts):
    result = tts.synthesize("The quick brown fox jumps over the lazy dog.")
    assert result.rtf < 1.0, f"Piper RTF {result.rtf:.3f} is not real-time"
