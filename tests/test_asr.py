import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.config import PipelineConfig
from pipeline.asr import ASREngine

TYPHOON_MODEL = os.path.join("models", "typhoon-asr.nemo")


@pytest.fixture(scope="module")
def asr():
    if not os.path.exists(TYPHOON_MODEL):
        pytest.skip("Typhoon model not found — run python setup.py --asr-only first")
    config = PipelineConfig()
    engine = ASREngine(config)
    engine.load()
    return engine


def test_transcribes_speech(asr):
    # 2 seconds of a 440Hz sine wave — not real Thai but verifies the pipeline runs
    sr = 16000
    t = np.linspace(0, 2, sr * 2, dtype=np.float32)
    audio = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    result = asr.transcribe(audio)
    assert isinstance(result.thai_text, str)
    assert result.latency_s > 0
    assert result.rtf >= 0


def test_silence_returns_string(asr):
    audio = np.zeros(16000, dtype=np.float32)
    result = asr.transcribe(audio)
    assert isinstance(result.thai_text, str)  # may be empty but should not crash


def test_rtf_reasonable(asr):
    sr = 16000
    audio = np.random.randn(sr * 5).astype(np.float32) * 0.01
    result = asr.transcribe(audio)
    assert result.rtf < 5.0, f"RTF {result.rtf:.2f} suspiciously high"
