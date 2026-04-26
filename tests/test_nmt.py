import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.config import PipelineConfig
from pipeline.nmt import NMTEngine

NMT_MODEL_DIR = os.path.join("models", "nllb-600m-int8")


@pytest.fixture(scope="module")
def nmt():
    if not os.path.exists(NMT_MODEL_DIR):
        pytest.skip("NLLB model not found — run python setup.py --nmt-only first")
    config = PipelineConfig()
    engine = NMTEngine(config)
    engine.load()
    return engine


def test_translates_thai(nmt):
    result = nmt.translate("สวัสดีครับ")
    assert result.english_text.strip() != ""
    assert len(result.english_text) > 2
    assert result.latency_s > 0


def test_empty_input(nmt):
    result = nmt.translate("")
    assert result.english_text == ""
    assert result.latency_s == 0.0


def test_latency_reasonable(nmt):
    result = nmt.translate("วันนี้อากาศดีมาก")
    assert result.latency_s < 30.0, f"NMT took {result.latency_s:.1f}s — too slow"
