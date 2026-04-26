---
title: Thai-English Voice Translation
emoji: 🇹🇭
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: false
---

# Real-Time Thai → English Voice Translation Pipeline

VAD → ASR → NMT → TTS pipeline that takes Thai speech and outputs English speech.

| Stage | Model | CPU Latency |
|---|---|---|
| VAD | Silero VAD (auto-download) | ~0s |
| ASR | Typhoon ASR (NeMo FastConformer-Transducer) | RTF 0.021 |
| NMT | NLLB-600M INT8 (CTranslate2) | ~1.15s/sentence |
| TTS | Piper en_US-lessac-medium (ONNX) | RTF 0.186 |
| **Total** | | **~2.2s per 5s utterance** |

---

## Setup

### 1. Install dependencies

```bash
uv sync
```

> If uv's resolver hits conflicts with `nemo_toolkit`, fall back to:
> ```bash
> uv pip install -r requirements.txt
> ```

### 2. Download and convert models (~2–3 GB, one-time)

```bash
# All models
uv run python setup.py

# Or individually
uv run python setup.py --asr-only   # Typhoon ASR (.nemo)
uv run python setup.py --nmt-only   # NLLB-600M INT8 (CTranslate2)
uv run python setup.py --tts-only   # Piper ONNX
```

---

## Gradio UI

```bash
uv run app.py
```

Opens a browser at `http://localhost:7860`. Record Thai speech or upload a WAV file — get Thai transcription, English translation, and English audio back.

To get a temporary public link (no deployment needed):

```python
# in app.py, change the last line to:
demo.launch(share=True)
```

---

## Running the Demo

### Audio file (no microphone required)

```bash
uv run python demo/file_demo.py path/to/thai_speech.wav
```

Save the translated English audio to a file:

```bash
uv run python demo/file_demo.py path/to/thai_speech.wav --output english_out.wav
```

### Live microphone

```bash
uv run python demo/mic_demo.py
```

Press **Ctrl+C** to stop. Each detected speech segment prints a result table:

```
╭─────────────── Segment 1 ────────────────╮
│ Thai:    สวัสดีครับ ผมชื่อวิชัย          │
│ English: Hello. My name is Vichai.       │
│ Latency: VAD 0.001s │ ASR 0.09s │        │
│          NMT 1.12s  │ TTS 0.78s          │
│ Total:   1.99s  │  Audio 4.2s  │ RTF 0.47x │
╰───────────────────────────────────────────╯
```

### GPU (if available)

Add `--device cuda` to either demo for faster inference (~1s per utterance):

```bash
uv run python demo/file_demo.py path/to/audio.wav --device cuda
uv run python demo/mic_demo.py --device cuda
```

---

## Running Tests

```bash
uv run pytest tests/
```

Tests skip automatically if models are not yet downloaded.

---

## Project Structure

```
full-pipeline/
├── pyproject.toml          # uv project config
├── requirements.txt        # pip fallback
├── setup.py                # model download/conversion
├── pipeline/
│   ├── config.py           # PipelineConfig
│   ├── vad.py              # Silero VAD (batch + streaming)
│   ├── asr.py              # Typhoon ASR wrapper
│   ├── nmt.py              # NLLB-600M translation
│   ├── tts.py              # Piper TTS wrapper
│   └── pipeline.py         # Orchestrator → PipelineResult
├── demo/
│   ├── file_demo.py        # File input demo
│   └── mic_demo.py         # Live mic demo
└── tests/
```
