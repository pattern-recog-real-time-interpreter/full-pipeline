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
| TTS | Kokoro-82M (ONNX) | RTF ~0.2 |
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
uv run python setup.py --tts-only   # Kokoro-82M ONNX
```

---

## Gradio UI

```bash
uv run app.py
```

Opens a browser at `http://localhost:7860`. Click the microphone, speak Thai, pause — translation appears automatically via streaming VAD.

```bash
uv run app.py --device cuda   # GPU inference
uv run app.py --share         # Gradio public tunnel link
uv run app.py --device cuda --share
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

## Evaluation (E2E)

Black-box evaluation on [FLEURS](https://huggingface.co/datasets/google/fleurs) Thai test set.
Input: real Thai speech. Output: English audio. No intermediate text used for scoring.

**Metrics**

| Metric | What it measures |
|--------|-----------------|
| BLEU | Translation accuracy (Whisper transcribes output audio → compare to FLEURS English ref) |
| chrF | Character-level translation accuracy (better for morphology) |
| RTF | Real-time factor — total latency / input audio duration |
| Latency p50/p95 | Per-sample pipeline wall time |

### 1. Install eval dependencies

```bash
pip install datasets sacrebleu openai-whisper
```

### 2. Run

```bash
# 100 samples, CPU, Whisper-small
uv run python eval/e2e_eval.py

# Options
uv run python eval/e2e_eval.py --n 200                      # more samples
uv run python eval/e2e_eval.py --device cuda                # GPU pipeline
uv run python eval/e2e_eval.py --whisper-model medium       # higher-quality transcription
uv run python eval/e2e_eval.py --out eval/my_results.json   # custom output path
```

### 3. Output

Prints a summary table:

```
============================================================
Black-box E2E evaluation  (n=100, device=cpu)
Whisper model : small

Translation quality  (Whisper(output_audio) vs FLEURS English ref)
  BLEU  : 18.34
  chrF  : 34.21

           mean      p50      p95
------------------------------------------
  Total    2.341s   2.198s   3.812s
  RTF      0.487×   0.461×   0.762×
============================================================
```

Per-sample results saved to `eval/results.json`.

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
│   ├── tts.py              # Kokoro TTS wrapper
│   └── pipeline.py         # Orchestrator → PipelineResult
├── app.py                  # Gradio streaming UI (VAD auto-detect)
├── demo/
│   ├── file_demo.py        # File input demo
│   └── mic_demo.py         # Live mic demo (terminal)
├── eval/
│   └── e2e_eval.py         # Black-box E2E eval (FLEURS, BLEU/chrF/RTF)
└── tests/
```
