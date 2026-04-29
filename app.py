"""
Gradio UI for the Thai → English voice translation pipeline.

Local:
    uv run app.py

Hugging Face Spaces:
    Push this repo — Spaces picks up app.py automatically.
    Set SDK: gradio in README.md front-matter.
"""
import asyncio
import sys

import numpy as np
import gradio as gr
import librosa

# Windows ProactorEventLoop throws ConnectionResetError on browser disconnect;
# SelectorEventLoop handles it silently.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from pipeline import PipelineConfig, ThaiToEnglishPipeline
from pipeline.vad import VADSegmenter

# ---------------------------------------------------------------------------
# Load pipeline + VAD once at startup
# ---------------------------------------------------------------------------
config = PipelineConfig(device="cpu")
pipeline = ThaiToEnglishPipeline(config)
vad = VADSegmenter(config)
pipeline.load()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_16k_float32(sr: int, data: np.ndarray) -> np.ndarray:
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype != np.float32:
        data = data.astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != config.sample_rate:
        data = librosa.resample(data, orig_sr=sr, target_sr=config.sample_rate)
    return data


def _run_pipeline(segment: np.ndarray):
    result = pipeline.process_segment(segment)
    latency_str = (
        f"VAD {result.vad_latency_s:.2f}s | ASR {result.asr_latency_s:.2f}s | "
        f"NMT {result.nmt_latency_s:.2f}s | TTS {result.tts_latency_s:.2f}s | "
        f"Total {result.total_latency_s:.2f}s  (RTF {result.end_to_end_rtf:.2f}×)"
    )
    audio_out = None
    if result.audio_output is not None and result.audio_output.size > 0:
        pcm = (result.audio_output * 32767).clip(-32768, 32767).astype(np.int16)
        audio_out = (result.tts_sample_rate, pcm)
    return result.thai_text, result.english_text, audio_out, latency_str


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------
def on_start():
    vad.reset()
    return "Listening…"


def process_chunk(audio):
    """Called by Gradio for every streaming audio chunk."""
    if audio is None:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    sr, data = audio
    data = _to_16k_float32(sr, data)
    segment = vad.push_chunk(data)
    if segment is None:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    return _run_pipeline(segment)


def on_stop():
    """Flush any remaining audio when the user stops recording."""
    segment = vad.flush()
    if segment is not None and len(segment) > 0:
        return _run_pipeline(segment)
    return gr.skip(), gr.skip(), gr.skip(), gr.skip()


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Thai → English Voice Translation") as demo:
    gr.Markdown(
        """
        # Thai → English Voice Translation
        **Pipeline:** Silero VAD → Typhoon ASR → NLLB-600M NMT → Kokoro TTS

        Click the microphone to start recording. Speak Thai, then pause —
        the translation will appear automatically. Each pause triggers one full pipeline run.
        """
    )

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources=["microphone"],
                streaming=True,
                type="numpy",
                label="Thai Speech Input",
            )
            status_out = gr.Textbox(
                label="Status",
                value="Click the mic to start",
                interactive=False,
                lines=1,
            )

        with gr.Column():
            thai_out = gr.Textbox(label="Thai Transcription (ASR)", lines=3)
            english_out = gr.Textbox(label="English Translation (NMT)", lines=3)
            audio_out = gr.Audio(
                label="English Speech (TTS)",
                type="numpy",
                autoplay=True,
            )
            latency_out = gr.Textbox(label="Latency", lines=1, interactive=False)

    audio_input.start_recording(
        fn=on_start,
        outputs=status_out,
    )
    audio_input.stream(
        fn=process_chunk,
        inputs=audio_input,
        outputs=[thai_out, english_out, audio_out, latency_out],
    )
    audio_input.stop_recording(
        fn=on_stop,
        outputs=[thai_out, english_out, audio_out, latency_out],
    )


if __name__ == "__main__":
    demo.launch()
