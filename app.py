"""
Gradio UI for the Thai → English voice translation pipeline.

Local:
    uv run app.py

Hugging Face Spaces:
    Push this repo — Spaces picks up app.py automatically.
    Set SDK: gradio in README.md front-matter (see below).
"""
import os
import numpy as np
import gradio as gr
import librosa

from pipeline import PipelineConfig, ThaiToEnglishPipeline

# ---------------------------------------------------------------------------
# Load pipeline once at startup
# ---------------------------------------------------------------------------
config = PipelineConfig(device="cpu")
pipeline = ThaiToEnglishPipeline(config)
pipeline.load()


# ---------------------------------------------------------------------------
# Inference function
# ---------------------------------------------------------------------------
def translate(audio):
    """Called by Gradio with (sample_rate, numpy_array) or None."""
    if audio is None:
        return "", "", None, ""

    sr, data = audio

    # Gradio may give int16 or float32; normalise to float32 [-1, 1]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype != np.float32:
        data = data.astype(np.float32)

    # Mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample to 16 kHz if needed
    if sr != config.sample_rate:
        data = librosa.resample(data, orig_sr=sr, target_sr=config.sample_rate)

    result = pipeline.process_segment(data)

    latency_str = (
        f"VAD {result.vad_latency_s:.2f}s | "
        f"ASR {result.asr_latency_s:.2f}s | "
        f"NMT {result.nmt_latency_s:.2f}s | "
        f"TTS {result.tts_latency_s:.2f}s | "
        f"Total {result.total_latency_s:.2f}s (RTF {result.end_to_end_rtf:.2f}×)"
    )

    audio_out = None
    if result.audio_output is not None and result.audio_output.size > 0:
        # Gradio Audio output expects (sample_rate, numpy int16 or float32)
        audio_out = (result.tts_sample_rate, result.audio_output)

    return result.thai_text, result.english_text, audio_out, latency_str


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Thai → English Voice Translation") as demo:
    gr.Markdown(
        """
        # 🇹🇭 Thai → English Voice Translation
        **Pipeline:** VAD → Typhoon ASR → NLLB-600M NMT → Piper TTS

        Record Thai speech or upload a WAV file — get English text and speech back.
        """
    )

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="numpy",
                label="Thai Speech Input",
            )
            submit_btn = gr.Button("Translate", variant="primary")

        with gr.Column():
            thai_out = gr.Textbox(label="Thai Transcription (ASR)", lines=3)
            english_out = gr.Textbox(label="English Translation (NMT)", lines=3)
            audio_out = gr.Audio(label="English Speech (TTS)", type="numpy")
            latency_out = gr.Textbox(label="Latency", lines=1, interactive=False)

    submit_btn.click(
        fn=translate,
        inputs=audio_input,
        outputs=[thai_out, english_out, audio_out, latency_out],
    )

    gr.Examples(
        examples=[["demo/sample_thai.wav"]] if os.path.exists("demo/sample_thai.wav") else [],
        inputs=audio_input,
    )

if __name__ == "__main__":
    demo.launch()
