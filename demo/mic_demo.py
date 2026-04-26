"""
Live microphone demo: Thai speech → English TTS output.

Usage:
    python demo/mic_demo.py [--device cpu|cuda]

Press Ctrl+C to stop.
"""
import argparse
import queue
import sys
import os
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import sounddevice as sd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live

from pipeline import PipelineConfig, ThaiToEnglishPipeline
from pipeline.vad import VADSegmenter


console = Console()
MIC_CHUNK_SECONDS = 0.1   # 100ms mic polling window
MIC_SAMPLERATE = 16000


def render_result(i: int, result) -> Panel:
    table = Table.grid(padding=(0, 1))
    table.add_column(style="bold cyan", min_width=8)
    table.add_column()
    table.add_row("Thai:", result.thai_text or "(empty)")
    table.add_row("English:", result.english_text or "(empty)")
    table.add_row(
        "Latency:",
        f"VAD {result.vad_latency_s:.3f}s │ "
        f"ASR {result.asr_latency_s:.3f}s │ "
        f"NMT {result.nmt_latency_s:.3f}s │ "
        f"TTS {result.tts_latency_s:.3f}s",
    )
    table.add_row(
        "Total:",
        f"{result.total_latency_s:.3f}s  │  "
        f"Audio {result.audio_duration_s:.1f}s  │  "
        f"E2E RTF {result.end_to_end_rtf:.2f}×",
    )
    return Panel(table, title=f"[bold]Segment {i}[/bold]", border_style="green")


def main():
    parser = argparse.ArgumentParser(description="Thai→English live mic demo")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    config = PipelineConfig(device=args.device)
    pipeline = ThaiToEnglishPipeline(config)
    vad = VADSegmenter(config)

    console.print("[bold yellow]Loading models...[/bold yellow]")
    pipeline.load()
    console.print("[bold green]Models loaded. Speak Thai into your microphone.[/bold green]")
    console.print("[dim]Press Ctrl+C to stop.[/dim]\n")

    segment_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=4)
    chunk_size = int(MIC_CHUNK_SECONDS * MIC_SAMPLERATE)

    def audio_callback(indata: np.ndarray, frames: int, time_info, status):
        chunk = indata[:, 0].copy()  # mono
        segment = vad.push_chunk(chunk)
        if segment is not None:
            try:
                segment_queue.put_nowait(segment)
            except queue.Full:
                pass  # drop segment if processing can't keep up

    seg_index = 0
    stream = sd.InputStream(
        samplerate=MIC_SAMPLERATE,
        channels=1,
        dtype="float32",
        blocksize=chunk_size,
        callback=audio_callback,
    )

    try:
        with stream:
            console.print("[dim]Listening...[/dim]")
            while True:
                try:
                    segment = segment_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                seg_index += 1
                result = pipeline.process_segment(segment)
                console.print(render_result(seg_index, result))

                if result.audio_output is not None and result.audio_output.size > 0:
                    sd.play(result.audio_output, samplerate=result.tts_sample_rate, blocking=False)

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Stopping...[/bold yellow]")
        # Flush any remaining buffered audio
        remaining = vad.flush()
        if remaining is not None and len(remaining) > 0:
            seg_index += 1
            result = pipeline.process_segment(remaining)
            console.print(render_result(seg_index, result))
            if result.audio_output is not None and result.audio_output.size > 0:
                sd.play(result.audio_output, samplerate=result.tts_sample_rate, blocking=True)

    console.print("[bold green]Done.[/bold green]")


if __name__ == "__main__":
    main()
