"""
Usage:
    python demo/file_demo.py path/to/audio.wav [--device cpu|cuda] [--output out.wav]
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import soundfile as sf
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from pipeline import PipelineConfig, ThaiToEnglishPipeline


console = Console()


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
    parser = argparse.ArgumentParser(description="Thai→English file translation demo")
    parser.add_argument("audio", help="Path to input WAV file (Thai speech)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output", default=None, help="Save concatenated English audio to WAV")
    args = parser.parse_args()

    config = PipelineConfig(device=args.device)
    pipeline = ThaiToEnglishPipeline(config)

    console.print("[bold yellow]Loading models...[/bold yellow]")
    pipeline.load()
    console.print("[bold green]Models loaded. Processing file...[/bold green]\n")

    results = pipeline.process_file(args.audio)

    if not results:
        console.print("[red]No speech detected in file.[/red]")
        return

    all_audio = []
    for i, result in enumerate(results, 1):
        console.print(render_result(i, result))
        if result.audio_output is not None:
            all_audio.append(result.audio_output)

    if args.output and all_audio:
        combined = np.concatenate(all_audio)
        sr = results[0].tts_sample_rate
        # Convert float32 → int16 for standard WAV
        out_int16 = (combined * 32767).clip(-32768, 32767).astype("int16")
        sf.write(args.output, out_int16, sr, subtype="PCM_16")
        console.print(f"\n[bold]Saved English audio to:[/bold] {args.output}")


if __name__ == "__main__":
    main()
