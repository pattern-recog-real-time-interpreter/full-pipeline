"""
Black-box E2E evaluation: Thai audio -> pipeline -> English audio.

Treats the pipeline as a black box. Evaluates output audio only -- no
intermediate text (ASR output, NMT output) is used for scoring.

Dataset
-------
  FLEURS (google/fleurs) -- real human Thai speech aligned with English text.
    th_th  ->  Thai audio input
    en_us  ->  English transcription (reference text for BLEU/chrF)

Metrics
-------
  BLEU + chrF   : Whisper transcribes output audio -> compare to FLEURS English reference
  UTMOS         : automated MOS (speech naturalness) on output audio, scale 1-5
  E2E latency   : total wall time per sample (mean / p50 / p95)
  RTF           : total_latency / input_audio_duration

Install eval deps:
    pip install datasets sacrebleu openai-whisper
    (UTMOS loads via torch.hub automatically on first run)

Usage:
    uv run python eval/e2e_eval.py [--n 100] [--device cpu] [--whisper-model small]
"""
import argparse
import json
import os
import statistics
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import librosa
import numpy as np
import sacrebleu
import soundfile as sf
import torch
import whisper
from datasets import load_dataset

from pipeline import PipelineConfig, ThaiToEnglishPipeline


def load_utmos():
    """Load UTMOS22 strong predictor via torch.hub (auto-downloads on first call)."""
    print("[eval] Loading UTMOS22 ...")
    predictor = torch.hub.load(
        "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
    )
    predictor.eval()
    return predictor


def compute_utmos(predictor, audio: np.ndarray, sr: int) -> float:
    """Predict MOS for audio (1-5 scale). Resamples to 16 kHz internally."""
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    wav = torch.tensor(audio).unsqueeze(0)  # (1, T)
    with torch.no_grad():
        score = predictor(wav, 16000)
    return float(score.mean())


def transcribe_audio(asr_model, audio: np.ndarray, sr: int) -> str:
    """Transcribe English audio with Whisper via temp WAV."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    try:
        sf.write(tmp, audio, sr)
        return asr_model.transcribe(tmp, language="en", fp16=False)["text"].strip()
    finally:
        os.unlink(tmp)


def latency_stats(values: list[float]) -> dict:
    s = sorted(values)
    n = len(s)
    return {
        "mean": statistics.mean(s),
        "p50":  s[n // 2],
        "p95":  s[min(int(n * 0.95), n - 1)],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",             type=int,   default=100)
    parser.add_argument("--device",        default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--whisper-model", default="small",
                        choices=["tiny", "base", "small", "medium"])
    parser.add_argument("--out",           default="eval/results.json")
    args = parser.parse_args()

    print("[eval] Loading translation pipeline ...")
    config = PipelineConfig(device=args.device)
    pipe = ThaiToEnglishPipeline(config)
    pipe.load()

    print(f"[eval] Loading Whisper {args.whisper_model} ...")
    asr = whisper.load_model(args.whisper_model)

    utmos = load_utmos()

    print("[eval] Fetching FLEURS th_th / en_us test splits ...")
    ds_th = load_dataset("google/fleurs", "th_th", split="test", trust_remote_code=True)
    ds_en = load_dataset("google/fleurs", "en_us", split="test", trust_remote_code=True)
    en_by_id = {row["id"]: row["transcription"] for row in ds_en}

    n = min(args.n, len(ds_th))

    transcripts: list[str] = []
    references:  list[str] = []
    records:     list[dict] = []

    for i, row in enumerate(list(ds_th)[:n]):
        en_ref_text = en_by_id.get(row["id"], "")
        if not en_ref_text:
            continue

        audio_in = np.array(row["audio"]["array"], dtype=np.float32)
        orig_sr  = row["audio"]["sampling_rate"]
        if orig_sr != config.sample_rate:
            audio_in = librosa.resample(audio_in, orig_sr=orig_sr,
                                        target_sr=config.sample_rate)

        dur = len(audio_in) / config.sample_rate
        print(f"\n[{i+1}/{n}] input_dur={dur:.1f}s")

        # ---- BLACK BOX ----
        result = pipe.process_segment(audio_in)

        if result.audio_output is None or result.audio_output.size == 0:
            print("  skip -- pipeline produced no audio output")
            continue

        transcript = transcribe_audio(asr, result.audio_output, result.tts_sample_rate)
        mos = compute_utmos(utmos, result.audio_output, result.tts_sample_rate)

        transcripts.append(transcript)
        references.append(en_ref_text)

        print(f"  Whisper : {transcript}")
        print(f"  Ref     : {en_ref_text}")
        print(f"  UTMOS   : {mos:.3f}")
        print(f"  RTF     : {result.end_to_end_rtf:.2f}x  total={result.total_latency_s:.2f}s")

        records.append({
            "id":        row["id"],
            "en_ref":    en_ref_text,
            "en_hyp":    transcript,
            "utmos":     mos,
            "audio_dur": result.audio_duration_s,
            "total_s":   result.total_latency_s,
            "rtf":       result.end_to_end_rtf,
            "vad_s":     result.vad_latency_s,
            "asr_s":     result.asr_latency_s,
            "nmt_s":     result.nmt_latency_s,
            "tts_s":     result.tts_latency_s,
        })

    actual_n = len(records)
    if actual_n == 0:
        print("No samples produced audio output.")
        return

    bleu     = sacrebleu.corpus_bleu(transcripts, [references])
    chrf     = sacrebleu.corpus_chrf(transcripts, [references])
    avg_mos  = statistics.mean(r["utmos"] for r in records)
    lat      = latency_stats([r["total_s"] for r in records])
    rtf      = latency_stats([r["rtf"]     for r in records])

    print("\n" + "=" * 60)
    print(f"Black-box E2E evaluation  (n={actual_n}, device={args.device})")
    print(f"Whisper model : {args.whisper_model}")
    print()
    print("Translation quality  (Whisper(output_audio) vs FLEURS English ref)")
    print(f"  BLEU  : {bleu.score:.2f}")
    print(f"  chrF  : {chrf.score:.2f}")
    print()
    print("Speech quality  (UTMOS22 on output audio, 1-5 scale)")
    print(f"  UTMOS : {avg_mos:.3f}")
    print()
    print(f"  {'':8}  {'mean':>8}  {'p50':>8}  {'p95':>8}")
    print("-" * 42)
    print(f"  {'Total':8}  {lat['mean']:>7.3f}s  {lat['p50']:>7.3f}s  {lat['p95']:>7.3f}s")
    print(f"  {'RTF':8}  {rtf['mean']:>7.3f}x  {rtf['p50']:>7.3f}x  {rtf['p95']:>7.3f}x")
    print("=" * 60)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "n":        actual_n,
            "device":   args.device,
            "bleu":     bleu.score,
            "chrf":     chrf.score,
            "avg_utmos": avg_mos,
            "latency":  {"mean": lat["mean"], "p50": lat["p50"], "p95": lat["p95"]},
            "rtf":      {"mean": rtf["mean"], "p50": rtf["p50"], "p95": rtf["p95"]},
            "samples":  records,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nPer-sample results -> {args.out}")


if __name__ == "__main__":
    main()
