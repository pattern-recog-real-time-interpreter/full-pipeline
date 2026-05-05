"""
Black-box E2E evaluation: Thai audio → pipeline → English audio.

Treats the pipeline as a black box. Evaluates output audio only — no
intermediate text (ASR output, NMT output) is used for scoring.

Metrics
-------
  BLEU + chrF   : Whisper transcribes output audio → compare to FLEURS English reference
  UTMOS         : automated MOS (speech naturalness) on output audio, scale 1–5
  E2E latency   : total wall time per sample (mean / p50 / p95)
  RTF           : total_latency / input_audio_duration

Install eval deps:
    pip install datasets sacrebleu openai-whisper speechmos

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
import whisper
from datasets import load_dataset
from speechmos import dnsmos  # UTMOS via speechmos package

from pipeline import PipelineConfig, ThaiToEnglishPipeline


def latency_stats(values: list[float]) -> dict:
    s = sorted(values)
    n = len(s)
    return {
        "mean": statistics.mean(s),
        "p50":  s[n // 2],
        "p95":  s[min(int(n * 0.95), n - 1)],
    }


def transcribe_audio(whisper_model, audio: np.ndarray, sr: int) -> str:
    """Write audio to temp WAV and transcribe with Whisper (English)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    try:
        sf.write(tmp, audio, sr)
        result = whisper_model.transcribe(tmp, language="en", fp16=False)
        return result["text"].strip()
    finally:
        os.unlink(tmp)


def mos_score(audio: np.ndarray, sr: int) -> float:
    """DNSMOS P.835 overall MOS on output audio (no reference needed)."""
    # speechmos expects 16 kHz
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    scores = dnsmos.run(audio, 16000)
    return float(scores["ovrl_mos"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",             type=int,   default=100)
    parser.add_argument("--device",        default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--whisper-model", default="small",
                        choices=["tiny", "base", "small", "medium"],
                        help="Whisper model for transcribing output audio")
    parser.add_argument("--out",           default="eval/results.json")
    args = parser.parse_args()

    # Load pipeline
    print("[eval] Loading translation pipeline ...")
    config = PipelineConfig(device=args.device)
    pipe = ThaiToEnglishPipeline(config)
    pipe.load()

    # Load Whisper for output audio transcription
    print(f"[eval] Loading Whisper {args.whisper_model} (for output audio transcription) ...")
    asr = whisper.load_model(args.whisper_model)

    # Load FLEURS
    print("[eval] Fetching FLEURS th_th / en_us test splits ...")
    ds_th = load_dataset("google/fleurs", "th_th", split="test", trust_remote_code=True)
    ds_en = load_dataset("google/fleurs", "en_us", split="test", trust_remote_code=True)
    en_by_id = {row["id"]: row["transcription"] for row in ds_en}

    n = min(args.n, len(ds_th))

    transcripts: list[str] = []   # Whisper output from TTS audio
    references:  list[str] = []   # FLEURS English ground truth
    records:     list[dict] = []

    for i, row in enumerate(list(ds_th)[:n]):
        en_ref = en_by_id.get(row["id"], "")
        if not en_ref:
            continue

        # Prepare input audio at 16 kHz
        audio_in = np.array(row["audio"]["array"], dtype=np.float32)
        orig_sr  = row["audio"]["sampling_rate"]
        if orig_sr != config.sample_rate:
            audio_in = librosa.resample(audio_in, orig_sr=orig_sr,
                                        target_sr=config.sample_rate)

        dur = len(audio_in) / config.sample_rate
        print(f"\n[{i+1}/{n}] input_dur={dur:.1f}s")

        # ---- BLACK BOX: only inspect result.audio_output ----
        result = pipe.process_segment(audio_in)

        if result.audio_output is None or result.audio_output.size == 0:
            print("  skip — pipeline produced no audio output")
            continue

        # Transcribe output audio with Whisper
        transcript = transcribe_audio(asr, result.audio_output, result.tts_sample_rate)

        # Automated MOS on output audio
        mos = mos_score(result.audio_output, result.tts_sample_rate)

        transcripts.append(transcript)
        references.append(en_ref)

        print(f"  Whisper transcript : {transcript}")
        print(f"  Reference          : {en_ref}")
        print(f"  DNSMOS MOS         : {mos:.2f}")
        print(f"  E2E RTF            : {result.end_to_end_rtf:.2f}×  "
              f"total={result.total_latency_s:.2f}s")

        records.append({
            "id":        row["id"],
            "en_ref":    en_ref,
            "en_hyp":    transcript,   # from Whisper on output audio
            "mos":       mos,
            "audio_dur": result.audio_duration_s,
            "total_s":   result.total_latency_s,
            "rtf":       result.end_to_end_rtf,
            # stage breakdown kept for analysis but NOT used in scoring
            "vad_s":     result.vad_latency_s,
            "asr_s":     result.asr_latency_s,
            "nmt_s":     result.nmt_latency_s,
            "tts_s":     result.tts_latency_s,
        })

    actual_n = len(records)
    if actual_n == 0:
        print("No samples produced audio output.")
        return

    # Translation quality
    bleu = sacrebleu.corpus_bleu(transcripts, [references])
    chrf = sacrebleu.corpus_chrf(transcripts, [references])

    # Speech quality
    avg_mos = statistics.mean(r["mos"] for r in records)

    # Latency
    lat  = latency_stats([r["total_s"] for r in records])
    rtf  = latency_stats([r["rtf"]     for r in records])

    print("\n" + "=" * 60)
    print(f"Black-box E2E evaluation  (n={actual_n}, device={args.device})")
    print(f"Whisper model: {args.whisper_model}  (transcribes output audio)")
    print()
    print("Translation quality  (Whisper(output audio) vs FLEURS English ref)")
    print(f"  BLEU  : {bleu.score:.2f}")
    print(f"  chrF  : {chrf.score:.2f}")
    print()
    print("Speech quality  (DNSMOS on output audio, 1–5 scale)")
    print(f"  MOS   : {avg_mos:.2f}")
    print()
    print("E2E latency  (input audio received → output audio ready)")
    print(f"  {'':8}  {'mean':>8}  {'p50':>8}  {'p95':>8}")
    print(f"  {'Total':8}  {lat['mean']:>7.3f}s  {lat['p50']:>7.3f}s  {lat['p95']:>7.3f}s")
    print(f"  {'RTF':8}  {rtf['mean']:>7.3f}×  {rtf['p50']:>7.3f}×  {rtf['p95']:>7.3f}×")
    print("=" * 60)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "n":       actual_n,
            "device":  args.device,
            "bleu":    bleu.score,
            "chrf":    chrf.score,
            "avg_mos": avg_mos,
            "latency": {"mean": lat["mean"], "p50": lat["p50"], "p95": lat["p95"]},
            "rtf":     {"mean": rtf["mean"], "p50": rtf["p50"], "p95": rtf["p95"]},
            "samples": records,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nPer-sample results → {args.out}")


if __name__ == "__main__":
    main()
