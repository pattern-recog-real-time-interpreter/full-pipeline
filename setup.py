"""
One-time model download and conversion.

Usage:
    python setup.py                  # download/convert everything
    python setup.py --asr-only
    python setup.py --nmt-only
    python setup.py --tts-only
"""
import argparse
import os
import subprocess
import sys


def setup_asr():
    """Download Typhoon ASR and save as .nemo file for fast loading."""
    out_path = os.path.join("models", "typhoon-asr.nemo")
    if os.path.exists(out_path):
        print(f"[ASR] Already exists: {out_path}")
        return

    print("[ASR] Downloading typhoon-ai/typhoon-asr-realtime via NeMo ...")
    import nemo.collections.asr as nemo_asr

    os.makedirs("models", exist_ok=True)
    model = nemo_asr.models.ASRModel.from_pretrained("typhoon-ai/typhoon-asr-realtime")
    model.save_to(out_path)
    print(f"[ASR] Saved to {out_path}")


def setup_nmt():
    """Convert NLLB-600M to CTranslate2 INT8."""
    out_dir = os.path.join("models", "nllb-600m-int8")
    if os.path.exists(out_dir):
        print(f"[NMT] Already exists: {out_dir}")
        return

    print("[NMT] Converting facebook/nllb-200-distilled-600M to CTranslate2 INT8 ...")
    os.makedirs("models", exist_ok=True)
    cmd = [
        sys.executable, "-m", "ctranslate2.converters.transformers",
        "--model", "facebook/nllb-200-distilled-600M",
        "--quantization", "int8",
        "--output_dir", out_dir,
        "--force",
    ]
    # Prefer the CLI entry point if available
    cli_cmd = [
        "ct2-transformers-converter",
        "--model", "facebook/nllb-200-distilled-600M",
        "--quantization", "int8",
        "--output_dir", out_dir,
        "--force",
    ]
    try:
        subprocess.run(cli_cmd, check=True)
    except FileNotFoundError:
        subprocess.run(cmd, check=True)
    print(f"[NMT] Saved to {out_dir}")


def setup_tts():
    """Download Piper ONNX model files."""
    piper_dir = os.path.join("models", "piper")
    onnx_path = os.path.join(piper_dir, "en_US-lessac-medium.onnx")
    json_path = os.path.join(piper_dir, "en_US-lessac-medium.onnx.json")

    if os.path.exists(onnx_path) and os.path.exists(json_path):
        print(f"[TTS] Already exists: {piper_dir}")
        return

    print("[TTS] Downloading Piper en_US-lessac-medium from HuggingFace ...")
    from huggingface_hub import hf_hub_download

    os.makedirs(piper_dir, exist_ok=True)
    base = "en/en_US/lessac/medium"
    for fname in ["en_US-lessac-medium.onnx", "en_US-lessac-medium.onnx.json"]:
        hf_hub_download(
            repo_id="rhasspy/piper-voices",
            filename=f"{base}/{fname}",
            local_dir=piper_dir,
            local_dir_use_symlinks=False,
        )
        # hf_hub_download nests by filename path — move to flat piper_dir
        nested = os.path.join(piper_dir, base, fname)
        dest = os.path.join(piper_dir, fname)
        if os.path.exists(nested) and not os.path.exists(dest):
            os.rename(nested, dest)
    print(f"[TTS] Saved to {piper_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download and convert pipeline models")
    parser.add_argument("--asr-only", action="store_true")
    parser.add_argument("--nmt-only", action="store_true")
    parser.add_argument("--tts-only", action="store_true")
    args = parser.parse_args()

    run_all = not any([args.asr_only, args.nmt_only, args.tts_only])

    if run_all or args.asr_only:
        setup_asr()
    if run_all or args.nmt_only:
        setup_nmt()
    if run_all or args.tts_only:
        setup_tts()

    print("\n[Setup] All done. You can now run:")
    print("  python demo/file_demo.py <path-to-thai-wav>")
    print("  python demo/mic_demo.py")


if __name__ == "__main__":
    main()
