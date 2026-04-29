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
    """Download Kokoro-82M ONNX model files from GitHub releases."""
    import urllib.request

    kokoro_dir = os.path.join("models", "kokoro")
    onnx_path = os.path.join(kokoro_dir, "kokoro-v1.0.onnx")
    voices_path = os.path.join(kokoro_dir, "voices-v1.0.bin")

    if os.path.exists(onnx_path) and os.path.exists(voices_path):
        print(f"[TTS] Already exists: {kokoro_dir}")
        return

    os.makedirs(kokoro_dir, exist_ok=True)
    base_url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
    for fname, dest in [("kokoro-v1.0.onnx", onnx_path), ("voices-v1.0.bin", voices_path)]:
        if os.path.exists(dest):
            print(f"[TTS] Already exists: {dest}")
            continue
        url = f"{base_url}/{fname}"
        print(f"[TTS] Downloading {fname} ...")
        urllib.request.urlretrieve(url, dest)
    print(f"[TTS] Saved to {kokoro_dir}")


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
