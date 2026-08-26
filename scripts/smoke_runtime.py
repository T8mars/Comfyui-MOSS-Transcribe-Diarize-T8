from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch
import torchaudio


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "comfyui_moss_transcribe_diarize_t8_smoke"


def load_package() -> None:
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        PLUGIN_ROOT / "__init__.py",
        submodule_search_locations=[str(PLUGIN_ROOT)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load plugin package.")
    package = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = package
    spec.loader.exec_module(package)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ComfyUI adapter without starting a ComfyUI server.")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=0, help="0 uses the duration-based automatic budget.")
    parser.add_argument("--smart-long", action="store_true", help="Exercise smart long-audio chunking and merge.")
    parser.add_argument("--target-chunk-seconds", type=float, default=480.0)
    parser.add_argument("--max-chunk-seconds", type=float, default=600.0)
    parser.add_argument("--overlap-seconds", type=float, default=1.0)
    parser.add_argument("--retry-policy", choices=("never", "invalid_format", "quality_failure"), default="never")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON result path.")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    load_package()
    inference = importlib.import_module(f"{PACKAGE_NAME}.runtime.inference")
    long_audio = importlib.import_module(f"{PACKAGE_NAME}.runtime.long_audio")
    model_cache = importlib.import_module(f"{PACKAGE_NAME}.runtime.model_cache")
    types = importlib.import_module(f"{PACKAGE_NAME}.runtime.types")
    audio_adapter = importlib.import_module(f"{PACKAGE_NAME}.vendor.moss_transcribe_diarize.audio_adapter")
    waveform, sample_rate = torchaudio.load(str(args.audio))
    comfy_audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
    handle = types.ModelHandle(args.model.resolve(), args.device, "auto")
    cuda_index = (
        int(args.device.split(":", 1)[1]) if ":" in args.device else torch.cuda.current_device()
    ) if args.device.startswith("cuda") and torch.cuda.is_available() else None
    if args.device.startswith("cuda") and torch.cuda.is_available():
        with torch.cuda.device(cuda_index):
            torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    chunk_report = None
    if args.smart_long:
        samples = audio_adapter.comfy_audio_to_numpy(comfy_audio, audio_adapter.TARGET_SAMPLE_RATE)
        payload, chunk_report = long_audio.transcribe_long_audio(
            handle,
            samples,
            None,
            max_new_tokens_per_chunk=args.max_new_tokens,
            target_seconds=args.target_chunk_seconds,
            max_seconds=args.max_chunk_seconds,
            overlap_seconds=args.overlap_seconds,
            split_strategy="vad",
            retry_policy=args.retry_policy,
            checkpoint_mode="read_write" if args.checkpoint_dir is not None else "off",
            checkpoint_dir=args.checkpoint_dir,
        )
    else:
        payload = inference.run_transcription(
            handle,
            comfy_audio,
            None,
            max_new_tokens=args.max_new_tokens,
            retry_policy=args.retry_policy,
        )
    elapsed = time.perf_counter() - started
    model_cache.MODEL_CACHE.release_all()
    result = payload.to_dict()
    peak_vram_gb = None
    if cuda_index is not None:
        with torch.cuda.device(cuda_index):
            peak_vram_gb = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    result["smoke_test"] = {
        "audio": str(args.audio.resolve()),
        "elapsed_seconds": round(elapsed, 3),
        "segment_count": len(payload.segments),
        "peak_vram_gb": peak_vram_gb,
        "mode": "smart_long_audio" if args.smart_long else "single_pass",
    }
    if chunk_report is not None:
        result["smoke_test"]["chunk_count"] = chunk_report["chunk_count"]
        result["smoke_test"]["resumed_chunks"] = chunk_report["resumed_chunks"]
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    chunk_metadata = [item.get("metadata", {}) for item in result["metadata"].get("chunks", [])]
    generated_tokens = result["metadata"].get("generated_tokens")
    max_new_tokens = result["metadata"].get("max_new_tokens")
    possibly_truncated = result["metadata"].get("possibly_truncated")
    if chunk_metadata:
        generated_tokens = sum(int(item.get("generated_tokens") or 0) for item in chunk_metadata)
        max_new_tokens = sum(int(item.get("max_new_tokens") or 0) for item in chunk_metadata)
        possibly_truncated = any(bool(item.get("possibly_truncated")) for item in chunk_metadata)
    shown = result["smoke_test"] | {
        "audio_duration_seconds": result["metadata"]["audio_duration_seconds"],
        "generated_tokens": generated_tokens,
        "max_new_tokens": max_new_tokens,
        "possibly_truncated": possibly_truncated,
        "diagnostics": result["diagnostics"],
    }
    print(json.dumps(shown if args.summary_only else result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
