from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "comfyui_moss_transcribe_diarize_t8_benchmark"


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


def configure_comfyui_path(comfyui_root: Path | None) -> None:
    if importlib.util.find_spec("comfy_api") is not None:
        return
    candidates = []
    if comfyui_root is not None:
        candidates.append(comfyui_root.resolve())
    if len(PLUGIN_ROOT.parents) >= 2:
        candidates.append(PLUGIN_ROOT.parents[1])
    for candidate in candidates:
        if (candidate / "comfy_api").is_dir():
            sys.path.insert(0, str(candidate))
            importlib.invalidate_caches()
            return
    raise RuntimeError(
        "Unable to import comfy_api. Run with ComfyUI's Python from an installed custom_nodes directory, "
        "or pass --comfyui-root D:\\path\\to\\ComfyUI."
    )


def evaluate_expectations(result: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    failures = []
    segment_count = int(result["metrics"]["segment_count"])
    quality = result.get("quality") or {}
    diagnostic_codes = set(result.get("diagnostic_codes") or [])
    if "min_segments" in expected and segment_count < int(expected["min_segments"]):
        failures.append(f"segment_count {segment_count} < {int(expected['min_segments'])}")
    if "min_speakers" in expected and int(result["metrics"].get("speaker_count") or 0) < int(
        expected["min_speakers"]
    ):
        failures.append(
            f"speaker_count {int(result['metrics'].get('speaker_count') or 0)} < {int(expected['min_speakers'])}"
        )
    if "usable" in expected and bool(quality.get("usable")) is not bool(expected["usable"]):
        failures.append(f"usable {bool(quality.get('usable'))} != {bool(expected['usable'])}")
    if "min_end_coverage" in expected:
        coverage = float(quality.get("end_coverage") or 0.0)
        if coverage < float(expected["min_end_coverage"]):
            failures.append(f"end_coverage {coverage:.4f} < {float(expected['min_end_coverage']):.4f}")
    forbidden = set(expected.get("forbidden_diagnostic_codes") or [])
    present = sorted(diagnostic_codes & forbidden)
    if present:
        failures.append(f"forbidden diagnostics present: {', '.join(present)}")
    if "max_real_time_factor" in expected:
        rtf = float(result["metrics"]["real_time_factor"])
        if rtf > float(expected["max_real_time_factor"]):
            failures.append(f"real_time_factor {rtf:.4f} > {float(expected['max_real_time_factor']):.4f}")
    if "max_peak_vram_gb" in expected and result["metrics"].get("peak_vram_gb") is not None:
        peak = float(result["metrics"]["peak_vram_gb"])
        if peak > float(expected["max_peak_vram_gb"]):
            failures.append(f"peak_vram_gb {peak:.3f} > {float(expected['max_peak_vram_gb']):.3f}")
    for expectation, metric, comparator in (
        ("min_word_alignment_coverage", "word_alignment_coverage", "minimum"),
        ("min_speaker_embedding_links", "speaker_embedding_links", "minimum"),
        ("max_speaker_embedding_failures", "speaker_embedding_failures", "maximum"),
    ):
        if expectation not in expected:
            continue
        actual = result["metrics"].get(metric)
        if actual is None:
            failures.append(f"{metric} unavailable (enable the matching auxiliary model)")
        elif comparator == "minimum" and float(actual) < float(expected[expectation]):
            failures.append(f"{metric} {float(actual):.4f} < {float(expected[expectation]):.4f}")
        elif comparator == "maximum" and float(actual) > float(expected[expectation]):
            failures.append(f"{metric} {float(actual):.4f} > {float(expected[expectation]):.4f}")
    for expectation, metric in (
        ("max_word_error_rate", "word_error_rate"),
        ("max_character_error_rate", "character_error_rate"),
    ):
        if expectation in expected:
            actual = result["metrics"].get(metric)
            if actual is None:
                failures.append(f"{metric} unavailable (reference transcript required)")
            elif float(actual) > float(expected[expectation]):
                failures.append(f"{metric} {float(actual):.4f} > {float(expected[expectation]):.4f}")
    missing_text = result.get("content_checks", {}).get("missing_required_text") or []
    if missing_text:
        failures.append(f"required text missing: {', '.join(str(item) for item in missing_text)}")
    return failures


def transcript_error_rates(hypothesis: str, reference: str) -> dict[str, float]:
    hypothesis_words = re.findall(r"\w+", hypothesis.casefold(), flags=re.UNICODE)
    reference_words = re.findall(r"\w+", reference.casefold(), flags=re.UNICODE)
    hypothesis_chars = list(re.sub(r"\s+", "", hypothesis.casefold()))
    reference_chars = list(re.sub(r"\s+", "", reference.casefold()))
    return {
        "word_error_rate": round(_edit_distance(hypothesis_words, reference_words) / max(1, len(reference_words)), 5),
        "character_error_rate": round(
            _edit_distance(hypothesis_chars, reference_chars) / max(1, len(reference_chars)), 5
        ),
    }


def _edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_item in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def _reference_text(case: dict[str, Any], manifest_path: Path) -> str | None:
    inline = case.get("reference_text")
    if inline is not None:
        return str(inline)
    relative = str(case.get("reference_text_file") or "").strip()
    if not relative:
        return None
    path = (manifest_path.parent / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Reference transcript not found for {case.get('id')}: {path}")
    return path.read_text(encoding="utf-8")


def main() -> None:
    import torchaudio

    parser = argparse.ArgumentParser(description="Run real MOSS audio regression benchmarks from a JSON manifest.")
    parser.add_argument("manifest", type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", type=Path, help="Local MOSS model directory.")
    source.add_argument("--endpoint", help="SGLang Omni/vLLM root URL or transcription endpoint.")
    parser.add_argument("--remote-model", default="OpenMOSS-Team/MOSS-Transcribe-Diarize")
    parser.add_argument("--allow-remote-upload", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--comfyui-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fail-on-regression", action="store_true")
    parser.add_argument("--word-alignment-model", help="Optional Whisper model ID/path for cases with word_alignment=true.")
    parser.add_argument("--word-alignment-revision", default="973afd24965f72e36ca33b3055d56a652f456b4d")
    parser.add_argument("--speaker-embedding-model", help="Optional X-Vector model ID/path for cases with speaker_embedding_link=true.")
    parser.add_argument("--speaker-embedding-revision", default="feb593a6c23c1cc3d9510425c29b0a14d2b07b1e")
    args = parser.parse_args()

    configure_comfyui_path(args.comfyui_root)
    load_package()
    inference = importlib.import_module(f"{PACKAGE_NAME}.runtime.inference")
    long_audio = importlib.import_module(f"{PACKAGE_NAME}.runtime.long_audio")
    model_cache = importlib.import_module(f"{PACKAGE_NAME}.runtime.model_cache")
    remote = importlib.import_module(f"{PACKAGE_NAME}.runtime.remote")
    types = importlib.import_module(f"{PACKAGE_NAME}.runtime.types")
    audio_adapter = importlib.import_module(f"{PACKAGE_NAME}.vendor.moss_transcribe_diarize.audio_adapter")
    alignment = importlib.import_module(f"{PACKAGE_NAME}.runtime.alignment")
    speaker_embeddings = importlib.import_module(f"{PACKAGE_NAME}.runtime.speaker_embeddings")

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "t8.moss-benchmark.v1" or not isinstance(manifest.get("cases"), list):
        raise ValueError("Benchmark manifest must use schema t8.moss-benchmark.v1 and contain a cases list.")
    if args.endpoint:
        if not args.allow_remote_upload:
            raise ValueError("Remote benchmark requires --allow-remote-upload.")
        handle = types.ModelHandle(
            Path("."),
            "remote",
            "server_managed",
            backend="openai_compatible",
            endpoint_url=remote.normalize_endpoint_url(args.endpoint),
            remote_model=args.remote_model,
        )
    else:
        handle = types.ModelHandle(args.model.resolve(), args.device, args.precision)
    alignment_handle = (
        alignment.WordAlignmentHandle(
            model_id=args.word_alignment_model,
            revision=args.word_alignment_revision,
            device=args.device,
            precision=args.precision,
        )
        if args.word_alignment_model
        else None
    )
    speaker_handle = (
        speaker_embeddings.SpeakerEmbeddingHandle(
            model_id=args.speaker_embedding_model,
            revision=args.speaker_embedding_revision,
            device=args.device,
            precision=args.precision,
        )
        if args.speaker_embedding_model
        else None
    )

    cuda_index = None
    if not handle.is_remote and args.device.startswith("cuda") and torch.cuda.is_available():
        cuda_index = int(args.device.split(":", 1)[1]) if ":" in args.device else torch.cuda.current_device()
    results = []
    try:
        for case in manifest["cases"]:
            case_id = str(case.get("id") or "").strip()
            if not case_id:
                raise ValueError("Every benchmark case requires a non-empty id.")
            audio_path = (manifest_path.parent / str(case["audio"])).resolve()
            if not audio_path.is_file():
                raise FileNotFoundError(f"Benchmark audio not found for {case_id}: {audio_path}")
            waveform, sample_rate = torchaudio.load(str(audio_path))
            comfy_audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
            prompt_config = case.get("prompt") or {}
            prompt = inference.build_prompt(
                str(prompt_config.get("base_prompt") or ""),
                str(prompt_config.get("hotwords") or ""),
                str(prompt_config.get("language_hint") or "auto"),
                bool(prompt_config.get("strict_format", True)),
                str(prompt_config.get("preset_id") or "default"),
                str(prompt_config.get("custom_language_hint") or ""),
            )
            if cuda_index is not None:
                with torch.cuda.device(cuda_index):
                    torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            mode = str(case.get("mode") or "single")
            options = case.get("options") or {}
            if mode == "smart_long":
                samples = audio_adapter.comfy_audio_to_numpy(comfy_audio, audio_adapter.TARGET_SAMPLE_RATE)
                payload, chunk_report = long_audio.transcribe_long_audio(
                    handle,
                    samples,
                    prompt,
                    max_new_tokens_per_chunk=int(options.get("max_new_tokens", 0)),
                    target_seconds=float(options.get("target_chunk_seconds", 480.0)),
                    max_seconds=float(options.get("max_chunk_seconds", 600.0)),
                    overlap_seconds=float(options.get("overlap_seconds", 1.0)),
                    split_strategy=str(options.get("split_strategy") or "vad"),
                    retry_policy=str(options.get("retry_policy") or "quality_failure"),
                    checkpoint_mode="off",
                    speaker_link_mode=str(options.get("speaker_link_mode") or "off"),
                )
            elif mode == "single":
                payload = inference.run_transcription(
                    handle,
                    comfy_audio,
                    prompt,
                    max_new_tokens=int(options.get("max_new_tokens", 0)),
                    retry_policy=str(options.get("retry_policy") or "quality_failure"),
                )
                chunk_report = None
            else:
                raise ValueError(f"Unsupported benchmark mode for {case_id}: {mode}")
            auxiliary_metrics = {}
            if bool(options.get("speaker_embedding_link")):
                if speaker_handle is None:
                    raise ValueError(
                        f"Benchmark case {case_id} requires --speaker-embedding-model."
                    )
                samples = audio_adapter.comfy_audio_to_numpy(comfy_audio, audio_adapter.TARGET_SAMPLE_RATE)
                payload, speaker_report = speaker_embeddings.link_speakers_by_voice(
                    speaker_handle,
                    samples,
                    payload,
                    similarity_threshold=float(options.get("speaker_similarity_threshold", 0.86)),
                )
                auxiliary_metrics["speaker_embedding_links"] = len(speaker_report["links"])
                auxiliary_metrics["speaker_embedding_failures"] = len(speaker_report["failures"])
            if bool(options.get("word_alignment")):
                if alignment_handle is None:
                    raise ValueError(f"Benchmark case {case_id} requires --word-alignment-model.")
                samples = audio_adapter.comfy_audio_to_numpy(comfy_audio, audio_adapter.TARGET_SAMPLE_RATE)
                payload, alignment_report = alignment.run_word_alignment(
                    alignment_handle,
                    samples,
                    payload,
                )
                auxiliary_metrics["word_alignment_coverage"] = alignment_report["model_match_coverage"]
                auxiliary_metrics["word_alignment_fallback_units"] = alignment_report["fallback_units"]
            elapsed = time.perf_counter() - started
            duration = float(payload.metadata.get("audio_duration_seconds") or 0.0)
            expected = case.get("expected") or {}
            reference = _reference_text(case, manifest_path)
            transcript_text = " ".join(str(item.get("text") or "") for item in payload.segments)
            accuracy = transcript_error_rates(transcript_text, reference) if reference is not None else {}
            required_text = [str(item) for item in expected.get("required_text") or []]
            missing_required_text = [item for item in required_text if item not in transcript_text]
            speakers = {
                str(item.get("speaker") or "S00")
                for item in payload.segments
                if str(item.get("speaker") or "S00") != "S00"
            }
            peak_vram_gb = None
            if cuda_index is not None:
                with torch.cuda.device(cuda_index):
                    peak_vram_gb = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
            result = {
                "id": case_id,
                "audio": str(audio_path),
                "mode": mode,
                "metrics": {
                    "audio_duration_seconds": duration,
                    "elapsed_seconds": round(elapsed, 3),
                    "real_time_factor": round(elapsed / duration, 5) if duration > 0 else 0.0,
                    "peak_vram_gb": peak_vram_gb,
                    "segment_count": len(payload.segments),
                    "speaker_count": len(speakers),
                    **auxiliary_metrics,
                    **accuracy,
                },
                "quality": payload.metadata.get("quality"),
                "diagnostic_codes": sorted({str(item.get("code") or "") for item in payload.diagnostics}),
                "chunk_report": chunk_report,
                "content_checks": {"missing_required_text": missing_required_text},
            }
            result["failures"] = evaluate_expectations(result, expected)
            result["passed"] = not result["failures"]
            results.append(result)
    finally:
        model_cache.MODEL_CACHE.release_all()
        alignment.release_alignment_model()
        speaker_embeddings.release_speaker_model()

    report = {
        "schema": "t8.moss-benchmark-report.v1",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "backend": handle.backend,
        },
        "passed": all(item["passed"] for item in results),
        "case_count": len(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_regression and not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
