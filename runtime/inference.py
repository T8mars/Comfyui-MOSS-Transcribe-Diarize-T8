from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .model_cache import MODEL_CACHE
from .types import ModelHandle, PromptConfig, TranscriptPayload
from ..vendor.moss_transcribe_diarize.audio_adapter import TARGET_SAMPLE_RATE, comfy_audio_to_numpy
from ..vendor.moss_transcribe_diarize.audio_preflight import analyze_audio_samples
from ..vendor.moss_transcribe_diarize.generation_budget import estimate_max_new_tokens
from ..vendor.moss_transcribe_diarize.inference_utils import (
    DEFAULT_PROMPT,
    build_transcription_messages,
    generate_transcription,
)
from ..vendor.moss_transcribe_diarize.transcript_validation import validate_transcript
from ..vendor.moss_transcribe_diarize.prompt_presets import compose_prompt


def _comfy_runtime_callbacks(total_tokens: int):
    """Return ComfyUI progress/cancellation callbacks without import-time coupling."""
    try:
        import comfy.model_management
        import comfy.utils
    except ImportError:
        return None, None

    progress = comfy.utils.ProgressBar(total_tokens)

    def token_callback(generated_tokens: int) -> None:
        progress.update_absolute(min(int(generated_tokens), total_tokens), total_tokens)

    def cancellation_callback() -> bool:
        comfy.model_management.throw_exception_if_processing_interrupted()
        return False

    return token_callback, cancellation_callback


def build_prompt(
    base_prompt: str,
    hotwords: str,
    language_hint: str,
    strict_format: bool,
    preset_id: str = "default",
) -> PromptConfig:
    text, words, resolved_language = compose_prompt(
        base_prompt=base_prompt,
        preset_id=preset_id,
        language_hint=language_hint,
        hotwords=hotwords,
        strict_format=strict_format,
    )
    return PromptConfig(text=text, hotwords=words, language_hint=resolved_language)


def run_transcription(
    handle: ModelHandle,
    audio: dict,
    prompt: PromptConfig | None,
    *,
    max_new_tokens: int,
    silence_policy: str = "warn",
) -> TranscriptPayload:
    samples = comfy_audio_to_numpy(audio, TARGET_SAMPLE_RATE)
    duration = samples.size / float(TARGET_SAMPLE_RATE)
    if silence_policy not in {"warn", "reject", "ignore"}:
        raise ValueError("silence_policy must be warn, reject, or ignore.")
    preflight = analyze_audio_samples(samples, TARGET_SAMPLE_RATE)
    preflight_diagnostics: list[dict] = []
    if preflight.should_warn and silence_policy != "ignore":
        code = "preflight_silent" if preflight.classification == "silent" else "preflight_mostly_silence"
        message = (
            "静音预检未发现有意义的语音能量。"
            if preflight.classification == "silent"
            else f"静音预检仅在 {preflight.active_ratio:.1%} 的帧检测到语音能量。"
        )
        preflight_diagnostics.append({"level": "warning", "code": code, "message": message})
        if silence_policy == "reject":
            raise ValueError(f"静音预检已拒绝推理：{message}")
    token_budget = int(max_new_tokens) if int(max_new_tokens) > 0 else estimate_max_new_tokens(duration)
    token_callback, cancellation_callback = _comfy_runtime_callbacks(token_budget)
    if cancellation_callback is not None:
        cancellation_callback()
    entry = MODEL_CACHE.acquire(handle)
    try:
        with entry.lock:
            if cancellation_callback is not None:
                cancellation_callback()
            result = generate_transcription(
                entry.model,
                entry.processor,
                build_transcription_messages(samples, (prompt.text if prompt else DEFAULT_PROMPT)),
                max_length=131072,
                max_new_tokens=token_budget,
                do_sample=False,
                device=entry.device,
                dtype=entry.dtype,
                token_callback=token_callback,
                cancellation_callback=cancellation_callback,
                attention_report=entry.attention_report,
            )
    finally:
        MODEL_CACHE.done(handle, entry)

    validation = validate_transcript(
        result["text"],
        media_duration=duration,
        generated_tokens=int(result["generated_tokens"]),
        max_new_tokens=token_budget,
        audio_rms=float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))) if samples.size else 0.0,
    )
    return TranscriptPayload(
        raw_text=result["text"],
        segments=tuple(
            {
                "id": f"seg-{index:05d}",
                "start": segment.start,
                "end": segment.end,
                "speaker": segment.speaker,
                "text": segment.text,
            }
            for index, segment in enumerate(validation.segments, start=1)
        ),
        diagnostics=tuple([*preflight_diagnostics, *(asdict(item) for item in validation.diagnostics)]),
        metadata={
            "audio_duration_seconds": duration,
            "sample_rate": TARGET_SAMPLE_RATE,
            "prompt_tokens": int(result["prompt_len"]),
            "generated_tokens": int(result["generated_tokens"]),
            "max_new_tokens": token_budget,
            "possibly_truncated": validation.possibly_truncated,
            "model_revision": handle.model_revision,
            "device": str(entry.device),
            "dtype": str(entry.dtype),
            "memory_policy": handle.effective_memory_policy,
            "attention": entry.attention_report,
            "audio_preflight": preflight.to_dict(),
        },
    )


__all__ = ["build_prompt", "estimate_max_new_tokens", "run_transcription"]
