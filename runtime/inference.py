from __future__ import annotations

from dataclasses import asdict
from typing import Callable

import numpy as np

from .model_cache import MODEL_CACHE
from .quality import evaluate_quality
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
from .remote import request_remote_transcription


STRICT_RETRY_SUFFIX = (
    "\nCritical output rules: return only complete [start_seconds][Sxx]text[end_seconds] segments in chronological "
    "order. Every spoken segment must include a speaker tag and two timestamps. Do not output commentary. "
    "Do not invent hotwords that are not spoken. If there is no speech, return an empty response."
)


def _comfy_runtime_callbacks(total_tokens: int):
    """Return ComfyUI progress/cancellation callbacks without import-time coupling."""
    try:
        import comfy.model_management
    except (ImportError, RuntimeError, AssertionError):
        return None, None

    native_progress = None
    try:
        from comfy_api.latest import ComfyAPISync

        api = ComfyAPISync()

        def report_native_progress(value: int, current_total: int) -> None:
            api.execution.set_progress(value, current_total)

        native_progress = report_native_progress
    except (ImportError, AttributeError, RuntimeError):
        native_progress = None

    legacy_progress = None
    try:
        import comfy.utils

        progress = comfy.utils.ProgressBar(total_tokens)

        def report_legacy_progress(value: int, current_total: int) -> None:
            progress.update_absolute(value, current_total)

        legacy_progress = report_legacy_progress
    except (ImportError, AttributeError):
        legacy_progress = None

    def token_callback(generated_tokens: int, current_total: int | None = None) -> None:
        resolved_total = max(1, int(current_total if current_total is not None else total_tokens))
        value = min(int(generated_tokens), resolved_total)
        if native_progress is not None:
            try:
                native_progress(value, resolved_total)
                return
            except (RuntimeError, ValueError, AttributeError):
                pass
        if legacy_progress is not None:
            legacy_progress(value, resolved_total)

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
    custom_language_hint: str = "",
) -> PromptConfig:
    text, words, resolved_language = compose_prompt(
        base_prompt=base_prompt,
        preset_id=preset_id,
        language_hint=language_hint,
        hotwords=hotwords,
        strict_format=strict_format,
        custom_language_hint=custom_language_hint,
    )
    return PromptConfig(text=text, hotwords=words, language_hint=resolved_language)


def run_transcription(
    handle: ModelHandle,
    audio: dict,
    prompt: PromptConfig | None,
    *,
    max_new_tokens: int,
    silence_policy: str = "warn",
    preflight_backend: str = "webrtc",
    vad_aggressiveness: int = 2,
    retry_policy: str = "invalid_format",
) -> TranscriptPayload:
    samples = comfy_audio_to_numpy(audio, TARGET_SAMPLE_RATE)
    return run_transcription_samples(
        handle,
        samples,
        prompt,
        max_new_tokens=max_new_tokens,
        silence_policy=silence_policy,
        preflight_backend=preflight_backend,
        vad_aggressiveness=vad_aggressiveness,
        retry_policy=retry_policy,
    )


def run_transcription_samples(
    handle: ModelHandle,
    samples: np.ndarray,
    prompt: PromptConfig | None,
    *,
    max_new_tokens: int,
    silence_policy: str = "warn",
    preflight_backend: str = "webrtc",
    vad_aggressiveness: int = 2,
    retry_policy: str = "invalid_format",
    progress_callback: Callable[[int, int], None] | None = None,
    cancellation_callback: Callable[[], bool] | None = None,
    cache_entry=None,
) -> TranscriptPayload:
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    duration = samples.size / float(TARGET_SAMPLE_RATE)
    if silence_policy not in {"warn", "reject", "ignore"}:
        raise ValueError("silence_policy must be warn, reject, or ignore.")
    if retry_policy not in {"never", "invalid_format", "quality_failure"}:
        raise ValueError("retry_policy must be never, invalid_format, or quality_failure.")
    preflight = analyze_audio_samples(
        samples,
        TARGET_SAMPLE_RATE,
        vad_backend=preflight_backend,
        vad_aggressiveness=vad_aggressiveness,
    )
    preflight_diagnostics: list[dict] = []
    if preflight.should_warn and silence_policy != "ignore":
        if preflight.classification == "silent":
            code = "preflight_silent"
            message = "静音预检未发现有意义的音频能量。"
        elif preflight.classification == "non_speech":
            code = "preflight_non_speech"
            message = f"VAD 仅在 {preflight.speech_ratio:.1%} 的帧检测到语音，输入可能是音乐或噪声。"
        else:
            code = "preflight_mostly_silence"
            message = f"预检仅在 {preflight.speech_ratio:.1%} 的帧检测到语音。"
        preflight_diagnostics.append({"level": "warning", "code": code, "message": message})
        if silence_policy == "reject":
            raise ValueError(f"音频预检已拒绝推理：{message}")
    token_budget = int(max_new_tokens) if int(max_new_tokens) > 0 else estimate_max_new_tokens(duration)
    progress_total = token_budget * (2 if retry_policy != "never" else 1)
    if progress_callback is None or cancellation_callback is None:
        comfy_progress, comfy_cancellation = _comfy_runtime_callbacks(progress_total)
        progress_callback = progress_callback or (
            (lambda value, total: comfy_progress(value, total)) if comfy_progress is not None else None
        )
        cancellation_callback = cancellation_callback or comfy_cancellation
    if cancellation_callback is not None:
        cancellation_callback()
    base_prompt = prompt.text if prompt else DEFAULT_PROMPT
    entry = None
    owns_cache_entry = False
    if handle.is_remote:
        def generate_attempt(prompt_text: str, progress_offset: int):
            return _generate_remote_and_validate(
                handle,
                samples,
                prompt_text,
                language_hint=prompt.language_hint if prompt else "auto",
                duration=duration,
                token_budget=token_budget,
                progress_offset=progress_offset,
                progress_total=progress_total,
                progress_callback=progress_callback,
                cancellation_callback=cancellation_callback,
            )

        runtime_device = "remote"
        runtime_dtype = "server_managed"
        attention_report = {"selected": "server_managed", "backend": handle.backend}
        result, validation, first_result, retry_result, retry_reason, retry_selected = _run_attempts(
            generate_attempt,
            base_prompt,
            token_budget,
            retry_policy,
        )
    else:
        owns_cache_entry = cache_entry is None
        entry = cache_entry if cache_entry is not None else MODEL_CACHE.acquire(handle)
        try:
            with entry.lock:
                if cancellation_callback is not None:
                    cancellation_callback()

                def generate_attempt(prompt_text: str, progress_offset: int):
                    return _generate_and_validate(
                        entry,
                        samples,
                        prompt_text,
                        duration=duration,
                        token_budget=token_budget,
                        progress_offset=progress_offset,
                        progress_total=progress_total,
                        progress_callback=progress_callback,
                        cancellation_callback=cancellation_callback,
                    )

                result, validation, first_result, retry_result, retry_reason, retry_selected = _run_attempts(
                    generate_attempt,
                    base_prompt,
                    token_budget,
                    retry_policy,
                )
            runtime_device = str(entry.device)
            runtime_dtype = str(entry.dtype)
            attention_report = entry.attention_report
        finally:
            if owns_cache_entry and entry is not None:
                MODEL_CACHE.done(handle, entry)

    retry_diagnostics: list[dict] = []
    if retry_reason is not None:
        retry_diagnostics.append(
            {
                "level": "info" if retry_selected else "warning",
                "code": "automatic_retry_selected" if retry_selected else "automatic_retry_not_selected",
                "message": f"严格格式自动重试已执行（原因：{retry_reason}），{'采用重试结果' if retry_selected else '保留首次结果'}。",
            }
        )
    payload = TranscriptPayload(
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
        diagnostics=tuple(
            [*preflight_diagnostics, *(asdict(item) for item in validation.diagnostics), *retry_diagnostics]
        ),
        metadata={
            "audio_duration_seconds": duration,
            "sample_rate": TARGET_SAMPLE_RATE,
            "prompt_tokens": int(result["prompt_len"]),
            "generated_tokens": int(result["generated_tokens"]),
            "max_new_tokens": token_budget,
            "possibly_truncated": validation.possibly_truncated,
            "model_revision": handle.model_revision,
            "backend": handle.backend,
            "device": runtime_device,
            "dtype": runtime_dtype,
            "memory_policy": handle.effective_memory_policy,
            "attention": attention_report,
            "remote": result.get("remote"),
            "audio_preflight": preflight.to_dict(),
            "retry": {
                "policy": retry_policy,
                "attempted": retry_reason is not None,
                "reason": retry_reason,
                "selected": retry_selected,
                "first_generated_tokens": int(first_result["generated_tokens"]),
                "retry_generated_tokens": int(retry_result["generated_tokens"]) if retry_result is not None else None,
            },
        },
    )
    payload.metadata["quality"] = evaluate_quality(payload)
    if progress_callback is not None:
        progress_callback(progress_total, progress_total)
    return payload


def _run_attempts(generate_attempt, base_prompt: str, token_budget: int, retry_policy: str):
    result, validation = generate_attempt(base_prompt, 0)
    first_result = result
    retry_reason = _retry_reason(validation, retry_policy)
    retry_result = None
    retry_selected = False
    if retry_reason is not None:
        retry_result, retry_validation = generate_attempt(base_prompt + STRICT_RETRY_SUFFIX, token_budget)
        if _validation_rank(retry_validation) > _validation_rank(validation):
            result, validation = retry_result, retry_validation
            retry_selected = True
    return result, validation, first_result, retry_result, retry_reason, retry_selected


def _generate_remote_and_validate(
    handle: ModelHandle,
    samples: np.ndarray,
    prompt_text: str,
    *,
    language_hint: str,
    duration: float,
    token_budget: int,
    progress_offset: int,
    progress_total: int,
    progress_callback: Callable[[int, int], None] | None,
    cancellation_callback: Callable[[], bool] | None,
):
    if cancellation_callback is not None:
        cancellation_callback()
    result = request_remote_transcription(
        handle,
        samples,
        sample_rate=TARGET_SAMPLE_RATE,
        prompt=prompt_text,
        language=language_hint,
        max_new_tokens=token_budget,
    )
    if cancellation_callback is not None:
        cancellation_callback()
    if progress_callback is not None:
        progress_callback(progress_offset + token_budget, progress_total)
    validation = validate_transcript(
        result["text"],
        media_duration=duration,
        generated_tokens=int(result["generated_tokens"]),
        max_new_tokens=token_budget,
        audio_rms=float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))) if samples.size else 0.0,
    )
    return result, validation


def _generate_and_validate(
    entry,
    samples: np.ndarray,
    prompt_text: str,
    *,
    duration: float,
    token_budget: int,
    progress_offset: int,
    progress_total: int,
    progress_callback: Callable[[int, int], None] | None,
    cancellation_callback: Callable[[], bool] | None,
):
    token_callback = None
    if progress_callback is not None:
        def report_token_progress(value: int) -> None:
            progress_callback(progress_offset + int(value), progress_total)

        token_callback = report_token_progress
    result = generate_transcription(
        entry.model,
        entry.processor,
        build_transcription_messages(samples, prompt_text),
        max_length=131072,
        max_new_tokens=token_budget,
        do_sample=False,
        device=entry.device,
        dtype=entry.dtype,
        token_callback=token_callback,
        cancellation_callback=cancellation_callback,
        attention_report=entry.attention_report,
    )
    validation = validate_transcript(
        result["text"],
        media_duration=duration,
        generated_tokens=int(result["generated_tokens"]),
        max_new_tokens=token_budget,
        audio_rms=float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))) if samples.size else 0.0,
    )
    return result, validation


def _retry_reason(validation, retry_policy: str) -> str | None:
    if retry_policy == "never":
        return None
    codes = {item.code for item in validation.diagnostics}
    if not validation.valid or "speaker_tag_missing" in codes:
        return "invalid_format"
    if retry_policy == "quality_failure" and codes & {
        "repeated_text",
        "possible_early_stop",
        "possible_silence_hallucination",
        "token_limit_reached",
        "timestamp_out_of_range",
    }:
        return "quality_failure"
    return None


def _validation_rank(validation) -> tuple[int, int, int, int, float, int]:
    errors = sum(item.level == "error" for item in validation.diagnostics)
    risk_weights = {
        "speaker_tag_missing": 1,
        "repeated_text": 2,
        "possible_early_stop": 2,
        "possible_silence_hallucination": 3,
        "token_limit_reached": 2,
        "timestamp_out_of_range": 4,
    }
    risks = sum(risk_weights.get(item.code, 0) for item in validation.diagnostics)
    unknown_speakers = sum(item.speaker == "S00" for item in validation.segments)
    last_end = max((item.end for item in validation.segments), default=0.0)
    return (int(validation.valid), -errors, -risks, -unknown_speakers, last_end, len(validation.segments))


__all__ = ["build_prompt", "estimate_max_new_tokens", "run_transcription", "run_transcription_samples"]
