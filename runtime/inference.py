from __future__ import annotations

from dataclasses import asdict

from .model_cache import MODEL_CACHE
from .types import ModelHandle, PromptConfig, TranscriptPayload
from ..vendor.moss_transcribe_diarize.audio_adapter import TARGET_SAMPLE_RATE, comfy_audio_to_numpy
from ..vendor.moss_transcribe_diarize.generation_budget import estimate_max_new_tokens
from ..vendor.moss_transcribe_diarize.inference_utils import (
    DEFAULT_PROMPT,
    build_transcription_messages,
    generate_transcription,
)
from ..vendor.moss_transcribe_diarize.transcript_validation import validate_transcript


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


def build_prompt(base_prompt: str, hotwords: str, language_hint: str, strict_format: bool) -> PromptConfig:
    text = base_prompt.strip() or DEFAULT_PROMPT
    words = tuple(dict.fromkeys(item.strip() for item in hotwords.replace("，", ",").split(",") if item.strip()))
    additions = []
    if language_hint != "auto":
        additions.append(f"主要语言提示：{language_hint}。")
    if words:
        additions.append("热词（仅在音频确实出现时采用）：" + "、".join(words) + "。")
    if strict_format:
        additions.append("只输出 [开始秒数][Sxx]正文[结束秒数] 段落，不要输出解释、标题或 Markdown。")
    if additions:
        text = text.rstrip() + "\n" + "\n".join(additions)
    return PromptConfig(text=text, hotwords=words, language_hint=language_hint)


def run_transcription(
    handle: ModelHandle,
    audio: dict,
    prompt: PromptConfig | None,
    *,
    max_new_tokens: int,
) -> TranscriptPayload:
    samples = comfy_audio_to_numpy(audio, TARGET_SAMPLE_RATE)
    duration = samples.size / float(TARGET_SAMPLE_RATE)
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
            )
    finally:
        MODEL_CACHE.done(handle, entry, release=handle.release_after_run)

    validation = validate_transcript(
        result["text"],
        media_duration=duration,
        generated_tokens=int(result["generated_tokens"]),
        max_new_tokens=token_budget,
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
        diagnostics=tuple(asdict(item) for item in validation.diagnostics),
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
        },
    )


__all__ = ["build_prompt", "estimate_max_new_tokens", "run_transcription"]
