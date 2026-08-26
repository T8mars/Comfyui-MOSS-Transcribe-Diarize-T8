from __future__ import annotations

from collections import Counter
import re
from typing import Any, Iterable

from .types import TranscriptPayload


def evaluate_quality(
    transcript: TranscriptPayload,
    *,
    min_end_coverage: float = 0.75,
    max_unknown_speaker_ratio: float = 0.50,
    reject_repetition: bool = True,
    reject_truncation: bool = True,
) -> dict[str, Any]:
    return evaluate_quality_components(
        transcript.segments,
        transcript.diagnostics,
        media_duration=_as_float(transcript.metadata.get("audio_duration_seconds")),
        speech_ratio=_as_float((transcript.metadata.get("audio_preflight") or {}).get("speech_ratio")),
        min_end_coverage=min_end_coverage,
        max_unknown_speaker_ratio=max_unknown_speaker_ratio,
        reject_repetition=reject_repetition,
        reject_truncation=reject_truncation,
    )


def evaluate_quality_components(
    segments: Iterable[Any],
    diagnostics: Iterable[dict[str, Any]],
    *,
    media_duration: float | None,
    speech_ratio: float | None = None,
    min_end_coverage: float = 0.75,
    max_unknown_speaker_ratio: float = 0.50,
    reject_repetition: bool = True,
    reject_truncation: bool = True,
) -> dict[str, Any]:
    items = list(segments)
    diagnostic_items = [dict(item) for item in diagnostics]
    codes = {str(item.get("code") or "") for item in diagnostic_items}
    error_count = sum(str(item.get("level") or "") == "error" for item in diagnostic_items)
    warning_count = sum(str(item.get("level") or "") == "warning" for item in diagnostic_items)

    last_end = max((_segment_number(item, "end") for item in items), default=0.0)
    end_coverage = None
    if media_duration is not None and media_duration > 0:
        end_coverage = min(1.0, max(0.0, last_end / media_duration))

    unknown_count = sum(_segment_text(item, "speaker") == "S00" for item in items)
    unknown_ratio = unknown_count / max(len(items), 1)
    normalized = [re.sub(r"\s+", " ", _segment_text(item, "text")).strip().casefold() for item in items]
    repetitions = Counter(text for text in normalized if len(text) >= 4)
    repeated_max = max(repetitions.values(), default=0)
    repeated_ratio = repeated_max / max(len(items), 1)

    reasons: list[str] = []
    if not items or error_count:
        reasons.append("invalid_transcript")
    if media_duration is not None and media_duration >= 30 and end_coverage is not None and end_coverage < min_end_coverage:
        reasons.append("insufficient_end_coverage")
    if unknown_ratio > max_unknown_speaker_ratio:
        reasons.append("too_many_unknown_speakers")
    if reject_repetition and ("repeated_text" in codes or (repeated_max >= 3 and repeated_ratio >= 0.25)):
        reasons.append("repeated_text")
    if reject_truncation and ({"token_limit_reached", "possible_early_stop"} & codes):
        reasons.append("possibly_truncated")
    if speech_ratio is not None and speech_ratio < 0.02 and items:
        reasons.append("non_speech_hallucination_risk")

    return {
        "usable": not reasons,
        "reasons": reasons,
        "segment_count": len(items),
        "error_count": error_count,
        "warning_count": warning_count,
        "last_end_seconds": last_end,
        "media_duration_seconds": media_duration,
        "end_coverage": end_coverage,
        "unknown_speaker_count": unknown_count,
        "unknown_speaker_ratio": unknown_ratio,
        "repeated_text_ratio": repeated_ratio,
        "speech_ratio": speech_ratio,
        "thresholds": {
            "min_end_coverage": min_end_coverage,
            "max_unknown_speaker_ratio": max_unknown_speaker_ratio,
            "reject_repetition": reject_repetition,
            "reject_truncation": reject_truncation,
        },
    }


def _segment_number(item: Any, key: str) -> float:
    value = item.get(key, 0.0) if isinstance(item, dict) else getattr(item, key, 0.0)
    return float(value or 0.0)


def _segment_text(item: Any, key: str) -> str:
    value = item.get(key, "") if isinstance(item, dict) else getattr(item, key, "")
    return str(value or "")


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = ["evaluate_quality", "evaluate_quality_components"]
