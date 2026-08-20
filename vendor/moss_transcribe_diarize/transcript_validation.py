from __future__ import annotations

from dataclasses import asdict, dataclass

from .transcript_parser import TranscriptSegment, parse_transcript


@dataclass(frozen=True, slots=True)
class TranscriptDiagnostic:
    level: str
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class TranscriptValidation:
    valid: bool
    segments: tuple[TranscriptSegment, ...]
    diagnostics: tuple[TranscriptDiagnostic, ...]
    possibly_truncated: bool

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "segments": [asdict(segment) for segment in self.segments],
            "diagnostics": [asdict(item) for item in self.diagnostics],
            "possibly_truncated": self.possibly_truncated,
        }


def validate_transcript(
    text: str,
    *,
    media_duration: float | None = None,
    generated_tokens: int | None = None,
    max_new_tokens: int | None = None,
) -> TranscriptValidation:
    diagnostics: list[TranscriptDiagnostic] = []
    segments = tuple(parse_transcript(text))
    if not text.strip():
        diagnostics.append(TranscriptDiagnostic("error", "empty_output", "The model returned no text."))
    elif not segments:
        diagnostics.append(
            TranscriptDiagnostic(
                "error",
                "invalid_format",
                "No complete [start][Sxx]text[end] segment could be parsed.",
            )
        )

    previous_start = -1.0
    for index, segment in enumerate(segments):
        if segment.start < previous_start:
            diagnostics.append(
                TranscriptDiagnostic("error", "timestamp_order", f"Segment {index + 1} starts before the previous segment.")
            )
        previous_start = segment.start
        if media_duration is not None and segment.end > media_duration + 0.25:
            diagnostics.append(
                TranscriptDiagnostic(
                    "warning",
                    "timestamp_out_of_range",
                    f"Segment {index + 1} ends at {segment.end:.2f}s beyond media duration {media_duration:.2f}s.",
                )
            )

    possibly_truncated = bool(
        generated_tokens is not None
        and max_new_tokens is not None
        and max_new_tokens > 0
        and generated_tokens >= max_new_tokens
    )
    if possibly_truncated:
        diagnostics.append(
            TranscriptDiagnostic(
                "warning",
                "token_limit_reached",
                "Generation reached max_new_tokens; the transcript may be incomplete.",
            )
        )
    valid = bool(segments) and not any(item.level == "error" for item in diagnostics)
    return TranscriptValidation(valid, segments, tuple(diagnostics), possibly_truncated)


__all__ = ["TranscriptDiagnostic", "TranscriptValidation", "validate_transcript"]
