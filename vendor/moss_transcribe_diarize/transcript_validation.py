# Modified by the T8star-Aix integration after OpenMOSS cb765f2; see CHANGELOG.md.
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
import re

from .transcript_parser import TranscriptSegment, TranscriptStreamParser


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
    audio_rms: float | None = None,
) -> TranscriptValidation:
    if media_duration is not None:
        try:
            media_duration = float(media_duration)
        except (TypeError, ValueError) as exc:
            raise ValueError("media_duration must be a finite non-negative number.") from exc
        if not math.isfinite(media_duration) or media_duration < 0:
            raise ValueError("media_duration must be a finite non-negative number.")
    if audio_rms is not None:
        try:
            audio_rms = float(audio_rms)
        except (TypeError, ValueError) as exc:
            raise ValueError("audio_rms must be a finite non-negative number.") from exc
        if not math.isfinite(audio_rms) or audio_rms < 0:
            raise ValueError("audio_rms must be a finite non-negative number.")
    diagnostics: list[TranscriptDiagnostic] = []
    parser = TranscriptStreamParser()
    parsed_segments = parser.feed(text)
    had_malformed_input = parser.had_malformed_input
    has_incomplete_segment = parser.has_incomplete_segment
    parsed_segments.extend(parser.close())
    segments = tuple(parsed_segments)
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
    else:
        if had_malformed_input:
            diagnostics.append(
                TranscriptDiagnostic(
                    "error",
                    "invalid_format",
                    "Malformed non-whitespace content was discarded between transcript segments.",
                )
            )
        if has_incomplete_segment:
            diagnostics.append(
                TranscriptDiagnostic(
                    "error",
                    "incomplete_segment",
                    "The transcript ended before the final segment was complete.",
                )
            )

    previous_start = -1.0
    unlabelled_count = sum(1 for segment in segments if segment.speaker == "S00")
    if unlabelled_count:
        diagnostics.append(
            TranscriptDiagnostic(
                "warning",
                "speaker_tag_missing",
                f"{unlabelled_count} segment(s) omitted [Sxx]; preserved as unknown speaker S00.",
            )
        )
    for index, segment in enumerate(segments):
        if segment.start < 0 or segment.end < segment.start:
            diagnostics.append(
                TranscriptDiagnostic("error", "invalid_timestamp", f"Segment {index + 1} has an invalid time range.")
            )
        elif segment.end == segment.start:
            diagnostics.append(
                TranscriptDiagnostic("error", "zero_duration", f"Segment {index + 1} has zero duration.")
            )
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

    if media_duration is not None:
        duration = max(0.0, float(media_duration))
        clamped_segments = []
        for segment in segments:
            start = min(max(0.0, segment.start), duration)
            end = max(start, min(max(0.0, segment.end), duration))
            clamped_segments.append(TranscriptSegment(start, end, segment.speaker, segment.text))
        segments = tuple(clamped_segments)

    if segments and media_duration is not None and media_duration > 0:
        tail_gap = max(0.0, media_duration - segments[-1].end)
        coverage = segments[-1].end / media_duration
        if coverage < 0.75:
            diagnostics.append(
                TranscriptDiagnostic(
                    "warning",
                    "possible_early_stop",
                    f"The final segment ends {tail_gap:.1f}s before the media ends; generation may have stopped early.",
                )
            )

    normalized = [re.sub(r"\s+", " ", segment.text).strip().casefold() for segment in segments]
    repeated = Counter(item for item in normalized if len(item) >= 4)
    repeated_count = max(repeated.values(), default=0)
    if repeated_count >= 3 and repeated_count / max(len(normalized), 1) >= 0.25:
        diagnostics.append(
            TranscriptDiagnostic(
                "warning",
                "repeated_text",
                "The same transcript text appears repeatedly; inspect the result for a generation loop.",
            )
        )

    if audio_rms is not None and audio_rms < 1e-4 and segments:
        diagnostics.append(
            TranscriptDiagnostic(
                "warning",
                "possible_silence_hallucination",
                "The input energy is near silence but the model returned speech; treat the transcript as unreliable.",
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
