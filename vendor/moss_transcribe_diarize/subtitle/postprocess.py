# Modified by the T8star-Aix integration after OpenMOSS cb765f2; see CHANGELOG.md.
from __future__ import annotations

from collections.abc import Iterable
from math import ceil, isfinite

from ..transcript_parser import TranscriptSegment, parse_transcript

from .models import SubtitleSegment


DEFAULT_MIN_DURATION = 1.0
DEFAULT_MAX_DURATION = 6.0
DEFAULT_MAX_CHARS = 24
DEFAULT_MERGE_GAP = 0.3
PUNCTUATION = "。！？!?；;，,、 "


def subtitle_segments_from_transcript(
    transcript: str,
    *,
    postprocess: bool = True,
    min_duration: float = DEFAULT_MIN_DURATION,
    max_duration: float = DEFAULT_MAX_DURATION,
    max_chars: int = DEFAULT_MAX_CHARS,
    merge_gap: float = DEFAULT_MERGE_GAP,
) -> list[SubtitleSegment]:
    return subtitle_segments_from_transcript_segments(
        parse_transcript(transcript),
        postprocess=postprocess,
        min_duration=min_duration,
        max_duration=max_duration,
        max_chars=max_chars,
        merge_gap=merge_gap,
    )


def subtitle_segments_from_transcript_segments(
    segments: Iterable[TranscriptSegment],
    *,
    postprocess: bool = True,
    min_duration: float = DEFAULT_MIN_DURATION,
    max_duration: float = DEFAULT_MAX_DURATION,
    max_chars: int = DEFAULT_MAX_CHARS,
    merge_gap: float = DEFAULT_MERGE_GAP,
) -> list[SubtitleSegment]:
    subtitle_segments = [
        SubtitleSegment(
            id=f"seg_{index:04d}",
            start=float(segment.start),
            end=float(segment.end),
            speaker=segment.speaker,
            text=segment.text,
        )
        for index, segment in enumerate(segments, start=1)
    ]
    if not postprocess:
        return subtitle_segments
    return normalize_segments(
        subtitle_segments,
        min_duration=min_duration,
        max_duration=max_duration,
        max_chars=max_chars,
        merge_gap=merge_gap,
        regenerate_ids=True,
    )


def coerce_subtitle_segments(segments: Iterable[SubtitleSegment | dict]) -> list[SubtitleSegment]:
    """Convert user/API payloads to subtitle segments without timing edits."""
    coerced: list[SubtitleSegment] = []
    for index, item in enumerate(segments, start=1):
        segment = item if isinstance(item, SubtitleSegment) else SubtitleSegment.from_dict(item, fallback_id=f"seg_{index:04d}")
        coerced.append(
            SubtitleSegment(
                id=segment.id or f"seg_{index:04d}",
                start=float(segment.start),
                end=float(segment.end),
                speaker=segment.speaker or "S00",
                text=segment.text,
            )
        )
    return coerced


def normalize_segments(
    segments: Iterable[SubtitleSegment | dict],
    *,
    min_duration: float = DEFAULT_MIN_DURATION,
    max_duration: float = DEFAULT_MAX_DURATION,
    max_chars: int = DEFAULT_MAX_CHARS,
    merge_gap: float = DEFAULT_MERGE_GAP,
    regenerate_ids: bool = False,
) -> list[SubtitleSegment]:
    if (
        not isfinite(float(min_duration))
        or not isfinite(float(max_duration))
        or not isfinite(float(merge_gap))
        or min_duration <= 0
        or max_duration <= 0
        or min_duration > max_duration
        or max_chars <= 0
        or merge_gap < 0
    ):
        raise ValueError("Subtitle timing, length, and merge limits must be finite and positive.")
    prepared = _prepare_segments(segments)
    prepared = _fix_overlaps(prepared)
    prepared = _merge_adjacent(prepared, merge_gap=merge_gap, max_chars=max_chars)
    prepared = _split_long_segments(
        prepared,
        min_duration=min_duration,
        max_duration=max_duration,
        max_chars=max_chars,
    )
    prepared = _fix_overlaps(prepared)
    if regenerate_ids:
        for index, segment in enumerate(prepared, start=1):
            segment.id = f"seg_{index:04d}"
    return prepared


def postprocess_subtitle_segments(
    segments: Iterable[SubtitleSegment | dict],
    *,
    min_duration: float = DEFAULT_MIN_DURATION,
    max_duration: float = DEFAULT_MAX_DURATION,
    max_chars_per_line: int = DEFAULT_MAX_CHARS,
    max_lines: int = 2,
    merge_gap: float = DEFAULT_MERGE_GAP,
) -> list[SubtitleSegment]:
    if max_chars_per_line <= 0 or max_lines <= 0:
        raise ValueError("Subtitle line limits must be positive.")
    processed = normalize_segments(
        segments,
        min_duration=min_duration,
        max_duration=max_duration,
        max_chars=max_chars_per_line * max_lines,
        merge_gap=merge_gap,
        regenerate_ids=True,
    )
    return [
        SubtitleSegment(
            id=segment.id,
            start=segment.start,
            end=segment.end,
            speaker=segment.speaker,
            text=_wrap_text(segment.text, max_chars_per_line=max_chars_per_line, max_lines=max_lines),
        )
        for segment in processed
    ]


def subtitle_readability_report(
    segments: Iterable[SubtitleSegment | dict],
    *,
    max_chars_per_second: float = 20.0,
) -> dict:
    if not isfinite(float(max_chars_per_second)) or max_chars_per_second < 0:
        raise ValueError("Subtitle reading-speed limit must be finite and non-negative.")
    prepared = coerce_subtitle_segments(segments)
    rates = []
    violations = []
    for segment in prepared:
        duration = max(0.0, segment.end - segment.start)
        character_count = len("".join(segment.text.split()))
        if duration <= 0:
            if max_chars_per_second > 0 and character_count:
                violations.append(
                    {
                        "segment_id": segment.id,
                        "chars_per_second": None,
                        "limit": float(max_chars_per_second),
                        "reason": "zero_duration",
                    }
                )
            continue
        rate = character_count / duration
        rates.append(rate)
        if max_chars_per_second > 0 and rate > max_chars_per_second:
            violations.append(
                {
                    "segment_id": segment.id,
                    "chars_per_second": round(rate, 3),
                    "limit": float(max_chars_per_second),
                }
            )
    return {
        "segment_count": len(prepared),
        "max_chars_per_second": round(max(rates), 3) if rates else 0.0,
        "average_chars_per_second": round(sum(rates) / len(rates), 3) if rates else 0.0,
        "limit": float(max_chars_per_second),
        "violation_count": len(violations),
        "violations": violations,
    }


def _prepare_segments(segments: Iterable[SubtitleSegment | dict]) -> list[SubtitleSegment]:
    prepared: list[SubtitleSegment] = []
    for index, item in enumerate(segments, start=1):
        segment = item if isinstance(item, SubtitleSegment) else SubtitleSegment.from_dict(item, fallback_id=f"seg_{index:04d}")
        text = segment.text.strip()
        if not text:
            continue
        start = max(0.0, float(segment.start))
        end = max(start, float(segment.end))
        if end <= start:
            continue
        prepared.append(
            SubtitleSegment(
                id=segment.id or f"seg_{index:04d}",
                start=start,
                end=end,
                speaker=segment.speaker or "S00",
                text=text,
            )
        )
    prepared.sort(key=lambda segment: (segment.start, segment.end))
    return prepared


def _fix_overlaps(segments: list[SubtitleSegment]) -> list[SubtitleSegment]:
    cursor = 0.0
    fixed: list[SubtitleSegment] = []
    for segment in segments:
        start = max(segment.start, cursor)
        end = max(start, segment.end)
        fixed.append(
            SubtitleSegment(
                id=segment.id,
                start=start,
                end=end,
                speaker=segment.speaker,
                text=segment.text,
            )
        )
        cursor = end
    return fixed


def _merge_adjacent(segments: list[SubtitleSegment], *, merge_gap: float, max_chars: int) -> list[SubtitleSegment]:
    if not segments:
        return []

    merged = [segments[0]]
    for segment in segments[1:]:
        previous = merged[-1]
        gap = segment.start - previous.end
        combined_text = _join_text(previous.text, segment.text)
        can_merge = (
            previous.speaker == segment.speaker
            and 0 <= gap <= merge_gap
            and len(combined_text) <= max_chars * 2
        )
        if can_merge:
            merged[-1] = SubtitleSegment(
                id=previous.id,
                start=previous.start,
                end=max(previous.end, segment.end),
                speaker=previous.speaker,
                text=combined_text,
            )
        else:
            merged.append(segment)
    return merged


def _split_long_segments(
    segments: list[SubtitleSegment],
    *,
    min_duration: float,
    max_duration: float,
    max_chars: int,
) -> list[SubtitleSegment]:
    output: list[SubtitleSegment] = []
    for segment in segments:
        duration = segment.end - segment.start
        if duration <= max_duration and len(segment.text) <= max_chars:
            output.append(segment)
            continue

        chunks = _split_text(segment.text, max_chars=max_chars)
        duration_parts = ceil(duration / max_duration)
        if duration_parts > len(chunks):
            chunks = _balanced_text_parts(
                segment.text,
                min(max(duration_parts, len(chunks)), len(segment.text)),
            )
        if len(chunks) <= 1:
            output.append(segment)
            continue

        weights = [max(len(chunk), 1) for chunk in chunks]
        total_chars = sum(weights)
        if duration_parts > 1:
            allocations = [duration / len(chunks)] * len(chunks)
        elif duration >= min_duration * len(chunks):
            residual = duration - min_duration * len(chunks)
            allocations = [min_duration + residual * weight / total_chars for weight in weights]
        else:
            allocations = [duration * weight / total_chars for weight in weights]
        cursor = segment.start
        for index, chunk in enumerate(chunks):
            if index == len(chunks) - 1:
                end = segment.end
            else:
                end = min(segment.end, cursor + allocations[index])
            output.append(
                SubtitleSegment(
                    id=f"{segment.id}_{index + 1}",
                    start=cursor,
                    end=end,
                    speaker=segment.speaker,
                    text=chunk,
                )
            )
            cursor = output[-1].end
    return output


def _split_text(text: str, *, max_chars: int) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    for ch in text:
        current.append(ch)
        should_cut = len(current) >= max_chars or (ch in PUNCTUATION and len(current) >= max_chars // 2)
        if should_cut:
            chunks.append("".join(current).strip())
            current.clear()
    if current:
        chunks.append("".join(current).strip())

    compact: list[str] = []
    for chunk in chunks:
        if not chunk:
            continue
        if compact and len(compact[-1]) + len(chunk) <= max_chars:
            compact[-1] = _join_text(compact[-1], chunk)
        else:
            compact.append(chunk)
    return compact


def _balanced_text_parts(text: str, count: int) -> list[str]:
    text = text.strip()
    count = max(1, min(int(count), len(text)))
    quotient, remainder = divmod(len(text), count)
    parts = []
    cursor = 0
    for index in range(count):
        width = quotient + (1 if index < remainder else 0)
        part = text[cursor : cursor + width].strip()
        if part:
            parts.append(part)
        cursor += width
    return parts


def _join_text(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    if left[-1].isascii() and right[0].isascii():
        return f"{left} {right}"
    return f"{left}{right}"


def _wrap_text(text: str, *, max_chars_per_line: int, max_lines: int) -> str:
    remaining = text.replace("\r", "").replace("\n", "").strip()
    lines = []
    while remaining and len(lines) < max_lines:
        slots_after = max_lines - len(lines) - 1
        if len(remaining) <= max_chars_per_line:
            lines.append(remaining)
            remaining = ""
            break
        minimum_cut = max(1, len(remaining) - slots_after * max_chars_per_line)
        maximum_cut = min(max_chars_per_line, len(remaining))
        cut = maximum_cut
        for index in range(maximum_cut - 1, minimum_cut - 2, -1):
            if remaining[index] in PUNCTUATION:
                cut = index + 1
                break
        lines.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        lines[-1] = _join_text(lines[-1], remaining)
    return "\n".join(line for line in lines if line)
