# Modified by the T8star-Aix integration after OpenMOSS cb765f2; see CHANGELOG.md.
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Iterator


@dataclass(slots=True, frozen=True)
class TranscriptSegment:
    start: float
    end: float
    speaker: str
    text: str


class TranscriptParseError(ValueError):
    """Raised for invalid parser usage."""


class TranscriptStreamParser:
    """Streaming parser for compact MOSS transcript output.

    Expected segment format:

        [start][Sxx]text[end]

    If a backend omits the speaker token, ``default_speaker`` preserves the
    otherwise valid timestamped text. ``S00`` is reserved by this integration
    for an unknown/unlabelled speaker; pass ``None`` for strict parsing.

    The parser deliberately avoids regular expressions. It scans characters
    once, keeps only the active token/text buffers, and emits a segment after
    an end timestamp is confirmed by the next segment start or by ``close()``.
    """

    _SEEK_START = 0
    _READ_START = 1
    _EXPECT_SPEAKER_OPEN = 2
    _READ_SPEAKER = 3
    _READ_TEXT = 4
    _READ_END = 5
    _AFTER_END = 6

    def __init__(
        self,
        *,
        strip_text: bool = True,
        skip_empty: bool = True,
        default_speaker: str | None = "S00",
    ):
        if default_speaker is not None and _parse_speaker(list(default_speaker)) is None:
            raise TranscriptParseError(
                f"default_speaker must look like Sxx, got {default_speaker!r}"
            )
        self.strip_text = strip_text
        self.skip_empty = skip_empty
        self.default_speaker = default_speaker
        self._state = self._SEEK_START
        self._token: list[str] = []
        self._text: list[str] = []
        self._pending_after_end: list[str] = []
        self._start: float | None = None
        self._end: float | None = None
        self._end_token = ""
        self._speaker: str | None = None
        self._had_malformed_input = False

    def reset(self) -> None:
        """Clear the active segment and all stream-level parse status."""
        self._reset_segment()
        self._had_malformed_input = False

    def _reset_segment(self) -> None:
        self._state = self._SEEK_START
        self._token.clear()
        self._text.clear()
        self._pending_after_end.clear()
        self._start = None
        self._end = None
        self._end_token = ""
        self._speaker = None

    @property
    def had_malformed_input(self) -> bool:
        """Whether non-whitespace input was discarded while recovering."""
        return self._had_malformed_input

    @property
    def has_incomplete_segment(self) -> bool:
        """Whether the stream currently ends inside an unfinished segment."""
        return self._state not in {self._SEEK_START, self._AFTER_END}

    def _discard_malformed_segment(self) -> None:
        self._had_malformed_input = True
        self._reset_segment()

    def feed(self, chunk: str) -> list[TranscriptSegment]:
        """Consume a text chunk and return any newly completed segments."""
        segments: list[TranscriptSegment] = []
        self.feed_into(chunk, segments.append)
        return segments

    def feed_into(self, chunk: str, emit: Callable[[TranscriptSegment], None]) -> None:
        """Consume a text chunk and send completed segments to ``emit``."""
        if not isinstance(chunk, str):
            raise TranscriptParseError(f"chunk must be str, got {type(chunk).__name__}")

        for ch in chunk:
            state = self._state
            if state == self._SEEK_START:
                self._seek_start(ch)
            elif state == self._READ_START:
                self._read_start(ch)
            elif state == self._EXPECT_SPEAKER_OPEN:
                self._expect_speaker_open(ch)
            elif state == self._READ_SPEAKER:
                self._read_speaker(ch)
            elif state == self._READ_TEXT:
                self._read_text(ch)
            elif state == self._READ_END:
                self._read_end(ch, emit)
            elif state == self._AFTER_END:
                self._after_end(ch, emit)

    def close(self) -> list[TranscriptSegment]:
        """Finish the stream and return a final segment if one is complete."""
        segments: list[TranscriptSegment] = []
        self.close_into(segments.append)
        return segments

    def close_into(self, emit: Callable[[TranscriptSegment], None]) -> None:
        """Finish the stream and send a final complete segment to ``emit``."""
        if self._state == self._AFTER_END:
            self._emit_segment(emit)
        self.reset()

    def _seek_start(self, ch: str) -> None:
        if ch == "[":
            self._token.clear()
            self._state = self._READ_START
        elif not ch.isspace():
            self._had_malformed_input = True

    def _read_start(self, ch: str) -> None:
        if ch == "]":
            start = _parse_timestamp(self._token)
            if start is None:
                self._discard_malformed_segment()
                return
            self._start = start
            self._state = self._EXPECT_SPEAKER_OPEN
            self._token.clear()
            return

        if _is_timestamp_char(ch):
            self._token.append(ch)
            if len(self._token) <= 32:
                return

        self._discard_malformed_segment()
        if ch == "[":
            self._state = self._READ_START

    def _expect_speaker_open(self, ch: str) -> None:
        if ch == "[":
            self._token.clear()
            self._state = self._READ_SPEAKER
        elif not ch.isspace() and self.default_speaker is not None:
            self._speaker = self.default_speaker
            self._text.clear()
            self._state = self._READ_TEXT
            self._read_text(ch)
        elif not ch.isspace():
            self._discard_malformed_segment()

    def _read_speaker(self, ch: str) -> None:
        if ch == "]":
            speaker = _parse_speaker(self._token)
            if speaker is None:
                self._discard_malformed_segment()
                return
            self._speaker = speaker
            self._text.clear()
            self._state = self._READ_TEXT
            self._token.clear()
            return

        if _is_speaker_char(ch):
            self._token.append(ch)
            if len(self._token) <= 16:
                return

        self._discard_malformed_segment()
        if ch == "[":
            self._state = self._READ_START

    def _read_text(self, ch: str) -> None:
        if ch == "[":
            self._token.clear()
            self._state = self._READ_END
        else:
            self._text.append(ch)

    def _read_end(self, ch: str, emit: Callable[[TranscriptSegment], None]) -> None:
        if ch == "]":
            end = _parse_timestamp(self._token)
            if end is not None and self._start is not None and end >= self._start:
                self._end = end
                self._end_token = "".join(self._token)
                self._pending_after_end.clear()
                self._state = self._AFTER_END
            else:
                self._text.append("[")
                self._text.extend(self._token)
                self._text.append("]")
                self._state = self._READ_TEXT
            self._token.clear()
            return

        if _is_timestamp_char(ch):
            self._token.append(ch)
            if len(self._token) <= 32:
                return

        self._text.append("[")
        self._text.extend(self._token)
        self._text.append(ch)
        self._token.clear()
        self._state = self._READ_TEXT

    def _after_end(self, ch: str, emit: Callable[[TranscriptSegment], None]) -> None:
        if ch == "[":
            self._emit_segment(emit)
            self._token.clear()
            self._state = self._READ_START
            return

        if ch.isspace():
            self._pending_after_end.append(ch)
            return

        self._text.append("[")
        self._text.append(self._end_token)
        self._text.append("]")
        self._text.extend(self._pending_after_end)
        self._text.append(ch)
        self._pending_after_end.clear()
        self._end = None
        self._end_token = ""
        self._state = self._READ_TEXT

    def _emit_segment(self, emit: Callable[[TranscriptSegment], None]) -> None:
        if self._start is None or self._end is None or self._speaker is None:
            self._discard_malformed_segment()
            return

        text = "".join(self._text)
        if self.strip_text:
            text = text.strip()
        if text or not self.skip_empty:
            emit(
                TranscriptSegment(
                    start=self._start,
                    end=self._end,
                    speaker=self._speaker,
                    text=text,
                )
            )

        self._reset_segment()


def iter_transcript_segments(chunks: Iterable[str], **parser_kwargs) -> Iterator[TranscriptSegment]:
    parser = TranscriptStreamParser(**parser_kwargs)
    for chunk in chunks:
        yield from parser.feed(chunk)
    yield from parser.close()


def parse_transcript(text: str, **parser_kwargs) -> list[TranscriptSegment]:
    parser = TranscriptStreamParser(**parser_kwargs)
    segments = parser.feed(text)
    segments.extend(parser.close())
    return segments


def _parse_timestamp(chars: list[str]) -> float | None:
    if not chars:
        return None

    dot_count = 0
    digit_count = 0
    for ch in chars:
        if "0" <= ch <= "9":
            digit_count += 1
        elif ch == ".":
            dot_count += 1
            if dot_count > 1:
                return None
        else:
            return None

    if digit_count == 0:
        return None
    return float("".join(chars))


def _parse_speaker(chars: list[str]) -> str | None:
    if len(chars) < 2 or chars[0] != "S":
        return None
    for ch in chars[1:]:
        if not ("0" <= ch <= "9"):
            return None
    return "".join(chars)


def _is_timestamp_char(ch: str) -> bool:
    return ("0" <= ch <= "9") or ch == "."


def _is_speaker_char(ch: str) -> bool:
    return ch == "S" or ("0" <= ch <= "9")


__all__ = [
    "TranscriptParseError",
    "TranscriptSegment",
    "TranscriptStreamParser",
    "iter_transcript_segments",
    "parse_transcript",
]
