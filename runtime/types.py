from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True, frozen=True)
class ModelHandle:
    model_dir: Path
    device: str
    precision: str
    release_after_run: bool = False
    model_revision: str = ""

    @property
    def cache_key(self) -> tuple[str, str, str]:
        return (str(self.model_dir.resolve()), self.device, self.precision)


@dataclass(slots=True, frozen=True)
class PromptConfig:
    text: str
    hotwords: tuple[str, ...] = ()
    language_hint: str = "auto"


@dataclass(slots=True, frozen=True)
class TranscriptPayload:
    raw_text: str
    segments: tuple[dict[str, Any], ...]
    diagnostics: tuple[dict[str, Any], ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "t8.moss-transcript.v1",
            "raw_text": self.raw_text,
            "segments": list(self.segments),
            "diagnostics": list(self.diagnostics),
            "metadata": self.metadata,
        }
