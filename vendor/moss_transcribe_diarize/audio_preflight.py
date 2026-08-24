from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class AudioPreflight:
    classification: str
    rms: float
    peak: float
    active_ratio: float
    duration_seconds: float
    sample_rate: int
    threshold: float

    @property
    def should_warn(self) -> bool:
        return self.classification in {"silent", "mostly_silence"}

    def to_dict(self) -> dict[str, float | int | str | bool]:
        return {**asdict(self), "should_warn": self.should_warn}


def analyze_audio_samples(
    samples,
    sample_rate: int,
    *,
    frame_seconds: float = 0.02,
    activity_threshold: float = 0.002,
) -> AudioPreflight:
    array = np.asarray(samples)
    if array.ndim > 1:
        sample_axis = int(np.argmax(array.shape))
        channel_axes = tuple(index for index in range(array.ndim) if index != sample_axis)
        array = np.mean(array, axis=channel_axes)
    array = np.asarray(array, dtype=np.float32).reshape(-1)
    if not np.isfinite(array).all():
        raise ValueError("Audio preflight received NaN or infinite samples.")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    duration = array.size / float(sample_rate)
    if not array.size:
        return AudioPreflight("silent", 0.0, 0.0, 0.0, 0.0, int(sample_rate), activity_threshold)
    rms = float(np.sqrt(np.mean(np.square(array, dtype=np.float64))))
    peak = float(np.max(np.abs(array)))
    frame_size = max(1, int(round(sample_rate * frame_seconds)))
    frame_count = int(np.ceil(array.size / frame_size))
    padded = np.pad(array, (0, frame_count * frame_size - array.size))
    frame_rms = np.sqrt(np.mean(np.square(padded.reshape(frame_count, frame_size), dtype=np.float64), axis=1))
    active_ratio = float(np.count_nonzero(frame_rms >= activity_threshold) / frame_count)
    if rms < 1e-4 or peak < 5e-4:
        classification = "silent"
    elif active_ratio < 0.05:
        classification = "mostly_silence"
    else:
        classification = "speech"
    return AudioPreflight(classification, rms, peak, active_ratio, duration, int(sample_rate), activity_threshold)


def analyze_audio_path(path: str | Path, *, sample_rate: int = 16000) -> AudioPreflight:
    from .inference_utils import load_audio_item

    samples = load_audio_item(str(Path(path).expanduser()), sampling_rate=sample_rate)
    return analyze_audio_samples(samples, sample_rate)


__all__ = ["AudioPreflight", "analyze_audio_path", "analyze_audio_samples"]
