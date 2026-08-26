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
    speech_ratio: float
    duration_seconds: float
    sample_rate: int
    threshold: float
    vad_backend: str
    vad_aggressiveness: int

    @property
    def should_warn(self) -> bool:
        return self.classification in {"silent", "mostly_silence", "non_speech"}

    def to_dict(self) -> dict[str, float | int | str | bool]:
        return {**asdict(self), "should_warn": self.should_warn}


def analyze_audio_samples(
    samples,
    sample_rate: int,
    *,
    frame_seconds: float = 0.02,
    activity_threshold: float = 0.002,
    vad_backend: str = "webrtc",
    vad_aggressiveness: int = 2,
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
        return AudioPreflight(
            "silent", 0.0, 0.0, 0.0, 0.0, 0.0, int(sample_rate), activity_threshold, vad_backend, vad_aggressiveness
        )
    rms = float(np.sqrt(np.mean(np.square(array, dtype=np.float64))))
    peak = float(np.max(np.abs(array)))
    frame_size = max(1, int(round(sample_rate * frame_seconds)))
    frame_count = int(np.ceil(array.size / frame_size))
    padded = np.pad(array, (0, frame_count * frame_size - array.size))
    frame_rms = np.sqrt(np.mean(np.square(padded.reshape(frame_count, frame_size), dtype=np.float64), axis=1))
    active_ratio = float(np.count_nonzero(frame_rms >= activity_threshold) / frame_count)
    speech_ratio, resolved_backend = _speech_ratio(
        array,
        sample_rate,
        frame_seconds=frame_seconds,
        backend=vad_backend,
        aggressiveness=vad_aggressiveness,
    )
    if rms < 1e-4 or peak < 5e-4:
        classification = "silent"
    elif resolved_backend == "webrtc" and speech_ratio < 0.02:
        classification = "non_speech"
    elif (resolved_backend == "webrtc" and speech_ratio < 0.05) or (
        resolved_backend != "webrtc" and active_ratio < 0.05
    ):
        classification = "mostly_silence"
    else:
        classification = "speech"
    return AudioPreflight(
        classification,
        rms,
        peak,
        active_ratio,
        speech_ratio,
        duration,
        int(sample_rate),
        activity_threshold,
        resolved_backend,
        vad_aggressiveness,
    )


def _speech_ratio(
    samples: np.ndarray,
    sample_rate: int,
    *,
    frame_seconds: float,
    backend: str,
    aggressiveness: int,
) -> tuple[float, str]:
    frames, _, resolved_backend = detect_speech_frames(
        samples,
        sample_rate,
        frame_seconds=frame_seconds,
        backend=backend,
        aggressiveness=aggressiveness,
    )
    return float(np.count_nonzero(frames) / max(frames.size, 1)), resolved_backend


def detect_speech_frames(
    samples: np.ndarray,
    sample_rate: int,
    *,
    frame_seconds: float = 0.02,
    backend: str = "webrtc",
    aggressiveness: int = 2,
    activity_threshold: float = 0.002,
) -> tuple[np.ndarray, int, str]:
    backend = str(backend or "webrtc").strip().lower()
    if backend not in {"webrtc", "energy"}:
        raise ValueError("vad_backend must be webrtc or energy.")
    if not 0 <= int(aggressiveness) <= 3:
        raise ValueError("vad_aggressiveness must be between 0 and 3.")
    array = np.asarray(samples, dtype=np.float32).reshape(-1)
    energy_frame_size = max(1, int(round(sample_rate * frame_seconds)))
    if not array.size:
        return np.zeros(0, dtype=bool), energy_frame_size, "webrtc" if backend == "webrtc" else "energy"

    def energy_frames(resolved_backend: str) -> tuple[np.ndarray, int, str]:
        count = int(np.ceil(array.size / energy_frame_size))
        padded = np.pad(array, (0, count * energy_frame_size - array.size))
        rms = np.sqrt(
            np.mean(np.square(padded.reshape(count, energy_frame_size), dtype=np.float64), axis=1)
        )
        return rms >= activity_threshold, energy_frame_size, resolved_backend

    if backend == "energy":
        return energy_frames("energy")
    if sample_rate not in {8000, 16000, 32000, 48000}:
        return energy_frames("energy_fallback")
    try:
        import webrtcvad
    except ImportError:
        return energy_frames("energy_fallback")

    frame_ms = min((10, 20, 30), key=lambda value: abs(value / 1000.0 - frame_seconds))
    frame_size = sample_rate * frame_ms // 1000
    clipped = np.clip(array, -1.0, 1.0)
    pcm = np.round(clipped * 32767.0).astype("<i2", copy=False)
    frame_count = int(np.ceil(pcm.size / frame_size))
    padded = np.pad(pcm, (0, frame_count * frame_size - pcm.size))
    vad = webrtcvad.Vad(int(aggressiveness))
    speech_frames = np.asarray(
        [bool(vad.is_speech(frame.tobytes(), sample_rate))
        for frame in padded.reshape(frame_count, frame_size)
        ],
        dtype=bool,
    )
    return speech_frames, frame_size, "webrtc"


def analyze_audio_path(path: str | Path, *, sample_rate: int = 16000) -> AudioPreflight:
    from .inference_utils import load_audio_item

    samples = load_audio_item(str(Path(path).expanduser()), sampling_rate=sample_rate)
    return analyze_audio_samples(samples, sample_rate)


__all__ = ["AudioPreflight", "analyze_audio_path", "analyze_audio_samples", "detect_speech_frames"]
