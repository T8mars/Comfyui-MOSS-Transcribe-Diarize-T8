from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


TARGET_SAMPLE_RATE = 16_000


def _finite_waveform(waveform: torch.Tensor) -> torch.Tensor:
    waveform = waveform.detach().to(device="cpu", dtype=torch.float32)
    if waveform.numel() == 0:
        raise ValueError("Audio input is empty.")
    if not torch.isfinite(waveform).all():
        raise ValueError("Audio input contains NaN or infinite samples.")
    return waveform


def normalize_waveform(waveform: torch.Tensor | np.ndarray) -> torch.Tensor:
    tensor = torch.as_tensor(waveform)
    tensor = _finite_waveform(tensor)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3:
        if tensor.shape[0] != 1:
            raise ValueError(f"Expected one audio item, got batch shape {tuple(tensor.shape)}.")
        tensor = tensor[0]
    elif tensor.ndim != 2:
        raise ValueError(f"Expected AUDIO waveform with 1-3 dimensions, got {tuple(tensor.shape)}.")
    if tensor.shape[0] > 1:
        tensor = tensor.mean(dim=0, keepdim=True)
    return tensor.contiguous()


def resample_waveform(waveform: torch.Tensor, source_rate: int, target_rate: int = TARGET_SAMPLE_RATE) -> torch.Tensor:
    source_rate = int(source_rate)
    target_rate = int(target_rate)
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Sample rates must be positive integers.")
    if source_rate == target_rate:
        return waveform
    try:
        import torchaudio.functional as audio_functional

        return audio_functional.resample(waveform, source_rate, target_rate)
    except (ImportError, OSError):
        import soxr

        values = soxr.resample(waveform.squeeze(0).numpy(), source_rate, target_rate)
        return torch.from_numpy(np.asarray(values, dtype=np.float32)).unsqueeze(0)


def comfy_audio_to_numpy(audio: Mapping[str, Any], target_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    if not isinstance(audio, Mapping):
        raise TypeError("ComfyUI AUDIO input must be a mapping.")
    if "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("ComfyUI AUDIO input requires waveform and sample_rate.")
    waveform = normalize_waveform(audio["waveform"])
    waveform = resample_waveform(waveform, int(audio["sample_rate"]), target_rate)
    values = waveform.squeeze(0).numpy().astype(np.float32, copy=False)
    if values.size == 0:
        raise ValueError("Audio input is empty after resampling.")
    return values


__all__ = ["TARGET_SAMPLE_RATE", "comfy_audio_to_numpy", "normalize_waveform", "resample_waveform"]
