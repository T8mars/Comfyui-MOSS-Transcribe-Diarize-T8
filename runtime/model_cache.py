from __future__ import annotations

import gc
import threading
from dataclasses import dataclass

import torch

from .types import ModelHandle


@dataclass(slots=True)
class CacheEntry:
    model: object
    processor: object
    device: torch.device
    dtype: torch.dtype
    lock: threading.RLock
    users: int = 0
    release_requested: bool = False


def _cuda_bf16_supported(device: torch.device) -> bool:
    index = device.index if device.index is not None else torch.cuda.current_device()
    try:
        return bool(torch.cuda.is_bf16_supported(index))
    except TypeError:
        with torch.cuda.device(index):
            return bool(torch.cuda.is_bf16_supported())


def resolve_dtype(precision: str, device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        return torch.float32
    if precision == "float32":
        return torch.float32
    if precision == "float16":
        return torch.float16
    if precision == "bfloat16":
        return torch.bfloat16 if device.type != "cuda" or _cuda_bf16_supported(device) else torch.float16
    if precision != "auto":
        raise ValueError(f"Unsupported precision: {precision}")
    if device.type == "cuda":
        return torch.bfloat16 if _cuda_bf16_supported(device) else torch.float16
    return torch.float32


class ModelCache:
    def __init__(self):
        self._entries: dict[tuple[str, str, str], CacheEntry] = {}
        self._lock = threading.RLock()

    def acquire(self, handle: ModelHandle) -> CacheEntry:
        with self._lock:
            entry = self._entries.get(handle.cache_key)
            if entry is None:
                from ..vendor.moss_transcribe_diarize.inference_utils import resolve_device
                from ..vendor.moss_transcribe_diarize.transformers_compat import load_local_model_and_processor

                device = resolve_device(handle.device)
                if handle.device.startswith("cuda") and device.type != "cuda":
                    raise RuntimeError("当前 ComfyUI 的 PyTorch 未检测到 CUDA。")
                dtype = resolve_dtype(handle.precision, device)
                model, processor = load_local_model_and_processor(handle.model_dir, load_dtype=dtype)
                model = model.to(device).eval()
                entry = CacheEntry(model, processor, device, dtype, threading.RLock())
                self._entries[handle.cache_key] = entry
            entry.users += 1
            return entry

    def done(self, handle: ModelHandle, entry: CacheEntry, *, release: bool = False) -> None:
        with self._lock:
            entry.users = max(0, entry.users - 1)
            entry.release_requested = entry.release_requested or release
            if entry.users == 0 and entry.release_requested:
                current = self._entries.get(handle.cache_key)
                if current is entry:
                    self._entries.pop(handle.cache_key, None)
                    self._dispose(entry)

    def release(self, handle: ModelHandle) -> bool:
        with self._lock:
            entry = self._entries.get(handle.cache_key)
            if entry is None:
                return False
            entry.release_requested = True
            if entry.users == 0:
                self._entries.pop(handle.cache_key, None)
                self._dispose(entry)
            return True

    def release_all(self) -> int:
        with self._lock:
            count = len(self._entries)
            for key, entry in list(self._entries.items()):
                entry.release_requested = True
                if entry.users == 0:
                    self._entries.pop(key, None)
                    self._dispose(entry)
            return count

    def report(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "model_dir": key[0],
                    "device": key[1],
                    "precision": key[2],
                    "users": entry.users,
                    "release_requested": entry.release_requested,
                }
                for key, entry in self._entries.items()
            ]

    @staticmethod
    def _dispose(entry: CacheEntry) -> None:
        entry.model = None
        entry.processor = None
        gc.collect()
        if entry.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


MODEL_CACHE = ModelCache()

__all__ = ["CacheEntry", "MODEL_CACHE", "ModelCache", "resolve_dtype"]
