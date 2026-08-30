from __future__ import annotations

import gc
import threading
from dataclasses import dataclass

import torch

from .types import ModelHandle


CacheKey = tuple[str, str, str, str, str]
LoadLockKey = tuple[str, str, str, str]


@dataclass(slots=True)
class CacheEntry:
    cache_key: CacheKey
    model: object
    processor: object
    device: torch.device
    dtype: torch.dtype
    lock: threading.RLock
    attention_report: dict
    users: int = 0
    release_requested: bool = False


@dataclass(slots=True)
class LoadGate:
    lock: threading.Lock
    users: int = 0


def _cuda_bf16_supported(device: torch.device) -> bool:
    index = device.index if device.index is not None else torch.cuda.current_device()
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
        self._entries: dict[CacheKey, CacheEntry] = {}
        self._load_locks: dict[LoadLockKey, LoadGate] = {}
        self._loading: dict[CacheKey, int] = {}
        self._release_epoch = 0
        self._release_key_epochs: dict[CacheKey, int] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _resolved(handle: ModelHandle) -> tuple[torch.device, torch.dtype, CacheKey]:
        from ..vendor.moss_transcribe_diarize.inference_utils import resolve_device
        from ..vendor.moss_transcribe_diarize.attention import normalize_attention_implementation

        device = resolve_device(handle.device)
        if handle.device.startswith("cuda") and device.type != "cuda":
            raise RuntimeError("当前 ComfyUI 的 PyTorch 未检测到 CUDA。")
        dtype = resolve_dtype(handle.precision, device)
        attention = normalize_attention_implementation(handle.attention_implementation)
        identity = handle.model_fingerprint or handle.model_revision
        key = (str(handle.model_dir.resolve()), str(device), str(dtype), identity, attention)
        return device, dtype, key

    @staticmethod
    def _coordinate_comfy_memory(device: torch.device) -> None:
        if device.type != "cuda":
            return
        try:
            import comfy.model_management as model_management

            model_management.soft_empty_cache()
        except (ImportError, AttributeError):
            return

    def acquire(self, handle: ModelHandle) -> CacheEntry:
        device, dtype, key = self._resolved(handle)
        with self._lock:
            self._retire_superseded_locked(key)
            entry = self._entries.get(key)
            if entry is not None:
                entry.users += 1
                return entry
            runtime_key = self._runtime_key(key)
            gate = self._load_locks.setdefault(runtime_key, LoadGate(threading.Lock()))
            gate.users += 1
            self._loading[key] = self._loading.get(key, 0) + 1
            release_epoch = self._release_epoch
            release_key_epoch = self._release_key_epochs.get(key, 0)

        try:
            with gate.lock:
                with self._lock:
                    self._retire_superseded_locked(key)
                    entry = self._entries.get(key)
                    if entry is not None:
                        entry.users += 1
                        return entry

                from ..vendor.moss_transcribe_diarize.transformers_compat import (
                    load_local_model_processor_with_attention,
                )

                self._coordinate_comfy_memory(device)
                try:
                    model, processor, attention_report = load_local_model_processor_with_attention(
                        handle.model_dir,
                        device=device,
                        load_dtype=dtype,
                        attention_implementation=handle.attention_implementation,
                    )
                    model = model.to(device).eval()
                except torch.cuda.OutOfMemoryError as exc:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    free, total = 0, 0
                    if device.type == "cuda":
                        try:
                            free, total = torch.cuda.mem_get_info(device)
                        except (TypeError, RuntimeError):
                            index = device.index if device.index is not None else torch.cuda.current_device()
                            with torch.cuda.device(index):
                                free, total = torch.cuda.mem_get_info()
                    raise RuntimeError(
                        f"MOSS 模型加载显存不足：可用 {free / 1024**3:.2f}GB / 总计 {total / 1024**3:.2f}GB。"
                        "请先释放其他 ComfyUI 模型、缩短音频或启用转写后释放。"
                    ) from exc
                entry = CacheEntry(
                    key,
                    model,
                    processor,
                    device,
                    dtype,
                    threading.RLock(),
                    attention_report,
                    users=1,
                )
                with self._lock:
                    entry.release_requested = bool(
                        self._release_epoch != release_epoch
                        or self._release_key_epochs.get(key, 0) != release_key_epoch
                    )
                    self._entries[key] = entry
                return entry
        finally:
            with self._lock:
                loading = max(0, self._loading.get(key, 0) - 1)
                if loading:
                    self._loading[key] = loading
                else:
                    self._loading.pop(key, None)
                gate.users = max(0, gate.users - 1)
                if gate.users == 0 and self._load_locks.get(runtime_key) is gate:
                    self._load_locks.pop(runtime_key, None)

    def _retire_superseded_locked(self, current_key: CacheKey) -> None:
        for key, entry in list(self._entries.items()):
            if key[0] != current_key[0] or key[3] == current_key[3]:
                continue
            entry.release_requested = True
            if entry.users == 0:
                self._entries.pop(key, None)
                self._dispose(entry)

    @staticmethod
    def _runtime_key(key: CacheKey) -> LoadLockKey:
        return (*key[:3], key[4])

    def done(self, handle: ModelHandle, entry: CacheEntry, *, release: bool = False) -> None:
        policy = handle.effective_memory_policy
        release = release or policy == "release_after_run"
        if policy == "release_under_pressure" and self._under_pressure(entry.device):
            release = True
        with self._lock:
            entry.users = max(0, entry.users - 1)
            entry.release_requested = entry.release_requested or release
            if entry.users == 0 and entry.release_requested:
                current = self._entries.get(entry.cache_key)
                if current is entry:
                    self._entries.pop(entry.cache_key, None)
                    self._dispose(entry)

    def release(self, handle: ModelHandle) -> bool:
        _, _, key = self._resolved(handle)
        with self._lock:
            entry = self._entries.get(key)
            loading = key in self._loading
            if entry is None and not loading:
                return False
            self._release_key_epochs[key] = self._release_key_epochs.get(key, 0) + 1
            if entry is not None:
                entry.release_requested = True
                if entry.users == 0:
                    self._entries.pop(key, None)
                    self._dispose(entry)
            return True

    def release_all(self) -> int:
        with self._lock:
            keys = set(self._entries) | set(self._loading)
            count = len(keys)
            self._release_epoch += 1
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
                    "model_identity": key[3],
                    "attention_requested": key[4],
                    "attention": entry.attention_report,
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

    @staticmethod
    def _under_pressure(device: torch.device) -> bool:
        if device.type != "cuda" or not torch.cuda.is_available():
            return False
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        except (TypeError, RuntimeError):
            return False
        return free_bytes < 2 * 1024**3 or free_bytes / max(total_bytes, 1) < 0.20


MODEL_CACHE = ModelCache()

__all__ = ["CacheEntry", "LoadGate", "MODEL_CACHE", "ModelCache", "_cuda_bf16_supported", "resolve_dtype"]
