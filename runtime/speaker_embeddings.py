from __future__ import annotations

from dataclasses import dataclass
import gc
import math
import threading
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional

from .model_cache import resolve_dtype
from .types import TranscriptPayload


DEFAULT_SPEAKER_MODEL = "microsoft/wavlm-base-plus-sv"
DEFAULT_SPEAKER_REVISION = "feb593a6c23c1cc3d9510425c29b0a14d2b07b1e"
SPEAKER_SAMPLE_RATE = 16000


@dataclass(frozen=True, slots=True)
class SpeakerEmbeddingHandle:
    model_id: str = DEFAULT_SPEAKER_MODEL
    revision: str = DEFAULT_SPEAKER_REVISION
    device: str = "cpu"
    precision: str = "auto"
    release_after_run: bool = False

    @property
    def cache_key(self) -> tuple[str, str, str, str]:
        return (self.model_id, self.revision, self.device, self.precision)


@dataclass(slots=True)
class _SpeakerEntry:
    processor: Any
    model: Any
    device: torch.device
    dtype: torch.dtype
    lock: threading.RLock


_CACHE: dict[tuple[str, str, str, str], _SpeakerEntry] = {}
_CACHE_LOCK = threading.RLock()


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("声纹模型请求了 CUDA，但当前 PyTorch 未检测到 CUDA。")
    return device


def _load_entry(handle: SpeakerEmbeddingHandle) -> _SpeakerEntry:
    key = handle.cache_key
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            return cached
        from transformers import AutoFeatureExtractor, AutoModelForAudioXVector

        source = handle.model_id.strip()
        if not source:
            raise ValueError("声纹模型不能为空。")
        revision = handle.revision.strip() or None
        options = {"revision": revision, "trust_remote_code": False}
        device = _resolve_device(handle.device)
        dtype = resolve_dtype(handle.precision, device)
        processor = AutoFeatureExtractor.from_pretrained(source, **options)
        model = AutoModelForAudioXVector.from_pretrained(
            source,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            **options,
        ).to(device).eval()
        entry = _SpeakerEntry(processor, model, device, dtype, threading.RLock())
        _CACHE[key] = entry
        return entry


def release_speaker_model(handle: SpeakerEmbeddingHandle | None = None) -> int:
    with _CACHE_LOCK:
        keys = [handle.cache_key] if handle is not None else list(_CACHE)
        removed = 0
        for key in keys:
            entry = _CACHE.pop(key, None)
            if entry is None:
                continue
            entry.model = None
            entry.processor = None
            removed += 1
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return removed


def speaker_cache_report() -> list[dict[str, str]]:
    with _CACHE_LOCK:
        return [
            {
                "model_id": key[0],
                "revision": key[1],
                "device": key[2],
                "precision": key[3],
            }
            for key in sorted(_CACHE)
        ]


def extract_speaker_embedding(
    handle: SpeakerEmbeddingHandle,
    samples: np.ndarray,
    *,
    sample_rate: int = SPEAKER_SAMPLE_RATE,
) -> np.ndarray:
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if sample_rate != SPEAKER_SAMPLE_RATE:
        raise ValueError("声纹模型要求 16 kHz 音频。")
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError("声纹音频为空或包含非有限值。")
    entry = _load_entry(handle)
    with entry.lock, torch.inference_mode():
        inputs = entry.processor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        prepared = {}
        for name, value in inputs.items():
            if hasattr(value, "to"):
                value = value.to(entry.device)
                if torch.is_floating_point(value):
                    value = value.to(entry.dtype)
            prepared[name] = value
        output = entry.model(**prepared)
        embedding = functional.normalize(output.embeddings, dim=-1)[0].detach().float().cpu().numpy()
    if not np.isfinite(embedding).all():
        raise RuntimeError("声纹模型产生了非有限嵌入。")
    return embedding.astype(np.float32, copy=False)


def link_speakers_by_voice(
    handle: SpeakerEmbeddingHandle,
    samples: np.ndarray,
    transcript: TranscriptPayload,
    *,
    sample_rate: int = SPEAKER_SAMPLE_RATE,
    similarity_threshold: float = 0.86,
    min_speech_seconds: float = 0.8,
    max_reference_seconds: float = 20.0,
    embedding_provider=None,
) -> tuple[TranscriptPayload, dict[str, Any]]:
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if sample_rate != SPEAKER_SAMPLE_RATE:
        raise ValueError("声纹关联要求 16 kHz 音频。")
    if not 0.0 < float(similarity_threshold) <= 1.0:
        raise ValueError("声纹相似度阈值必须在 (0, 1]。")
    if min_speech_seconds <= 0 or max_reference_seconds < min_speech_seconds:
        raise ValueError("声纹参考时长设置无效。")
    provider = embedding_provider or (
        lambda value: extract_speaker_embedding(handle, value, sample_rate=sample_rate)
    )
    groups = _speaker_groups(
        transcript,
        audio,
        sample_rate=sample_rate,
        min_speech_seconds=float(min_speech_seconds),
        max_reference_seconds=float(max_reference_seconds),
    )
    embeddings: dict[str, np.ndarray] = {}
    failures: list[dict[str, str]] = []
    try:
        for speaker, group in groups.items():
            if group["eligible"]:
                try:
                    embeddings[speaker] = np.asarray(provider(group["samples"]), dtype=np.float32).reshape(-1)
                except Exception as exc:  # noqa: BLE001 - optional auxiliary models need per-speaker isolation
                    failures.append({"speaker": speaker, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        if handle.release_after_run:
            release_speaker_model(handle)

    mapping, links, rejected = _cluster_embeddings(
        embeddings,
        groups,
        similarity_threshold=float(similarity_threshold),
    )
    segments = []
    for source in transcript.segments:
        segment = dict(source)
        original = str(segment.get("speaker") or "S00")
        resolved = mapping.get(original, original)
        if resolved != original:
            segment["voice_original_speaker"] = original
            segment["speaker"] = resolved
        segments.append(segment)

    report = {
        "schema": "t8.moss-speaker-embedding-link.v1",
        "backend": "wavlm_xvector",
        "model_id": handle.model_id,
        "revision": handle.revision,
        "similarity_threshold": float(similarity_threshold),
        "speaker_count": len(groups),
        "embedded_speaker_count": len(embeddings),
        "links": links,
        "rejected_candidates": rejected,
        "failures": failures,
        "references": {
            speaker: {
                "duration_seconds": round(float(group["duration_seconds"]), 3),
                "chunk_ids": sorted(group["chunk_ids"]),
                "eligible": bool(group["eligible"]),
            }
            for speaker, group in groups.items()
        },
    }
    diagnostics = list(transcript.diagnostics)
    diagnostics.append(
        {
            "level": "info" if not failures else "warning",
            "code": "speaker_embedding_linking",
            "message": f"声纹模型为 {len(embeddings)} 个局部说话人生成嵌入并建立 {len(links)} 个跨分块关联。",
            "links": len(links),
            "failures": len(failures),
        }
    )
    metadata = dict(transcript.metadata)
    metadata["speaker_embedding_link"] = report
    metadata["speaker_links"] = [*(metadata.get("speaker_links") or []), *links]
    if links:
        metadata["speaker_scope"] = "voice_embedding_linked"
    return (
        TranscriptPayload(
            raw_text=_render_raw_transcript(segments),
            segments=tuple(segments),
            diagnostics=tuple(diagnostics),
            metadata=metadata,
        ),
        report,
    )


def _speaker_groups(
    transcript: TranscriptPayload,
    audio: np.ndarray,
    *,
    sample_rate: int,
    min_speech_seconds: float,
    max_reference_seconds: float,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    max_samples = max(1, int(round(max_reference_seconds * sample_rate)))
    for segment in transcript.segments:
        speaker = str(segment.get("speaker") or "S00")
        if speaker == "S00":
            continue
        start = max(0, int(round(_finite(segment.get("start"), 0.0) * sample_rate)))
        end = min(audio.size, int(round(_finite(segment.get("end"), 0.0) * sample_rate)))
        if end <= start:
            continue
        group = grouped.setdefault(speaker, {"parts": [], "chunk_ids": set(), "first_start": start})
        group["parts"].append(audio[start:end])
        group["first_start"] = min(group["first_start"], start)
        chunk_id = str(segment.get("chunk_id") or "").strip() or "__single_pass__"
        group["chunk_ids"].add(chunk_id)
    output = {}
    for speaker, group in grouped.items():
        concatenated = np.concatenate(group["parts"])[:max_samples]
        duration = concatenated.size / float(sample_rate)
        output[speaker] = {
            "samples": concatenated,
            "duration_seconds": duration,
            "eligible": duration >= min_speech_seconds,
            "chunk_ids": set(group["chunk_ids"]),
            "first_start": int(group["first_start"]),
        }
    return output


def _cluster_embeddings(
    embeddings: dict[str, np.ndarray],
    groups: dict[str, dict[str, Any]],
    *,
    similarity_threshold: float,
) -> tuple[dict[str, str], list[dict[str, Any]], list[dict[str, Any]]]:
    speakers = sorted(embeddings, key=lambda item: (int(groups[item]["first_start"]), item))
    parent = {speaker: speaker for speaker in speakers}
    chunks = {speaker: set(groups[speaker]["chunk_ids"]) for speaker in speakers}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    candidates = []
    for left_index, left in enumerate(speakers):
        for right in speakers[left_index + 1 :]:
            similarity = _cosine(embeddings[left], embeddings[right])
            if similarity >= similarity_threshold:
                candidates.append((similarity, left, right))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for similarity, left, right in candidates:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            continue
        shared_chunks = chunks[left_root] & chunks[right_root]
        if shared_chunks:
            rejected.append(
                {
                    "left": left,
                    "right": right,
                    "similarity": round(similarity, 5),
                    "reason": "same_chunk_collision",
                    "chunk_ids": sorted(shared_chunks),
                }
            )
            continue
        canonical, merged = sorted(
            (left_root, right_root),
            key=lambda item: (int(groups[item]["first_start"]), item),
        )
        parent[merged] = canonical
        chunks[canonical] |= chunks[merged]
        accepted.append(
            {
                "from_speaker": merged,
                "speaker": canonical,
                "method": "wavlm_xvector_cosine",
                "confidence": round(similarity, 5),
            }
        )
    mapping = {speaker: find(speaker) for speaker in speakers}
    for link in accepted:
        link["speaker"] = mapping.get(str(link["speaker"]), str(link["speaker"]))
    return mapping, accepted, rejected


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float32).reshape(-1)
    right = np.asarray(right, dtype=np.float32).reshape(-1)
    if left.size != right.size or not left.size:
        raise ValueError("声纹嵌入维度不一致。")
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0 else -1.0


def _render_raw_transcript(segments: list[dict[str, Any]]) -> str:
    return "".join(
        f"[{_finite(item.get('start'), 0.0):.2f}][{item.get('speaker') or 'S00'}]"
        f"{str(item.get('text') or '').strip()}[{_finite(item.get('end'), 0.0):.2f}]"
        for item in segments
    )


def _finite(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


__all__ = [
    "DEFAULT_SPEAKER_MODEL",
    "DEFAULT_SPEAKER_REVISION",
    "SPEAKER_SAMPLE_RATE",
    "SpeakerEmbeddingHandle",
    "extract_speaker_embedding",
    "link_speakers_by_voice",
    "release_speaker_model",
    "speaker_cache_report",
]
