from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import gc
import math
import re
import threading
from typing import Any

import numpy as np
import torch

from .model_cache import resolve_dtype
from .types import TranscriptPayload


DEFAULT_ALIGNMENT_MODEL = "openai/whisper-small"
DEFAULT_ALIGNMENT_REVISION = "973afd24965f72e36ca33b3055d56a652f456b4d"
ALIGNMENT_SAMPLE_RATE = 16000


@dataclass(frozen=True, slots=True)
class WordAlignmentHandle:
    model_id: str = DEFAULT_ALIGNMENT_MODEL
    revision: str = DEFAULT_ALIGNMENT_REVISION
    device: str = "cpu"
    precision: str = "auto"
    language: str = "auto"
    chunk_length_seconds: float = 30.0
    release_after_run: bool = False

    @property
    def cache_key(self) -> tuple[str, str, str, str]:
        return (self.model_id, self.revision, self.device, self.precision)


@dataclass(slots=True)
class _AlignmentEntry:
    processor: Any
    model: Any
    pipeline: Any
    device: torch.device
    lock: threading.RLock


_CACHE: dict[tuple[str, str, str, str], _AlignmentEntry] = {}
_CACHE_LOCK = threading.RLock()


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("词级对齐请求了 CUDA，但当前 PyTorch 未检测到 CUDA。")
    return device


def _load_entry(handle: WordAlignmentHandle) -> _AlignmentEntry:
    key = handle.cache_key
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            return cached
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        device = _resolve_device(handle.device)
        dtype = resolve_dtype(handle.precision, device)
        source = handle.model_id.strip()
        if not source:
            raise ValueError("词级对齐模型不能为空。")
        revision = handle.revision.strip() or None
        load_options = {"revision": revision, "trust_remote_code": False}
        processor = AutoProcessor.from_pretrained(source, **load_options)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            source,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            **load_options,
        ).to(device).eval()
        aligner = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device,
            chunk_length_s=float(handle.chunk_length_seconds),
        )
        entry = _AlignmentEntry(processor, model, aligner, device, threading.RLock())
        _CACHE[key] = entry
        return entry


def release_alignment_model(handle: WordAlignmentHandle | None = None) -> int:
    with _CACHE_LOCK:
        keys = [handle.cache_key] if handle is not None else list(_CACHE)
        removed = 0
        for key in keys:
            entry = _CACHE.pop(key, None)
            if entry is None:
                continue
            entry.pipeline = None
            entry.model = None
            entry.processor = None
            removed += 1
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return removed


def alignment_cache_report() -> list[dict[str, str]]:
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


def run_whisper_word_timestamps(
    handle: WordAlignmentHandle,
    samples: np.ndarray,
    *,
    sample_rate: int = ALIGNMENT_SAMPLE_RATE,
) -> list[dict[str, Any]]:
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if sample_rate != ALIGNMENT_SAMPLE_RATE:
        raise ValueError("词级对齐要求 16 kHz 音频。")
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError("词级对齐音频为空或包含非有限值。")
    entry = _load_entry(handle)
    generate_kwargs: dict[str, str] = {}
    language = handle.language.strip()
    if language and language.casefold() != "auto":
        generate_kwargs["language"] = language
    try:
        with entry.lock, torch.inference_mode():
            result = entry.pipeline(
                {"raw": audio, "sampling_rate": sample_rate},
                return_timestamps="word",
                generate_kwargs=generate_kwargs,
            )
    finally:
        if handle.release_after_run:
            release_alignment_model(handle)
    chunks = result.get("chunks") if isinstance(result, dict) else None
    if not isinstance(chunks, list):
        raise RuntimeError("对齐模型没有返回词级时间戳；请确认所选模型支持 Whisper word timestamps。")
    words = []
    duration = audio.size / float(sample_rate)
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        timestamp = chunk.get("timestamp")
        if not isinstance(timestamp, (list, tuple)) or len(timestamp) != 2:
            continue
        start = _finite_time(timestamp[0], 0.0)
        end = _finite_time(timestamp[1], start)
        start = min(duration, max(0.0, start))
        end = min(duration, max(start, end))
        text = str(chunk.get("text") or "").strip()
        if text:
            words.append({"word": text, "start": start, "end": end, "confidence": None})
    if not words:
        raise RuntimeError("对齐模型未产生可用的词级时间戳。")
    return words


def align_transcript_words(
    transcript: TranscriptPayload,
    timed_words: list[dict[str, Any]],
    *,
    model_id: str,
    revision: str = "",
) -> tuple[TranscriptPayload, dict[str, Any]]:
    normalized_source = []
    for index, item in enumerate(timed_words):
        word = str(item.get("word") or item.get("text") or "").strip()
        start = _finite_time(item.get("start"), 0.0)
        end = _finite_time(item.get("end"), start)
        if not word or end < start:
            continue
        source_units = _text_units(word)
        if not source_units:
            continue
        width = max(0.0, end - start) / len(source_units)
        for unit_index, unit in enumerate(source_units):
            normalized_source.append(
                {
                    "index": index,
                    "source_unit_index": unit_index,
                    "word": unit,
                    "normalized": _normalize_unit(unit),
                    "start": start + width * unit_index,
                    "end": start + width * (unit_index + 1),
                    "confidence": item.get("confidence"),
                }
            )

    aligned_segments = []
    total_units = 0
    model_matched_units = 0
    for segment_index, source_segment in enumerate(transcript.segments, 1):
        segment = dict(source_segment)
        start = _finite_time(segment.get("start"), 0.0)
        end = max(start, _finite_time(segment.get("end"), start))
        units = _text_units(str(segment.get("text") or ""))
        candidates = [
            item
            for item in normalized_source
            if item["end"] >= start and item["start"] <= end and item["normalized"]
        ]
        words, matched = _align_segment_units(units, candidates, start, end)
        segment["words"] = words
        segment["word_alignment"] = {
            "model_matched_units": matched,
            "unit_count": len(units),
            "coverage": round(matched / max(1, len(units)), 5),
        }
        aligned_segments.append(segment)
        total_units += len(units)
        model_matched_units += matched

    coverage = model_matched_units / max(1, total_units)
    report = {
        "schema": "t8.moss-word-alignment.v1",
        "backend": "whisper_word_timestamps",
        "model_id": model_id,
        "revision": revision,
        "source_word_count": len(normalized_source),
        "output_word_count": total_units,
        "model_matched_units": model_matched_units,
        "model_match_coverage": round(coverage, 5),
        "fallback_units": max(0, total_units - model_matched_units),
    }
    diagnostics = list(transcript.diagnostics)
    if total_units and coverage < 0.5:
        diagnostics.append(
            {
                "level": "warning",
                "code": "word_alignment_low_coverage",
                "message": "独立对齐模型与 MOSS 文本的匹配覆盖率低于 50%；未匹配词已在相邻模型锚点之间插值。",
                "coverage": round(coverage, 5),
            }
        )
    metadata = dict(transcript.metadata)
    metadata["word_alignment"] = report
    return (
        TranscriptPayload(
            raw_text=transcript.raw_text,
            segments=tuple(aligned_segments),
            diagnostics=tuple(diagnostics),
            metadata=metadata,
        ),
        report,
    )


def run_word_alignment(
    handle: WordAlignmentHandle,
    samples: np.ndarray,
    transcript: TranscriptPayload,
    *,
    sample_rate: int = ALIGNMENT_SAMPLE_RATE,
) -> tuple[TranscriptPayload, dict[str, Any]]:
    timed_words = run_whisper_word_timestamps(handle, samples, sample_rate=sample_rate)
    return align_transcript_words(
        transcript,
        timed_words,
        model_id=handle.model_id,
        revision=handle.revision,
    )


def _align_segment_units(
    units: list[str],
    candidates: list[dict[str, Any]],
    segment_start: float,
    segment_end: float,
) -> tuple[list[dict[str, Any]], int]:
    if not units:
        return [], 0
    target_norm = [_normalize_unit(item) for item in units]
    source_norm = [str(item["normalized"]) for item in candidates]
    matcher = SequenceMatcher(None, target_norm, source_norm, autojunk=False)
    anchors: dict[int, dict[str, Any]] = {}
    for target_index, source_index, size in matcher.get_matching_blocks():
        for offset in range(size):
            if target_norm[target_index + offset]:
                anchors[target_index + offset] = candidates[source_index + offset]

    boundaries = [segment_start] * (len(units) + 1)
    boundary_anchors: dict[int, float] = {0: segment_start, len(units): segment_end}
    for index, item in anchors.items():
        boundary_anchors[index] = max(segment_start, min(segment_end, float(item["start"])))
        boundary_anchors[index + 1] = max(segment_start, min(segment_end, float(item["end"])))
    ordered = sorted(boundary_anchors.items())
    for (left_index, left_time), (right_index, right_time) in zip(ordered, ordered[1:], strict=False):
        right_time = max(left_time, right_time)
        width = max(1, right_index - left_index)
        for position in range(left_index, right_index + 1):
            fraction = (position - left_index) / width
            boundaries[position] = left_time + (right_time - left_time) * fraction
    for index in range(1, len(boundaries)):
        boundaries[index] = max(boundaries[index - 1], min(segment_end, boundaries[index]))

    output = []
    for index, unit in enumerate(units):
        anchor = anchors.get(index)
        output.append(
            {
                "word": unit,
                "start": round(boundaries[index], 3),
                "end": round(boundaries[index + 1], 3),
                "confidence": anchor.get("confidence") if anchor is not None else None,
                "source": "alignment_model" if anchor is not None else "interpolated_between_model_anchors",
            }
        )
    return output, len(anchors)


def _text_units(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*|[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]|[^\s]", text)


def _normalize_unit(text: str) -> str:
    return re.sub(r"[^0-9a-z\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]+", "", text.casefold())


def _finite_time(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


__all__ = [
    "ALIGNMENT_SAMPLE_RATE",
    "DEFAULT_ALIGNMENT_MODEL",
    "DEFAULT_ALIGNMENT_REVISION",
    "WordAlignmentHandle",
    "align_transcript_words",
    "alignment_cache_report",
    "release_alignment_model",
    "run_whisper_word_timestamps",
    "run_word_alignment",
]
