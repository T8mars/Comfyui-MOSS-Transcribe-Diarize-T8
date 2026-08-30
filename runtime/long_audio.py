from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
from typing import Any
import uuid

from filelock import FileLock, Timeout
import numpy as np

from .inference import _comfy_runtime_callbacks, run_transcription_samples
from .model_cache import MODEL_CACHE
from .quality import evaluate_quality
from .types import ModelHandle, PromptConfig, TranscriptPayload
from ..vendor.moss_transcribe_diarize.audio_preflight import detect_speech_frames
from ..vendor.moss_transcribe_diarize.generation_budget import estimate_max_new_tokens


CHECKPOINT_SCHEMA = "t8.moss-long-audio-checkpoint.v1"
LONG_AUDIO_ALGORITHM_VERSION = "t8.moss-long-audio.v2"
MAX_CHECKPOINT_NAME_LENGTH = 96


@dataclass(frozen=True, slots=True)
class AudioChunk:
    index: int
    chunk_id: str
    start_sample: int
    end_sample: int
    start_seconds: float
    end_seconds: float

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


def plan_audio_chunks(
    samples: np.ndarray,
    sample_rate: int,
    *,
    target_seconds: float = 480.0,
    max_seconds: float = 600.0,
    overlap_seconds: float = 1.0,
    strategy: str = "vad",
    vad_aggressiveness: int = 2,
) -> list[AudioChunk]:
    array = np.asarray(samples, dtype=np.float32).reshape(-1)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    if target_seconds <= 0 or max_seconds <= 0 or target_seconds > max_seconds:
        raise ValueError("Chunk target must be positive and no greater than the maximum.")
    if overlap_seconds < 0 or overlap_seconds >= target_seconds / 2:
        raise ValueError("Chunk overlap must be non-negative and less than half the target duration.")
    if strategy not in {"vad", "fixed"}:
        raise ValueError("strategy must be vad or fixed.")
    if not array.size:
        raise ValueError("Long-audio input is empty.")

    total = array.size
    target = max(1, int(round(target_seconds * sample_rate)))
    maximum = max(target, int(round(max_seconds * sample_rate)))
    overlap = int(round(overlap_seconds * sample_rate))
    speech_frames = None
    frame_size = max(1, int(round(0.02 * sample_rate)))
    if strategy == "vad":
        speech_frames, frame_size, _ = detect_speech_frames(
            array,
            sample_rate,
            backend="webrtc",
            aggressiveness=vad_aggressiveness,
        )

    chunks: list[AudioChunk] = []
    start = 0
    while start < total:
        hard_end = min(total, start + maximum)
        if hard_end >= total:
            end = total
        else:
            ideal_end = min(hard_end, start + target)
            end = ideal_end
            if speech_frames is not None:
                candidate = _find_silence_boundary(
                    speech_frames,
                    frame_size,
                    sample_rate,
                    start_sample=start,
                    ideal_sample=ideal_end,
                    hard_end_sample=hard_end,
                )
                if candidate is not None:
                    end = candidate
        if end <= start:
            end = min(total, start + target)
        index = len(chunks) + 1
        chunks.append(
            AudioChunk(
                index,
                f"part{index:03d}",
                start,
                end,
                start / float(sample_rate),
                end / float(sample_rate),
            )
        )
        if end >= total:
            break
        next_start = max(start + 1, end - overlap)
        start = next_start
    return chunks


def _find_silence_boundary(
    speech_frames: np.ndarray,
    frame_size: int,
    sample_rate: int,
    *,
    start_sample: int,
    ideal_sample: int,
    hard_end_sample: int,
) -> int | None:
    search_radius = int(round(30.0 * sample_rate))
    minimum_chunk = start_sample + int(round(60.0 * sample_rate))
    left = max(minimum_chunk, ideal_sample - search_radius)
    right = min(hard_end_sample, ideal_sample + search_radius)
    first_frame = max(0, left // frame_size)
    last_frame = min(speech_frames.size, int(np.ceil(right / frame_size)))
    if last_frame <= first_frame:
        return None

    candidates: list[tuple[float, int, int]] = []
    run_start = None
    for frame_index in range(first_frame, last_frame + 1):
        silent = frame_index < last_frame and not bool(speech_frames[frame_index])
        if silent and run_start is None:
            run_start = frame_index
        elif not silent and run_start is not None:
            run_end = frame_index
            run_samples = (run_end - run_start) * frame_size
            if run_samples >= int(0.20 * sample_rate):
                midpoint = ((run_start + run_end) * frame_size) // 2
                score = abs(midpoint - ideal_sample) - min(run_samples, sample_rate * 2) * 0.25
                candidates.append((score, -run_samples, midpoint))
            run_start = None
    if not candidates:
        return None
    boundary = min(candidates)[2]
    return min(hard_end_sample, max(start_sample + 1, boundary))


def transcribe_long_audio(
    handle: ModelHandle,
    samples: np.ndarray,
    prompt: PromptConfig | None,
    *,
    sample_rate: int = 16000,
    max_new_tokens_per_chunk: int = 0,
    target_seconds: float = 480.0,
    max_seconds: float = 600.0,
    overlap_seconds: float = 1.0,
    split_strategy: str = "vad",
    silence_policy: str = "warn",
    preflight_backend: str = "webrtc",
    vad_aggressiveness: int = 2,
    retry_policy: str = "quality_failure",
    checkpoint_mode: str = "read_write",
    checkpoint_dir: Path | None = None,
    checkpoint_id: str = "",
) -> tuple[TranscriptPayload, dict[str, Any]]:
    array = np.asarray(samples, dtype=np.float32).reshape(-1)
    if sample_rate != 16000:
        raise ValueError("Smart long-audio runtime expects 16 kHz samples.")
    if checkpoint_mode not in {"off", "read_write", "restart"}:
        raise ValueError("checkpoint_mode must be off, read_write, or restart.")
    chunks = plan_audio_chunks(
        array,
        sample_rate,
        target_seconds=target_seconds,
        max_seconds=max_seconds,
        overlap_seconds=overlap_seconds,
        strategy=split_strategy,
        vad_aggressiveness=vad_aggressiveness,
    )
    budgets = [
        int(max_new_tokens_per_chunk)
        if int(max_new_tokens_per_chunk) > 0
        else estimate_max_new_tokens(chunk.duration_seconds)
        for chunk in chunks
    ]
    fingerprint = _job_fingerprint(
        array,
        handle,
        prompt,
        chunks,
        budgets,
        split_strategy=split_strategy,
        overlap_seconds=overlap_seconds,
        silence_policy=silence_policy,
        preflight_backend=preflight_backend,
        vad_aggressiveness=vad_aggressiveness,
        retry_policy=retry_policy,
    )
    checkpoint_path = _checkpoint_path(checkpoint_dir, checkpoint_id, fingerprint, checkpoint_mode)
    total_budget = max(1, sum(budgets))
    progress, cancellation = _comfy_runtime_callbacks(total_budget)
    checkpoint_lock = _acquire_checkpoint_lock(checkpoint_path, cancellation)
    entry = None
    try:
        if checkpoint_mode == "restart" and checkpoint_path is not None and checkpoint_path.exists():
            checkpoint_path.unlink()
        if checkpoint_mode == "read_write":
            completed, checkpoint_status = _load_checkpoint(checkpoint_path, fingerprint)
        elif checkpoint_mode == "restart":
            completed, checkpoint_status = {}, "restarted"
        else:
            completed, checkpoint_status = {}, "off"

        payloads: dict[int, TranscriptPayload] = dict(completed)
        if progress is not None and completed:
            progress(sum(budgets[index - 1] for index in completed if 1 <= index <= len(budgets)))
        for chunk, budget in zip(chunks, budgets, strict=True):
            if cancellation is not None:
                cancellation()
            if chunk.index in payloads:
                continue
            if entry is None:
                entry = MODEL_CACHE.acquire(handle)
            base = sum(budgets[: chunk.index - 1])
            high_water = [base]

            def report_chunk(
                value: int,
                current_total: int,
                *,
                base_value: int = base,
                chunk_budget: int = budget,
                retry_allowed: bool = retry_policy != "never",
            ) -> None:
                resolved_total = max(1, int(current_total))
                clamped_value = min(max(int(value), 0), resolved_total)
                fraction = clamped_value / resolved_total
                if retry_allowed and resolved_total <= chunk_budget:
                    fraction *= 0.5
                completed_units = min(chunk_budget, max(0, round(fraction * chunk_budget)))
                absolute = base_value + completed_units
                high_water[0] = max(high_water[0], absolute)
                if progress is not None:
                    progress(high_water[0])

            payload = run_transcription_samples(
                handle,
                array[chunk.start_sample : chunk.end_sample],
                prompt,
                max_new_tokens=budget,
                silence_policy=silence_policy,
                preflight_backend=preflight_backend,
                vad_aggressiveness=vad_aggressiveness,
                retry_policy=retry_policy,
                progress_callback=report_chunk,
                cancellation_callback=cancellation,
                cache_entry=entry,
            )
            high_water[0] = base + budget
            if progress is not None:
                progress(high_water[0])
            payloads[chunk.index] = payload
            _save_checkpoint(checkpoint_path, fingerprint, chunks, payloads)
    finally:
        try:
            if entry is not None:
                MODEL_CACHE.done(handle, entry)
        finally:
            if checkpoint_lock is not None:
                checkpoint_lock.release()

    ordered_payloads = [payloads[chunk.index] for chunk in chunks]
    merged = merge_chunk_payloads(chunks, ordered_payloads, total_duration=array.size / float(sample_rate))
    merged.metadata.update(
        {
            "mode": "smart_long_audio",
            "split_strategy": split_strategy,
            "target_chunk_seconds": target_seconds,
            "max_chunk_seconds": max_seconds,
            "overlap_seconds": overlap_seconds,
            "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
            "checkpoint_fingerprint": fingerprint,
            "checkpoint_status": checkpoint_status,
            "resumed_chunks": sorted(completed),
        }
    )
    merged.metadata["quality"] = evaluate_quality(merged)
    report = {
        "chunk_count": len(chunks),
        "chunks": [asdict(chunk) for chunk in chunks],
        "resumed_chunks": sorted(completed),
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "checkpoint_status": checkpoint_status,
        "quality": merged.metadata["quality"],
    }
    return merged, report


def merge_chunk_payloads(
    chunks: list[AudioChunk],
    payloads: list[TranscriptPayload],
    *,
    total_duration: float,
) -> TranscriptPayload:
    if len(chunks) != len(payloads):
        raise ValueError("chunks and payloads must have equal length.")
    merged_segments: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    chunk_reports: list[dict[str, Any]] = []
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}

    for chunk, payload in zip(chunks, payloads, strict=True):
        for diagnostic in payload.diagnostics:
            diagnostics.append({**diagnostic, "chunk_id": chunk.chunk_id})
        chunk_reports.append(
            {
                "chunk_id": chunk.chunk_id,
                "start_seconds": chunk.start_seconds,
                "end_seconds": chunk.end_seconds,
                "segment_count": len(payload.segments),
                "metadata": payload.metadata,
            }
        )
        for local_index, item in enumerate(payload.segments, 1):
            candidate = {
                "id": f"{chunk.chunk_id}/seg-{local_index:05d}",
                "start": min(total_duration, chunk.start_seconds + float(item.get("start", 0.0))),
                "end": min(total_duration, chunk.start_seconds + float(item.get("end", 0.0))),
                "speaker": _namespace_speaker(str(item.get("speaker") or "S00"), chunk.index),
                "text": str(item.get("text") or "").strip(),
                "chunk_id": chunk.chunk_id,
                "local_speaker": str(item.get("speaker") or "S00"),
            }
            if candidate["end"] < candidate["start"]:
                candidate["end"] = candidate["start"]
            if _is_overlap_duplicate(candidate, merged_segments, chunks_by_id):
                continue
            merged_segments.append(candidate)

    merged_segments.sort(key=lambda item: (float(item["start"]), float(item["end"]), str(item["id"])))
    raw_text = "".join(
        f"[{float(item['start']):.2f}][{item['speaker']}]{item['text']}[{float(item['end']):.2f}]"
        for item in merged_segments
    )
    diagnostics.insert(
        0,
        {
            "level": "info",
            "code": "smart_long_audio_chunking",
            "message": f"长音频已按 {len(chunks)} 个分片处理；说话人编号按分片命名空间隔离。",
        },
    )
    return TranscriptPayload(
        raw_text=raw_text,
        segments=tuple(merged_segments),
        diagnostics=tuple(diagnostics),
        metadata={
            "audio_duration_seconds": total_duration,
            "sample_rate": 16000,
            "speaker_scope": "chunk_namespaced",
            "chunks": chunk_reports,
            "audio_preflight": _aggregate_audio_preflight(chunks, payloads),
        },
    )


def _aggregate_audio_preflight(
    chunks: list[AudioChunk],
    payloads: list[TranscriptPayload],
) -> dict[str, Any] | None:
    weighted_speech = 0.0
    weighted_seconds = 0.0
    backends: set[str] = set()
    classifications: list[str] = []
    preflight_chunk_count = 0
    for chunk, payload in zip(chunks, payloads, strict=True):
        preflight = payload.metadata.get("audio_preflight")
        if not isinstance(preflight, dict):
            continue
        try:
            speech_ratio = min(1.0, max(0.0, float(preflight["speech_ratio"])))
        except (KeyError, TypeError, ValueError):
            continue
        weight = max(0.0, chunk.duration_seconds)
        weighted_speech += speech_ratio * weight
        weighted_seconds += weight
        preflight_chunk_count += 1
        backend = str(preflight.get("vad_backend") or preflight.get("backend") or "").strip()
        classification = str(preflight.get("classification") or "").strip()
        if backend:
            backends.add(backend)
        if classification:
            classifications.append(classification)
    if weighted_seconds <= 0:
        return None
    speech_ratio = weighted_speech / weighted_seconds
    return {
        "backend": "chunk_aggregate",
        "source_backends": sorted(backends),
        "classification": _aggregate_preflight_classification(speech_ratio, classifications),
        "speech_ratio": speech_ratio,
        "chunk_count": preflight_chunk_count,
    }


def _aggregate_preflight_classification(speech_ratio: float, classifications: list[str]) -> str:
    if classifications and all(item == "silent" for item in classifications):
        return "silent"
    if speech_ratio < 0.02:
        return "non_speech"
    if speech_ratio < 0.10:
        return "mostly_silence"
    return "speech"


def _is_overlap_duplicate(
    candidate: dict[str, Any],
    previous: list[dict[str, Any]],
    chunks_by_id: dict[str, AudioChunk],
) -> bool:
    text = _normalize_text(str(candidate.get("text") or ""))
    if not text:
        return True
    candidate_chunk = chunks_by_id.get(str(candidate.get("chunk_id") or ""))
    if candidate_chunk is None:
        return False
    for item in reversed(previous):
        if item.get("chunk_id") == candidate.get("chunk_id"):
            continue
        previous_chunk = chunks_by_id.get(str(item.get("chunk_id") or ""))
        if previous_chunk is None:
            continue
        if previous_chunk.end_seconds <= candidate_chunk.start_seconds:
            break
        overlap_start = max(previous_chunk.start_seconds, candidate_chunk.start_seconds)
        overlap_end = min(previous_chunk.end_seconds, candidate_chunk.end_seconds)
        if overlap_end <= overlap_start:
            continue
        if not _intersects_window(item, overlap_start, overlap_end):
            continue
        if not _intersects_window(candidate, overlap_start, overlap_end):
            continue
        if not _segments_overlap_in_time(item, candidate):
            continue
        other = _normalize_text(str(item.get("text") or ""))
        if text == other:
            return True
        if min(len(text), len(other)) >= 12 and SequenceMatcher(None, text, other).ratio() >= 0.90:
            return True
    return False


def _intersects_window(segment: dict[str, Any], start: float, end: float) -> bool:
    return float(segment["end"]) >= start and float(segment["start"]) <= end


def _segments_overlap_in_time(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_start, left_end = float(left["start"]), float(left["end"])
    right_start, right_end = float(right["start"]), float(right["end"])
    intersection = min(left_end, right_end) - max(left_start, right_start)
    if intersection > 0:
        shorter_duration = min(max(0.0, left_end - left_start), max(0.0, right_end - right_start))
        return shorter_duration <= 0 or intersection / shorter_duration >= 0.20
    if left_start == left_end and right_start == right_end:
        return abs(left_start - right_start) <= 0.25
    return False


def _normalize_text(text: str) -> str:
    return re.sub(r"[^\w\u3400-\u9fff]+", "", text.casefold())


def _namespace_speaker(speaker: str, chunk_index: int) -> str:
    if speaker == "S00":
        return "S00"
    digits = "".join(ch for ch in speaker if ch.isdigit()) or "0"
    return f"S{chunk_index:03d}{int(digits):03d}"


def _job_fingerprint(
    samples: np.ndarray,
    handle: ModelHandle,
    prompt: PromptConfig | None,
    chunks: list[AudioChunk],
    budgets: list[int],
    **config: Any,
) -> str:
    digest = hashlib.sha256()
    contiguous = np.ascontiguousarray(samples, dtype=np.float32)
    digest.update(memoryview(contiguous).cast("B"))
    digest.update(
        json.dumps(
            {
                "model_revision": handle.model_revision,
                "model_fingerprint": handle.model_fingerprint,
                "long_audio_algorithm_version": LONG_AUDIO_ALGORITHM_VERSION,
                "model_dir": str(handle.model_dir.resolve()),
                "device": handle.device,
                "precision": handle.precision,
                "attention_implementation": handle.attention_implementation,
                "prompt": prompt.text if prompt else "",
                "chunks": [asdict(chunk) for chunk in chunks],
                "budgets": budgets,
                **config,
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _checkpoint_path(
    checkpoint_dir: Path | None,
    checkpoint_id: str,
    fingerprint: str,
    checkpoint_mode: str,
) -> Path | None:
    if checkpoint_mode == "off":
        return None
    if checkpoint_dir is None:
        raise ValueError("checkpoint_dir is required when checkpointing is enabled.")
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", checkpoint_id).strip("._") or fingerprint[:20]
    if len(safe) > MAX_CHECKPOINT_NAME_LENGTH:
        suffix = hashlib.sha256(safe.encode("utf-8")).hexdigest()[:12]
        safe = f"{safe[: MAX_CHECKPOINT_NAME_LENGTH - len(suffix) - 1]}-{suffix}"
    return Path(checkpoint_dir).resolve() / f"{safe}.json"


def _acquire_checkpoint_lock(path: Path | None, cancellation_callback) -> FileLock | None:
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(path) + ".lock")
    while True:
        try:
            lock.acquire(timeout=0.25)
            return lock
        except Timeout:
            if cancellation_callback is not None:
                cancellation_callback()


def _load_checkpoint(path: Path | None, fingerprint: str) -> tuple[dict[int, TranscriptPayload], str]:
    if path is None or not path.exists():
        return {}, "new"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}, "invalid"
    if not isinstance(data, dict) or data.get("schema") != CHECKPOINT_SCHEMA:
        return {}, "invalid"
    if data.get("fingerprint") != fingerprint:
        return {}, "configuration_changed"
    raw_completed = data.get("completed") or {}
    if not isinstance(raw_completed, dict):
        return {}, "invalid"
    completed: dict[int, TranscriptPayload] = {}
    try:
        for key, item in raw_completed.items():
            index = int(key)
            if index <= 0 or not isinstance(item, dict):
                return {}, "invalid"
            segments = item.get("segments") or ()
            diagnostics = item.get("diagnostics") or ()
            metadata = item.get("metadata") or {}
            if not isinstance(segments, (list, tuple)) or not all(isinstance(value, dict) for value in segments):
                return {}, "invalid"
            if not isinstance(diagnostics, (list, tuple)) or not all(
                isinstance(value, dict) for value in diagnostics
            ):
                return {}, "invalid"
            if not isinstance(metadata, dict):
                return {}, "invalid"
            completed[index] = TranscriptPayload(
                raw_text=str(item.get("raw_text") or ""),
                segments=tuple(segments),
                diagnostics=tuple(diagnostics),
                metadata=dict(metadata),
            )
    except (TypeError, ValueError):
        return {}, "invalid"
    indices = sorted(completed)
    if indices and indices != list(range(1, indices[-1] + 1)):
        return {}, "invalid"
    return completed, "resumed" if completed else "empty"


def _save_checkpoint(
    path: Path | None,
    fingerprint: str,
    chunks: list[AudioChunk],
    completed: dict[int, TranscriptPayload],
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "fingerprint": fingerprint,
        "chunks": [asdict(chunk) for chunk in chunks],
        "completed": {str(index): item.to_dict() for index, item in sorted(completed.items())},
    }
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    write_lock = FileLock(str(path) + ".write.lock")
    with write_lock:
        try:
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


__all__ = ["AudioChunk", "merge_chunk_payloads", "plan_audio_chunks", "transcribe_long_audio"]
