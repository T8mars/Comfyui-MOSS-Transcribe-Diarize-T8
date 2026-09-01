from __future__ import annotations

import concurrent.futures
import importlib
import json
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_runtime import PACKAGE_NAME


attention = importlib.import_module(f"{PACKAGE_NAME}.vendor.moss_transcribe_diarize.attention")
audio_preflight = importlib.import_module(f"{PACKAGE_NAME}.vendor.moss_transcribe_diarize.audio_preflight")
inference = importlib.import_module(f"{PACKAGE_NAME}.runtime.inference")
long_audio = importlib.import_module(f"{PACKAGE_NAME}.runtime.long_audio")
nodes_module = importlib.import_module(f"{PACKAGE_NAME}.nodes_v3")
quality = importlib.import_module(f"{PACKAGE_NAME}.runtime.quality")
speaker_mapping = importlib.import_module(f"{PACKAGE_NAME}.vendor.moss_transcribe_diarize.speaker_mapping")
subtitle = importlib.import_module(f"{PACKAGE_NAME}.vendor.moss_transcribe_diarize.subtitle")
types_module = importlib.import_module(f"{PACKAGE_NAME}.runtime.types")
validation_module = importlib.import_module(
    f"{PACKAGE_NAME}.vendor.moss_transcribe_diarize.transcript_validation"
)


def _payload(text: str, segments: tuple[dict, ...], *, duration: float = 10.0):
    return types_module.TranscriptPayload(
        raw_text=text,
        segments=segments,
        diagnostics=(),
        metadata={"audio_duration_seconds": duration, "audio_preflight": {"speech_ratio": 0.8}},
    )


def test_attention_auto_prefers_sdpa_and_explicit_failure_never_falls_back():
    seen = []

    def loader(_path, **kwargs):
        seen.append(kwargs["attn_implementation"])
        config = SimpleNamespace(_attn_implementation="sdpa", text_config=None, audio_config=None)
        return SimpleNamespace(config=config)

    _model, report = attention.load_model_with_attention_fallback(
        ".", device=torch.device("cpu"), dtype=torch.float32, requested="auto", model_loader=loader
    )
    assert seen == ["sdpa"]
    assert report["policy"] == "automatic_fallback"
    assert report["selected"] == "sdpa"
    assert report["attempts"][0]["backend"] == "flash_attention_2"
    assert report["attempts"][0]["status"] == "skipped"

    calls = []

    def failing_loader(_path, **kwargs):
        calls.append(kwargs["attn_implementation"])
        raise RuntimeError("requested backend failed")

    with pytest.raises(RuntimeError, match="No usable attention"):
        attention.load_model_with_attention_fallback(
            ".", device=torch.device("cpu"), dtype=torch.float32, requested="eager", model_loader=failing_loader
        )
    assert calls == ["eager"]


def test_attention_auto_falls_back_to_eager_without_accepting_silent_rewrite():
    calls = []

    def loader(_path, **kwargs):
        requested = kwargs["attn_implementation"]
        calls.append(requested)
        if requested == "sdpa":
            config = SimpleNamespace(_attn_implementation="eager", text_config=None, audio_config=None)
            return SimpleNamespace(config=config)
        config = SimpleNamespace(_attn_implementation="eager", text_config=None, audio_config=None)
        return SimpleNamespace(config=config)

    _model, report = attention.load_model_with_attention_fallback(
        ".", device=torch.device("cpu"), dtype=torch.float32, requested="auto", model_loader=loader
    )

    assert calls == ["sdpa", "eager"]
    assert report["selected"] == "eager"
    assert report["attempts"][-2]["backend"] == "sdpa"
    assert report["attempts"][-2]["status"] == "failed"
    assert report["attempts"][-1] == {"backend": "eager", "status": "selected"}


def test_attention_auto_prefers_flash_attention_2_when_preflight_passes(monkeypatch):
    calls = []
    monkeypatch.setattr(attention, "_flash_preflight", lambda *_args, **_kwargs: None)

    def loader(_path, **kwargs):
        requested = kwargs["attn_implementation"]
        calls.append(requested)
        config = SimpleNamespace(_attn_implementation=requested, text_config=None, audio_config=None)
        return SimpleNamespace(config=config)

    _model, report = attention.load_model_with_attention_fallback(
        ".", device=torch.device("cuda"), dtype=torch.float16, requested="auto", model_loader=loader
    )

    assert calls == ["flash_attention_2"]
    assert report["selected"] == "flash_attention_2"
    assert report["policy"] == "automatic_fallback"


def test_flash_attention_2_rejects_pre_ampere_devices(monkeypatch):
    monkeypatch.setattr(attention, "_device_capability", lambda _device: (7, 5))
    monkeypatch.setattr(attention, "_module_available", lambda _name: True)
    monkeypatch.setattr(attention, "_transformers_flash_available", lambda _implementation: True)

    reason = attention._flash_preflight("flash_attention_2", torch.device("cuda:0"), torch.float16)

    assert reason == "requires compute capability >= 8.x, got (7, 5)"


def test_webrtc_vad_and_energy_fallback_are_observable():
    empty, frame_size, backend = audio_preflight.detect_speech_frames(
        np.zeros(0, dtype=np.float32), 16000, backend="webrtc"
    )
    assert empty.size == 0 and frame_size == 320 and backend == "webrtc"

    time = np.arange(16000, dtype=np.float32) / 16000.0
    waveform = 0.25 * np.sin(2 * np.pi * 180 * time)
    frames, frame_size, backend = audio_preflight.detect_speech_frames(waveform, 16000, backend="webrtc")
    assert backend == "webrtc"
    assert frames.dtype == np.bool_ and frames.size == 50 and frame_size == 320

    energy, _, backend = audio_preflight.detect_speech_frames(waveform, 11025, backend="webrtc")
    assert backend == "energy_fallback" and energy.any()


def test_quality_gate_detects_coverage_unknown_speakers_and_repetition():
    segments = tuple(
        {"start": index * 2.0, "end": index * 2.0 + 1.0, "speaker": "S00", "text": "repeat me"}
        for index in range(4)
    )
    report = quality.evaluate_quality(_payload("", segments, duration=60.0))
    assert report["usable"] is False
    assert {
        "insufficient_end_coverage",
        "too_many_unknown_speakers",
        "repeated_text",
    } <= set(report["reasons"])


def test_quality_gate_enforces_end_coverage_for_short_audio():
    payload = _payload(
        "[0.00][S01]short[1.00]",
        ({"start": 0.0, "end": 1.0, "speaker": "S01", "text": "short"},),
        duration=29.9,
    )

    report = quality.evaluate_quality(payload, min_end_coverage=0.75)

    assert report["usable"] is False
    assert "insufficient_end_coverage" in report["reasons"]


def test_zero_duration_segment_is_invalid_and_unusable():
    result = validation_module.validate_transcript("[10][S01]hello[10]", media_duration=10.0)
    payload = types_module.TranscriptPayload(
        raw_text="[10][S01]hello[10]",
        segments=tuple(
            {"start": item.start, "end": item.end, "speaker": item.speaker, "text": item.text}
            for item in result.segments
        ),
        diagnostics=tuple(
            {"level": item.level, "code": item.code, "message": item.message}
            for item in result.diagnostics
        ),
        metadata={"audio_duration_seconds": 10.0},
    )

    assert result.valid is False
    assert "zero_duration" in {item.code for item in result.diagnostics}
    assert quality.evaluate_quality(payload)["usable"] is False


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_media_duration_is_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        validation_module.validate_transcript("[0][S01]hello[1]", media_duration=value)

    payload = _payload(
        "[0][S01]hello[1]",
        ({"start": 0.0, "end": 1.0, "speaker": "S01", "text": "hello"},),
        duration=value,
    )
    with pytest.raises(ValueError, match="finite"):
        quality.evaluate_quality(payload)


def test_out_of_range_timestamps_are_rejected_and_never_outrank_a_safer_result():
    first = validation_module.validate_transcript(
        "[0.00]accurate words[8.00]",
        media_duration=10.0,
    )
    retry = validation_module.validate_transcript(
        "[0.00][S01]invented words[999.00]",
        media_duration=10.0,
    )
    payload = types_module.TranscriptPayload(
        raw_text="[0.00][S01]invented words[999.00]",
        segments=tuple(
            {"start": item.start, "end": item.end, "speaker": item.speaker, "text": item.text}
            for item in retry.segments
        ),
        diagnostics=tuple(
            {"level": item.level, "code": item.code, "message": item.message}
            for item in retry.diagnostics
        ),
        metadata={"audio_duration_seconds": 10.0},
    )

    assert inference._validation_rank(first) > inference._validation_rank(retry)
    assert inference._retry_reason(retry, "quality_failure") == "quality_failure"
    assert quality.evaluate_quality(payload)["reasons"] == ["timestamp_out_of_range"]


def test_partial_transcript_with_incomplete_tail_is_invalid_and_retried():
    result = validation_module.validate_transcript(
        "[0.00][S01]complete[1.00][1.00][S02]unfinished tail",
        media_duration=2.0,
        generated_tokens=20,
        max_new_tokens=100,
    )

    assert [item.text for item in result.segments] == ["complete"]
    assert result.valid is False
    assert {item.code for item in result.diagnostics} == {"incomplete_segment", "possible_early_stop"}
    assert inference._retry_reason(result, "invalid_format") == "invalid_format"


def test_malformed_content_between_valid_segments_is_not_silently_ignored():
    result = validation_module.validate_transcript(
        "[0.00][S01]first[1.00][broken][1.00][S02]second[2.00]"
    )

    assert [item.text for item in result.segments] == ["first", "second"]
    assert result.valid is False
    assert "invalid_format" in {item.code for item in result.diagnostics}


def test_smart_chunk_planning_merge_and_overlap_deduplication():
    chunks = long_audio.plan_audio_chunks(
        np.ones(5 * 16000, dtype=np.float32),
        16000,
        target_seconds=2.0,
        max_seconds=2.0,
        overlap_seconds=0.5,
        strategy="fixed",
    )
    assert [(item.start_seconds, item.end_seconds) for item in chunks] == [
        (0.0, 2.0),
        (1.5, 3.5),
        (3.0, 5.0),
    ]

    pair = [
        long_audio.AudioChunk(1, "part001", 0, 160000, 0.0, 10.0),
        long_audio.AudioChunk(2, "part002", 144000, 304000, 9.0, 19.0),
    ]
    first = _payload(
        "",
        ({"start": 9.0, "end": 10.0, "speaker": "S01", "text": "same boundary sentence"},),
    )
    second = _payload(
        "",
        (
            {"start": 0.0, "end": 1.0, "speaker": "S02", "text": "same boundary sentence"},
            {"start": 1.2, "end": 2.0, "speaker": "S02", "text": "new content"},
        ),
    )
    merged = long_audio.merge_chunk_payloads(pair, [first, second], total_duration=19.0)
    assert [item["text"] for item in merged.segments] == ["same boundary sentence", "new content"]
    assert [item["speaker"] for item in merged.segments] == ["S001001", "S002002"]


def test_long_audio_quality_aggregates_chunk_speech_ratio():
    chunk = long_audio.AudioChunk(1, "part001", 0, 160000, 0.0, 10.0)
    payload = types_module.TranscriptPayload(
        raw_text="",
        segments=({"start": 0.0, "end": 10.0, "speaker": "S01", "text": "hallucinated speech"},),
        diagnostics=({"level": "warning", "code": "preflight_non_speech", "message": "no speech"},),
        metadata={
            "audio_duration_seconds": 10.0,
            "audio_preflight": {
                "vad_backend": "webrtc",
                "classification": "non_speech",
                "speech_ratio": 0.0,
            },
        },
    )

    merged = long_audio.merge_chunk_payloads([chunk], [payload], total_duration=10.0)
    report = quality.evaluate_quality(merged)

    assert merged.metadata["audio_preflight"]["speech_ratio"] == 0.0
    assert merged.metadata["audio_preflight"]["source_backends"] == ["webrtc"]
    assert merged.metadata["audio_preflight"]["chunk_count"] == 1
    assert report["usable"] is False
    assert "non_speech_hallucination_risk" in report["reasons"]


def test_cross_chunk_speaker_mapping_targets_namespaced_long_audio_speakers():
    chunk = long_audio.AudioChunk(1, "part001", 0, 160000, 0.0, 10.0)
    payload = _payload(
        "",
        ({"start": 0.0, "end": 1.0, "speaker": "S01", "text": "hello"},),
    )
    merged = long_audio.merge_chunk_payloads([chunk], [payload], total_duration=10.0)

    names = speaker_mapping.resolve_speaker_names(
        {},
        {"part001:S01": "Host"},
        segments=merged.segments,
    )

    assert names == {"S001001": "Host"}

    output = nodes_module.T8MossSubtitleExport.execute(
        merged,
        "{}",
        True,
        "moss_transcript",
        False,
        cross_chunk_speaker_map_json='{"part001:S01":"Host"}',
    )
    json_text, txt_text, srt_text, ass_text, files_text, vtt_text, rttm_text = output.result
    json_payload = json.loads(json_text)
    assert json_payload[0]["speaker"] == "S001001"
    assert json_payload[0]["speaker_name"] == "Host"
    assert "[Host]hello" in txt_text
    assert "Host: hello" in srt_text
    assert "Host: hello" in ass_text
    assert "Host: hello" in vtt_text
    assert "S001001" in rttm_text
    assert files_text == "{}"


def test_transcript_validate_preserves_long_audio_provenance_and_risk_diagnostics():
    transcript = types_module.TranscriptPayload(
        raw_text="[0.00][S001001]hello[10.00]",
        segments=(
            {
                "id": "part001/seg-00001",
                "start": 0.0,
                "end": 10.0,
                "speaker": "S001001",
                "text": "hello",
                "chunk_id": "part001",
                "local_speaker": "S01",
            },
        ),
        diagnostics=(
            {"level": "info", "code": "smart_long_audio_chunking", "message": "chunked"},
            {
                "level": "warning",
                "code": "token_limit_reached",
                "message": "truncated",
                "chunk_id": "part001",
            },
        ),
        metadata={"audio_duration_seconds": 10.0, "speaker_scope": "chunk_namespaced"},
    )

    validated = nodes_module.T8MossTranscriptValidate.execute("", 0.0, 0, 0, transcript).result[0]
    report = quality.evaluate_quality(validated)
    exported = nodes_module.T8MossSubtitleExport.execute(
        validated,
        "{}",
        True,
        "moss_transcript",
        False,
        cross_chunk_speaker_map_json='{"part001:S01":"Host"}',
    )

    assert validated.segments == transcript.segments
    assert {item["code"] for item in validated.diagnostics} >= {
        "smart_long_audio_chunking",
        "token_limit_reached",
    }
    assert validated.metadata["possibly_truncated"] is True
    assert report["usable"] is False and "possibly_truncated" in report["reasons"]
    assert json.loads(exported.result[0])[0]["speaker_name"] == "Host"
    assert "Host: hello" in exported.result[2]


def test_transcript_validate_can_be_chained_when_optional_metadata_is_unknown():
    transcript = types_module.TranscriptPayload(
        raw_text="[0.00][S01]hello[1.00]",
        segments=(),
        diagnostics=(),
        metadata={},
    )

    first = nodes_module.T8MossTranscriptValidate.execute("", 0.0, 0, 0, transcript).result[0]
    second = nodes_module.T8MossTranscriptValidate.execute("", 0.0, 0, 0, first).result[0]

    assert second.segments == first.segments
    assert "audio_duration_seconds" not in second.metadata
    assert "generated_tokens" not in second.metadata
    assert "max_new_tokens" not in second.metadata


def test_long_audio_merge_preserves_repeated_phrases_without_chunk_overlap():
    chunks = [
        long_audio.AudioChunk(1, "part001", 0, 160000, 0.0, 10.0),
        long_audio.AudioChunk(2, "part002", 160000, 320000, 10.0, 20.0),
    ]
    first = _payload("", ({"start": 8.0, "end": 9.0, "speaker": "S01", "text": "yes"},))
    second = _payload("", ({"start": 0.0, "end": 1.0, "speaker": "S02", "text": "yes"},))

    merged = long_audio.merge_chunk_payloads(chunks, [first, second], total_duration=20.0)

    assert [item["text"] for item in merged.segments] == ["yes", "yes"]


def test_long_audio_merge_preserves_distinct_repetitions_inside_overlap_window():
    chunks = [
        long_audio.AudioChunk(1, "part001", 0, 160000, 0.0, 10.0),
        long_audio.AudioChunk(2, "part002", 144000, 304000, 9.0, 19.0),
    ]
    first = _payload("", ({"start": 9.0, "end": 9.2, "speaker": "S01", "text": "yes"},))
    second = _payload("", ({"start": 0.7, "end": 0.9, "speaker": "S02", "text": "yes"},))

    merged = long_audio.merge_chunk_payloads(chunks, [first, second], total_duration=19.0)

    assert [item["text"] for item in merged.segments] == ["yes", "yes"]


def test_overlap_deduplication_checks_every_segment_in_the_overlap_window():
    chunks = [
        long_audio.AudioChunk(1, "part001", 0, 320000, 0.0, 20.0),
        long_audio.AudioChunk(2, "part002", 160000, 480000, 10.0, 30.0),
    ]
    first_segments = tuple(
        {
            "start": 10.0 + index * 0.4,
            "end": 10.2 + index * 0.4,
            "speaker": "S01",
            "text": "boundary duplicate" if index == 0 else f"unique {index}",
        }
        for index in range(21)
    )
    payloads = [
        _payload("", first_segments, duration=20.0),
        _payload(
            "",
            ({"start": 0.0, "end": 0.2, "speaker": "S02", "text": "boundary duplicate"},),
            duration=20.0,
        ),
    ]

    merged = long_audio.merge_chunk_payloads(chunks, payloads, total_duration=30.0)

    assert [item["text"] for item in merged.segments].count("boundary duplicate") == 1


def test_long_audio_checkpoint_resumes_after_interruption(monkeypatch, tmp_path: Path):
    samples = np.ones(5 * 16000, dtype=np.float32) * 0.1
    handle = types_module.ModelHandle(tmp_path / "model", "cpu", "float32", model_revision="fixed")
    cache_entry = SimpleNamespace()
    monkeypatch.setattr(long_audio, "_comfy_runtime_callbacks", lambda _total: (None, None))
    monkeypatch.setattr(long_audio.MODEL_CACHE, "acquire", lambda _handle: cache_entry)
    monkeypatch.setattr(long_audio.MODEL_CACHE, "done", lambda *_args: None)
    calls = []

    def interrupted(_handle, chunk, _prompt, **_kwargs):
        calls.append(len(chunk))
        if len(calls) == 2:
            raise RuntimeError("interrupted")
        duration = len(chunk) / 16000.0
        return _payload("", ({"start": 0.0, "end": duration, "speaker": "S01", "text": "chunk one"},), duration=duration)

    monkeypatch.setattr(long_audio, "run_transcription_samples", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        long_audio.transcribe_long_audio(
            handle,
            samples,
            None,
            target_seconds=2.0,
            max_seconds=2.0,
            overlap_seconds=0.0,
            split_strategy="fixed",
            checkpoint_dir=tmp_path,
            checkpoint_id="resume-test",
        )
    assert (tmp_path / "resume-test.json").is_file()

    resumed_calls = []

    def resumed(_handle, chunk, _prompt, **_kwargs):
        resumed_calls.append(len(chunk))
        duration = len(chunk) / 16000.0
        index = len(resumed_calls) + 1
        return _payload(
            "",
            ({"start": 0.0, "end": duration, "speaker": "S01", "text": f"chunk {index}"},),
            duration=duration,
        )

    monkeypatch.setattr(long_audio, "run_transcription_samples", resumed)
    merged, report = long_audio.transcribe_long_audio(
        handle,
        samples,
        None,
        target_seconds=2.0,
        max_seconds=2.0,
        overlap_seconds=0.0,
        split_strategy="fixed",
        checkpoint_dir=tmp_path,
        checkpoint_id="resume-test",
    )
    assert report["resumed_chunks"] == [1]
    assert len(resumed_calls) == 2
    assert len(merged.segments) == 3


def test_long_audio_checkpoint_configuration_change_reprocesses_every_chunk(monkeypatch, tmp_path: Path):
    samples = np.ones(2 * 16000, dtype=np.float32) * 0.1
    model_dir = tmp_path / "model"
    first_handle = types_module.ModelHandle(
        model_dir,
        "cpu",
        "float32",
        model_revision="fixed",
        attention_implementation="sdpa",
    )
    changed_handle = types_module.ModelHandle(
        model_dir,
        "cpu",
        "bfloat16",
        model_revision="fixed",
        attention_implementation="eager",
    )
    monkeypatch.setattr(long_audio, "_comfy_runtime_callbacks", lambda _total: (None, None))
    monkeypatch.setattr(long_audio.MODEL_CACHE, "acquire", lambda _handle: SimpleNamespace())
    monkeypatch.setattr(long_audio.MODEL_CACHE, "done", lambda *_args: None)

    first_calls = []

    def initial(_handle, chunk, _prompt, **kwargs):
        first_calls.append(kwargs["silence_policy"])
        duration = len(chunk) / 16000.0
        return _payload(
            "",
            ({"start": 0.0, "end": duration, "speaker": "S01", "text": "old result"},),
            duration=duration,
        )

    monkeypatch.setattr(long_audio, "run_transcription_samples", initial)
    long_audio.transcribe_long_audio(
        first_handle,
        samples,
        None,
        target_seconds=1.0,
        max_seconds=1.0,
        overlap_seconds=0.0,
        split_strategy="fixed",
        silence_policy="warn",
        preflight_backend="webrtc",
        vad_aggressiveness=2,
        checkpoint_dir=tmp_path,
        checkpoint_id="configuration-change",
    )
    assert first_calls == ["warn", "warn"]

    changed_calls = []

    def changed(_handle, chunk, _prompt, **kwargs):
        changed_calls.append(
            (kwargs["silence_policy"], kwargs["preflight_backend"], kwargs["vad_aggressiveness"])
        )
        duration = len(chunk) / 16000.0
        return _payload(
            "",
            ({"start": 0.0, "end": duration, "speaker": "S01", "text": "new result"},),
            duration=duration,
        )

    monkeypatch.setattr(long_audio, "run_transcription_samples", changed)
    merged, report = long_audio.transcribe_long_audio(
        changed_handle,
        samples,
        None,
        target_seconds=1.0,
        max_seconds=1.0,
        overlap_seconds=0.0,
        split_strategy="fixed",
        silence_policy="reject",
        preflight_backend="energy",
        vad_aggressiveness=3,
        checkpoint_dir=tmp_path,
        checkpoint_id="configuration-change",
    )

    assert changed_calls == [("reject", "energy", 3), ("reject", "energy", 3)]
    assert report["checkpoint_status"] == "configuration_changed"
    assert report["resumed_chunks"] == []
    assert {item["text"] for item in merged.segments} == {"new result"}


def test_long_audio_checkpoint_fingerprint_covers_every_inference_setting(tmp_path: Path, monkeypatch):
    samples = np.ones(16000, dtype=np.float32) * 0.1
    chunks = [long_audio.AudioChunk(1, "part001", 0, 16000, 0.0, 1.0)]
    budgets = [256]
    handle = types_module.ModelHandle(
        tmp_path / "model",
        "cpu",
        "float32",
        model_revision="fixed",
        attention_implementation="sdpa",
    )
    config = {
        "split_strategy": "fixed",
        "overlap_seconds": 0.0,
        "silence_policy": "warn",
        "preflight_backend": "webrtc",
        "vad_aggressiveness": 2,
        "retry_policy": "quality_failure",
    }

    def fingerprint(selected_handle=handle, **overrides):
        return long_audio._job_fingerprint(
            samples,
            selected_handle,
            None,
            chunks,
            budgets,
            **{**config, **overrides},
        )

    baseline = fingerprint()
    assert fingerprint(replace(handle, device="cuda:0")) != baseline
    assert fingerprint(replace(handle, precision="bfloat16")) != baseline
    assert fingerprint(replace(handle, attention_implementation="eager")) != baseline
    assert fingerprint(silence_policy="reject") != baseline
    assert fingerprint(preflight_backend="energy") != baseline
    assert fingerprint(vad_aggressiveness=3) != baseline
    monkeypatch.setattr(long_audio, "LONG_AUDIO_ALGORITHM_VERSION", "t8.moss-long-audio.test-change")
    assert fingerprint() != baseline


@pytest.mark.parametrize(
    "checkpoint",
    [
        "{broken-json",
        json.dumps(
            {
                "schema": long_audio.CHECKPOINT_SCHEMA,
                "fingerprint": "fixed",
                "completed": {
                    "2": {
                        "raw_text": "[0][S01]gap[1]",
                        "segments": [],
                        "diagnostics": [],
                        "metadata": {},
                    }
                },
            }
        ),
    ],
)
def test_invalid_or_noncontiguous_checkpoint_is_safely_reprocessed(tmp_path: Path, checkpoint: str):
    path = tmp_path / "checkpoint.json"
    path.write_text(checkpoint, encoding="utf-8")

    completed, status = long_audio._load_checkpoint(path, "fixed")

    assert completed == {}
    assert status == "invalid"


def test_checkpoint_names_are_bounded_and_atomic_concurrent_writes_do_not_collide(tmp_path: Path):
    fingerprint = "f" * 64
    path = long_audio._checkpoint_path(tmp_path, "x" * 300, fingerprint, "read_write")
    chunk = long_audio.AudioChunk(1, "part001", 0, 16000, 0.0, 1.0)
    barrier = threading.Barrier(8)

    def save(index: int):
        payload = _payload(
            f"[0.00][S01]worker {index}[1.00]",
            ({"start": 0.0, "end": 1.0, "speaker": "S01", "text": f"worker {index}"},),
            duration=1.0,
        )
        barrier.wait()
        long_audio._save_checkpoint(path, f"fingerprint-{index}", [chunk], {1: payload})

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(save, range(8)))

    assert len(path.name) <= long_audio.MAX_CHECKPOINT_NAME_LENGTH + len(".json")
    assert json.loads(path.read_text(encoding="utf-8"))["schema"] == long_audio.CHECKPOINT_SCHEMA
    assert not list(tmp_path.glob("*.tmp"))


def test_checkpoint_job_lock_serializes_same_checkpoint(tmp_path: Path):
    path = tmp_path / "shared.json"
    first = long_audio._acquire_checkpoint_lock(path, None)
    attempted = threading.Event()
    acquired = threading.Event()

    def acquire_second():
        attempted.set()
        second = long_audio._acquire_checkpoint_lock(path, None)
        acquired.set()
        second.release()

    thread = threading.Thread(target=acquire_second)
    thread.start()
    assert attempted.wait(1.0)
    assert not acquired.wait(0.3)
    first.release()
    assert acquired.wait(2.0)
    thread.join(timeout=2.0)
    assert not thread.is_alive()


def test_long_subtitle_filename_prefix_is_safely_bounded(monkeypatch, tmp_path: Path):
    folder_paths = importlib.import_module("folder_paths")
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    transcript = _payload(
        "[0.00][S01]hello[1.00]",
        ({"start": 0.0, "end": 1.0, "speaker": "S01", "text": "hello"},),
        duration=1.0,
    )

    output = nodes_module.T8MossSubtitleExport.execute(
        transcript,
        '{"S01":"Host"}',
        True,
        "x" * 300,
        True,
    )
    files = json.loads(output.result[4])

    assert set(files) == {"json", "txt", "srt", "ass", "vtt", "rttm"}
    assert all(Path(path).is_file() for path in files.values())
    assert all(len(Path(path).name) < 255 for path in files.values())


def test_invalid_format_is_retried_once_and_better_result_is_selected(monkeypatch, tmp_path: Path):
    invalid = validation_module.TranscriptValidation(
        False,
        (),
        (validation_module.TranscriptDiagnostic("error", "invalid_format", "bad"),),
        False,
    )
    valid_segment = SimpleNamespace(start=0.0, end=1.0, speaker="S01", text="hello")
    valid = validation_module.TranscriptValidation(True, (valid_segment,), (), False)
    results = [
        ({"text": "bad", "prompt_len": 4, "generated_tokens": 2}, invalid),
        ({"text": "[0.00][S01]hello[1.00]", "prompt_len": 8, "generated_tokens": 6}, valid),
    ]
    prompts = []

    def generate(_entry, _samples, prompt_text, **kwargs):
        prompts.append(prompt_text)
        kwargs["progress_callback"](
            kwargs["progress_offset"] + kwargs["token_budget"],
            kwargs["progress_total"],
        )
        return results.pop(0)

    monkeypatch.setattr(inference, "_generate_and_validate", generate)
    entry = SimpleNamespace(
        lock=threading.Lock(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        attention_report={"selected": "sdpa"},
    )
    handle = types_module.ModelHandle(tmp_path, "cpu", "float32")
    progress = []
    payload = inference.run_transcription_samples(
        handle,
        np.ones(16000, dtype=np.float32) * 0.1,
        None,
        max_new_tokens=16,
        silence_policy="ignore",
        retry_policy="invalid_format",
        progress_callback=lambda value, total: progress.append((value, total)),
        cancellation_callback=lambda: False,
        cache_entry=entry,
    )
    assert len(prompts) == 2 and inference.STRICT_RETRY_SUFFIX in prompts[1]
    assert payload.metadata["retry"]["selected"] is True
    assert payload.raw_text.endswith("hello[1.00]")
    assert progress[-1] == (32, 32)
    assert progress == sorted(progress)
    assert progress[0][0] < progress[0][1]


def test_success_without_retry_reserves_retry_progress_capacity(monkeypatch, tmp_path: Path):
    valid_segment = SimpleNamespace(start=0.0, end=1.0, speaker="S01", text="hello")
    valid = validation_module.TranscriptValidation(True, (valid_segment,), (), False)

    def generate(_entry, _samples, _prompt_text, **kwargs):
        kwargs["progress_callback"](4, kwargs["progress_total"])
        return {"text": "[0.00][S01]hello[1.00]", "prompt_len": 4, "generated_tokens": 4}, valid

    monkeypatch.setattr(inference, "_generate_and_validate", generate)
    entry = SimpleNamespace(
        lock=threading.Lock(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        attention_report={"selected": "sdpa"},
    )
    progress = []
    handle = types_module.ModelHandle(tmp_path, "cpu", "float32")

    inference.run_transcription_samples(
        handle,
        np.ones(16000, dtype=np.float32) * 0.1,
        None,
        max_new_tokens=16,
        silence_policy="ignore",
        retry_policy="invalid_format",
        progress_callback=lambda value, total: progress.append((value, total)),
        cancellation_callback=lambda: False,
        cache_entry=entry,
    )

    assert progress == [(4, 32), (32, 32)]


def test_token_limit_quality_failure_triggers_retry():
    validation = validation_module.validate_transcript(
        "[0.00][S01]partial[10.00]",
        media_duration=10.0,
        generated_tokens=128,
        max_new_tokens=128,
    )

    assert inference._retry_reason(validation, "quality_failure") == "quality_failure"


def test_long_audio_retry_progress_does_not_finish_before_retry_work(monkeypatch, tmp_path: Path):
    progress = []
    monkeypatch.setattr(long_audio, "_comfy_runtime_callbacks", lambda _total: (progress.append, None))
    monkeypatch.setattr(long_audio.MODEL_CACHE, "acquire", lambda _handle: SimpleNamespace())
    monkeypatch.setattr(long_audio.MODEL_CACHE, "done", lambda *_args: None)

    def retrying(_handle, samples, _prompt, **kwargs):
        budget = kwargs["max_new_tokens"]
        callback = kwargs["progress_callback"]
        callback(budget, budget)
        assert progress[-1] < budget
        callback(budget + budget // 2, budget * 2)
        assert progress[-1] < budget
        callback(budget * 2, budget * 2)
        return _payload(
            "[0.00][S01]hello[1.00]",
            ({"start": 0.0, "end": 1.0, "speaker": "S01", "text": "hello"},),
            duration=len(samples) / 16000.0,
        )

    monkeypatch.setattr(long_audio, "run_transcription_samples", retrying)
    handle = types_module.ModelHandle(tmp_path, "cpu", "float32", model_revision="fixed")

    long_audio.transcribe_long_audio(
        handle,
        np.ones(16000, dtype=np.float32),
        None,
        max_new_tokens_per_chunk=16,
        target_seconds=2.0,
        max_seconds=2.0,
        overlap_seconds=0.0,
        split_strategy="fixed",
        checkpoint_mode="off",
    )

    assert progress[-1] == 16
    assert progress == sorted(progress)


def test_retry_ranking_prefers_clean_speaker_labels_and_no_generation_loop():
    risky_segments = tuple(
        SimpleNamespace(start=float(index), end=float(index + 1), speaker="S00", text="repeat")
        for index in range(4)
    )
    risky = validation_module.TranscriptValidation(
        True,
        risky_segments,
        (
            validation_module.TranscriptDiagnostic("warning", "speaker_tag_missing", "missing"),
            validation_module.TranscriptDiagnostic("warning", "repeated_text", "loop"),
        ),
        False,
    )
    clean = validation_module.TranscriptValidation(
        True,
        (SimpleNamespace(start=0.0, end=3.0, speaker="S01", text="clean"),),
        (),
        False,
    )
    assert inference._validation_rank(clean) > inference._validation_rank(risky)


def test_overlap_deduplication_never_removes_repeated_speech_inside_one_chunk():
    chunk = long_audio.AudioChunk(1, "part001", 0, 48000, 0.0, 3.0)
    payload = _payload(
        "",
        (
            {"start": 0.0, "end": 1.0, "speaker": "S01", "text": "yes, yes"},
            {"start": 1.1, "end": 2.0, "speaker": "S01", "text": "yes, yes"},
        ),
        duration=3.0,
    )
    merged = long_audio.merge_chunk_payloads([chunk], [payload], total_duration=3.0)
    assert len(merged.segments) == 2


def test_subtitle_style_controls_resolution_and_auto_font_size():
    segment = subtitle.SubtitleSegment("1", 0.0, 1.0, "S01", "Hello")
    style = subtitle.SubtitleStyle(font_name="Inter", font_size=None, video_width=1280, video_height=720)
    ass = subtitle.export_ass([segment], style=style)
    assert "PlayResX: 1280" in ass and "PlayResY: 720" in ass
    assert "Style: Default,Inter,32" in ass


def test_subtitle_export_hides_speaker_in_txt_when_requested():
    transcript = _payload(
        "[0.00][S01]hello[1.00]",
        ({"start": 0.0, "end": 1.0, "speaker": "S01", "text": "hello"},),
        duration=1.0,
    )

    output = nodes_module.T8MossSubtitleExport.execute(
        transcript,
        '{"S01":"Host"}',
        False,
        "moss_transcript",
        False,
    )

    assert output.result[1] == "[0.00]hello[1.00]\n"
    assert "Host" not in output.result[2]
    assert "Host" not in output.result[3]


@pytest.mark.parametrize(
    "style, message",
    [
        (subtitle.SubtitleStyle(font_name="Arial,Injected"), "font_name"),
        (subtitle.SubtitleStyle(font_name="Arial\nStyle: Injected"), "font_name"),
        (subtitle.SubtitleStyle(primary_color="#ffffff"), "primary_color"),
    ],
)
def test_ass_style_rejects_fields_that_can_break_the_file(style, message):
    segment = subtitle.SubtitleSegment("1", 0.0, 1.0, "S01", "Hello")

    with pytest.raises(ValueError, match=message):
        subtitle.export_ass([segment], style=style)


@pytest.mark.parametrize("line_break", ["\n", "\r\n", "\r"])
def test_ass_export_normalizes_every_line_break(line_break):
    segment = subtitle.SubtitleSegment("1", 0.0, 1.0, "S01", f"Hello{line_break}Dialogue: injected")

    ass = subtitle.export_ass([segment])
    dialogue_lines = [line for line in ass.splitlines() if line.startswith("Dialogue:")]

    assert len(dialogue_lines) == 1
    assert "\\NDialogue: injected" in dialogue_lines[0]
    assert "\r" not in ass


def test_subtitle_splitting_never_expands_original_time_range():
    original = subtitle.SubtitleSegment("1", 0.0, 0.1, "S01", "abcdefghijklmnopqrstuvwxyz")

    segments = subtitle.normalize_segments([original], min_duration=1.0, max_duration=6.0, max_chars=5)

    assert len(segments) > 1
    assert segments[0].start == pytest.approx(0.0)
    assert segments[-1].end == pytest.approx(0.1)
    assert all(0.0 <= item.start <= item.end <= 0.1 for item in segments)


def test_transcript_revalidation_drops_stale_top_level_diagnostics_but_keeps_chunk_diagnostics():
    source = "[0][S01]hello[8]"
    transcript = types_module.TranscriptPayload(
        raw_text=source,
        segments=({"id": "original", "start": 0.0, "end": 5.0, "speaker": "S01", "text": "hello"},),
        diagnostics=(
            {"level": "warning", "code": "timestamp_out_of_range", "message": "stale"},
            {
                "level": "warning",
                "code": "timestamp_out_of_range",
                "message": "chunk evidence",
                "chunk_id": "part001",
            },
        ),
        metadata={"audio_duration_seconds": 5.0},
    )

    output = nodes_module.T8MossTranscriptValidate.execute(source, 10.0, 0, 0, transcript).result[0]
    diagnostics = list(output.diagnostics)

    assert output.segments[0]["end"] == pytest.approx(8.0)
    assert not any(item.get("message") == "stale" for item in diagnostics)
    assert any(item.get("chunk_id") == "part001" for item in diagnostics)


def test_quality_gate_can_stop_unusable_results():
    transcript = _payload(
        "[0][S01]short[1]",
        ({"start": 0.0, "end": 1.0, "speaker": "S01", "text": "short"},),
        duration=10.0,
    )

    output = nodes_module.T8MossQualityGate.execute(transcript, 0.75, 0.5, True, True, False)
    assert output.result[1] is False
    with pytest.raises(ValueError, match="质量门拒绝"):
        nodes_module.T8MossQualityGate.execute(transcript, 0.75, 0.5, True, True, True)


def test_model_loader_reports_requested_and_effective_cpu_precision(monkeypatch, tmp_path: Path):
    report = SimpleNamespace(require_valid=lambda: None)
    monkeypatch.setattr(nodes_module, "resolve_model", lambda *_args: tmp_path)
    monkeypatch.setattr(nodes_module, "validate_model_dir", lambda *_args, **_kwargs: report)
    monkeypatch.setattr(nodes_module, "_resolve_device", lambda _requested: "cpu")
    monkeypatch.setattr(nodes_module, "load_manifest", lambda: {"revision": "fixed-revision"})
    monkeypatch.setattr(nodes_module, "model_fingerprint", lambda _path: "fingerprint")

    output = nodes_module.T8MossModelLoader.execute("model", "cpu", "float16", False, False)

    assert "precision=float16 -> dtype=torch.float32" in output.result[1]
