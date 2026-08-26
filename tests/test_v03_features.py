from __future__ import annotations

import importlib
import threading
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
quality = importlib.import_module(f"{PACKAGE_NAME}.runtime.quality")
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


def test_attention_auto_omits_override_and_explicit_failure_never_falls_back():
    seen = []

    def loader(_path, **kwargs):
        seen.append(kwargs.get("attn_implementation", "omitted"))
        config = SimpleNamespace(_attn_implementation="sdpa", text_config=None, audio_config=None)
        return SimpleNamespace(config=config)

    _model, report = attention.load_model_with_attention_fallback(
        ".", device=torch.device("cpu"), dtype=torch.float32, requested="auto", model_loader=loader
    )
    assert seen == ["omitted"]
    assert report["policy"] == "upstream_default"
    assert report["selected"] == "sdpa"

    calls = []

    def failing_loader(_path, **kwargs):
        calls.append(kwargs["attn_implementation"])
        raise RuntimeError("requested backend failed")

    with pytest.raises(RuntimeError, match="No usable attention"):
        attention.load_model_with_attention_fallback(
            ".", device=torch.device("cpu"), dtype=torch.float32, requested="eager", model_loader=failing_loader
        )
    assert calls == ["eager"]


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

    def generate(_entry, _samples, prompt_text, **_kwargs):
        prompts.append(prompt_text)
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
