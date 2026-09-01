from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from test_runtime import PACKAGE_NAME


inference = importlib.import_module(f"{PACKAGE_NAME}.runtime.inference")
long_audio = importlib.import_module(f"{PACKAGE_NAME}.runtime.long_audio")
nodes = importlib.import_module(f"{PACKAGE_NAME}.nodes_v3")
remote = importlib.import_module(f"{PACKAGE_NAME}.runtime.remote")
subtitle = importlib.import_module(f"{PACKAGE_NAME}.vendor.moss_transcribe_diarize.subtitle")
types_module = importlib.import_module(f"{PACKAGE_NAME}.runtime.types")


def _payload(segments: tuple[dict, ...], *, duration: float = 20.0):
    return types_module.TranscriptPayload(
        raw_text="",
        segments=segments,
        diagnostics=(),
        metadata={"audio_duration_seconds": duration, "audio_preflight": {"speech_ratio": 0.8}},
    )


def test_language_hints_support_english_defaults_and_custom_labels():
    english = inference.build_prompt("", "OpenMOSS,T8star-Aix", "English", True)
    assert "transcribe the audio" in english.text.lower()
    assert "Primary language: English" in english.text
    assert "preserve the supplied spelling and capitalization" in english.text

    custom = inference.build_prompt("", "", "自定义 / Custom", True, custom_language_hint="pt-BR")
    assert custom.language_hint == "pt-BR"
    assert "Primary language: pt-BR" in custom.text

    with pytest.raises(ValueError, match="自定义语言"):
        inference.build_prompt("", "", "自定义 / Custom", True)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("http://127.0.0.1:8000", "http://127.0.0.1:8000/v1/audio/transcriptions"),
        ("https://moss.example/api", "https://moss.example/api/v1/audio/transcriptions"),
        ("https://moss.example/v1/audio/transcriptions", "https://moss.example/v1/audio/transcriptions"),
    ],
)
def test_remote_endpoint_normalization(value: str, expected: str):
    assert remote.normalize_endpoint_url(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "http://moss.example:8000",
        "ftp://127.0.0.1/model",
        "https://user:secret@moss.example",
        "https://moss.example/path?token=secret",
        "https://moss.example:not-a-port",
    ],
)
def test_remote_endpoint_rejects_insecure_or_secret_bearing_urls(value: str):
    with pytest.raises(ValueError):
        remote.normalize_endpoint_url(value)


def test_remote_request_posts_pcm_wav_and_fixed_environment_secret(monkeypatch, tmp_path: Path):
    captured = {}

    class Response:
        status = 200
        reason = "OK"

        @staticmethod
        def getheader(_name):
            return None

        @staticmethod
        def read(_amount):
            return json.dumps(
                {"text": "[0.00][S01]hello[1.00]", "usage": {"input_tokens": 7, "output_tokens": 9}}
            ).encode()

    class Connection:
        def __init__(self, host, port, timeout):
            captured.update(host=host, port=port, timeout=timeout)

        def request(self, method, path, body, headers):
            captured.update(method=method, path=path, body=body, headers=headers)

        @staticmethod
        def getresponse():
            return Response()

        @staticmethod
        def close():
            return None

    monkeypatch.setattr(remote, "HTTPConnection", Connection)
    monkeypatch.setenv(remote.REMOTE_API_KEY_ENV, "test-token")
    handle = types_module.ModelHandle(
        tmp_path,
        "remote",
        "server_managed",
        backend="openai_compatible",
        endpoint_url="http://127.0.0.1:8000",
        remote_model="test/model",
        timeout_seconds=42.0,
    )

    result = remote.request_remote_transcription(
        handle,
        np.array([-2.0, 0.0, 2.0], dtype=np.float32),
        sample_rate=16000,
        prompt="prompt",
        language="English",
        max_new_tokens=128,
    )

    assert captured["method"] == "POST"
    assert captured["path"] == "/v1/audio/transcriptions"
    assert captured["headers"]["Authorization"] == "Bearer test-token"
    assert b"RIFF" in captured["body"]
    assert b"test/model" in captured["body"]
    assert b'name="language"\r\n\r\nen' in captured["body"]
    assert result["text"].startswith("[0.00]")
    assert result["prompt_len"] == 7
    assert result["generated_tokens"] == 9
    assert "test-token" not in json.dumps(result)
    assert remote.normalize_remote_language("中文") == "zh"
    assert remote.normalize_remote_language("pt-BR") == "pt-BR"
    assert remote.normalize_remote_language("auto") == ""


def test_remote_loader_requires_explicit_audio_upload_consent():
    with pytest.raises(ValueError, match="未授权"):
        nodes.T8MossRemoteModelLoader.execute("http://127.0.0.1:8000", "test/model", 30.0, False)

    output = nodes.T8MossRemoteModelLoader.execute(
        "http://127.0.0.1:8000", "test/model", 30.0, True
    )
    handle = output.result[0]
    assert handle.is_remote
    assert handle.effective_memory_policy == "server_managed"
    assert json.loads(output.result[1])["audio_leaves_comfyui_host"] is True

    with pytest.raises(ValueError, match="超时"):
        nodes.T8MossRemoteModelLoader.execute(
            "http://127.0.0.1:8000", "test/model", float("nan"), True
        )


def test_remote_transcription_never_acquires_local_model_cache(monkeypatch, tmp_path: Path):
    handle = types_module.ModelHandle(
        tmp_path,
        "remote",
        "server_managed",
        backend="openai_compatible",
        endpoint_url="http://127.0.0.1:8000",
        remote_model="test/model",
    )
    monkeypatch.setattr(
        inference,
        "request_remote_transcription",
        lambda *_args, **_kwargs: {
            "text": "[0.00][S01]hello world[1.00]",
            "prompt_len": 3,
            "generated_tokens": 8,
            "remote": {"backend": "openai_compatible"},
        },
    )
    monkeypatch.setattr(
        inference.MODEL_CACHE,
        "acquire",
        lambda _handle: (_ for _ in ()).throw(AssertionError("local cache must not be acquired")),
    )

    payload = inference.run_transcription_samples(
        handle,
        np.full(16000, 0.2, dtype=np.float32),
        types_module.PromptConfig("prompt", language_hint="English"),
        max_new_tokens=64,
        silence_policy="ignore",
        retry_policy="never",
    )

    assert payload.segments[0]["speaker"] == "S01"
    assert payload.metadata["backend"] == "openai_compatible"
    assert payload.metadata["remote"]["backend"] == "openai_compatible"


def test_remote_transcription_uses_shared_strict_retry_selection(monkeypatch, tmp_path: Path):
    handle = types_module.ModelHandle(
        tmp_path,
        "remote",
        "server_managed",
        backend="openai_compatible",
        endpoint_url="http://127.0.0.1:8000",
        remote_model="test/model",
    )
    responses = iter(("not a transcript", "[0.00][S01]recovered text[1.00]"))
    monkeypatch.setattr(
        inference,
        "request_remote_transcription",
        lambda *_args, **_kwargs: {
            "text": next(responses),
            "prompt_len": 1,
            "generated_tokens": 8,
            "remote": {"backend": "openai_compatible"},
        },
    )
    progress = []
    payload = inference.run_transcription_samples(
        handle,
        np.full(16000, 0.2, dtype=np.float32),
        types_module.PromptConfig("prompt"),
        max_new_tokens=64,
        silence_policy="ignore",
        retry_policy="invalid_format",
        progress_callback=lambda value, total: progress.append((value, total)),
        cancellation_callback=lambda: False,
    )
    assert payload.raw_text.endswith("[1.00]")
    assert payload.metadata["retry"] == {
        "policy": "invalid_format",
        "attempted": True,
        "reason": "invalid_format",
        "selected": True,
        "first_generated_tokens": 8,
        "retry_generated_tokens": 8,
    }
    assert progress[-1] == (128, 128)


def test_subtitle_postprocess_wraps_lines_and_reports_reading_speed():
    segment = subtitle.SubtitleSegment("seg-1", 0.0, 1.0, "S01", "abcdefghijklmnopqrst")
    processed = subtitle.postprocess_subtitle_segments(
        [segment],
        min_duration=0.1,
        max_duration=10.0,
        max_chars_per_line=5,
        max_lines=2,
        merge_gap=0.0,
    )

    assert len(processed) == 2
    assert all(len(line) <= 5 for item in processed for line in item.text.splitlines())
    report = subtitle.subtitle_readability_report(processed, max_chars_per_second=2.0)
    assert report["violation_count"] == 2

    duration_limited = subtitle.postprocess_subtitle_segments(
        [subtitle.SubtitleSegment("seg-2", 0.0, 20.0, "S01", "hello")],
        min_duration=0.1,
        max_duration=6.0,
        max_chars_per_line=20,
        max_lines=2,
        merge_gap=0.0,
    )
    assert len(duration_limited) == 4
    assert all(item.end - item.start <= 6.0 for item in duration_limited)

    zero_duration = subtitle.subtitle_readability_report(
        [subtitle.SubtitleSegment("zero", 1.0, 1.0, "S01", "text")]
    )
    assert zero_duration["violations"][0]["reason"] == "zero_duration"
    assert "Infinity" not in json.dumps(zero_duration)

    node_output = nodes.T8MossSubtitlePostprocess.execute(
        _payload(({"start": 0.0, "end": 1.0, "speaker": "S01", "text": "abcdefghijklmnopqrst"},)),
        0.1,
        10.0,
        5,
        2,
        0.0,
        2.0,
    )
    assert json.loads(node_output.result[1])["readability"]["violation_count"] == 2
    assert "quality" in node_output.result[0].metadata


def test_subtitle_exports_webvtt_and_rttm():
    segments = [subtitle.SubtitleSegment("seg-1", 1.25, 3.5, "S01", "hello")]
    vtt = subtitle.export_vtt(segments, speaker_names={"S01": "Host"})
    rttm = subtitle.export_rttm(segments, file_id="meeting 01")

    assert vtt.startswith("WEBVTT\n")
    assert "00:00:01.250 --> 00:00:03.500" in vtt
    assert "Host: hello" in vtt
    assert rttm == "SPEAKER meeting_01 1 1.250 2.250 <NA> <NA> S01 <NA> <NA>\n"


def test_overlap_only_speaker_linking_uses_duplicate_boundary_evidence():
    chunks = [
        long_audio.AudioChunk(1, "part001", 0, 160000, 0.0, 10.0),
        long_audio.AudioChunk(2, "part002", 128000, 288000, 8.0, 18.0),
    ]
    payloads = [
        _payload(({"start": 8.0, "end": 9.0, "speaker": "S01", "text": "same overlap"},)),
        _payload(
            (
                {"start": 0.0, "end": 1.0, "speaker": "S02", "text": "same overlap"},
                {"start": 1.2, "end": 2.0, "speaker": "S02", "text": "continuation"},
            )
        ),
    ]

    isolated = long_audio.merge_chunk_payloads(chunks, payloads, total_duration=18.0)
    linked = long_audio.merge_chunk_payloads(
        chunks, payloads, total_duration=18.0, speaker_link_mode="overlap_only"
    )

    assert isolated.segments[-1]["speaker"] == "S002002"
    assert linked.segments[-1]["speaker"] == "S001001"
    assert linked.metadata["speaker_links"][0]["method"] == "overlap_text_time"
    assert linked.metadata["speaker_scope"] == "chunk_namespaced_with_overlap_links"


def test_short_overlap_text_does_not_link_speakers():
    chunks = [
        long_audio.AudioChunk(1, "part001", 0, 160000, 0.0, 10.0),
        long_audio.AudioChunk(2, "part002", 128000, 288000, 8.0, 18.0),
    ]
    payloads = [
        _payload(({"start": 8.0, "end": 9.0, "speaker": "S01", "text": "yes"},)),
        _payload(
            (
                {"start": 0.0, "end": 1.0, "speaker": "S02", "text": "yes"},
                {"start": 1.2, "end": 2.0, "speaker": "S02", "text": "next sentence"},
            )
        ),
    ]

    linked = long_audio.merge_chunk_payloads(
        chunks, payloads, total_duration=18.0, speaker_link_mode="overlap_only"
    )
    assert linked.segments[-1]["speaker"] == "S002002"
    assert linked.metadata["speaker_links"] == []


def test_speaker_linking_refuses_many_local_speakers_to_one_target():
    chunks = [
        long_audio.AudioChunk(1, "part001", 0, 160000, 0.0, 10.0),
        long_audio.AudioChunk(2, "part002", 128000, 288000, 8.0, 18.0),
    ]
    payloads = [
        _payload(
            (
                {"start": 8.0, "end": 8.8, "speaker": "S01", "text": "alpha overlap"},
                {"start": 8.9, "end": 9.7, "speaker": "S01", "text": "beta overlap"},
            )
        ),
        _payload(
            (
                {"start": 0.0, "end": 0.8, "speaker": "S02", "text": "alpha overlap"},
                {"start": 0.9, "end": 1.7, "speaker": "S03", "text": "beta overlap"},
                {"start": 2.1, "end": 2.8, "speaker": "S02", "text": "speaker two"},
                {"start": 2.9, "end": 3.6, "speaker": "S03", "text": "speaker three"},
            )
        ),
    ]

    linked = long_audio.merge_chunk_payloads(
        chunks, payloads, total_duration=18.0, speaker_link_mode="overlap_only"
    )
    assert {item["speaker"] for item in linked.segments[-2:]} == {"S002002", "S002003"}
    assert linked.metadata["speaker_links"] == []
    assert {item["reason"] for item in linked.metadata["speaker_link_conflicts"]} == {
        "multiple_local_speakers_target_same_speaker"
    }

    with pytest.raises(ValueError, match="speaker_link_mode"):
        long_audio.merge_chunk_payloads(chunks, payloads, total_duration=18.0, speaker_link_mode="unsafe")


def test_benchmark_expectation_evaluator_reports_regressions():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_benchmarks.py"
    spec = importlib.util.spec_from_file_location("t8_benchmark_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    result = {
        "metrics": {
            "segment_count": 1,
            "speaker_count": 1,
            "real_time_factor": 2.0,
            "peak_vram_gb": 12.5,
            "word_error_rate": 0.5,
            "word_alignment_coverage": 0.4,
            "speaker_embedding_links": 0,
            "speaker_embedding_failures": 2,
        },
        "quality": {"usable": False, "end_coverage": 0.5},
        "diagnostic_codes": ["repeated_text"],
        "content_checks": {"missing_required_text": ["OpenMOSS"]},
    }
    failures = module.evaluate_expectations(
        result,
        {
            "min_segments": 2,
            "min_speakers": 2,
            "usable": True,
            "min_end_coverage": 0.9,
            "forbidden_diagnostic_codes": ["repeated_text"],
            "max_real_time_factor": 1.0,
            "max_peak_vram_gb": 10.0,
            "max_word_error_rate": 0.2,
            "max_character_error_rate": 0.2,
            "min_word_alignment_coverage": 0.7,
            "min_speaker_embedding_links": 1,
            "max_speaker_embedding_failures": 0,
        },
    )
    assert len(failures) == 13
    assert module.transcript_error_rates("hello brave world", "hello world") == {
        "word_error_rate": 0.5,
        "character_error_rate": 0.5,
    }
