from __future__ import annotations

import importlib

import numpy as np

from test_runtime import PACKAGE_NAME


alignment = importlib.import_module(f"{PACKAGE_NAME}.runtime.alignment")
speaker_embeddings = importlib.import_module(f"{PACKAGE_NAME}.runtime.speaker_embeddings")
types_module = importlib.import_module(f"{PACKAGE_NAME}.runtime.types")


def _payload(segments: tuple[dict, ...], duration: float = 4.0):
    return types_module.TranscriptPayload(
        raw_text="",
        segments=segments,
        diagnostics=(),
        metadata={"audio_duration_seconds": duration},
    )


def test_word_alignment_uses_model_anchors_and_marks_interpolated_units():
    payload = _payload(
        (
            {
                "id": "seg-1",
                "start": 0.0,
                "end": 2.0,
                "speaker": "S01",
                "text": "Hello 世界!",
            },
        ),
        2.0,
    )
    aligned, report = alignment.align_transcript_words(
        payload,
        [
            {"word": "Hello", "start": 0.10, "end": 0.60, "confidence": 0.9},
            {"word": "世", "start": 0.80, "end": 1.05, "confidence": 0.8},
            {"word": "界", "start": 1.05, "end": 1.30, "confidence": 0.85},
        ],
        model_id="test/whisper",
        revision="abc",
    )

    words = aligned.segments[0]["words"]
    assert [item["word"] for item in words] == ["Hello", "世", "界", "!"]
    assert [item["source"] for item in words] == [
        "alignment_model",
        "alignment_model",
        "alignment_model",
        "interpolated_between_model_anchors",
    ]
    assert all(0.0 <= item["start"] <= item["end"] <= 2.0 for item in words)
    assert all(words[index]["end"] <= words[index + 1]["start"] for index in range(len(words) - 1))
    assert report["model_matched_units"] == 3
    assert report["fallback_units"] == 1
    assert aligned.metadata["word_alignment"] == report


def test_word_alignment_surfaces_low_model_match_coverage():
    payload = _payload(
        ({"start": 0.0, "end": 1.0, "speaker": "S01", "text": "one two three four"},),
        1.0,
    )
    aligned, report = alignment.align_transcript_words(
        payload,
        [{"word": "one", "start": 0.0, "end": 0.2}],
        model_id="test/whisper",
    )
    assert report["model_match_coverage"] == 0.25
    assert any(item.get("code") == "word_alignment_low_coverage" for item in aligned.diagnostics)


def test_voice_embeddings_link_across_chunks_without_same_chunk_collision():
    sample_rate = 16000
    audio = np.zeros(sample_rate * 4, dtype=np.float32)
    audio[0:sample_rate] = 0.1
    audio[sample_rate : sample_rate * 2] = 0.1
    audio[sample_rate * 2 : sample_rate * 3] = 0.9
    payload = _payload(
        (
            {"id": "a", "start": 0.0, "end": 1.0, "speaker": "S001001", "text": "first", "chunk_id": "part001"},
            {"id": "b", "start": 1.0, "end": 2.0, "speaker": "S002001", "text": "same", "chunk_id": "part002"},
            {"id": "c", "start": 2.0, "end": 3.0, "speaker": "S002002", "text": "other", "chunk_id": "part002"},
        )
    )
    handle = speaker_embeddings.SpeakerEmbeddingHandle(model_id="test/speaker", revision="abc")

    def provider(samples):
        return np.array([1.0, 0.0], dtype=np.float32) if float(np.mean(samples)) < 0.5 else np.array([0.0, 1.0], dtype=np.float32)

    linked, report = speaker_embeddings.link_speakers_by_voice(
        handle,
        audio,
        payload,
        similarity_threshold=0.8,
        embedding_provider=provider,
    )

    assert [item["speaker"] for item in linked.segments] == ["S001001", "S001001", "S002002"]
    assert linked.segments[1]["voice_original_speaker"] == "S002001"
    assert report["links"][0]["method"] == "wavlm_xvector_cosine"
    assert linked.metadata["speaker_scope"] == "voice_embedding_linked"


def test_voice_embedding_cluster_rejects_two_speakers_from_same_chunk():
    embeddings = {
        "S001001": np.array([1.0, 0.0], dtype=np.float32),
        "S001002": np.array([0.999, 0.001], dtype=np.float32),
    }
    groups = {
        "S001001": {"chunk_ids": {"part001"}, "first_start": 0},
        "S001002": {"chunk_ids": {"part001"}, "first_start": 1},
    }
    mapping, links, rejected = speaker_embeddings._cluster_embeddings(
        embeddings,
        groups,
        similarity_threshold=0.8,
    )
    assert mapping == {"S001001": "S001001", "S001002": "S001002"}
    assert links == []
    assert rejected[0]["reason"] == "same_chunk_collision"
