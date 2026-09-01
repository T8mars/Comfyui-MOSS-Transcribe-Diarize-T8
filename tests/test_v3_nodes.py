from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


pytest.importorskip("comfy_api.latest")


def _load_plugin():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "comfyui_moss_transcribe_diarize_t8_test",
        root / "__init__.py",
        submodule_search_locations=[str(root)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_registers_fifteen_pure_v3_nodes():
    plugin = _load_plugin()
    extension = asyncio.run(plugin.comfy_entrypoint())
    nodes = asyncio.run(extension.get_node_list())
    schemas = [node.GET_SCHEMA() for node in nodes]
    assert [schema.node_id for schema in schemas] == [
        "T8_MOSS_ModelLoader",
        "T8_MOSS_RemoteModelLoader",
        "T8_MOSS_PromptHotwords",
        "T8_MOSS_TranscribeDiarize",
        "T8_MOSS_SmartLongAudio",
        "T8_MOSS_WordAlignmentModelLoader",
        "T8_MOSS_WordAlignment",
        "T8_MOSS_SpeakerEmbeddingModelLoader",
        "T8_MOSS_SpeakerEmbeddingLink",
        "T8_MOSS_TranscriptValidate",
        "T8_MOSS_QualityGate",
        "T8_MOSS_SubtitlePostprocess",
        "T8_MOSS_SubtitleStyle",
        "T8_MOSS_SubtitleExport",
        "T8_MOSS_EnvironmentRelease",
    ]
    assert all(schema.category == "T8star-Aix/Audio/MOSS Transcribe Diarize" for schema in schemas)
    assert schemas[3].outputs[0].io_type == "AUDIO"
    assert schemas[13].is_output_node is True
    assert schemas[13].not_idempotent is True
    assert schemas[14].is_output_node is True
    assert schemas[14].not_idempotent is True
    assert nodes[13].OUTPUT_NODE is True
    assert nodes[14].OUTPUT_NODE is True
    assert not hasattr(plugin, "NODE_CLASS_MAPPINGS")
    # ComfyUI's current V3 API exposes the stable input identifier as ``id``;
    # the lightweight compatibility stub used by isolated tests calls it
    # ``name``.  Accept both so this assertion runs against the real API too.
    input_names = [
        {getattr(item, "id", getattr(item, "name", None)) for item in schema.inputs}
        for schema in schemas
    ]
    assert {"memory_policy", "release_after_run", "attention_implementation"} <= input_names[0]
    assert {"endpoint_url", "allow_remote_upload"} <= input_names[1]
    assert {"preset_id", "custom_language_hint"} <= input_names[2]
    assert "silence_policy" in input_names[3]
    assert {"preflight_backend", "vad_aggressiveness", "retry_policy"} <= input_names[3]
    assert {"split_strategy", "checkpoint_mode", "checkpoint_id", "speaker_link_mode"} <= input_names[4]
    assert {"model_id", "revision", "language"} <= input_names[5]
    assert {"aligner", "audio", "transcript"} <= input_names[6]
    assert {"model_id", "revision", "release_after_run"} <= input_names[7]
    assert {"speaker_model", "audio", "transcript", "similarity_threshold"} <= input_names[8]
    assert {"min_end_coverage", "max_unknown_speaker_ratio", "fail_on_unusable"} <= input_names[10]
    assert {"max_chars_per_line", "max_chars_per_second"} <= input_names[11]
    assert {"font_name", "video_width", "video_height"} <= input_names[12]
    assert {"chunk_id", "cross_chunk_speaker_map_json", "style"} <= input_names[13]
    assert [getattr(item, "id", getattr(item, "name", None)) for item in schemas[13].outputs][-2:] == [
        "vtt",
        "rttm",
    ]


def test_locales_cover_every_v3_node_and_output():
    plugin = _load_plugin()
    extension = asyncio.run(plugin.comfy_entrypoint())
    nodes = asyncio.run(extension.get_node_list())
    root = Path(__file__).resolve().parents[1] / "locales"
    for language in ("en", "zh"):
        translations = json.loads((root / language / "nodeDefs.json").read_text(encoding="utf-8"))
        assert set(translations) == {node.GET_SCHEMA().node_id for node in nodes}
        for node in nodes:
            schema = node.GET_SCHEMA()
            translated = translations[schema.node_id]
            assert set(translated.get("inputs", {})) == {
                getattr(item, "id", getattr(item, "name", None)) for item in schema.inputs
            }
            assert set(translated.get("outputs", {})) == {str(index) for index in range(len(schema.outputs))}


def test_attention_tooltips_describe_the_explicit_auto_fallback_order():
    root = Path(__file__).resolve().parents[1]
    source = (root / "nodes_v3.py").read_text(encoding="utf-8")
    assert "FlashAttention-2、SDPA、eager" in source
    for language in ("en", "zh"):
        translations = json.loads((root / "locales" / language / "nodeDefs.json").read_text(encoding="utf-8"))
        tooltip = translations["T8_MOSS_ModelLoader"]["inputs"]["attention_implementation"]["tooltip"]
        assert "FlashAttention-2" in tooltip
        assert "SDPA" in tooltip
        assert "eager" in tooltip


def test_subtitle_output_stems_remain_unique_with_identical_clock_values(monkeypatch):
    plugin = _load_plugin()
    nodes = sys.modules[f"{plugin.__name__}.nodes_v3"]
    identifiers = iter(("aaaaaaaa00000000", "bbbbbbbb00000000"))
    monkeypatch.setattr(nodes.time, "time_ns", lambda: 123456789)
    monkeypatch.setattr(nodes.uuid, "uuid4", lambda: SimpleNamespace(hex=next(identifiers)))

    first = nodes._unique_output_stem("moss_transcript")
    second = nodes._unique_output_stem("moss_transcript")

    assert first != second
    assert first.startswith("moss_transcript_123456789_")


def test_api_examples_have_queue_eligible_output_nodes():
    plugin = _load_plugin()
    extension = asyncio.run(plugin.comfy_entrypoint())
    nodes = asyncio.run(extension.get_node_list())
    node_types = {node.GET_SCHEMA().node_id: node for node in nodes}
    root = Path(__file__).resolve().parents[1] / "example_workflows" / "api"
    for path in sorted(root.glob("*.json")):
        prompt = json.loads(path.read_text(encoding="utf-8"))
        outputs = [
            node_id
            for node_id, value in prompt.items()
            if value["class_type"] in node_types and node_types[value["class_type"]].OUTPUT_NODE is True
        ]
        assert outputs, f"{path.name} has no queue-eligible output node"


def test_long_audio_examples_stop_before_writing_unusable_results():
    root = Path(__file__).resolve().parents[1] / "example_workflows"
    api = json.loads((root / "api" / "02_long_audio_diagnostics.json").read_text(encoding="utf-8"))
    ui = json.loads((root / "ui" / "02_long_audio_diagnostics.json").read_text(encoding="utf-8"))
    quality = next(node for node in ui["nodes"] if node["type"] == "T8_MOSS_QualityGate")

    assert api["5"]["inputs"]["fail_on_unusable"] is True
    assert quality["widgets_values"][-1] is True
