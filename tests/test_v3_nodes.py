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


def test_registers_nine_pure_v3_nodes():
    plugin = _load_plugin()
    extension = asyncio.run(plugin.comfy_entrypoint())
    nodes = asyncio.run(extension.get_node_list())
    schemas = [node.GET_SCHEMA() for node in nodes]
    assert [schema.node_id for schema in schemas] == [
        "T8_MOSS_ModelLoader",
        "T8_MOSS_PromptHotwords",
        "T8_MOSS_TranscribeDiarize",
        "T8_MOSS_SmartLongAudio",
        "T8_MOSS_TranscriptValidate",
        "T8_MOSS_QualityGate",
        "T8_MOSS_SubtitleStyle",
        "T8_MOSS_SubtitleExport",
        "T8_MOSS_EnvironmentRelease",
    ]
    assert all(schema.category == "T8star-Aix/Audio/MOSS Transcribe Diarize" for schema in schemas)
    assert schemas[2].outputs[0].io_type == "AUDIO"
    assert schemas[7].is_output_node is True
    assert schemas[7].not_idempotent is True
    assert schemas[8].is_output_node is True
    assert schemas[8].not_idempotent is True
    assert nodes[7].OUTPUT_NODE is True
    assert nodes[8].OUTPUT_NODE is True
    assert not hasattr(plugin, "NODE_CLASS_MAPPINGS")
    # ComfyUI's current V3 API exposes the stable input identifier as ``id``;
    # the lightweight compatibility stub used by isolated tests calls it
    # ``name``.  Accept both so this assertion runs against the real API too.
    input_names = [
        {getattr(item, "id", getattr(item, "name", None)) for item in schema.inputs}
        for schema in schemas
    ]
    assert {"memory_policy", "release_after_run", "attention_implementation"} <= input_names[0]
    assert "preset_id" in input_names[1]
    assert "silence_policy" in input_names[2]
    assert {"preflight_backend", "vad_aggressiveness", "retry_policy"} <= input_names[2]
    assert {"split_strategy", "checkpoint_mode", "checkpoint_id"} <= input_names[3]
    assert {"min_end_coverage", "max_unknown_speaker_ratio"} <= input_names[5]
    assert {"font_name", "video_width", "video_height"} <= input_names[6]
    assert {"chunk_id", "cross_chunk_speaker_map_json", "style"} <= input_names[7]


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
