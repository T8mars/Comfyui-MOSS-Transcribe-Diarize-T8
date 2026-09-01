from __future__ import annotations

import json
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[1]


def test_basic_ui_and_api_workflows_are_present_and_connected():
    ui = json.loads((PLUGIN_ROOT / "example_workflows" / "ui" / "01_basic_transcribe.json").read_text(encoding="utf-8"))
    api = json.loads((PLUGIN_ROOT / "example_workflows" / "api" / "01_basic_transcribe.json").read_text(encoding="utf-8"))
    assert ui["version"] == 0.4
    assert ui["last_node_id"] == max(node["id"] for node in ui["nodes"])
    assert ui["last_link_id"] == len(ui["links"])
    assert {node["type"] for node in ui["nodes"]}.issuperset({"T8_MOSS_TranscribeDiarize", "T8_MOSS_SubtitleExport"})
    assert {node["class_type"] for node in api.values()}.issuperset({"T8_MOSS_TranscribeDiarize", "T8_MOSS_SubtitleExport"})
    transcribe = next(node for node in api.values() if node["class_type"] == "T8_MOSS_TranscribeDiarize")
    assert transcribe["inputs"]["max_new_tokens"] == 0


def test_long_audio_ui_and_api_workflows_include_diagnostics():
    ui = json.loads((PLUGIN_ROOT / "example_workflows" / "ui" / "02_long_audio_diagnostics.json").read_text(encoding="utf-8"))
    api = json.loads((PLUGIN_ROOT / "example_workflows" / "api" / "02_long_audio_diagnostics.json").read_text(encoding="utf-8"))
    assert ui["last_node_id"] == max(node["id"] for node in ui["nodes"])
    assert ui["last_link_id"] == len(ui["links"])
    required = {
        "T8_MOSS_SmartLongAudio",
        "T8_MOSS_QualityGate",
        "T8_MOSS_EnvironmentRelease",
        "T8_MOSS_SubtitleExport",
    }
    assert {node["type"] for node in ui["nodes"]}.issuperset(required)
    assert {node["class_type"] for node in api.values()}.issuperset(required)
    transcribe = next(node for node in api.values() if node["class_type"] == "T8_MOSS_SmartLongAudio")
    quality_gate = next(node for node in api.values() if node["class_type"] == "T8_MOSS_QualityGate")
    assert transcribe["inputs"]["max_new_tokens_per_chunk"] == 0
    assert transcribe["inputs"]["checkpoint_mode"] == "read_write"
    assert transcribe["inputs"]["speaker_link_mode"] == "overlap_only"
    assert quality_gate["inputs"]["transcript"] == ["4", 5]


def test_examples_cover_subtitle_postprocess_and_safe_remote_opt_in():
    api_root = PLUGIN_ROOT / "example_workflows" / "api"
    basic = json.loads((api_root / "01_basic_transcribe.json").read_text(encoding="utf-8"))
    long_audio = json.loads((api_root / "02_long_audio_diagnostics.json").read_text(encoding="utf-8"))
    remote = json.loads((api_root / "03_remote_transcribe.json").read_text(encoding="utf-8"))

    assert any(node["class_type"] == "T8_MOSS_SubtitlePostprocess" for node in basic.values())
    assert any(node["class_type"] == "T8_MOSS_SubtitlePostprocess" for node in long_audio.values())
    remote_loader = next(node for node in remote.values() if node["class_type"] == "T8_MOSS_RemoteModelLoader")
    assert remote_loader["inputs"]["allow_remote_upload"] is False
    assert remote_loader["inputs"]["endpoint_url"].startswith("http://127.0.0.1")


def test_advanced_api_example_connects_real_auxiliary_models():
    path = PLUGIN_ROOT / "example_workflows" / "api" / "04_word_alignment_voice_link.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    types = {node["class_type"] for node in workflow.values()}
    assert {
        "T8_MOSS_SmartLongAudio",
        "T8_MOSS_SpeakerEmbeddingModelLoader",
        "T8_MOSS_SpeakerEmbeddingLink",
        "T8_MOSS_WordAlignmentModelLoader",
        "T8_MOSS_WordAlignment",
        "T8_MOSS_SubtitleExport",
    }.issubset(types)
    speaker_link = next(node for node in workflow.values() if node["class_type"] == "T8_MOSS_SpeakerEmbeddingLink")
    word_alignment = next(node for node in workflow.values() if node["class_type"] == "T8_MOSS_WordAlignment")
    assert speaker_link["inputs"]["transcript"] == ["4", 5]
    assert word_alignment["inputs"]["transcript"] == ["6", 0]


def test_ui_workflow_links_reference_existing_nodes_and_slots():
    for path in sorted((PLUGIN_ROOT / "example_workflows" / "ui").glob("*.json")):
        workflow = json.loads(path.read_text(encoding="utf-8"))
        nodes = {node["id"]: node for node in workflow["nodes"]}
        link_ids = {link[0] for link in workflow["links"]}
        assert link_ids == set(range(1, workflow["last_link_id"] + 1))
        for link_id, source_id, source_slot, target_id, target_slot, link_type in workflow["links"]:
            assert source_id in nodes and target_id in nodes, (path.name, link_id)
            assert source_slot < len(nodes[source_id].get("outputs", [])), (path.name, link_id)
            assert target_slot < len(nodes[target_id].get("inputs", [])), (path.name, link_id)
            assert nodes[source_id]["outputs"][source_slot]["type"] == link_type
            assert nodes[target_id]["inputs"][target_slot]["type"] == link_type
