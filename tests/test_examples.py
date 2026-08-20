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
    required = {"T8_MOSS_TranscriptValidate", "T8_MOSS_EnvironmentRelease", "T8_MOSS_SubtitleExport"}
    assert {node["type"] for node in ui["nodes"]}.issuperset(required)
    assert {node["class_type"] for node in api.values()}.issuperset(required)
    transcribe = next(node for node in api.values() if node["class_type"] == "T8_MOSS_TranscribeDiarize")
    validator = next(node for node in api.values() if node["class_type"] == "T8_MOSS_TranscriptValidate")
    assert transcribe["inputs"]["max_new_tokens"] == 0
    assert validator["inputs"]["transcript"] == ["4", 5]
