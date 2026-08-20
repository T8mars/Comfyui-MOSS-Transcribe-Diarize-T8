from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

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


def test_registers_six_pure_v3_nodes():
    plugin = _load_plugin()
    extension = asyncio.run(plugin.comfy_entrypoint())
    nodes = asyncio.run(extension.get_node_list())
    schemas = [node.GET_SCHEMA() for node in nodes]
    assert [schema.node_id for schema in schemas] == [
        "T8_MOSS_ModelLoader",
        "T8_MOSS_PromptHotwords",
        "T8_MOSS_TranscribeDiarize",
        "T8_MOSS_TranscriptValidate",
        "T8_MOSS_SubtitleExport",
        "T8_MOSS_EnvironmentRelease",
    ]
    assert all(schema.category == "T8star-Aix/Audio/MOSS Transcribe Diarize" for schema in schemas)
    assert schemas[2].outputs[0].io_type == "AUDIO"
    assert not hasattr(plugin, "NODE_CLASS_MAPPINGS")
