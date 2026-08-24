from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PLUGIN_ROOT / "scripts" / "download_models.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("moss_t8_download_models_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_target_uses_explicit_comfyui_root(tmp_path: Path, monkeypatch):
    module = _load_script()
    monkeypatch.setitem(sys.modules, "folder_paths", None)
    root = tmp_path / "ComfyUI"
    (root / "models").mkdir(parents=True)
    assert module.default_target(root) == (
        root / "models" / "moss_transcribe_diarize" / "MOSS-Transcribe-Diarize"
    )


def test_default_target_never_silently_uses_plugin_model_downloads(monkeypatch):
    module = _load_script()
    monkeypatch.setitem(sys.modules, "folder_paths", None)
    monkeypatch.setattr(module, "discover_comfyui_root", lambda: None)
    with pytest.raises(RuntimeError, match="Cannot locate ComfyUI"):
        module.default_target()
