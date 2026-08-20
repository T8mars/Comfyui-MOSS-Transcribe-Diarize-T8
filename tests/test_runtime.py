from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
import types
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "comfyui_moss_t8_runtime_test"
SPEC = importlib.util.spec_from_file_location(
    PACKAGE_NAME,
    PLUGIN_ROOT / "__init__.py",
    submodule_search_locations=[str(PLUGIN_ROOT)],
)
assert SPEC is not None and SPEC.loader is not None
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[PACKAGE_NAME] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
inference = importlib.import_module(f"{PACKAGE_NAME}.runtime.inference")
model_store = importlib.import_module(f"{PACKAGE_NAME}.services.model_store")
build_prompt = inference.build_prompt
estimate_max_new_tokens = inference.estimate_max_new_tokens


def _metadata(data: bytes) -> dict:
    return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def test_prompt_deduplicates_hotwords_and_enforces_format():
    result = build_prompt("识别音频。", "ComfyUI，OpenMOSS,ComfyUI", "中文", True)
    assert result.hotwords == ("ComfyUI", "OpenMOSS")
    assert "主要语言提示：中文" in result.text
    assert "只输出" in result.text


def test_auto_token_budget_is_bounded_and_duration_sensitive():
    assert estimate_max_new_tokens(1) == 2048
    assert estimate_max_new_tokens(3600) > estimate_max_new_tokens(600)
    assert estimate_max_new_tokens(100000) == 65536


def test_model_directory_discovery_and_validation(tmp_path: Path, monkeypatch):
    model_dir = tmp_path / "MOSS-Transcribe-Diarize"
    model_dir.mkdir()
    files = {
        "config.json": b"{}",
        "model-00000-of-00001.safetensors": b"weights",
        "tokenizer.json": b"tokenizer",
    }
    for relative, data in files.items():
        (model_dir / relative).write_bytes(data)
    monkeypatch.setattr(
        model_store,
        "load_manifest",
        lambda: {"revision": "fixed", "files": {name: _metadata(data) for name, data in files.items()}},
    )
    assert model_store.discover_models([tmp_path]) == {"MOSS-Transcribe-Diarize": model_dir.resolve()}
    assert model_store.validate_model_dir(model_dir).valid
    assert model_store.validate_model_dir(model_dir, verify_hashes=True).valid
    (model_dir / "tokenizer.json").write_bytes(b"broken")
    assert model_store.validate_model_dir(model_dir).mismatched == ("tokenizer.json",)


def test_requirements_never_replace_comfyui_torch_stack():
    requirements = (model_store.PLUGIN_ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    active = [line.strip() for line in requirements.splitlines() if line.strip() and not line.startswith("#")]
    assert not any(line.startswith(("torch", "torchaudio", "torchvision")) for line in active)
    assert not any(line.startswith("transformers") for line in active)
    assert any(line.startswith("filelock") for line in active)
    assert any(line.startswith("huggingface-hub") for line in active)
    pyproject = (model_store.PLUGIN_ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert '"transformers' not in pyproject
    assert '"torch' not in pyproject
    optional = (model_store.PLUGIN_ROOT / "requirements-transformers-v4.txt").read_text(encoding="utf-8").lower()
    assert "transformers==4.57.6" in optional


def test_manifest_is_pinned_to_reviewed_revisions():
    manifest = model_store.load_manifest()
    assert manifest["revision"] == "e8681d68e7042738ffca8ac8212bc8fcb1131ab8"
    assert manifest["code_revision"] == "0e3d1403fd8f1f1c674e883ece96b9f630794ebe"
    assert manifest["files"]["model-00000-of-00001.safetensors"]["size"] == 1817113576


def test_comfy_callbacks_forward_progress_and_interrupt(monkeypatch):
    progress_updates = []
    interrupt_checks = []

    class FakeProgressBar:
        def __init__(self, total):
            assert total == 64

        def update_absolute(self, value, total):
            progress_updates.append((value, total))

    fake_comfy = types.ModuleType("comfy")
    fake_utils = types.ModuleType("comfy.utils")
    fake_management = types.ModuleType("comfy.model_management")
    fake_utils.ProgressBar = FakeProgressBar
    fake_management.throw_exception_if_processing_interrupted = lambda: interrupt_checks.append(True)
    fake_comfy.utils = fake_utils
    fake_comfy.model_management = fake_management
    monkeypatch.setitem(sys.modules, "comfy", fake_comfy)
    monkeypatch.setitem(sys.modules, "comfy.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "comfy.model_management", fake_management)
    token_callback, cancellation_callback = inference._comfy_runtime_callbacks(64)
    token_callback(9)
    assert cancellation_callback() is False
    assert progress_updates == [(9, 64)]
    assert interrupt_checks == [True]


def test_vendored_runtime_has_no_host_project_absolute_imports():
    vendor_root = model_store.PLUGIN_ROOT / "vendor" / "moss_transcribe_diarize"
    offenders = []
    for path in vendor_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "from moss_transcribe_diarize" in text or "import moss_transcribe_diarize" in text:
            offenders.append(path.relative_to(vendor_root).as_posix())
    assert offenders == []
