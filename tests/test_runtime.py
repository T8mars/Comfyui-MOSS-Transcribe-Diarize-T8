from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest


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


def test_prompt_scenario_preset_adds_reviewed_instruction():
    result = build_prompt("识别音频。", "", "auto", True, "zh_meeting")
    assert result.language_hint == "中文"
    assert "中文会议" in result.text


def test_silence_rejects_before_model_cache_acquisition(monkeypatch, tmp_path: Path):
    types_module = importlib.import_module(f"{PACKAGE_NAME}.runtime.types")
    handle = types_module.ModelHandle(tmp_path, "cpu", "float32")
    monkeypatch.setattr(inference, "comfy_audio_to_numpy", lambda *_args: np.zeros(16000, dtype=np.float32))
    acquired = []
    monkeypatch.setattr(inference.MODEL_CACHE, "acquire", lambda _handle: acquired.append(True))
    with pytest.raises(ValueError, match="预检已拒绝推理"):
        inference.run_transcription(handle, {"waveform": None}, None, max_new_tokens=16, silence_policy="reject")
    assert acquired == []


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


def test_explicit_sha256_verification_never_reuses_a_stale_digest(tmp_path: Path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    path = model_dir / "weights.bin"
    original = b"a" * 32
    replacement = b"b" * 32
    path.write_bytes(original)
    original_stat = path.stat()
    monkeypatch.setattr(
        model_store,
        "load_manifest",
        lambda: {"revision": "fixed", "files": {"weights.bin": _metadata(original)}},
    )

    assert model_store.validate_model_dir(model_dir, verify_hashes=True).valid
    original_fingerprint = model_store.model_fingerprint(model_dir)
    path.write_bytes(replacement)
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    assert model_store._sha256(path) == hashlib.sha256(replacement).hexdigest()
    assert model_store.validate_model_dir(model_dir, verify_hashes=True).mismatched == ("weights.bin",)
    assert model_store.model_fingerprint(model_dir) != original_fingerprint


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
    assert not (model_store.PLUGIN_ROOT / "requirements-transformers-v4.txt").exists()
    optional = (model_store.PLUGIN_ROOT / "requirements-transformers-v5.txt").read_text(encoding="utf-8").lower()
    assert "transformers==5.15.1" in optional
    checker = (model_store.PLUGIN_ROOT / "scripts" / "check_transformers.py").read_text(encoding="utf-8")
    assert 'Version("5.5.0")' in checker
    assert "published security advisories" in checker


def test_ui_workflows_reference_current_node_package_version():
    import json

    workflows = model_store.PLUGIN_ROOT / "example_workflows" / "ui"
    for path in sorted(workflows.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        versions = {
            node.get("properties", {}).get("ver")
            for node in payload.get("nodes", [])
            if node.get("properties", {}).get("cnr_id") == "comfyui-moss-transcribe-diarize-t8"
        }
        assert versions == {"0.3.5"}, f"{path.name} has stale node versions: {versions}"


def test_manifest_is_pinned_to_reviewed_revisions():
    manifest = model_store.load_manifest()
    assert manifest["revision"] == "e8681d68e7042738ffca8ac8212bc8fcb1131ab8"
    assert manifest["code_revision"] == "cb765f2b0fe6f7a298aa2002e2281ae693d1f3c3"
    assert "never silently fall back" in manifest["integration_attention_policy"]
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


def test_comfy_callbacks_prefer_native_v3_progress(monkeypatch):
    latest = importlib.import_module("comfy_api.latest")
    native_updates = []
    legacy_updates = []

    class FakeExecution:
        def set_progress(self, value, total):
            native_updates.append((value, total))

    class FakeAPISync:
        def __init__(self):
            self.execution = FakeExecution()

    class FakeProgressBar:
        def __init__(self, _total):
            pass

        def update_absolute(self, value, total):
            legacy_updates.append((value, total))

    fake_comfy = types.ModuleType("comfy")
    fake_utils = types.ModuleType("comfy.utils")
    fake_management = types.ModuleType("comfy.model_management")
    fake_utils.ProgressBar = FakeProgressBar
    fake_management.throw_exception_if_processing_interrupted = lambda: None
    fake_comfy.utils = fake_utils
    fake_comfy.model_management = fake_management
    monkeypatch.setattr(latest, "ComfyAPISync", FakeAPISync)
    monkeypatch.setitem(sys.modules, "comfy", fake_comfy)
    monkeypatch.setitem(sys.modules, "comfy.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "comfy.model_management", fake_management)

    token_callback, _ = inference._comfy_runtime_callbacks(32)
    token_callback(7)
    assert native_updates == [(7, 32)]
    assert legacy_updates == []


def test_model_cache_normalizes_equivalent_cpu_precision(tmp_path: Path):
    model_cache = importlib.import_module(f"{PACKAGE_NAME}.runtime.model_cache")
    types_module = importlib.import_module(f"{PACKAGE_NAME}.runtime.types")
    common = dict(
        model_dir=tmp_path,
        device="cpu",
        release_after_run=False,
        model_revision="fixed",
    )
    automatic = types_module.ModelHandle(precision="auto", **common)
    explicit = types_module.ModelHandle(precision="float32", **common)
    assert model_cache.ModelCache._resolved(automatic)[2] == model_cache.ModelCache._resolved(explicit)[2]


def test_bf16_capability_is_checked_inside_the_requested_cuda_device(monkeypatch):
    model_cache = importlib.import_module(f"{PACKAGE_NAME}.runtime.model_cache")
    torch = importlib.import_module("torch")
    active = {"index": 0}
    checked = []

    class DeviceContext:
        def __init__(self, index):
            self.index = index
            self.previous = None

        def __enter__(self):
            self.previous = active["index"]
            active["index"] = self.index

        def __exit__(self, *_args):
            active["index"] = self.previous

    monkeypatch.setattr(torch.cuda, "device", DeviceContext)

    def supported():
        checked.append(active["index"])
        return active["index"] == 3

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", supported)

    assert model_cache._cuda_bf16_supported(torch.device("cuda:3")) is True
    assert checked == [3]
    assert active["index"] == 0


def test_model_cache_discards_idle_load_gates_after_success_and_failure(tmp_path: Path, monkeypatch):
    model_cache = importlib.import_module(f"{PACKAGE_NAME}.runtime.model_cache")
    compat = importlib.import_module(
        f"{PACKAGE_NAME}.vendor.moss_transcribe_diarize.transformers_compat"
    )
    types_module = importlib.import_module(f"{PACKAGE_NAME}.runtime.types")
    cache = model_cache.ModelCache()
    handle = types_module.ModelHandle(tmp_path, "cpu", "float32", model_revision="fixed")

    class FakeModel:
        def to(self, _device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(
        compat,
        "load_local_model_processor_with_attention",
        lambda *_args, **_kwargs: (FakeModel(), object(), {"selected": "sdpa"}),
    )
    entry = cache.acquire(handle)

    assert cache._load_locks == {}
    cache.done(handle, entry, release=True)

    failing = model_cache.ModelCache()
    monkeypatch.setattr(
        compat,
        "load_local_model_processor_with_attention",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("load failed")),
    )
    with pytest.raises(RuntimeError, match="load failed"):
        failing.acquire(handle)
    assert failing._load_locks == {}


def test_attention_policy_is_part_of_comfy_model_cache_identity(tmp_path: Path):
    model_cache = importlib.import_module(f"{PACKAGE_NAME}.runtime.model_cache")
    types_module = importlib.import_module(f"{PACKAGE_NAME}.runtime.types")
    automatic = types_module.ModelHandle(tmp_path, "cpu", "float32", attention_implementation="auto")
    eager = types_module.ModelHandle(tmp_path, "cpu", "float32", attention_implementation="eager")
    auto_key = model_cache.ModelCache._resolved(automatic)[2]
    eager_key = model_cache.ModelCache._resolved(eager)[2]
    assert auto_key[-1] == "auto"
    assert eager_key[-1] == "eager"
    assert auto_key != eager_key


def test_model_fingerprint_is_part_of_comfy_model_cache_identity(tmp_path: Path):
    model_cache = importlib.import_module(f"{PACKAGE_NAME}.runtime.model_cache")
    types_module = importlib.import_module(f"{PACKAGE_NAME}.runtime.types")
    first = types_module.ModelHandle(tmp_path, "cpu", "float32", model_fingerprint="fingerprint-a")
    updated = types_module.ModelHandle(tmp_path, "cpu", "float32", model_fingerprint="fingerprint-b")

    first_key = model_cache.ModelCache._resolved(first)[2]
    updated_key = model_cache.ModelCache._resolved(updated)[2]

    assert first_key != updated_key
    assert first_key[-1] == updated_key[-1] == "auto"


def test_model_cache_retires_superseded_idle_weights(tmp_path: Path):
    import threading
    import torch

    model_cache = importlib.import_module(f"{PACKAGE_NAME}.runtime.model_cache")
    types_module = importlib.import_module(f"{PACKAGE_NAME}.runtime.types")
    cache = model_cache.ModelCache()
    first = types_module.ModelHandle(
        tmp_path,
        "cpu",
        "float32",
        model_fingerprint="fingerprint-a",
        attention_implementation="eager",
    )
    updated = types_module.ModelHandle(
        tmp_path,
        "cpu",
        "float32",
        model_fingerprint="fingerprint-b",
        attention_implementation="eager",
    )
    first_key = cache._resolved(first)[2]
    updated_key = cache._resolved(updated)[2]
    entry = model_cache.CacheEntry(
        first_key,
        object(),
        object(),
        torch.device("cpu"),
        torch.float32,
        threading.RLock(),
        {},
    )
    cache._entries[first_key] = entry

    with cache._lock:
        cache._retire_superseded_locked(updated_key)

    assert first_key not in cache._entries
    assert entry.model is None and entry.processor is None


def test_model_handle_memory_policy_keeps_legacy_release_compatibility(tmp_path: Path):
    types_module = importlib.import_module(f"{PACKAGE_NAME}.runtime.types")
    keep = types_module.ModelHandle(tmp_path, "cpu", "float32", memory_policy="keep")
    pressure = types_module.ModelHandle(tmp_path, "cpu", "float32", memory_policy="release_under_pressure")
    legacy = types_module.ModelHandle(tmp_path, "cpu", "float32", release_after_run=True, memory_policy="keep")
    assert keep.effective_memory_policy == "keep"
    assert pressure.effective_memory_policy == "release_under_pressure"
    assert legacy.effective_memory_policy == "release_after_run"


def test_comfy_coordination_never_globally_unloads_unrelated_models(monkeypatch):
    model_cache = importlib.import_module(f"{PACKAGE_NAME}.runtime.model_cache")
    calls = []
    fake_comfy = types.ModuleType("comfy")
    fake_management = types.ModuleType("comfy.model_management")
    fake_management.unload_all_models = lambda: calls.append("unload")
    fake_management.soft_empty_cache = lambda: calls.append("soft")
    fake_comfy.model_management = fake_management
    monkeypatch.setitem(sys.modules, "comfy", fake_comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", fake_management)
    model_cache.ModelCache._coordinate_comfy_memory(importlib.import_module("torch").device("cuda:0"))
    assert calls == ["soft"]


def test_vendored_runtime_has_no_host_project_absolute_imports():
    vendor_root = model_store.PLUGIN_ROOT / "vendor" / "moss_transcribe_diarize"
    offenders = []
    for path in vendor_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "from moss_transcribe_diarize" in text or "import moss_transcribe_diarize" in text:
            offenders.append(path.relative_to(vendor_root).as_posix())
    assert offenders == []
    assert (vendor_root / "attention.py").is_file()
