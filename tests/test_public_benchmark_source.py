from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_public_benchmarks.py"
SPEC = importlib.util.spec_from_file_location("t8_public_benchmark_source_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def _source(revision: str) -> dict:
    return {"dataset": "google/fleurs", "split": "validation", "revision": revision}


def test_public_benchmark_revision_gate_accepts_exact_dataset_head(monkeypatch):
    revision = "a" * 40
    monkeypatch.setattr(module, "_json_request", lambda _url: {"sha": revision})
    module._verify_source_revision(_source(revision))


def test_public_benchmark_revision_gate_rejects_dataset_drift(monkeypatch):
    monkeypatch.setattr(module, "_json_request", lambda _url: {"sha": "b" * 40})
    with pytest.raises(RuntimeError, match="moved from the audited"):
        module._verify_source_revision(_source("a" * 40))
