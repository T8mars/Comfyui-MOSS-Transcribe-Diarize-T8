from __future__ import annotations

import shutil
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PLUGIN_ROOT.parent
SOURCE_PACKAGE = PROJECT_ROOT / "moss_transcribe_diarize"
TARGET_PACKAGE = PLUGIN_ROOT / "vendor" / "moss_transcribe_diarize"
FILES = (
    "__init__.py",
    "audio_adapter.py",
    "audio_preflight.py",
    "attention.py",
    "configuration_moss_transcribe_diarize.py",
    "generation_budget.py",
    "inference_utils.py",
    "modeling_moss_transcribe_diarize.py",
    "model_store.py",
    "prompt_presets.py",
    "processing_moss_transcribe_diarize.py",
    "speaker_mapping.py",
    "transcript_parser.py",
    "transcript_validation.py",
    "transformers_compat.py",
    "subtitle/__init__.py",
    "subtitle/export.py",
    "subtitle/layout.py",
    "subtitle/models.py",
    "subtitle/postprocess.py",
)


def sync() -> list[Path]:
    copied = []
    for relative in FILES:
        source = SOURCE_PACKAGE / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        target = TARGET_PACKAGE / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(target)
    manifest_source = PROJECT_ROOT / "manifests" / "model_0_9b.json"
    for target in (
        PLUGIN_ROOT / "manifests" / "model_0_9b.json",
        PLUGIN_ROOT / "vendor" / "manifests" / "model_0_9b.json",
    ):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_source, target)
        copied.append(target)
    for filename in ("LICENSE", "DISCLAIMER", "THIRD_PARTY_NOTICES.md"):
        shutil.copy2(PROJECT_ROOT / filename, PLUGIN_ROOT / filename)
        copied.append(PLUGIN_ROOT / filename)
    return copied


if __name__ == "__main__":
    paths = sync()
    print(f"Synced {len(paths)} audited runtime files into {PLUGIN_ROOT}")
