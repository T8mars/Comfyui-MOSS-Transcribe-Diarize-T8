from __future__ import annotations

import json
import zipfile
from pathlib import Path

from scripts import build_release, verify_pinned_revisions


EXPECTED_CODE_REVISION = "cde3c13af82c3001a21cf085d37ebc7d81e8981d"
EXPECTED_MODEL_REVISION = "e8681d68e7042738ffca8ac8212bc8fcb1131ab8"


def test_revision_targets_use_full_resolvable_urls():
    targets = verify_pinned_revisions.revision_targets(verify_pinned_revisions.load_manifest())
    assert targets == (
        (
            "OpenMOSS code",
            EXPECTED_CODE_REVISION,
            f"https://api.github.com/repos/OpenMOSS/MOSS-Transcribe-Diarize/commits/{EXPECTED_CODE_REVISION}",
        ),
        (
            "Hugging Face model",
            EXPECTED_MODEL_REVISION,
            "https://huggingface.co/api/models/OpenMOSS-Team/MOSS-Transcribe-Diarize/"
            f"revision/{EXPECTED_MODEL_REVISION}",
        ),
    )


def test_release_build_is_reproducible_and_excludes_tests(tmp_path: Path):
    first = build_release.build_release(tmp_path / "first", expected_tag="v0.3.1")
    second = build_release.build_release(tmp_path / "second", expected_tag="v0.3.1")
    assert [path.name for path in first] == [path.name for path in second]
    assert [path.read_bytes() for path in first] == [path.read_bytes() for path in second]

    archive, manifest_path, sums = first
    with zipfile.ZipFile(archive) as package:
        names = package.namelist()
    assert "README.md" in names
    assert "README_EN.md" in names
    assert "scripts/build_release.py" in names
    assert not any(name.startswith("tests/") for name in names)
    assert not any("__pycache__" in name for name in names)
    assert not any(name.startswith(".compat/") for name in names)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["node_version"] == "0.3.1"
    assert manifest["upstream_code_revision"] == EXPECTED_CODE_REVISION
    assert manifest["archive"]["name"] == archive.name
    assert manifest["archive"]["sha256"] == build_release.sha256(archive)
    assert archive.name in sums.read_text(encoding="utf-8")
