from __future__ import annotations

import json
import zipfile
from pathlib import Path

from scripts import build_release, verify_pinned_revisions


EXPECTED_CODE_REVISION = "cb765f2b0fe6f7a298aa2002e2281ae693d1f3c3"
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


def test_release_build_is_reproducible_and_excludes_development_only_files(tmp_path: Path):
    first = build_release.build_release(tmp_path / "first", expected_tag="v0.3.4")
    second = build_release.build_release(tmp_path / "second", expected_tag="v0.3.4")
    assert [path.name for path in first] == [path.name for path in second]
    assert [path.read_bytes() for path in first] == [path.read_bytes() for path in second]

    archive, manifest_path, sums = first
    with zipfile.ZipFile(archive) as package:
        names = package.namelist()
    assert "README.md" in names
    assert "README_EN.md" in names
    assert "requirements-transformers-v5.txt" in names
    assert "requirements-transformers-v4.txt" not in names
    assert "scripts/download_models.py" in names
    assert "scripts/build_release.py" not in names
    assert "scripts/sync_vendor.py" not in names
    assert "scripts/verify_pinned_revisions.py" not in names
    assert not any(name.startswith("tests/") for name in names)
    assert not any(name.startswith(".github/") for name in names)
    assert not any("__pycache__" in name for name in names)
    assert not any(name.startswith(".compat/") for name in names)

    with zipfile.ZipFile(archive) as package:
        packaged_python = "\n".join(
            package.read(name).decode("utf-8")
            for name in names
            if name.endswith(".py")
        )
    assert "subprocess.run(" not in packaged_python
    assert "urllib.request.urlopen(" not in packaged_python
    assert "os.environ.get(" not in packaged_python

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["node_version"] == "0.3.4"
    assert manifest["upstream_code_revision"] == EXPECTED_CODE_REVISION
    assert manifest["archive"]["name"] == archive.name
    assert manifest["archive"]["sha256"] == build_release.sha256(archive)
    assert archive.name in sums.read_text(encoding="utf-8")


def test_release_and_compatibility_workflows_keep_automatic_delivery_observable():
    root = Path(__file__).resolve().parents[1]
    release = (root / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    retry = (root / ".github" / "workflows" / "publish.yml").read_text(encoding="utf-8")
    validate = (root / ".github" / "workflows" / "validate.yml").read_text(encoding="utf-8")

    assert "publish-registry:" in release
    assert "needs: release" in release
    assert "Comfy-Org/publish-node-action@" in release
    assert "release_ref:" in retry
    assert "default: main" not in retry
    assert "Registry publication cannot continue" in retry
    assert "exit 1" in retry
    assert "types: [published]" not in retry
    assert 'cron: "15 20 * * 0"' in validate
    assert "GITHUB_STEP_SUMMARY" in validate
