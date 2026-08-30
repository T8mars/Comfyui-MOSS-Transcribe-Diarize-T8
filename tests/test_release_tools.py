from __future__ import annotations

import json
from types import SimpleNamespace
import zipfile
from pathlib import Path

import pytest

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
    first = build_release.build_release(tmp_path / "first")
    second = build_release.build_release(tmp_path / "second")
    assert [path.name for path in first] == [path.name for path in second]
    assert [path.read_bytes() for path in first] == [path.read_bytes() for path in second]

    archive, manifest_path, sums = first
    with zipfile.ZipFile(archive) as package:
        names = package.namelist()
        assert b"\r\n" not in package.read("LICENSE")
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
    assert manifest["node_version"] == "0.3.7"
    assert b"\r\n" not in manifest_path.read_bytes()
    assert b"\r\n" not in sums.read_bytes()
    assert manifest["upstream_code_revision"] == EXPECTED_CODE_REVISION
    assert manifest["archive"]["name"] == archive.name
    assert manifest["archive"]["sha256"] == build_release.sha256(archive)
    assert isinstance(manifest["source_dirty"], bool)
    assert archive.name in sums.read_text(encoding="utf-8")


def test_release_selection_excludes_untracked_files():
    probe = build_release.PLUGIN_ROOT / ".release-untracked-secret-probe"
    assert not probe.exists()
    try:
        probe.write_text("must not be packaged\n", encoding="utf-8")
        assert probe not in build_release.release_files()
    finally:
        probe.unlink(missing_ok=True)


def test_release_selection_rejects_tracked_symlinks(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        build_release.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=b"unsafe-link\0"),
    )
    original_is_symlink = Path.is_symlink
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda path: path.name == "unsafe-link" or original_is_symlink(path),
    )

    with pytest.raises(RuntimeError, match="symbolic links"):
        build_release.release_files()


def test_third_party_notices_match_the_source_only_node_package():
    root = Path(__file__).resolve().parents[1]
    notices = (root / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert "does **not** bundle" in notices
    assert "manifests/ffmpeg_windows_x64.json" not in notices
    assert "nvidia-runtime-inventory.json" not in notices
    modified = (
        "audio_adapter.py",
        "inference_utils.py",
        "modeling_moss_transcribe_diarize.py",
        "subtitle/__init__.py",
        "subtitle/export.py",
        "subtitle/models.py",
        "subtitle/postprocess.py",
        "transcript_parser.py",
        "transcript_validation.py",
    )
    vendor = root / "vendor" / "moss_transcribe_diarize"
    for relative in modified:
        first_line = (vendor / relative).read_text(encoding="utf-8").splitlines()[0]
        assert "Modified by the T8star-Aix integration" in first_line


def test_tagged_release_requires_matching_clean_head(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(build_release, "source_commit", lambda: "head")
    monkeypatch.setattr(build_release, "tagged_commit", lambda tag: "other")
    with pytest.raises(RuntimeError, match="points to other"):
        build_release.validate_release_source("v0.3.7")

    monkeypatch.setattr(build_release, "tagged_commit", lambda tag: "head")
    monkeypatch.setattr(build_release, "source_is_dirty", lambda: True)
    with pytest.raises(RuntimeError, match="clean working tree"):
        build_release.validate_release_source("v0.3.7")


def test_release_and_compatibility_workflows_keep_automatic_delivery_observable():
    root = Path(__file__).resolve().parents[1]
    release = (root / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    retry = (root / ".github" / "workflows" / "publish.yml").read_text(encoding="utf-8")
    validate = (root / ".github" / "workflows" / "validate.yml").read_text(encoding="utf-8")

    assert "publish-registry:" in release
    assert "needs: release" in release
    assert "name: Validate and build" in release
    assert "name: Attest and publish GitHub Release" in release
    assert "actions/upload-artifact@b7c566a772e6b6bfb58ed0dc250532a479d7789f" in release
    assert "actions/download-artifact@634f93cb2916e3fdff6788551b99b062d0335ce0" in release
    assert release.count("persist-credentials: false") == 3
    assert "python-version: \"3.12\"" in release
    assert "Comfy-Org/publish-node-action@" not in release
    assert "Comfy-Org/publish-node-action@" not in retry
    assert "comfy-cli==1.16.0" in release
    assert "comfy-cli==1.16.0" in retry
    assert "refs/tags/${RELEASE_REF}^{commit}" in retry
    assert 'gh release view "$RELEASE_REF"' in retry
    assert "release_ref:" in retry
    assert "persist-credentials: false" in retry
    assert "default: main" not in retry
    assert "Registry publication cannot continue" in retry
    assert "exit 1" in retry
    assert "types: [published]" not in retry
    assert 'cron: "15 20 * * 0"' in validate
    assert 'transformers: "4.52.1"' not in validate
    assert 'transformers: "4.57.6"' not in validate
    assert 'transformers: "5.5.0"' in validate
    assert "GITHUB_STEP_SUMMARY" in validate
