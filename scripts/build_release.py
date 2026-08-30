from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import stat
import subprocess
import zipfile
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PLUGIN_ROOT / "dist"
ARCHIVE_STEM = "comfyui-MOSS-Transcribe-Diarize-T8"
REGISTRY_ID = "comfyui-moss-transcribe-diarize-t8"
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
ALWAYS_IGNORED = (".git", ".compat", "dist", ".artifacts")
TEXT_FILENAMES = {".comfyignore", ".gitattributes", ".gitignore", "DISCLAIMER", "LICENSE"}
TEXT_SUFFIXES = {".cfg", ".ini", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def project_version() -> str:
    payload = tomllib.loads((PLUGIN_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return str(payload["project"]["version"])


def package_version() -> str:
    source = (PLUGIN_ROOT / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"$', source, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError("Unable to read __version__ from __init__.py")
    return match.group(1)


def workflow_versions() -> set[str]:
    versions: set[str] = set()
    for path in sorted((PLUGIN_ROOT / "example_workflows" / "ui").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        versions.update(
            str(node.get("properties", {}).get("ver"))
            for node in payload.get("nodes", [])
            if node.get("properties", {}).get("cnr_id") == REGISTRY_ID
        )
    return versions


def validate_versions(expected_tag: str = "") -> str:
    version = project_version()
    if package_version() != version:
        raise RuntimeError(f"pyproject.toml and __init__.py versions differ: {version} != {package_version()}")
    versions = workflow_versions()
    if versions != {version}:
        raise RuntimeError(f"Example workflow versions are stale: {sorted(versions)} != {[version]}")
    if expected_tag and expected_tag != f"v{version}":
        raise RuntimeError(f"Release tag {expected_tag!r} does not match package version v{version}")
    return version


def ignore_patterns() -> tuple[str, ...]:
    patterns = list(ALWAYS_IGNORED)
    for raw in (PLUGIN_ROOT / ".comfyignore").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            patterns.append(line.rstrip("/"))
    return tuple(patterns)


def is_ignored(relative: str, patterns: tuple[str, ...]) -> bool:
    return any(
        relative == pattern
        or relative.startswith(f"{pattern}/")
        or fnmatch.fnmatch(relative, pattern)
        for pattern in patterns
    )


def release_files() -> list[Path]:
    result = subprocess.run(
        ["git", "-C", str(PLUGIN_ROOT), "ls-files", "--cached", "-z"],
        check=True,
        capture_output=True,
    )
    patterns = ignore_patterns()
    files = []
    for raw in result.stdout.decode("utf-8").split("\0"):
        if not raw:
            continue
        relative = raw.replace("\\", "/")
        path = PLUGIN_ROOT / relative
        if path.is_symlink():
            raise RuntimeError(f"Release archives do not allow symbolic links: {relative}")
        if path.is_file() and not is_ignored(relative, patterns):
            files.append(path)
    return sorted(files, key=lambda item: item.relative_to(PLUGIN_ROOT).as_posix())


def source_commit() -> str:
    result = subprocess.run(
        ["git", "-C", str(PLUGIN_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def source_is_dirty() -> bool:
    result = subprocess.run(
        ["git", "-C", str(PLUGIN_ROOT), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def tagged_commit(tag: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(PLUGIN_ROOT), "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Release tag does not exist locally: {tag}") from exc
    return result.stdout.strip()


def validate_release_source(expected_tag: str = "") -> None:
    if not expected_tag:
        return
    head = source_commit()
    resolved_tag = tagged_commit(expected_tag)
    if resolved_tag != head:
        raise RuntimeError(f"Release tag {expected_tag} points to {resolved_tag}, but HEAD is {head}")
    if source_is_dirty():
        raise RuntimeError("Tagged release builds require a clean working tree")


def release_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    if path.name in TEXT_FILENAMES or path.suffix.lower() in TEXT_SUFFIXES:
        return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return data


def write_archive(path: Path, files: list[Path]) -> tuple[int, int]:
    unpacked_size = 0
    with zipfile.ZipFile(
        path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=True,
    ) as archive:
        for source in files:
            relative = source.relative_to(PLUGIN_ROOT).as_posix()
            data = release_bytes(source)
            unpacked_size += len(data)
            info = zipfile.ZipInfo(relative, date_time=FIXED_ZIP_TIME)
            info.create_system = 3
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return len(files), unpacked_size


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_bytes((json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def build_release(output: Path, *, expected_tag: str = "") -> list[Path]:
    version = validate_versions(expected_tag)
    validate_release_source(expected_tag)
    output.mkdir(parents=True, exist_ok=True)
    files = release_files()
    if not files:
        raise RuntimeError("No release files were selected")

    archive = output / f"{ARCHIVE_STEM}-v{version}.zip"
    manifest = output / f"release-manifest-v{version}.json"
    sums = output / f"SHA256SUMS-v{version}.txt"
    for stale in (archive, manifest, sums):
        stale.unlink(missing_ok=True)

    file_count, unpacked_size = write_archive(archive, files)
    archive_digest = sha256(archive)
    model_manifest = json.loads((PLUGIN_ROOT / "manifests" / "model_0_9b.json").read_text(encoding="utf-8"))
    write_json(
        manifest,
        {
            "archive": {
                "file_count": file_count,
                "name": archive.name,
                "sha256": archive_digest,
                "size": archive.stat().st_size,
                "unpacked_size": unpacked_size,
            },
            "model_revision": model_manifest["revision"],
            "node_version": version,
            "schema_version": 1,
            "source_commit": source_commit(),
            "source_dirty": source_is_dirty(),
            "source_repository": "T8mars/Comfyui-MOSS-Transcribe-Diarize-T8",
            "upstream_code_revision": model_manifest["code_revision"],
        },
    )
    sums.write_bytes(
        f"{archive_digest}  {archive.name}\n{sha256(manifest)}  {manifest.name}\n".encode("utf-8")
    )
    return [archive, manifest, sums]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic ComfyUI node release archive.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-tag", default="")
    args = parser.parse_args()
    for path in build_release(args.output.resolve(), expected_tag=args.expected_tag):
        print(f"{sha256(path)}  {path}")


if __name__ == "__main__":
    main()
