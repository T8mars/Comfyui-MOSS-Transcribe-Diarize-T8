from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from filelock import FileLock


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "manifests" / "model_0_9b.json"
DEFAULT_MODEL_DIR = ROOT / "pretrained" / "moss-transcribe-diarize"


@dataclass(frozen=True, slots=True)
class ModelFileSpec:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ModelManifest:
    model_id: str
    revision: str
    code_repository: str
    code_revision: str
    license: str
    files: tuple[ModelFileSpec, ...]
    source_path: Path

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MANIFEST) -> "ModelManifest":
        source = Path(path).expanduser().resolve()
        data = json.loads(source.read_text(encoding="utf-8"))
        files = tuple(
            ModelFileSpec(path=name, size=int(meta["size"]), sha256=str(meta["sha256"]).lower())
            for name, meta in data["files"].items()
        )
        return cls(
            model_id=str(data["model_id"]),
            revision=str(data["revision"]),
            code_repository=str(data["code_repository"]),
            code_revision=str(data["code_revision"]),
            license=str(data["license"]),
            files=files,
            source_path=source,
        )


@dataclass(frozen=True, slots=True)
class ModelValidationIssue:
    path: str
    code: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "code": self.code, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class ModelValidationReport:
    model_dir: Path
    revision: str
    checked_hashes: bool
    issues: tuple[ModelValidationIssue, ...]

    @property
    def valid(self) -> bool:
        return not self.issues

    def require_valid(self) -> None:
        if self.valid:
            return
        summary = "; ".join(f"{issue.path}: {issue.detail}" for issue in self.issues[:5])
        raise RuntimeError(f"Model validation failed: {summary}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_dir": str(self.model_dir),
            "revision": self.revision,
            "checked_hashes": self.checked_hashes,
            "valid": self.valid,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model_dir(
    model_dir: str | Path,
    *,
    manifest: ModelManifest | None = None,
    verify_hashes: bool = False,
) -> ModelValidationReport:
    manifest = manifest or ModelManifest.load()
    model_dir = Path(model_dir).expanduser().resolve()
    issues: list[ModelValidationIssue] = []
    if not model_dir.is_dir():
        issues.append(ModelValidationIssue(".", "model_dir_missing", "Model directory does not exist."))
        return ModelValidationReport(model_dir, manifest.revision, verify_hashes, tuple(issues))

    for spec in manifest.files:
        path = model_dir.joinpath(*spec.path.split("/"))
        if not path.is_file():
            issues.append(ModelValidationIssue(spec.path, "missing", "Required file is missing."))
            continue
        actual_size = path.stat().st_size
        if actual_size != spec.size:
            issues.append(
                ModelValidationIssue(
                    spec.path,
                    "size_mismatch",
                    f"Expected {spec.size} bytes, found {actual_size}.",
                )
            )
            continue
        if verify_hashes:
            actual_hash = sha256_file(path)
            if actual_hash != spec.sha256:
                issues.append(
                    ModelValidationIssue(
                        spec.path,
                        "sha256_mismatch",
                        f"Expected {spec.sha256}, found {actual_hash}.",
                    )
                )
    return ModelValidationReport(model_dir, manifest.revision, verify_hashes, tuple(issues))


def download_model_snapshot(
    target_dir: str | Path = DEFAULT_MODEL_DIR,
    *,
    manifest: ModelManifest | None = None,
    token: str | None = None,
    verify_hashes: bool = True,
    progress: Callable[[str], None] | None = None,
) -> ModelValidationReport:
    """Download the pinned snapshot, validate it, and atomically expose a new model directory."""
    manifest = manifest or ModelManifest.load()
    target = Path(target_dir).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_name(f"{target.name}.download.lock")

    def report(message: str) -> None:
        if progress is not None:
            progress(message)

    with FileLock(str(lock_path)):
        existing = validate_model_dir(target, manifest=manifest, verify_hashes=verify_hashes)
        if existing.valid:
            report("Model snapshot is already complete.")
            return existing

        is_repair = target.exists()
        destination = target if is_repair else target.with_name(f"{target.name}.partial")
        destination.mkdir(parents=True, exist_ok=True)
        report(f"Downloading pinned model revision {manifest.revision}.")
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("Install huggingface-hub to download the model snapshot.") from exc

        snapshot_download(
            repo_id=manifest.model_id,
            revision=manifest.revision,
            local_dir=str(destination),
            allow_patterns=[spec.path for spec in manifest.files],
            token=token,
        )
        validated = validate_model_dir(destination, manifest=manifest, verify_hashes=verify_hashes)
        validated.require_valid()
        if not is_repair:
            os.replace(destination, target)
            validated = validate_model_dir(target, manifest=manifest, verify_hashes=verify_hashes)
        report("Model snapshot is ready.")
        return validated


__all__ = [
    "DEFAULT_MANIFEST",
    "DEFAULT_MODEL_DIR",
    "ModelFileSpec",
    "ModelManifest",
    "ModelValidationIssue",
    "ModelValidationReport",
    "download_model_snapshot",
    "sha256_file",
    "validate_model_dir",
]
