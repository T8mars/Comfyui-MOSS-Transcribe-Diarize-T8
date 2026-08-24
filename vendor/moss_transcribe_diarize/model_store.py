from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from filelock import FileLock


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "manifests" / "model_0_9b.json"
DEFAULT_MODEL_DIR = ROOT / "pretrained" / "moss-transcribe-diarize"
DOWNLOAD_REPORT_NAME = ".t8-download-report.json"
DOWNLOAD_EVENT_PREFIX = "T8_DOWNLOAD_EVENT "


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
    event_callback: Callable[[dict[str, Any]], None] | None = None,
) -> ModelValidationReport:
    """Download the pinned snapshot, validate it, and atomically expose a new model directory."""
    manifest = manifest or ModelManifest.load()
    target = Path(target_dir).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_name(f"{target.name}.download.lock")

    def report(message: str) -> None:
        if progress is not None:
            progress(message)

    def emit(event: dict[str, Any]) -> None:
        if event_callback is not None:
            event_callback(dict(event))

    with FileLock(str(lock_path)):
        destination = target.with_name(f"{target.name}.partial")
        previous = target.with_name(f"{target.name}.previous")
        if not target.exists() and previous.exists():
            os.replace(previous, target)
        existing = validate_model_dir(target, manifest=manifest, verify_hashes=verify_hashes)
        if existing.valid:
            _remove_path(destination)
            _remove_path(previous)
            report("Model snapshot is already complete.")
            telemetry = _download_telemetry_base(manifest, target, verify_hashes)
            telemetry.update({
                "phase": "already_complete",
                "finished_utc": _utc_now(),
                "required_bytes": sum(spec.size for spec in manifest.files),
                "verified_hashes": verify_hashes,
                "hash_results": _hash_results(manifest, verify_hashes),
            })
            _write_download_report(target / DOWNLOAD_REPORT_NAME, telemetry)
            emit(telemetry)
            return existing

        if destination.exists() and not destination.is_dir():
            _remove_path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        staging_bytes_before = _directory_bytes(destination)
        required_bytes = sum(spec.size for spec in manifest.files)
        expected_existing_bytes = sum(
            min((destination.joinpath(*spec.path.split("/")).stat().st_size if destination.joinpath(*spec.path.split("/")).is_file() else 0), spec.size)
            for spec in manifest.files
        )
        disk = shutil.disk_usage(target.parent)
        estimated_additional_bytes = max(0, required_bytes - expected_existing_bytes)
        safety_margin_bytes = 512 * 1024 * 1024
        telemetry = _download_telemetry_base(manifest, target, verify_hashes)
        telemetry.update({
            "phase": "starting",
            "required_bytes": required_bytes,
            "staging_bytes_before": staging_bytes_before,
            "expected_file_bytes_before": expected_existing_bytes,
            "estimated_additional_bytes": estimated_additional_bytes,
            "disk_free_bytes_before": disk.free,
            "safety_margin_bytes": safety_margin_bytes,
        })
        _write_download_report(destination / DOWNLOAD_REPORT_NAME, telemetry)
        emit(telemetry)
        if disk.free < estimated_additional_bytes + safety_margin_bytes:
            telemetry.update({"phase": "failed", "finished_utc": _utc_now(), "error_type": "InsufficientDiskSpace"})
            _write_download_report(destination / DOWNLOAD_REPORT_NAME, telemetry)
            emit(telemetry)
            raise RuntimeError(
                "Insufficient disk space for the pinned model download: "
                f"need about {(estimated_additional_bytes + safety_margin_bytes) / 1024**3:.2f}GB, "
                f"have {disk.free / 1024**3:.2f}GB free."
            )
        report(f"Downloading pinned model revision {manifest.revision}.")
        try:
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
            staging_bytes_after = _directory_bytes(destination)
            validated = validate_model_dir(destination, manifest=manifest, verify_hashes=verify_hashes)
            validated.require_valid()
            disk_after = shutil.disk_usage(target.parent)
            telemetry.update({
                "phase": "complete",
                "finished_utc": _utc_now(),
                "staging_bytes_after": staging_bytes_after,
                "bytes_added_to_staging": max(0, staging_bytes_after - staging_bytes_before),
                "resume_candidate_bytes_before": staging_bytes_before,
                "disk_free_bytes_after": disk_after.free,
                "verified_hashes": verify_hashes,
                "hash_results": _hash_results(manifest, verify_hashes),
            })
            _write_download_report(destination / DOWNLOAD_REPORT_NAME, telemetry)
            moved_previous = False
            if target.exists():
                _remove_path(previous)
                os.replace(target, previous)
                moved_previous = True
            try:
                os.replace(destination, target)
            except Exception:
                if moved_previous and previous.exists() and not target.exists():
                    os.replace(previous, target)
                raise
            _remove_path(previous)
            validated = validate_model_dir(target, manifest=manifest, verify_hashes=verify_hashes)
            report("Model snapshot is ready.")
            emit(telemetry)
            return validated
        except Exception as exc:
            telemetry.update({
                "phase": "failed",
                "finished_utc": _utc_now(),
                "error_type": type(exc).__name__,
                "staging_bytes_after": _directory_bytes(destination),
            })
            if destination.exists():
                _write_download_report(destination / DOWNLOAD_REPORT_NAME, telemetry)
            emit(telemetry)
            raise


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _directory_bytes(path: Path) -> int:
    total = 0
    if not path.is_dir():
        return total
    for item in path.rglob("*"):
        if item.is_file() and item.name != DOWNLOAD_REPORT_NAME:
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total


def _download_telemetry_base(manifest: ModelManifest, target: Path, verify_hashes: bool) -> dict[str, Any]:
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    endpoint_host = urlsplit(endpoint).hostname or "custom"
    return {
        "schema": "t8.moss-download.v1",
        "started_utc": _utc_now(),
        "model_id": manifest.model_id,
        "revision": manifest.revision,
        "audited_code_revision": manifest.code_revision,
        "download_backend": "huggingface_hub",
        "endpoint_host": endpoint_host,
        "proxy_configured": any(os.environ.get(name) for name in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY")),
        "offline_mode": any(os.environ.get(name) == "1" for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")),
        "verify_hashes_requested": verify_hashes,
        "target_volume": target.drive or target.anchor,
        "remote_analytics": False,
    }


def _hash_results(manifest: ModelManifest, verified: bool) -> list[dict[str, Any]]:
    status = "match" if verified else "not_checked"
    return [
        {"path": spec.path, "sha256": spec.sha256, "status": status}
        for spec in manifest.files
    ]


def _write_download_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


__all__ = [
    "DEFAULT_MANIFEST",
    "DEFAULT_MODEL_DIR",
    "DOWNLOAD_EVENT_PREFIX",
    "DOWNLOAD_REPORT_NAME",
    "ModelFileSpec",
    "ModelManifest",
    "ModelValidationIssue",
    "ModelValidationReport",
    "download_model_snapshot",
    "sha256_file",
    "validate_model_dir",
]
