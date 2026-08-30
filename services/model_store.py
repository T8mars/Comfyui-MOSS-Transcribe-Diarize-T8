from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PLUGIN_ROOT / "manifests" / "model_0_9b.json"
MODEL_FOLDER_NAME = "MOSS-Transcribe-Diarize"
MODEL_PATH_KEY = "moss_transcribe_diarize"
MISSING_MODEL_OPTION = "[未找到] 请将模型放入 models/moss_transcribe_diarize"


@dataclass(frozen=True, slots=True)
class ValidationReport:
    model_dir: Path
    missing: tuple[str, ...] = ()
    mismatched: tuple[str, ...] = ()
    hashes_verified: bool = False

    @property
    def valid(self) -> bool:
        return not self.missing and not self.mismatched

    def require_valid(self) -> None:
        if self.valid:
            return
        details = []
        if self.missing:
            details.append("缺少：" + ", ".join(self.missing))
        if self.mismatched:
            details.append("大小或哈希不匹配：" + ", ".join(self.mismatched))
        raise FileNotFoundError(f"MOSS 模型目录不完整：{self.model_dir}；" + "；".join(details))


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def register_model_paths() -> None:
    import folder_paths

    default_root = Path(folder_paths.models_dir) / MODEL_PATH_KEY
    default_root.mkdir(parents=True, exist_ok=True)
    folder_paths.add_model_folder_path(MODEL_PATH_KEY, str(default_root), is_default=True)


def configured_model_roots() -> list[Path]:
    import folder_paths

    register_model_paths()
    return [Path(item).expanduser().resolve() for item in folder_paths.get_folder_paths(MODEL_PATH_KEY)]


def _looks_like_model(path: Path) -> bool:
    return (path / "config.json").is_file() and (path / "model-00000-of-00001.safetensors").is_file()


def discover_models(roots: Iterable[Path] | None = None) -> dict[str, Path]:
    roots = list(roots) if roots is not None else configured_model_roots()
    candidates: list[tuple[str, Path]] = []
    for root_value in roots:
        root = Path(root_value).expanduser().resolve()
        if _looks_like_model(root):
            candidates.append((root.name, root))
        if not root.is_dir():
            continue
        for current, directories, files in os.walk(root):
            model_dir = Path(current).resolve()
            try:
                relative = model_dir.relative_to(root)
            except ValueError:
                continue
            if len(relative.parts) >= 3:
                directories.clear()
            else:
                directories[:] = [name for name in directories if name not in {".git", "__pycache__", ".cache"}]
            if "config.json" in files and _looks_like_model(model_dir):
                candidates.append((root.name if relative == Path(".") else relative.as_posix(), model_dir))

    result: dict[str, Path] = {}
    seen: set[Path] = set()
    for label, path in sorted(candidates, key=lambda item: (item[0].lower(), str(item[1]).lower())):
        if path in seen:
            continue
        seen.add(path)
        option = label or path.name
        suffix = 2
        while option in result:
            option = f"{label} [{suffix}]"
            suffix += 1
        result[option] = path
    return result


def model_options() -> list[str]:
    return list(discover_models()) or [MISSING_MODEL_OPTION]


def resolve_model(model_name: str, custom_model_path: str = "") -> Path:
    if custom_model_path.strip():
        return Path(custom_model_path.strip().strip('"')).expanduser().resolve()
    models = discover_models()
    if model_name == MISSING_MODEL_OPTION or model_name not in models:
        expected = configured_model_roots()[0] / MODEL_FOLDER_NAME
        raise FileNotFoundError(
            "未找到完整的 MOSS Transcribe Diarize 模型。请运行节点目录中的 scripts/download_models.py，"
            f"或把模型放到：{expected}"
        )
    return models[model_name]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model_dir(model_dir: Path | str, *, verify_hashes: bool = False) -> ValidationReport:
    model_dir = Path(model_dir).expanduser().resolve()
    missing: list[str] = []
    mismatched: list[str] = []
    for relative, metadata in load_manifest()["files"].items():
        path = model_dir.joinpath(*relative.split("/"))
        if not path.is_file():
            missing.append(relative)
        elif path.stat().st_size != int(metadata["size"]):
            mismatched.append(relative)
        elif verify_hashes and _sha256(path).lower() != str(metadata["sha256"]).lower():
            mismatched.append(relative)
    return ValidationReport(model_dir, tuple(missing), tuple(mismatched), verify_hashes)


def model_fingerprint(model_dir: Path | str) -> str:
    model_dir = Path(model_dir)
    manifest = load_manifest()
    parts = [manifest["revision"]]
    for relative in manifest["files"]:
        path = model_dir.joinpath(*relative.split("/"))
        try:
            stat = path.stat()
            parts.append(
                f"{relative}:{stat.st_size}:{stat.st_mtime_ns}:"
                f"{getattr(stat, 'st_ctime_ns', 0)}:{getattr(stat, 'st_ino', 0)}:"
                f"{_sha256(path)}"
            )
        except FileNotFoundError:
            parts.append(f"{relative}:missing")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


__all__ = [
    "MISSING_MODEL_OPTION",
    "ValidationReport",
    "configured_model_roots",
    "discover_models",
    "load_manifest",
    "model_fingerprint",
    "model_options",
    "register_model_paths",
    "resolve_model",
    "validate_model_dir",
]
