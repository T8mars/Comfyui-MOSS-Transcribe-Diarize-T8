from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PLUGIN_ROOT / "manifests" / "model_0_9b.json"
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def revision_targets(manifest: dict[str, Any]) -> tuple[tuple[str, str, str], ...]:
    code_revision = str(manifest["code_revision"])
    model_revision = str(manifest["revision"])
    for label, revision in (("OpenMOSS code", code_revision), ("Hugging Face model", model_revision)):
        if not FULL_SHA.fullmatch(revision):
            raise ValueError(f"{label} revision must be a full lowercase 40-character SHA: {revision!r}")

    code_repository = urllib.parse.quote(str(manifest["code_repository"]), safe="/")
    model_id = urllib.parse.quote(str(manifest["model_id"]), safe="/")
    return (
        (
            "OpenMOSS code",
            code_revision,
            f"https://api.github.com/repos/{code_repository}/commits/{code_revision}",
        ),
        (
            "Hugging Face model",
            model_revision,
            f"https://huggingface.co/api/models/{model_id}/revision/{model_revision}",
        ),
    )


def fetch_json(url: str, *, github_token: str = "", timeout: float = 30.0) -> dict[str, Any]:
    headers = {
        "Accept": "application/vnd.github+json" if url.startswith("https://api.github.com/") else "application/json",
        "User-Agent": "comfyui-MOSS-Transcribe-Diarize-T8-revision-verifier",
    }
    if github_token and url.startswith("https://api.github.com/"):
        headers["Authorization"] = f"Bearer {github_token}"
        headers["X-GitHub-Api-Version"] = "2022-11-28"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed HTTPS hosts
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Pinned revision did not resolve: {url} (HTTP {exc.code})") from None
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Unable to verify pinned revision at {url}: {exc.reason}") from None
    if not isinstance(payload, dict):
        raise RuntimeError(f"Revision endpoint returned an unexpected payload: {url}")
    return payload


def verify_manifest(path: Path = DEFAULT_MANIFEST, *, github_token: str = "") -> list[str]:
    manifest = load_manifest(path)
    verified: list[str] = []
    for label, revision, url in revision_targets(manifest):
        payload = fetch_json(url, github_token=github_token)
        resolved = str(payload.get("sha", ""))
        if resolved != revision:
            raise RuntimeError(f"{label} resolved to {resolved!r}, expected {revision!r}")
        verified.append(f"{label}: {revision}")
    return verified


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify that pinned GitHub and Hugging Face revisions exist.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    for result in verify_manifest(args.manifest.resolve(), github_token=os.environ.get("GITHUB_TOKEN", "")):
        print(f"Verified {result}")


if __name__ == "__main__":
    main()
