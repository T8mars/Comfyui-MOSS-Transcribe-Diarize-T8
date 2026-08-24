from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from vendor.moss_transcribe_diarize.model_store import DOWNLOAD_EVENT_PREFIX, ModelManifest, download_model_snapshot


def discover_comfyui_root() -> Path | None:
    configured = os.environ.get("COMFYUI_ROOT", "").strip()
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if (candidate / "models").is_dir() or (candidate / "main.py").is_file():
            return candidate

    # Normal Git/ZIP install: ComfyUI/custom_nodes/<this package>.
    custom_nodes = PLUGIN_ROOT.parent
    if custom_nodes.name.lower() == "custom_nodes":
        candidate = custom_nodes.parent.resolve()
        if (candidate / "models").is_dir() or (candidate / "main.py").is_file():
            return candidate
    return None


def default_target(comfyui_root: Path | None = None) -> Path:
    if comfyui_root is not None:
        root = Path(comfyui_root).expanduser().resolve()
        return root / "models" / "moss_transcribe_diarize" / "MOSS-Transcribe-Diarize"
    try:
        import folder_paths

        return Path(folder_paths.models_dir) / "moss_transcribe_diarize" / "MOSS-Transcribe-Diarize"
    except ImportError:
        root = discover_comfyui_root()
        if root is None:
            raise RuntimeError(
                "Cannot locate ComfyUI. Run with --comfyui-root <ComfyUI directory> or "
                "--target <ComfyUI/models/moss_transcribe_diarize/MOSS-Transcribe-Diarize>."
            )
        return root / "models" / "moss_transcribe_diarize" / "MOSS-Transcribe-Diarize"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify the pinned MOSS model snapshot.")
    parser.add_argument("--comfyui-root", type=Path, default=None, help="ComfyUI root used to derive the models directory.")
    parser.add_argument("--target", type=Path, default=None, help="Explicit model target directory.")
    parser.add_argument("--skip-hash", action="store_true", help="Only verify required file sizes.")
    args = parser.parse_args()
    target = args.target or default_target(args.comfyui_root)
    print(f"Model target: {target}")
    manifest = ModelManifest.load(PLUGIN_ROOT / "manifests" / "model_0_9b.json")
    report = download_model_snapshot(
        target,
        manifest=manifest,
        verify_hashes=not args.skip_hash,
        progress=print,
        event_callback=lambda event: print(
            DOWNLOAD_EVENT_PREFIX + json.dumps(event, ensure_ascii=False, separators=(",", ":")),
            flush=True,
        ),
    )
    print(f"Model ready: {report.model_dir}")
    print(f"Pinned revision: {manifest.revision}")


if __name__ == "__main__":
    main()
