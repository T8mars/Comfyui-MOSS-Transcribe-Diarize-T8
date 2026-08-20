from __future__ import annotations

import argparse
import sys
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from vendor.moss_transcribe_diarize.model_store import ModelManifest, download_model_snapshot


def default_target() -> Path:
    try:
        import folder_paths

        return Path(folder_paths.models_dir) / "moss_transcribe_diarize" / "MOSS-Transcribe-Diarize"
    except ImportError:
        return PLUGIN_ROOT / "model-downloads" / "MOSS-Transcribe-Diarize"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify the pinned MOSS model snapshot.")
    parser.add_argument("--target", type=Path, default=default_target())
    parser.add_argument("--skip-hash", action="store_true", help="Only verify required file sizes.")
    args = parser.parse_args()
    manifest = ModelManifest.load(PLUGIN_ROOT / "manifests" / "model_0_9b.json")
    report = download_model_snapshot(args.target, manifest=manifest, verify_hashes=not args.skip_hash, progress=print)
    print(f"Model ready: {report.model_dir}")
    print(f"Pinned revision: {manifest.revision}")


if __name__ == "__main__":
    main()
