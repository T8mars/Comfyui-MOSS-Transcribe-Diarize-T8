from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


SAMPLE_RATE = 16000


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic non-speech MOSS benchmark fixtures.")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/generated"))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260830)
    fixtures = {
        "silence.wav": np.zeros(SAMPLE_RATE * 8, dtype=np.float32),
        "noise.wav": np.clip(rng.normal(0.0, 0.04, SAMPLE_RATE * 8), -1.0, 1.0).astype(np.float32),
        "music_like.wav": _music_like(12.0),
        "long_silence_noise.wav": np.concatenate(
            [
                np.zeros(SAMPLE_RATE * 20, dtype=np.float32),
                np.clip(rng.normal(0.0, 0.03, SAMPLE_RATE * 20), -1.0, 1.0).astype(np.float32),
                np.zeros(SAMPLE_RATE * 20, dtype=np.float32),
            ]
        ),
    }
    for name, samples in fixtures.items():
        sf.write(output / name, samples, SAMPLE_RATE, subtype="PCM_16")
    manifest = {
        "schema": "t8.moss-benchmark.v1",
        "description": "Deterministic non-speech guardrail fixtures; real speech cases live in benchmarks/cases.example.json.",
        "cases": [
            {
                "id": "silence-reject",
                "audio": "silence.wav",
                "mode": "single",
                "expected": {"usable": False, "min_segments": 0},
            },
            {
                "id": "noise-guardrail",
                "audio": "noise.wav",
                "mode": "single",
                "expected": {"usable": False},
            },
            {
                "id": "music-guardrail",
                "audio": "music_like.wav",
                "mode": "single",
                "expected": {"usable": False},
            },
            {
                "id": "long-nonspeech-chunking",
                "audio": "long_silence_noise.wav",
                "mode": "smart_long",
                "options": {"target_chunk_seconds": 20, "max_chunk_seconds": 25},
                "expected": {"usable": False},
            },
        ],
    }
    (output / "cases.synthetic.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Generated {len(fixtures)} fixtures and {output / 'cases.synthetic.json'}")


def _music_like(duration: float) -> np.ndarray:
    time_axis = np.arange(round(duration * SAMPLE_RATE), dtype=np.float64) / SAMPLE_RATE
    envelope = np.minimum(1.0, np.minimum(time_axis * 4.0, (duration - time_axis) * 4.0))
    signal = sum(np.sin(2 * np.pi * frequency * time_axis) for frequency in (220.0, 277.18, 329.63)) / 3.0
    return (signal * envelope * 0.18).astype(np.float32)


if __name__ == "__main__":
    main()
