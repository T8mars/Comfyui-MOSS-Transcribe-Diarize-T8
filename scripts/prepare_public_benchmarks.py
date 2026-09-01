from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "benchmarks" / "public_fleurs.json"
DATASET_SERVER = "https://datasets-server.huggingface.co/rows"
DATASET_API = "https://huggingface.co/api/datasets/google/fleurs"
SAMPLE_RATE = 16000


def _json_request(url: str, *, attempts: int = 4) -> dict:
    last_error = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.load(response)
        except Exception as exc:  # noqa: BLE001 - bounded public corpus retry
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Unable to read public corpus metadata: {url}") from last_error


def _rows(config: str, *, offset: int, length: int) -> list[dict]:
    query = urllib.parse.urlencode(
        {
            "dataset": "google/fleurs",
            "config": config,
            "split": "validation",
            "offset": offset,
            "length": length,
        }
    )
    payload = _json_request(f"{DATASET_SERVER}?{query}")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError(f"FLEURS rows response is invalid for {config}:{offset}.")
    return rows


def _verify_source_revision(source: dict) -> None:
    if source.get("dataset") != "google/fleurs" or source.get("split") != "validation":
        raise ValueError("Public benchmark currently supports only google/fleurs validation rows.")
    expected = str(source.get("revision") or "").strip()
    if len(expected) != 40:
        raise ValueError("Public FLEURS manifest must pin a full 40-character revision.")
    actual = str(_json_request(DATASET_API).get("sha") or "").strip()
    if actual != expected:
        raise RuntimeError(
            "The dataset-viewer rows API serves only the current dataset revision, and google/fleurs "
            f"moved from the audited {expected} to {actual or 'unknown'}. Review and update "
            "benchmarks/public_fleurs.json before generating new benchmark media."
        )


def _download(url: str, destination: Path, *, attempts: int = 4) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return _sha256(destination)
    temporary = destination.with_suffix(destination.suffix + ".part")
    last_error = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "T8-MOSS-public-benchmark/0.4.0"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            temporary.replace(destination)
            return _sha256(destination)
        except Exception as exc:  # noqa: BLE001 - bounded public audio retry
            last_error = exc
            temporary.unlink(missing_ok=True)
            if attempt + 1 < attempts:
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"Unable to download public corpus audio: {url}") from last_error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_pcm(path: Path) -> np.ndarray:
    samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz FLEURS WAV: {path}")
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    return np.round(np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)


def _write_pcm(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(path),
        np.asarray(samples, dtype=np.float32) / 32768.0,
        SAMPLE_RATE,
        subtype="PCM_16",
    )


def _compose(items: list[dict], *, silence_seconds: float, noise_snr_db: float | None = None) -> np.ndarray:
    silence = np.zeros(max(0, int(round(silence_seconds * SAMPLE_RATE))), dtype=np.int16)
    parts = []
    for index, item in enumerate(items):
        if index:
            parts.append(silence)
        parts.append(_read_pcm(Path(item["local_audio"])))
    samples = np.concatenate(parts) if parts else np.empty(0, dtype=np.int16)
    if noise_snr_db is None or not samples.size:
        return samples
    signal = samples.astype(np.float32) / 32768.0
    signal_power = float(np.mean(signal**2))
    if signal_power <= 0:
        return samples
    noise_power = signal_power / (10.0 ** (float(noise_snr_db) / 10.0))
    noise = np.random.default_rng(8040).normal(0.0, np.sqrt(noise_power), signal.size).astype(np.float32)
    mixed = np.clip(signal + noise, -1.0, 1.0)
    return np.round(mixed * 32767.0).astype(np.int16)


def _materialize_rows(config: str, wanted: list[int], cache_dir: Path) -> list[dict]:
    wanted_set = set(wanted)
    found = {}
    for offset in range(0, max(wanted_set, default=0) + 1, 100):
        for item in _rows(config, offset=offset, length=min(100, max(wanted_set) - offset + 1)):
            row_index = int(item["row_idx"])
            if row_index not in wanted_set:
                continue
            row = item["row"]
            audio = row.get("audio") or []
            if not audio or not audio[0].get("src"):
                raise RuntimeError(f"FLEURS row has no audio URL: {config}:{row_index}")
            destination = cache_dir / config / f"{row_index:05d}.wav"
            digest = _download(str(audio[0]["src"]), destination)
            found[row_index] = {
                "config": config,
                "row": row_index,
                "dataset_id": row.get("id"),
                "gender": row.get("gender"),
                "text": str(row.get("raw_transcription") or row.get("transcription") or "").strip(),
                "num_samples": int(row.get("num_samples") or 0),
                "sha256": digest,
                "local_audio": str(destination),
            }
    missing = wanted_set - set(found)
    if missing:
        raise RuntimeError(f"FLEURS rows not found for {config}: {sorted(missing)}")
    return [found[index] for index in wanted]


def _long_rows(configs: list[str], target_seconds: float, cache_dir: Path) -> list[dict]:
    selected = []
    duration = 0.0
    for page in range(3):
        for config in configs:
            page_rows = _rows(config, offset=page * 100, length=100)
            indices = [int(item["row_idx"]) for item in page_rows]
            rows = _materialize_rows(config, indices, cache_dir)
            for row in rows:
                selected.append(row)
                duration += int(row["num_samples"]) / float(SAMPLE_RATE) + 0.35
                if duration >= target_seconds:
                    return selected
    raise RuntimeError(f"Public FLEURS validation rows provide only {duration / 60.0:.1f} minutes.")


def prepare(source_path: Path, output_root: Path, *, long_minutes: float) -> tuple[Path, Path]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("schema") != "t8.moss-public-benchmark-source.v1":
        raise ValueError("Unsupported public benchmark source manifest.")
    _verify_source_revision(source)
    audio_dir = output_root / "audio"
    reference_dir = output_root / "references"
    cache_dir = output_root / "source-cache"
    reference_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    provenance_cases = {}
    for case_id, requested in source["cases"].items():
        by_config: dict[str, list[int]] = {}
        for item in requested:
            by_config.setdefault(str(item["config"]), []).append(int(item["row"]))
        materialized = {
            (item["config"], item["row"]): item
            for config, indices in by_config.items()
            for item in _materialize_rows(config, indices, cache_dir)
        }
        rows = [materialized[(str(item["config"]), int(item["row"]))] for item in requested]
        audio_path = audio_dir / f"{case_id}.wav"
        _write_pcm(audio_path, _compose(rows, silence_seconds=0.45))
        reference_path = reference_dir / f"{case_id}.txt"
        reference_path.write_text(" ".join(item["text"] for item in rows) + "\n", encoding="utf-8")
        if case_id == "zh-two-speaker-meeting":
            prompt = {"preset_id": "zh_meeting", "language_hint": "中文", "strict_format": True}
            expected = {"usable": True, "min_segments": 2, "min_speakers": 2, "min_end_coverage": 0.8, "max_character_error_rate": 0.45, "min_word_alignment_coverage": 0.5}
        elif case_id == "en-hotwords-capitalization":
            prompt = {"preset_id": "en_meeting", "language_hint": "English", "hotwords": "Javanese, Lakkha Singh", "strict_format": True}
            expected = {
                "usable": True,
                "min_segments": 2,
                "min_end_coverage": 0.8,
                "required_text": ["Lakkha Singh"],
                "max_word_error_rate": 0.4,
                "min_word_alignment_coverage": 0.5,
            }
        else:
            prompt = {"preset_id": "multilingual", "language_hint": "auto", "strict_format": True}
            expected = {"usable": True, "min_segments": 3, "min_end_coverage": 0.75, "max_character_error_rate": 0.5, "min_word_alignment_coverage": 0.5}
        cases.append(
            {
                "id": case_id,
                "audio": f"audio/{audio_path.name}",
                "reference_text_file": f"references/{reference_path.name}",
                "mode": "single",
                "options": {"word_alignment": True},
                "prompt": prompt,
                "expected": expected,
            }
        )
        provenance_cases[case_id] = rows

    long_config = source["long_case"]
    configured_minutes = float(long_config.get("target_minutes") or 30.0)
    long_id = str(long_config["id"])
    if abs(long_minutes - configured_minutes) > 0.001:
        long_id = f"noisy-multilingual-smoke-{long_minutes:g}min"
    long_rows = _long_rows(list(long_config["configs"]), long_minutes * 60.0, cache_dir)
    long_audio = _compose(
        long_rows,
        silence_seconds=float(long_config["silence_between_clips_seconds"]),
        noise_snr_db=float(long_config["noise_snr_db"]),
    )[: int(round(long_minutes * 60.0 * SAMPLE_RATE))]
    long_audio_path = audio_dir / f"{long_id}.wav"
    _write_pcm(long_audio_path, long_audio)
    long_reference = reference_dir / f"{long_id}.txt"
    long_reference.write_text(" ".join(item["text"] for item in long_rows) + "\n", encoding="utf-8")
    cases.append(
        {
            "id": long_id,
            "audio": f"audio/{long_audio_path.name}",
            "reference_text_file": f"references/{long_reference.name}",
            "mode": "smart_long",
            "prompt": {"preset_id": "multilingual", "language_hint": "auto", "strict_format": True},
            "options": {
                "target_chunk_seconds": 480,
                "max_chunk_seconds": 600,
                "overlap_seconds": 2,
                "speaker_link_mode": "overlap_only",
                "speaker_embedding_link": True,
                "speaker_similarity_threshold": 0.86,
            },
            "expected": {
                "usable": True,
                "min_segments": max(2, int(round(long_minutes * 2.0 / 3.0))),
                "min_speakers": 2,
                "min_end_coverage": 0.7,
                "max_character_error_rate": 0.6,
                "min_speaker_embedding_links": 1 if long_minutes >= configured_minutes else 0,
                "max_speaker_embedding_failures": 0,
                "forbidden_diagnostic_codes": ["invalid_format", "timestamp_out_of_range"],
            },
        }
    )
    provenance_cases[long_id] = long_rows
    manifest_path = output_root / "cases.public-real.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "t8.moss-benchmark.v1",
                "description": f"Public FLEURS real-speech regression matrix ({long_minutes:g}-minute noisy long case).",
                "cases": cases,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    provenance_path = output_root / "provenance.public-real.json"
    provenance_path.write_text(
        json.dumps(
            {
                "schema": "t8.moss-public-benchmark-provenance.v1",
                "dataset": source["dataset"],
                "revision": source["revision"],
                "license": source["license"],
                "dataset_card": source["dataset_card"],
                "attribution": source["attribution"],
                "cases": provenance_cases,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path, provenance_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and compose licensed FLEURS real-speech benchmarks.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=ROOT / "benchmarks" / "generated" / "public-real")
    parser.add_argument("--long-minutes", type=float, default=30.0)
    args = parser.parse_args()
    if args.long_minutes < 1.0 or args.long_minutes > 60.0:
        raise ValueError("--long-minutes must be between 1 and 60.")
    manifest, provenance = prepare(args.source.resolve(), args.output_root.resolve(), long_minutes=args.long_minutes)
    print(json.dumps({"manifest": str(manifest), "provenance": str(provenance)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
