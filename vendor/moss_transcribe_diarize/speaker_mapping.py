from __future__ import annotations

from collections.abc import Mapping


def normalize_speaker_mapping(value: Mapping | None) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("Speaker mapping must be a JSON object.")
    result = {}
    for key, target in value.items():
        source = str(key).strip()
        resolved = str(target).strip()
        if source and resolved:
            result[source] = resolved
    return result


def resolve_speaker_names(
    local_names: Mapping | None,
    cross_chunk_mapping: Mapping | None,
    *,
    chunk_id: str = "",
) -> dict[str, str]:
    names = normalize_speaker_mapping(local_names)
    mapping = normalize_speaker_mapping(cross_chunk_mapping)
    prefix = f"{chunk_id.strip()}:" if chunk_id.strip() else ""
    for key, target in mapping.items():
        if ":" in key:
            if prefix and key.startswith(prefix):
                names[key[len(prefix):]] = target
        else:
            names[key] = target
    return names


__all__ = ["normalize_speaker_mapping", "resolve_speaker_names"]
