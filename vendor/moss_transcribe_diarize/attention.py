"""Observable attention selection for long-form local inference.

``auto`` deliberately prefers efficient, bounded-memory implementations in the
same order as the audited upstream loader: FlashAttention-2, SDPA, then eager
as a clearly reported last resort. Explicit selections remain available for
diagnostics and controlled deployments, but are never silently rewritten or
cascaded to a different implementation.
"""

from __future__ import annotations

import gc
import importlib.util
import logging
import warnings
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F


LOGGER = logging.getLogger(__name__)

AUTO_ATTENTION_IMPLEMENTATION = "auto"
ATTENTION_IMPLEMENTATIONS = (
    "flash_attention_4",
    "flash_attention_3",
    "flash_attention_2",
    "sdpa",
    "eager",
)
_FLASH_IMPLEMENTATIONS = ATTENTION_IMPLEMENTATIONS[:3]


def normalize_attention_implementation(value: str | None) -> str:
    normalized = (value or AUTO_ATTENTION_IMPLEMENTATION).strip().lower()
    normalized = {
        "default": AUTO_ATTENTION_IMPLEMENTATION,
        "flash4": "flash_attention_4",
        "flash3": "flash_attention_3",
        "flash2": "flash_attention_2",
    }.get(normalized, normalized)
    if normalized not in (AUTO_ATTENTION_IMPLEMENTATION, *ATTENTION_IMPLEMENTATIONS):
        choices = ", ".join((AUTO_ATTENTION_IMPLEMENTATION, *ATTENTION_IMPLEMENTATIONS))
        raise ValueError(f"Unsupported attention implementation {value!r}; choose one of: {choices}.")
    return normalized


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _transformers_flash_available(implementation: str) -> bool | None:
    version = implementation.rsplit("_", 1)[-1]
    try:
        from transformers import utils as transformers_utils

        checker = getattr(transformers_utils, f"is_flash_attn_{version}_available", None)
        return None if checker is None else bool(checker())
    except Exception:  # noqa: BLE001 - version-specific optional capability
        return None


def _device_capability(device: torch.device) -> tuple[int, int] | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    try:
        return tuple(torch.cuda.get_device_capability(device))
    except (RuntimeError, AssertionError, ValueError):
        return None


def _flash_preflight(implementation: str, device: torch.device, dtype: torch.dtype) -> str | None:
    if device.type != "cuda":
        return "requires a CUDA device"
    if dtype not in (torch.float16, torch.bfloat16):
        return f"requires float16/bfloat16, got {dtype}"

    capability = _device_capability(device)
    if capability is None:
        return "CUDA device capability could not be determined"
    major, _ = capability
    if implementation == "flash_attention_4" and major < 9:
        return f"requires compute capability >= 9.x, got {capability}"
    if implementation == "flash_attention_3" and major < 8:
        return f"requires compute capability >= 8.x, got {capability}"
    if implementation == "flash_attention_2" and major < 8:
        return f"requires compute capability >= 8.x, got {capability}"

    module_name = {
        "flash_attention_4": "flash_attn",
        "flash_attention_3": "flash_attn_interface",
        "flash_attention_2": "flash_attn",
    }[implementation]
    if not _module_available(module_name):
        return f"optional package/module {module_name!r} is not installed"
    if _transformers_flash_available(implementation) is False:
        return f"Transformers reports {implementation} unavailable in this environment"
    return None


def _sdpa_available() -> bool:
    return callable(getattr(F, "scaled_dot_product_attention", None))


def _candidate_list(
    requested: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[str | None], list[dict[str, str]]]:
    if requested == AUTO_ATTENTION_IMPLEMENTATION:
        # Do not delegate ``auto`` to Transformers: some supported versions can
        # silently resolve the model to eager attention, whose memory use grows
        # quadratically with long audio. Keep the audited upstream order while
        # recording why optional candidates were skipped.
        candidates: list[str | None] = []
        attempts: list[dict[str, str]] = []

        flash_reason = _flash_preflight("flash_attention_2", device, dtype)
        if flash_reason is None:
            candidates.append("flash_attention_2")
        else:
            attempts.append({"backend": "flash_attention_2", "status": "skipped", "reason": flash_reason})

        if _sdpa_available():
            candidates.append("sdpa")
        else:
            attempts.append(
                {
                    "backend": "sdpa",
                    "status": "skipped",
                    "reason": "torch.nn.functional.scaled_dot_product_attention is unavailable",
                }
            )

        candidates.append("eager")
        return candidates, attempts

    if requested in _FLASH_IMPLEMENTATIONS:
        reason = _flash_preflight(requested, device, dtype)
        if reason is not None:
            return [], [{"backend": requested, "status": "failed", "reason": reason}]
    elif requested == "sdpa" and not _sdpa_available():
        return [], [
            {
                "backend": "sdpa",
                "status": "failed",
                "reason": "torch.nn.functional.scaled_dot_product_attention is unavailable",
            }
        ]
    return [requested], []


def _config_attention_values(model: Any) -> dict[str, str | None]:
    config = getattr(model, "config", None)
    values: dict[str, str | None] = {}
    for name, item in (
        ("model", config),
        ("text", getattr(config, "text_config", None)),
        ("audio", getattr(config, "audio_config", None)),
    ):
        value = None
        if item is not None:
            value = getattr(item, "_attn_implementation", None)
            if value is None:
                value = getattr(item, "_attn_implementation_internal", None)
        values[name] = None if value is None else str(value)
    return values


def _attention_family(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.lower()
    if value == "eager" or value.endswith("|eager"):
        return "eager"
    for version in ("4", "3", "2"):
        if f"flash_attention_{version}" in value or f"flash-attn{version}" in value:
            return f"flash_attention_{version}"
    if value == "sdpa" or value.endswith("|sdpa"):
        return "sdpa"
    return None


def _contains_eager(values: dict[str, str | None]) -> bool:
    return any(_attention_family(value) == "eager" for value in values.values())


def _resolved_attention(values: dict[str, str | None]) -> str:
    families = [_attention_family(value) for value in values.values()]
    concrete = [value for value in families if value]
    if not concrete:
        return "upstream_default"
    if len(set(concrete)) == 1:
        return concrete[0]
    return "+".join(dict.fromkeys(concrete))


def _resolution_mismatch(requested: str, values: dict[str, str | None]) -> str | None:
    families = {_attention_family(value) for value in values.values() if value is not None}
    families.discard(None)
    expected = _attention_family(requested)
    if families and expected is not None and families != {expected}:
        return f"requested {requested}, resolved families={sorted(families)} ({values})"
    return None


def _release_failed_model(model: Any, device: torch.device) -> None:
    if model is not None:
        del model
    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - cleanup must not mask load errors
            pass


def probe_sdpa_kernels(device: torch.device, dtype: torch.dtype) -> tuple[str, ...]:
    """Probe native SDPA kernels with the Qwen3 16/8 GQA head shape."""
    if device.type != "cuda" or not _sdpa_available():
        return ("math",) if _sdpa_available() else ()
    if dtype not in (torch.float16, torch.bfloat16):
        return ("math",)

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except (ImportError, AttributeError):
        return ("math",)

    backends = (
        ("flash", getattr(SDPBackend, "FLASH_ATTENTION", None)),
        ("cudnn", getattr(SDPBackend, "CUDNN_ATTENTION", None)),
        ("efficient", getattr(SDPBackend, "EFFICIENT_ATTENTION", None)),
        ("math", getattr(SDPBackend, "MATH", None)),
    )
    query = key = value = None
    available: list[str] = []
    try:
        query = torch.randn((1, 16, 128, 128), device=device, dtype=dtype)
        key = torch.randn((1, 8, 128, 128), device=device, dtype=dtype)
        value = torch.randn_like(key)
        with torch.inference_mode():
            for name, backend in backends:
                if backend is None:
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        with sdpa_kernel(backend):
                            F.scaled_dot_product_attention(
                                query,
                                key,
                                value,
                                is_causal=True,
                                enable_gqa=True,
                            )
                    torch.cuda.synchronize(device)
                    available.append(name)
                except Exception:  # noqa: BLE001 - capability probe
                    continue
    finally:
        del query, key, value
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
    return tuple(available)


@contextmanager
def attention_execution_context(report: dict[str, Any] | None):
    """Apply the probed native SDPA priority during generation."""
    if not report or report.get("selected") != "sdpa":
        with nullcontext():
            yield
        return

    kernels = tuple(report.get("sdpa_kernels") or ())
    if report.get("device_type") != "cuda" or not kernels:
        with nullcontext():
            yield
        return

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except (ImportError, AttributeError):
        with nullcontext():
            yield
        return

    backend_names = {
        "flash": "FLASH_ATTENTION",
        "cudnn": "CUDNN_ATTENTION",
        "efficient": "EFFICIENT_ATTENTION",
        "math": "MATH",
    }
    backends = [
        getattr(SDPBackend, backend_names[name])
        for name in kernels
        if name in backend_names and getattr(SDPBackend, backend_names[name], None) is not None
    ]
    if not backends:
        with nullcontext():
            yield
        return

    try:
        context = sdpa_kernel(backends, set_priority=True)
    except TypeError:
        try:
            context = sdpa_kernel(backends)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[MOSS attention] could not apply SDPA priority (%s)", exc)
            with nullcontext():
                yield
            return
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[MOSS attention] could not apply SDPA priority (%s)", exc)
        with nullcontext():
            yield
        return
    with context:
        yield


def load_model_with_attention_fallback(
    model_path: str | Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
    requested: str = AUTO_ATTENTION_IMPLEMENTATION,
    model_loader: Callable[..., Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load a model through an explicit and observable attention policy.

    ``model_loader`` is used by the distribution to load the audited local
    implementation.  The optional default preserves standalone upstream-style
    usage, but desktop and ComfyUI never execute unpinned Hub code.
    """
    requested = normalize_attention_implementation(requested)
    candidates, attempts = _candidate_list(requested, device=device, dtype=dtype)

    if model_loader is None:
        from transformers import AutoModelForCausalLM

        def model_loader(path: str, **kwargs: Any):
            kwargs = {key: value for key, value in kwargs.items() if value is not None}
            return AutoModelForCausalLM.from_pretrained(
                path,
                trust_remote_code=True,
                dtype="auto",
                **kwargs,
            )

    for implementation in candidates:
        backend = implementation or "upstream_default"
        model = None
        try:
            load_kwargs = {} if implementation is None else {"attn_implementation": implementation}
            model = model_loader(str(model_path), **load_kwargs)
            config_values = _config_attention_values(model)
            if implementation is not None and implementation != "eager" and _contains_eager(config_values):
                raise RuntimeError(f"Transformers resolved {implementation} to eager: {config_values}")
            if implementation is not None:
                mismatch = _resolution_mismatch(implementation, config_values)
                if mismatch is not None:
                    raise RuntimeError(f"Transformers rewrote the requested attention implementation: {mismatch}")

            selected = _resolved_attention(config_values) if implementation is None else implementation

            report: dict[str, Any] = {
                "requested": requested,
                "policy": "automatic_fallback" if requested == AUTO_ATTENTION_IMPLEMENTATION else "explicit",
                "selected": selected,
                "config": config_values,
                "attempts": [*attempts, {"backend": backend, "status": "selected"}],
                "device_type": device.type,
            }
            if selected == "sdpa":
                try:
                    kernels = probe_sdpa_kernels(device, dtype)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("[MOSS attention] SDPA kernel probe failed: %s", exc)
                    kernels = ()
                report["sdpa_kernels"] = list(kernels)
                if device.type == "cuda" and not any(kernel != "math" for kernel in kernels):
                    LOGGER.warning(
                        "[MOSS attention] selected sdpa but no fused CUDA kernel passed the probe; "
                        "long prompts may use quadratic math attention"
                    )
                else:
                    LOGGER.info("[MOSS attention] SDPA kernel preference: %s", kernels)
            if selected == "eager":
                LOGGER.warning(
                    "[MOSS attention] selected eager attention; long audio can require quadratic attention memory"
                )
            else:
                LOGGER.info("[MOSS attention] selected %s (requested=%s)", selected, requested)
            return model, report
        except Exception as exc:  # noqa: BLE001 - each candidate is isolated
            reason = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
            attempts.append({"backend": backend, "status": "failed", "reason": reason})
            LOGGER.warning("[MOSS attention] %s unavailable: %s", backend, reason)
            _release_failed_model(model, device)

    details = "; ".join(f"{item['backend']}: {item.get('reason', item['status'])}" for item in attempts)
    raise RuntimeError(f"No usable attention implementation was found ({details})")


__all__ = [
    "ATTENTION_IMPLEMENTATIONS",
    "AUTO_ATTENTION_IMPLEMENTATION",
    "attention_execution_context",
    "load_model_with_attention_fallback",
    "normalize_attention_implementation",
    "probe_sdpa_kernels",
]
