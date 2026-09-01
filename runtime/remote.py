from __future__ import annotations

from http.client import HTTPConnection, HTTPSConnection, HTTPException
from io import BytesIO
import json
import os
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit
import uuid
import wave

import numpy as np

from .types import ModelHandle


REMOTE_API_KEY_ENV = "MOSS_TRANSCRIBE_API_KEY"
DEFAULT_REMOTE_MODEL = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
TRANSCRIPTION_PATH = "/v1/audio/transcriptions"
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
REMOTE_LANGUAGE_CODES = {
    "中文": "zh",
    "English": "en",
    "日本語": "ja",
    "한국어": "ko",
    "粤语": "yue",
    "Español": "es",
    "Français": "fr",
    "Deutsch": "de",
    "Italiano": "it",
    "Português": "pt",
    "Русский": "ru",
    "ไทย": "th",
    "Tiếng Việt": "vi",
    "Tagalog": "tl",
    "हिन्दी": "hi",
    "मराठी": "mr",
    "اردو": "ur",
    "العربية": "ar",
    "Türkçe": "tr",
    "Polski": "pl",
    "Nederlands": "nl",
    "Bahasa Indonesia": "id",
    "Bahasa Melayu": "ms",
}


def normalize_endpoint_url(value: str) -> str:
    raw = str(value or "").strip().rstrip("/")
    if not raw:
        raise ValueError("远程服务地址不能为空。")
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("远程服务地址必须是有效的 http:// 或 https:// URL。")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError("远程服务地址包含无效端口。") from exc
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("远程服务地址不能包含凭据、查询参数或片段。")
    if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("非本机远程服务必须使用 HTTPS，避免音频明文传输。")
    path = parsed.path.rstrip("/")
    if not path.endswith(TRANSCRIPTION_PATH):
        path = f"{path}{TRANSCRIPTION_PATH}" if path else TRANSCRIPTION_PATH
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def remote_api_key_configured() -> bool:
    return bool(os.getenv(REMOTE_API_KEY_ENV, "").strip())


def request_remote_transcription(
    handle: ModelHandle,
    samples: np.ndarray,
    *,
    sample_rate: int,
    prompt: str,
    language: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    if not handle.is_remote:
        raise ValueError("Remote transcription requires a remote ModelHandle.")
    endpoint = normalize_endpoint_url(handle.endpoint_url)
    fields = {
        "model": handle.remote_model or DEFAULT_REMOTE_MODEL,
        "response_format": "verbose_json",
        "temperature": "0",
        "max_new_tokens": str(max(1, int(max_new_tokens))),
    }
    if prompt.strip():
        fields["prompt"] = prompt.strip()
    remote_language = normalize_remote_language(language)
    if remote_language:
        fields["language"] = remote_language
    body, content_type = _multipart_body(fields, "audio.wav", _wav_bytes(samples, sample_rate))
    parsed = urlsplit(endpoint)
    connection_type = HTTPSConnection if parsed.scheme == "https" else HTTPConnection
    timeout = _validated_timeout(handle.timeout_seconds)
    connection = connection_type(parsed.hostname, parsed.port, timeout=timeout)
    headers = {"Content-Type": content_type, "Content-Length": str(len(body)), "Accept": "application/json"}
    api_key = os.getenv(REMOTE_API_KEY_ENV, "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    path = parsed.path or TRANSCRIPTION_PATH
    try:
        connection.request("POST", path, body=body, headers=headers)
        response = connection.getresponse()
        content_length = response.getheader("Content-Length")
        if content_length:
            try:
                declared_length = int(content_length)
            except ValueError as exc:
                raise RuntimeError("远程转写响应包含无效的 Content-Length。") from exc
            if declared_length < 0:
                raise RuntimeError("远程转写响应包含无效的 Content-Length。")
            if declared_length > MAX_RESPONSE_BYTES:
                raise RuntimeError("远程转写响应超过 16 MiB 安全上限。")
        raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise RuntimeError("远程转写响应超过 16 MiB 安全上限。")
        if response.status < 200 or response.status >= 300:
            detail = _safe_error_text(raw)
            raise RuntimeError(f"远程转写请求失败：HTTP {response.status} {response.reason}; {detail}")
    except (OSError, TimeoutError, HTTPException) as exc:
        raise RuntimeError(f"无法连接远程转写服务：{exc}") from exc
    finally:
        connection.close()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("远程转写服务没有返回有效 JSON。") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("远程转写服务返回的 JSON 顶层必须是对象。")
    text = payload.get("text")
    if not isinstance(text, str):
        raise RuntimeError("远程转写响应缺少字符串字段 text。")
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    return {
        "text": text,
        "prompt_len": _safe_int(usage.get("prompt_tokens") or usage.get("input_tokens")),
        "generated_tokens": _safe_int(
            usage.get("completion_tokens") or usage.get("output_tokens") or payload.get("generated_tokens")
        ),
        "remote": {
            "backend": handle.backend,
            "endpoint": _redacted_endpoint(endpoint),
            "model": fields["model"],
            "response_format": fields["response_format"],
        },
    }


def _wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    array = np.asarray(samples, dtype=np.float32).reshape(-1)
    pcm = np.round(np.clip(array, -1.0, 1.0) * 32767.0).astype("<i2", copy=False)
    buffer = BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(int(sample_rate))
        output.writeframes(pcm.tobytes())
    return buffer.getvalue()


def normalize_remote_language(value: str) -> str:
    language = str(value or "").strip()
    if not language or language.casefold() == "auto":
        return ""
    return REMOTE_LANGUAGE_CODES.get(language, language)


def _multipart_body(fields: dict[str, str], filename: str, file_bytes: bytes) -> tuple[bytes, str]:
    boundary = f"----T8Moss{uuid.uuid4().hex}"
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.extend(
            [
                f"--{boundary}\r\n".encode("ascii"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("ascii"),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )
    parts.extend(
        [
            f"--{boundary}\r\n".encode("ascii"),
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode("ascii"),
            b"Content-Type: audio/wav\r\n\r\n",
            file_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode("ascii"),
        ]
    )
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _safe_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _validated_timeout(value: Any) -> float:
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Remote timeout must be a finite number between 1 and 7200 seconds.") from exc
    if not np.isfinite(timeout) or timeout < 1.0 or timeout > 7200.0:
        raise ValueError("Remote timeout must be a finite number between 1 and 7200 seconds.")
    return timeout


def _safe_error_text(raw: bytes) -> str:
    text = raw[:500].decode("utf-8", errors="replace")
    return re.sub(r"[^\x20-\x7E\u3400-\u9fff]+", " ", text).strip() or "no response body"


def _redacted_endpoint(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


__all__ = [
    "DEFAULT_REMOTE_MODEL",
    "REMOTE_API_KEY_ENV",
    "normalize_endpoint_url",
    "normalize_remote_language",
    "remote_api_key_configured",
    "request_remote_transcription",
]
