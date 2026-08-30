from __future__ import annotations

import hashlib
import json
import logging
import math
import platform
import re
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path

import torch
from comfy_api.latest import ComfyExtension, io

from .runtime.inference import build_prompt, run_transcription
from .runtime.long_audio import transcribe_long_audio
from .runtime.model_cache import MODEL_CACHE, _cuda_bf16_supported, resolve_dtype
from .runtime.quality import evaluate_quality
from .runtime.types import ModelHandle, PromptConfig, TranscriptPayload
from .services.model_store import (
    MISSING_MODEL_OPTION,
    load_manifest,
    model_fingerprint,
    model_options,
    register_model_paths,
    resolve_model,
    validate_model_dir,
)
from .vendor.moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT
from .vendor.moss_transcribe_diarize.audio_adapter import TARGET_SAMPLE_RATE, comfy_audio_to_numpy
from .vendor.moss_transcribe_diarize.attention import ATTENTION_IMPLEMENTATIONS, AUTO_ATTENTION_IMPLEMENTATION
from .vendor.moss_transcribe_diarize.prompt_presets import PROMPT_PRESETS
from .vendor.moss_transcribe_diarize.speaker_mapping import resolve_speaker_names
from .vendor.moss_transcribe_diarize.subtitle import (
    SubtitleSegment,
    SubtitleStyle,
    export_ass,
    export_json,
    export_srt,
    validate_ass_style,
)
from .vendor.moss_transcribe_diarize.transcript_validation import validate_transcript
from .vendor.moss_transcribe_diarize.transformers_compat import compatibility_report


LOGGER = logging.getLogger("comfyui-MOSS-Transcribe-Diarize-T8")
CATEGORY = "T8star-Aix/Audio/MOSS Transcribe Diarize"
MAX_OUTPUT_PREFIX_LENGTH = 120
ModelType = io.Custom("T8_MOSS_TRANSCRIBE_MODEL")
PromptType = io.Custom("T8_MOSS_PROMPT")
TranscriptType = io.Custom("T8_MOSS_TRANSCRIPT")
SubtitleStyleType = io.Custom("T8_MOSS_SUBTITLE_STYLE")
REVALIDATED_DIAGNOSTIC_CODES = {
    "empty_output",
    "invalid_format",
    "incomplete_segment",
    "speaker_tag_missing",
    "invalid_timestamp",
    "zero_duration",
    "timestamp_order",
    "timestamp_out_of_range",
    "possible_early_stop",
    "repeated_text",
    "token_limit_reached",
}


def _device_options() -> list[str]:
    values = ["auto"]
    if torch.cuda.is_available():
        values.extend(f"cuda:{index}" for index in range(torch.cuda.device_count()))
    values.append("cpu")
    return values


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("当前 ComfyUI 的 PyTorch 未检测到 CUDA。")
        return requested
    try:
        import comfy.model_management

        selected = str(comfy.model_management.get_torch_device())
    except Exception:
        selected = "cuda:0" if torch.cuda.is_available() else "cpu"
    if selected == "cuda":
        selected = f"cuda:{torch.cuda.current_device()}"
    if not (selected.startswith("cuda") or selected.startswith("cpu")):
        raise RuntimeError(f"MOSS 首版节点暂不支持 ComfyUI 当前设备：{selected}")
    return selected


def _subtitle_segments(payload: TranscriptPayload) -> list[SubtitleSegment]:
    return [SubtitleSegment.from_dict(item, fallback_id=f"seg-{index:05d}") for index, item in enumerate(payload.segments, 1)]


def _transcript_json(payload: TranscriptPayload) -> str:
    return json.dumps(payload.to_dict(), ensure_ascii=False, indent=2)


def _merge_validation_diagnostics(
    original: tuple[dict, ...],
    generated: tuple[dict, ...],
) -> tuple[dict, ...]:
    merged: list[dict] = []
    seen: set[str] = set()
    retained_original = (
        item
        for item in original
        if item.get("chunk_id") is not None or str(item.get("code") or "") not in REVALIDATED_DIAGNOSTIC_CODES
    )
    for item in (*retained_original, *generated):
        diagnostic = dict(item)
        signature = json.dumps(diagnostic, ensure_ascii=False, sort_keys=True, default=str)
        if signature in seen:
            continue
        seen.add(signature)
        merged.append(diagnostic)
    return tuple(merged)


def _merge_validated_segments(
    validated_segments,
    original_segments: tuple[dict, ...],
) -> tuple[dict, ...]:
    validated_segments = tuple(validated_segments)
    merged: list[dict] = []
    for index, segment in enumerate(validated_segments, 1):
        candidate = original_segments[index - 1] if index <= len(original_segments) else None
        preserves_original = bool(
            candidate
            and str(candidate.get("speaker") or "S00") == segment.speaker
            and str(candidate.get("text") or "").strip() == segment.text
        )
        original = dict(candidate) if preserves_original else {}
        merged.append(
            {
                **original,
                "id": str(original.get("id") or f"seg-{index:05d}"),
                "start": segment.start,
                "end": segment.end,
                "speaker": segment.speaker,
                "text": segment.text,
            }
        )
    return tuple(merged)


def _subtitle_outputs(payload: TranscriptPayload) -> tuple[str, str]:
    segments = _subtitle_segments(payload)
    return export_srt(segments), export_ass(segments)


def _unique_output_stem(safe_prefix: str) -> str:
    return f"{safe_prefix}_{time.time_ns()}_{uuid.uuid4().hex[:8]}"


def _safe_output_prefix(filename_prefix: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z._\u4e00-\u9fff-]+", "_", filename_prefix).strip("._") or "moss_transcript"
    if len(safe) > MAX_OUTPUT_PREFIX_LENGTH:
        suffix = hashlib.sha256(safe.encode("utf-8")).hexdigest()[:12]
        safe = f"{safe[: MAX_OUTPUT_PREFIX_LENGTH - len(suffix) - 1]}-{suffix}"
    return safe


def _optional_float(value) -> float:
    if value in (None, ""):
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        raise ValueError("Numeric inputs must be finite.")
    return number


def _optional_int(value) -> int:
    if value in (None, ""):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


class T8MossModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        options = model_options()
        return io.Schema(
            node_id="T8_MOSS_ModelLoader",
            display_name="MOSS 转写说话人模型加载器 · T8star-Aix",
            category=CATEGORY,
            search_aliases=["MOSS ASR", "diarization", "转写", "说话人识别", "T8star-Aix"],
            description="发现并校验 models/moss_transcribe_diarize 下的固定版本模型；权重在转写时按需载入。",
            inputs=[
                io.Combo.Input("model_name", display_name="模型", options=options, default=options[0]),
                io.Combo.Input("device", display_name="推理设备", options=_device_options(), default="auto"),
                io.Combo.Input(
                    "precision",
                    display_name="精度",
                    options=["auto", "bfloat16", "float16", "float32"],
                    default="auto",
                    tooltip="auto：支持 BF16 的 CUDA 使用 BF16，否则 CUDA 使用 FP16，CPU 使用 FP32。",
                ),
                io.Boolean.Input(
                    "release_after_run",
                    display_name="旧工作流：转写后释放",
                    default=False,
                    advanced=True,
                    tooltip="兼容旧工作流；开启时覆盖显存驻留策略。",
                ),
                io.Boolean.Input(
                    "verify_hashes",
                    display_name="完整 SHA-256 校验",
                    default=False,
                    advanced=True,
                    tooltip="模型身份始终读取全文件 SHA-256；开启后还会与固定 manifest 的预期摘要逐项比对。",
                ),
                io.Combo.Input(
                    "memory_policy",
                    display_name="显存驻留策略",
                    options=["keep", "release_under_pressure", "release_after_run"],
                    default="keep",
                    tooltip="常驻最快；压力释放会在可用显存低于 2GB 或 20% 时释放；每次释放最省显存。",
                ),
                io.Combo.Input(
                    "attention_implementation",
                    display_name="Attention 后端",
                    options=[AUTO_ATTENTION_IMPLEMENTATION, *ATTENTION_IMPLEMENTATIONS],
                    default=AUTO_ATTENTION_IMPLEMENTATION,
                    advanced=True,
                    tooltip="auto 按 FlashAttention-2、SDPA、eager 顺序显式尝试并记录结果；显式后端失败时直接报错，不静默回退。",
                ),
                io.String.Input(
                    "custom_model_path",
                    display_name="自定义模型绝对路径",
                    default="",
                    optional=True,
                    advanced=True,
                ),
            ],
            outputs=[
                ModelType.Output("model", display_name="MOSS 模型"),
                io.String.Output("model_info", display_name="模型信息"),
            ],
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        model_name: str,
        device: str,
        precision: str,
        release_after_run: bool,
        verify_hashes: bool,
        memory_policy: str = "keep",
        attention_implementation: str = AUTO_ATTENTION_IMPLEMENTATION,
        custom_model_path: str = "",
    ) -> str:
        try:
            fingerprint = model_fingerprint(resolve_model(model_name, custom_model_path))
            return ":".join(
                (
                    fingerprint,
                    device,
                    precision,
                    memory_policy,
                    attention_implementation,
                    str(bool(release_after_run)),
                    str(bool(verify_hashes)),
                )
            )
        except Exception as exc:
            return f"missing:{model_name}:{custom_model_path}:{exc}"

    @classmethod
    def validate_inputs(cls, model_name: str, custom_model_path: str = "", **kwargs) -> bool | str:
        try:
            transformers = compatibility_report()
        except RuntimeError as exc:
            return f"Transformers 环境不可用：{exc}。请运行 scripts/check_transformers.py。"
        if not transformers.supported:
            return (
                f"Transformers {transformers.installed} 不受支持，需要 >= {transformers.minimum}, "
                f"< {transformers.maximum_exclusive}。请运行 scripts/check_transformers.py，"
                "不要替换 ComfyUI 的 Torch。"
            )
        if model_name == MISSING_MODEL_OPTION and not custom_model_path.strip():
            return "未找到 MOSS 模型；请运行节点目录中的 scripts/download_models.py。"
        return True

    @classmethod
    def execute(
        cls,
        model_name: str,
        device: str,
        precision: str,
        release_after_run: bool,
        verify_hashes: bool,
        memory_policy: str = "keep",
        attention_implementation: str = AUTO_ATTENTION_IMPLEMENTATION,
        custom_model_path: str = "",
    ) -> io.NodeOutput:
        model_dir = resolve_model(model_name, custom_model_path)
        report = validate_model_dir(model_dir, verify_hashes=verify_hashes)
        report.require_valid()
        resolved_device = _resolve_device(device)
        effective_dtype = resolve_dtype(precision, torch.device(resolved_device))
        manifest = load_manifest()
        handle = ModelHandle(
            model_dir=model_dir,
            device=resolved_device,
            precision=precision,
            release_after_run=bool(release_after_run),
            model_revision=manifest["revision"],
            model_fingerprint=model_fingerprint(model_dir),
            memory_policy=memory_policy,
            attention_implementation=attention_implementation,
        )
        warning = ""
        if resolved_device.startswith("cuda"):
            index = int(resolved_device.split(":", 1)[1]) if ":" in resolved_device else 0
            total_gb = torch.cuda.get_device_properties(index).total_memory / 1024**3
            if total_gb < 12:
                warning = f" | 警告：{total_gb:.1f}GB 低于正式支持的 12GB 基线，仅建议短音频"
        info = (
            f"MOSS Transcribe Diarize | {model_dir} | device={resolved_device} | "
            f"precision={precision} -> dtype={effective_dtype} | "
            f"memory={handle.effective_memory_policy} | "
            f"attention={handle.attention_implementation} | "
            f"{'SHA-256 已校验' if verify_hashes else '文件大小已校验'} | revision={manifest['revision'][:12]}{warning}"
        )
        return io.NodeOutput(handle, info)


class T8MossPromptHotwords(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_MOSS_PromptHotwords",
            display_name="MOSS 提示词与热词 · T8star-Aix",
            category=CATEGORY,
            search_aliases=["hotwords", "prompt", "热词", "专有名词"],
            description="构建格式约束、语言提示与热词；热词只作为提示，不保证强制命中。",
            inputs=[
                io.String.Input(
                    "base_prompt",
                    display_name="基础提示词",
                    multiline=True,
                    default=DEFAULT_PROMPT,
                    dynamic_prompts=False,
                ),
                io.Combo.Input(
                    "preset_id",
                    display_name="场景预设",
                    options=[preset.id for preset in PROMPT_PRESETS],
                    default="default",
                ),
                io.String.Input(
                    "hotwords",
                    display_name="热词（逗号分隔）",
                    multiline=True,
                    default="OpenMOSS, ComfyUI, T8star-Aix",
                    dynamic_prompts=False,
                ),
                io.Combo.Input(
                    "language_hint",
                    display_name="主要语言提示",
                    options=["auto", "中文", "English", "日本語", "한국어", "粤语"],
                    default="auto",
                ),
                io.Boolean.Input("strict_format", display_name="严格段落格式", default=True),
            ],
            outputs=[
                PromptType.Output("prompt", display_name="MOSS 提示配置"),
                io.String.Output("prompt_text", display_name="最终提示词"),
            ],
        )

    @classmethod
    def execute(
        cls,
        base_prompt: str,
        hotwords: str,
        language_hint: str,
        strict_format: bool,
        preset_id: str = "default",
    ) -> io.NodeOutput:
        prompt = build_prompt(base_prompt, hotwords, language_hint, strict_format, preset_id)
        return io.NodeOutput(prompt, prompt.text)


class T8MossTranscribeDiarize(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_MOSS_TranscribeDiarize",
            display_name="MOSS 转写与说话人分离 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            search_aliases=["ASR", "speaker diarization", "字幕", "语音识别"],
            description=(
                "标准 ComfyUI AUDIO 直传；自动下混并重采样到 16kHz。优先整段推理以维持全程说话人编号。"
                "输出为句/段级时间戳，不是逐词时间戳。"
            ),
            inputs=[
                ModelType.Input("model", display_name="MOSS 模型"),
                io.Audio.Input("audio", display_name="待转写音频"),
                io.Int.Input(
                    "max_new_tokens",
                    display_name="最大新 token（0=自动）",
                    default=0,
                    min=0,
                    max=65536,
                    step=256,
                    tooltip="自动模式依据音频时长估算；达到上限会在诊断中明确提示可能截断。",
                ),
                PromptType.Input("prompt", display_name="提示词与热词", optional=True),
                io.Combo.Input(
                    "silence_policy",
                    display_name="静音预检",
                    options=["warn", "reject", "ignore"],
                    default="warn",
                    tooltip="warn 继续并给出诊断；reject 在载入模型前拒绝；ignore 仅记录检测结果。",
                ),
                io.Combo.Input(
                    "preflight_backend",
                    display_name="语音预检后端",
                    options=["webrtc", "energy"],
                    default="webrtc",
                    advanced=True,
                    tooltip="webrtc 检测真实语音帧；energy 仅按能量判断，兼容旧环境。",
                ),
                io.Int.Input(
                    "vad_aggressiveness",
                    display_name="VAD 严格度",
                    default=2,
                    min=0,
                    max=3,
                    advanced=True,
                ),
                io.Combo.Input(
                    "retry_policy",
                    display_name="自动重试",
                    options=["never", "invalid_format", "quality_failure"],
                    default="invalid_format",
                    advanced=True,
                    tooltip="格式无效或缺失说话人标签时，可用更严格提示自动重试一次。",
                ),
            ],
            outputs=[
                io.Audio.Output("audio", display_name="原音频透传"),
                io.String.Output("raw_text", display_name="模型原始文本"),
                io.String.Output("transcript_json", display_name="结构化 JSON"),
                io.String.Output("srt", display_name="SRT 字幕"),
                io.String.Output("ass", display_name="ASS 字幕"),
                TranscriptType.Output("transcript", display_name="MOSS_TRANSCRIPT"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model: ModelHandle,
        audio: dict,
        max_new_tokens: int,
        prompt: PromptConfig | None = None,
        silence_policy: str = "warn",
        preflight_backend: str = "webrtc",
        vad_aggressiveness: int = 2,
        retry_policy: str = "invalid_format",
    ) -> io.NodeOutput:
        payload = run_transcription(
            model,
            audio,
            prompt,
            max_new_tokens=int(max_new_tokens),
            silence_policy=silence_policy,
            preflight_backend=preflight_backend,
            vad_aggressiveness=int(vad_aggressiveness),
            retry_policy=retry_policy,
        )
        srt, ass = _subtitle_outputs(payload)
        return io.NodeOutput(audio, payload.raw_text, _transcript_json(payload), srt, ass, payload)


class T8MossSmartLongAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_MOSS_SmartLongAudio",
            display_name="MOSS 智能长音频转写 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            search_aliases=["long audio", "VAD chunk", "resume transcription", "长音频", "断点续跑"],
            description="按 VAD 静音边界安全分片、重叠去重、分片说话人隔离，并支持中断后从检查点续跑。",
            inputs=[
                ModelType.Input("model", display_name="MOSS 模型"),
                io.Audio.Input("audio", display_name="长音频"),
                io.Int.Input(
                    "max_new_tokens_per_chunk",
                    display_name="每片最大新 token（0=自动）",
                    default=0,
                    min=0,
                    max=65536,
                    step=256,
                ),
                PromptType.Input("prompt", display_name="提示词与热词", optional=True),
                io.Float.Input(
                    "target_chunk_minutes",
                    display_name="目标分片分钟",
                    default=8.0,
                    min=1.0,
                    max=30.0,
                    step=0.5,
                ),
                io.Float.Input(
                    "max_chunk_minutes",
                    display_name="最长分片分钟",
                    default=10.0,
                    min=1.0,
                    max=40.0,
                    step=0.5,
                ),
                io.Float.Input(
                    "overlap_seconds",
                    display_name="分片重叠秒数",
                    default=1.0,
                    min=0.0,
                    max=10.0,
                    step=0.25,
                ),
                io.Combo.Input(
                    "split_strategy",
                    display_name="分片策略",
                    options=["vad", "fixed"],
                    default="vad",
                ),
                io.Combo.Input(
                    "silence_policy",
                    display_name="无语音分片策略",
                    options=["warn", "reject", "ignore"],
                    default="warn",
                ),
                io.Int.Input(
                    "vad_aggressiveness",
                    display_name="VAD 严格度",
                    default=2,
                    min=0,
                    max=3,
                ),
                io.Combo.Input(
                    "retry_policy",
                    display_name="失败自动重试",
                    options=["never", "invalid_format", "quality_failure"],
                    default="quality_failure",
                ),
                io.Combo.Input(
                    "checkpoint_mode",
                    display_name="检查点模式",
                    options=["off", "read_write", "restart"],
                    default="read_write",
                    tooltip="read_write 自动续跑；restart 忽略并覆盖旧检查点；off 不写磁盘。",
                ),
                io.String.Input(
                    "checkpoint_id",
                    display_name="检查点名称（空=音频指纹）",
                    default="",
                    optional=True,
                    advanced=True,
                ),
            ],
            outputs=[
                io.Audio.Output("audio", display_name="原音频透传"),
                io.String.Output("raw_text", display_name="合并原始文本"),
                io.String.Output("transcript_json", display_name="结构化 JSON"),
                io.String.Output("srt", display_name="SRT 字幕"),
                io.String.Output("ass", display_name="ASS 字幕"),
                TranscriptType.Output("transcript", display_name="MOSS_TRANSCRIPT"),
                io.String.Output("chunk_report", display_name="分片报告 JSON"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model: ModelHandle,
        audio: dict,
        max_new_tokens_per_chunk: int,
        target_chunk_minutes: float,
        max_chunk_minutes: float,
        overlap_seconds: float,
        split_strategy: str,
        silence_policy: str,
        vad_aggressiveness: int,
        retry_policy: str,
        checkpoint_mode: str,
        prompt: PromptConfig | None = None,
        checkpoint_id: str = "",
    ) -> io.NodeOutput:
        if float(target_chunk_minutes) > float(max_chunk_minutes):
            raise ValueError("目标分片分钟不能大于最长分片分钟。")
        samples = comfy_audio_to_numpy(audio, TARGET_SAMPLE_RATE)
        checkpoint_dir = None
        if checkpoint_mode != "off":
            import folder_paths

            checkpoint_dir = (
                Path(folder_paths.get_output_directory()).resolve()
                / "moss_transcribe_diarize"
                / "checkpoints"
            )
        payload, report = transcribe_long_audio(
            model,
            samples,
            prompt,
            sample_rate=TARGET_SAMPLE_RATE,
            max_new_tokens_per_chunk=int(max_new_tokens_per_chunk),
            target_seconds=float(target_chunk_minutes) * 60.0,
            max_seconds=float(max_chunk_minutes) * 60.0,
            overlap_seconds=float(overlap_seconds),
            split_strategy=split_strategy,
            silence_policy=silence_policy,
            preflight_backend="webrtc",
            vad_aggressiveness=int(vad_aggressiveness),
            retry_policy=retry_policy,
            checkpoint_mode=checkpoint_mode,
            checkpoint_dir=checkpoint_dir,
            checkpoint_id=checkpoint_id,
        )
        srt, ass = _subtitle_outputs(payload)
        return io.NodeOutput(
            audio,
            payload.raw_text,
            _transcript_json(payload),
            srt,
            ass,
            payload,
            json.dumps(report, ensure_ascii=False, indent=2),
        )


class T8MossTranscriptValidate(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_MOSS_TranscriptValidate",
            display_name="MOSS 转写解析与校验 · T8star-Aix",
            category=CATEGORY,
            search_aliases=["validate transcript", "时间戳检查", "截断检查"],
            description="检查格式、时间戳顺序/越界和 token 上限；不会隐藏静音幻觉或截断风险。",
            inputs=[
                io.String.Input("raw_text", display_name="模型原始文本", multiline=True, default="", dynamic_prompts=False),
                io.Float.Input("media_duration_seconds", display_name="媒体时长（0=未知）", default=0.0, min=0.0, max=86400.0, step=0.1),
                io.Int.Input("generated_tokens", display_name="已生成 token（0=未知）", default=0, min=0, max=131072),
                io.Int.Input("max_new_tokens", display_name="token 上限（0=未知）", default=0, min=0, max=131072),
                TranscriptType.Input("transcript", display_name="已有 MOSS_TRANSCRIPT", optional=True),
            ],
            outputs=[
                TranscriptType.Output("transcript", display_name="已校验 MOSS_TRANSCRIPT"),
                io.String.Output("report_json", display_name="校验报告 JSON"),
                io.String.Output("segments_json", display_name="段落 JSON"),
            ],
        )

    @classmethod
    def execute(
        cls,
        raw_text: str,
        media_duration_seconds: float,
        generated_tokens: int,
        max_new_tokens: int,
        transcript: TranscriptPayload | None = None,
    ) -> io.NodeOutput:
        source = transcript.raw_text if transcript is not None else raw_text
        duration = _optional_float(media_duration_seconds) or (
            _optional_float(transcript.metadata.get("audio_duration_seconds")) if transcript else 0.0
        )
        used_tokens = _optional_int(generated_tokens) or (
            _optional_int(transcript.metadata.get("generated_tokens")) if transcript else 0
        )
        limit = _optional_int(max_new_tokens) or (
            _optional_int(transcript.metadata.get("max_new_tokens")) if transcript else 0
        )
        result = validate_transcript(
            source,
            media_duration=duration or None,
            generated_tokens=used_tokens or None,
            max_new_tokens=limit or None,
        )
        original_segments = transcript.segments if transcript is not None else ()
        original_diagnostics = transcript.diagnostics if transcript is not None else ()
        segments = _merge_validated_segments(result.segments, original_segments)
        diagnostics = _merge_validation_diagnostics(
            original_diagnostics,
            tuple(asdict(item) for item in result.diagnostics),
        )
        possibly_truncated = bool(
            result.possibly_truncated
            or any(item.get("code") == "token_limit_reached" for item in diagnostics)
        )
        validation_passed = bool(
            result.valid and not any(str(item.get("level") or "") == "error" for item in diagnostics)
        )
        metadata = dict(transcript.metadata) if transcript else {}
        for key, value in (
            ("audio_duration_seconds", duration),
            ("generated_tokens", used_tokens),
            ("max_new_tokens", limit),
        ):
            if value:
                metadata[key] = value
            else:
                metadata.pop(key, None)
        metadata.update(
            {
                "possibly_truncated": possibly_truncated,
                "validation_passed": validation_passed,
            }
        )
        payload = TranscriptPayload(
            raw_text=source,
            segments=segments,
            diagnostics=diagnostics,
            metadata=metadata,
        )
        payload.metadata["quality"] = evaluate_quality(payload)
        report = {
            "valid": validation_passed,
            "possibly_truncated": possibly_truncated,
            "segment_count": len(payload.segments),
            "diagnostics": list(payload.diagnostics),
        }
        return io.NodeOutput(
            payload,
            json.dumps(report, ensure_ascii=False, indent=2),
            json.dumps(list(payload.segments), ensure_ascii=False, indent=2),
        )


class T8MossQualityGate(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_MOSS_QualityGate",
            display_name="MOSS 转写质量门 · T8star-Aix",
            category=CATEGORY,
            search_aliases=["quality gate", "coverage", "hallucination", "质量检查"],
            description="按格式错误、尾部覆盖、未知说话人、重复文本和截断风险输出可用性布尔值。",
            inputs=[
                TranscriptType.Input("transcript", display_name="MOSS_TRANSCRIPT"),
                io.Float.Input(
                    "min_end_coverage",
                    display_name="最低尾部覆盖率",
                    default=0.75,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                ),
                io.Float.Input(
                    "max_unknown_speaker_ratio",
                    display_name="未知说话人比例上限",
                    default=0.50,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                ),
                io.Boolean.Input("reject_repetition", display_name="拒绝重复循环", default=True),
                io.Boolean.Input("reject_truncation", display_name="拒绝疑似截断", default=True),
                io.Boolean.Input(
                    "fail_on_unusable",
                    display_name="不可用时终止工作流",
                    default=False,
                    advanced=True,
                    tooltip="启用后质量检查失败会抛出错误，阻止后续字幕写盘节点执行。",
                ),
            ],
            outputs=[
                TranscriptType.Output("transcript", display_name="已评估 MOSS_TRANSCRIPT"),
                io.Boolean.Output("is_usable", display_name="结果可用"),
                io.String.Output("quality_report", display_name="质量报告 JSON"),
            ],
        )

    @classmethod
    def execute(
        cls,
        transcript: TranscriptPayload,
        min_end_coverage: float,
        max_unknown_speaker_ratio: float,
        reject_repetition: bool,
        reject_truncation: bool,
        fail_on_unusable: bool = False,
    ) -> io.NodeOutput:
        report = evaluate_quality(
            transcript,
            min_end_coverage=float(min_end_coverage),
            max_unknown_speaker_ratio=float(max_unknown_speaker_ratio),
            reject_repetition=bool(reject_repetition),
            reject_truncation=bool(reject_truncation),
        )
        payload = TranscriptPayload(
            raw_text=transcript.raw_text,
            segments=transcript.segments,
            diagnostics=transcript.diagnostics,
            metadata={**transcript.metadata, "quality_gate": report},
        )
        if fail_on_unusable and not report["usable"]:
            reasons = ", ".join(str(item) for item in report["reasons"])
            raise ValueError(f"MOSS 转写质量门拒绝结果：{reasons}")
        return io.NodeOutput(payload, bool(report["usable"]), json.dumps(report, ensure_ascii=False, indent=2))


class T8MossSubtitleStyle(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_MOSS_SubtitleStyle",
            display_name="MOSS 字幕样式 · T8star-Aix",
            category=CATEGORY,
            search_aliases=["ASS style", "subtitle style", "字幕样式", "说话人颜色"],
            description="集中配置 ASS 分辨率、字体、布局、描边和说话人配色；可复用到多个字幕导出节点。",
            inputs=[
                io.String.Input("font_name", display_name="字体名称", default="Noto Sans CJK SC"),
                io.Int.Input(
                    "font_size",
                    display_name="字号（0=按分辨率自动）",
                    default=0,
                    min=0,
                    max=512,
                    step=1,
                ),
                io.Int.Input("alignment", display_name="ASS 对齐（1-9）", default=2, min=1, max=9, step=1),
                io.Int.Input("margin_v", display_name="垂直边距", default=56, min=0, max=2160, step=1),
                io.Boolean.Input("speaker_colors", display_name="按说话人配色", default=True),
                io.String.Input("primary_color", display_name="主色（ASS BGR）", default="&H00FFFFFF"),
                io.String.Input("outline_color", display_name="描边色（ASS BGR）", default="&H00000000"),
                io.String.Input("back_color", display_name="背景色（ASS BGR）", default="&H64000000"),
                io.Int.Input("outline", display_name="描边宽度", default=3, min=0, max=20, step=1),
                io.Int.Input("shadow", display_name="阴影宽度", default=1, min=0, max=20, step=1),
                io.Int.Input("video_width", display_name="视频宽度", default=1920, min=16, max=16384, step=2),
                io.Int.Input("video_height", display_name="视频高度", default=1080, min=16, max=16384, step=2),
            ],
            outputs=[
                SubtitleStyleType.Output("style", display_name="MOSS 字幕样式"),
                io.String.Output("style_json", display_name="样式 JSON"),
            ],
        )

    @classmethod
    def execute(
        cls,
        font_name: str,
        font_size: int,
        alignment: int,
        margin_v: int,
        speaker_colors: bool,
        primary_color: str,
        outline_color: str,
        back_color: str,
        outline: int,
        shadow: int,
        video_width: int,
        video_height: int,
    ) -> io.NodeOutput:
        style = SubtitleStyle(
            font_name=str(font_name).strip() or "Noto Sans CJK SC",
            font_size=None if int(font_size) <= 0 else int(font_size),
            alignment=int(alignment),
            margin_v=int(margin_v),
            speaker_colors=bool(speaker_colors),
            primary_color=str(primary_color),
            outline_color=str(outline_color),
            back_color=str(back_color),
            outline=int(outline),
            shadow=int(shadow),
            video_width=int(video_width),
            video_height=int(video_height),
        )
        validate_ass_style(style)
        return io.NodeOutput(style, json.dumps(style.to_dict(), ensure_ascii=False, indent=2))


class T8MossSubtitleExport(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_MOSS_SubtitleExport",
            display_name="MOSS 字幕导出 · T8star-Aix",
            category=CATEGORY,
            is_output_node=True,
            not_idempotent=True,
            search_aliases=["SRT", "ASS", "JSON", "speaker names", "字幕导出"],
            description="输出 JSON/TXT/SRT/ASS 内容，并可安全写入 ComfyUI/output/moss_transcribe_diarize。",
            inputs=[
                TranscriptType.Input("transcript", display_name="MOSS_TRANSCRIPT"),
                io.String.Input(
                    "speaker_names_json",
                    display_name="说话人重命名 JSON",
                    multiline=True,
                    default='{"S01": "主持人", "S02": "嘉宾"}',
                    dynamic_prompts=False,
                ),
                io.Boolean.Input("show_speaker", display_name="字幕显示说话人", default=True),
                io.String.Input("filename_prefix", display_name="文件名前缀", default="moss_transcript"),
                io.Boolean.Input("write_files", display_name="写入 output 目录", default=True),
                io.String.Input("chunk_id", display_name="分片 ID（可选）", default="", advanced=True),
                io.String.Input(
                    "cross_chunk_speaker_map_json",
                    display_name="跨分片说话人映射 JSON",
                    multiline=True,
                    default='{"part001:S01": "主持人", "part002:S02": "主持人"}',
                    dynamic_prompts=False,
                    advanced=True,
                ),
                SubtitleStyleType.Input("style", display_name="字幕样式（可选）", optional=True),
            ],
            outputs=[
                io.String.Output("json", display_name="JSON 内容"),
                io.String.Output("txt", display_name="TXT 内容"),
                io.String.Output("srt", display_name="SRT 内容"),
                io.String.Output("ass", display_name="ASS 内容"),
                io.String.Output("files", display_name="导出文件路径 JSON"),
            ],
        )

    @classmethod
    def execute(
        cls,
        transcript: TranscriptPayload,
        speaker_names_json: str,
        show_speaker: bool,
        filename_prefix: str,
        write_files: bool,
        chunk_id: str = "",
        cross_chunk_speaker_map_json: str = "{}",
        style: SubtitleStyle | None = None,
    ) -> io.NodeOutput:
        try:
            names = json.loads(speaker_names_json or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"说话人重命名 JSON 无效：{exc}") from exc
        if not isinstance(names, dict):
            raise ValueError("说话人重命名必须是 JSON 对象。")
        try:
            cross_chunk_mapping = json.loads(cross_chunk_speaker_map_json or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"跨分片说话人映射 JSON 无效：{exc}") from exc
        if not isinstance(cross_chunk_mapping, dict):
            raise ValueError("跨分片说话人映射必须是 JSON 对象。")
        names = resolve_speaker_names(
            names,
            cross_chunk_mapping,
            chunk_id=chunk_id,
            segments=transcript.segments,
        )
        segments = _subtitle_segments(transcript)
        style = replace(
            style or SubtitleStyle(),
            show_speaker=bool(show_speaker),
            speaker_names=names or None,
        )
        json_text = export_json(segments, speaker_names=names or None)
        txt_text = "\n".join(
            f"[{item.start:.2f}]"
            f"{'[' + str(names.get(item.speaker, item.speaker)) + ']' if style.show_speaker else ''}"
            f"{item.text}[{item.end:.2f}]"
            for item in segments
        ) + ("\n" if segments else "")
        srt_text = export_srt(segments, show_speaker=style.show_speaker, speaker_names=style.speaker_names)
        ass_text = export_ass(segments, style=style)
        files: dict[str, str] = {}
        if write_files:
            import folder_paths

            safe_prefix = _safe_output_prefix(filename_prefix)
            output_dir = Path(folder_paths.get_output_directory()).resolve() / "moss_transcribe_diarize"
            output_dir.mkdir(parents=True, exist_ok=True)
            stem = _unique_output_stem(safe_prefix)
            payloads = {"json": json_text, "txt": txt_text, "srt": srt_text, "ass": ass_text}
            for kind, text in payloads.items():
                path = output_dir / f"{stem}.{kind}"
                path.write_text(text, encoding="utf-8-sig" if kind in {"srt", "ass"} else "utf-8")
                files[kind] = str(path)
        return io.NodeOutput(json_text, txt_text, srt_text, ass_text, json.dumps(files, ensure_ascii=False, indent=2))


class T8MossEnvironmentRelease(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_MOSS_EnvironmentRelease",
            display_name="MOSS 环境诊断与模型释放 · T8star-Aix",
            category=CATEGORY,
            is_output_node=True,
            not_idempotent=True,
            search_aliases=["environment", "VRAM", "unload", "环境诊断", "释放显存"],
            description="报告 Transformers/PyTorch/CUDA/缓存状态，并且只释放本节点包加载的 MOSS 模型。",
            inputs=[
                io.Combo.Input(
                    "action",
                    display_name="动作",
                    options=["report_only", "release_selected", "release_all_moss"],
                    default="report_only",
                ),
                ModelType.Input("model", display_name="指定 MOSS 模型", optional=True),
            ],
            outputs=[io.String.Output("environment_report", display_name="环境报告 JSON")],
        )

    @classmethod
    def execute(cls, action: str, model: ModelHandle | None = None) -> io.NodeOutput:
        action_result = "no_change"
        if action == "release_selected":
            if model is None:
                raise ValueError("release_selected 需要连接模型加载器。")
            action_result = "released_or_marked" if MODEL_CACHE.release(model) else "not_loaded"
        elif action == "release_all_moss":
            action_result = f"released_or_marked:{MODEL_CACHE.release_all()}"
        elif action != "report_only":
            raise ValueError(f"未知动作：{action}")

        cuda = []
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                cuda.append({
                    "index": index,
                    "name": props.name,
                    "total_vram_gb": round(props.total_memory / 1024**3, 2),
                    "bf16_supported": _cuda_bf16_supported(torch.device(f"cuda:{index}")),
                    "meets_12gb_baseline": props.total_memory >= 12 * 1024**3,
                    "meets_10gb_baseline": props.total_memory >= 10 * 1024**3,
                })
        report = {
            "plugin": "comfyui-MOSS-Transcribe-Diarize-T8",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": compatibility_report().to_dict(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "gpus": cuda,
            "cache": MODEL_CACHE.report(),
            "action": action,
            "action_result": action_result,
            "model_revision": load_manifest()["revision"],
            "limitations": [
                "句/段级时间戳，不是逐词时间戳",
                "分片后的 speaker 编号仅在各分片内部有效",
                "静音、长静音和长音频可能产生提前结束、重复或幻觉，必须检查诊断",
            ],
        }
        return io.NodeOutput(json.dumps(report, ensure_ascii=False, indent=2))


class T8MossExtension(ComfyExtension):
    async def on_load(self) -> None:
        register_model_paths()
        try:
            report = compatibility_report()
            if not report.supported:
                LOGGER.error("Unsupported Transformers %s: %s", report.installed, report.reason)
        except RuntimeError as exc:
            LOGGER.error("Transformers compatibility check failed: %s", exc)
        LOGGER.info("Loaded comfyui-MOSS-Transcribe-Diarize-T8 (V3 nodes)")

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            T8MossModelLoader,
            T8MossPromptHotwords,
            T8MossTranscribeDiarize,
            T8MossSmartLongAudio,
            T8MossTranscriptValidate,
            T8MossQualityGate,
            T8MossSubtitleStyle,
            T8MossSubtitleExport,
            T8MossEnvironmentRelease,
        ]


async def comfy_entrypoint() -> T8MossExtension:
    return T8MossExtension()


__all__ = [
    "T8MossEnvironmentRelease",
    "T8MossExtension",
    "T8MossModelLoader",
    "T8MossPromptHotwords",
    "T8MossQualityGate",
    "T8MossSmartLongAudio",
    "T8MossSubtitleExport",
    "T8MossSubtitleStyle",
    "T8MossTranscriptValidate",
    "T8MossTranscribeDiarize",
    "comfy_entrypoint",
]
