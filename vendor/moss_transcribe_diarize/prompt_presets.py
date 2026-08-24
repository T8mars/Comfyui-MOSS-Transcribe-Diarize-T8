from __future__ import annotations

from dataclasses import asdict, dataclass

from .inference_utils import DEFAULT_PROMPT


@dataclass(frozen=True, slots=True)
class PromptPreset:
    id: str
    label_zh: str
    label_en: str
    language_hint: str
    instruction: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


PROMPT_PRESETS = (
    PromptPreset("default", "默认通用", "Default", "auto", ""),
    PromptPreset(
        "zh_meeting",
        "中文会议",
        "Chinese meeting",
        "中文",
        "主要内容为中文会议。保留专有名词、数字、日期和行动项，不要翻译。",
    ),
    PromptPreset(
        "en_meeting",
        "英文会议",
        "English meeting",
        "English",
        "The recording is primarily an English meeting. Preserve names, numbers, dates, and action items; do not translate.",
    ),
    PromptPreset(
        "interview",
        "访谈/播客",
        "Interview / podcast",
        "auto",
        "这是访谈或播客。优先保持提问者与回答者的说话人编号稳定，保留口语语气但去除无意义重复。",
    ),
    PromptPreset(
        "subtitle_clean",
        "精简字幕",
        "Clean subtitles",
        "auto",
        "输出适合字幕阅读的简洁断句；不要添加音频中不存在的解释、标题或总结。",
    ),
    PromptPreset(
        "multilingual",
        "多语言原文",
        "Multilingual verbatim",
        "auto",
        "音频可能包含多种语言。每段保持原语言转写，不要翻译，并保留语言切换位置。",
    ),
)
PRESET_BY_ID = {preset.id: preset for preset in PROMPT_PRESETS}


def list_prompt_presets() -> list[dict[str, str]]:
    return [preset.to_dict() for preset in PROMPT_PRESETS]


def compose_prompt(
    *,
    base_prompt: str = DEFAULT_PROMPT,
    preset_id: str = "default",
    language_hint: str = "auto",
    hotwords: str = "",
    strict_format: bool = True,
) -> tuple[str, tuple[str, ...], str]:
    try:
        preset = PRESET_BY_ID[preset_id]
    except KeyError as exc:
        raise ValueError(f"Unknown prompt preset: {preset_id}") from exc
    resolved_language = language_hint if language_hint != "auto" else preset.language_hint
    words = tuple(dict.fromkeys(item.strip() for item in hotwords.replace("，", ",").split(",") if item.strip()))
    additions = []
    if preset.instruction:
        additions.append(preset.instruction)
    if resolved_language != "auto":
        additions.append(f"主要语言提示：{resolved_language}。")
    if words:
        additions.append("热词（仅在音频确实出现时采用）：" + "、".join(words) + "。")
    if strict_format:
        additions.append("只输出 [开始秒数][Sxx]正文[结束秒数] 段落，不要输出解释、标题或 Markdown。")
    text = base_prompt.strip() or DEFAULT_PROMPT
    if additions:
        text = text.rstrip() + "\n" + "\n".join(additions)
    return text, words, resolved_language


__all__ = ["PRESET_BY_ID", "PROMPT_PRESETS", "PromptPreset", "compose_prompt", "list_prompt_presets"]
