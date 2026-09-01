# T8star-Aix MOSS Transcribe Diarize Desktop v0.2.3

这是发布在 `T8mars/Comfyui-MOSS-Transcribe-Diarize-T8` 仓库中的 Windows 便携整合包预发布。配套 ComfyUI 节点继续使用已发布的 `v0.4.0`，本次不会重复发布 Registry 节点。

## 中文

- 修复无任务且侧栏折叠时，最大化窗口会把主工作区压缩到约 12px、造成界面空白的问题。
- 导入与处理中界面在普通窗口、跨 1360px 断点、最大化和还原时保持一致的滚动与布局行为。
- 增加真实打包 EXE 的 Windows 原生最大化/还原回归检查；实测主视图从普通窗口 `1171/1186px` 扩展到最大化 `2048/2048px`，还原后恢复正常。
- 中英文界面明确说明“主要语言”仅用于提高识别准确率；本项目提供转写、说话人分离、时间戳、声学事件和字幕处理，不执行翻译，也不包含 TTS。
- 137 项测试通过，1 项按环境跳过，275 个子测试通过；轻量版和完整离线版均通过运行时清单及分卷 ZIP 校验。

## 下载

- 完整离线版：下载 `full.zip.001` 至 `.006` 到同一目录，用 7-Zip 打开 `.001`；包含固定 revision 的模型。
- 轻量版：下载 `lightweight.zip.001` 至 `.005` 到同一目录，用 7-Zip 打开 `.001`；首次使用需要下载固定模型。
- ComfyUI 节点：继续使用单独发布的 [`v0.4.0`](https://github.com/T8mars/Comfyui-MOSS-Transcribe-Diarize-T8/releases/tag/v0.4.0)。
- 下载后请使用 `SHA256SUMS.txt` 校验每个分卷。

## English

- Fixed the empty wide-window state where a hidden sidebar and the collapsed-sidebar grid rule compressed the main workspace to roughly 12px.
- Stabilized import and processing layouts across normal size, the 1360px breakpoint, maximize, and restore.
- Added a native Windows packaged-EXE regression test for maximize/restore. The main view measured `1171/1186px` normally, `2048/2048px` maximized, and returned to the normal width after restore.
- Clarified in both UI languages that the language selector is a recognition hint. The product performs transcription, diarization, timestamping, acoustic-event recognition, and subtitle work; translation and TTS are not included.
- Passed 137 tests, 275 subtests, both packaged runtime inventory gates, and split ZIP reconstruction/CRC/content verification.

The Windows artifacts are unsigned pre-release builds. The companion ComfyUI node remains the separately published [`v0.4.0`](https://github.com/T8mars/Comfyui-MOSS-Transcribe-Diarize-T8/releases/tag/v0.4.0).
