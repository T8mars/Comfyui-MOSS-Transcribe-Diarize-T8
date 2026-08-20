# Changelog

## 0.1.0 - 2026-08-20

- 首次公开版本，提供六个 `comfy_api.latest` V3 节点。
- 支持标准 ComfyUI `AUDIO`、16kHz 安全重采样、透传音频和本地 BF16/FP16/FP32 推理。
- 支持 Transformers 4.52.1 至 5.x，不自动替换 ComfyUI 的 Torch/Transformers 环境。
- 增加固定模型 revision、文件大小/SHA-256 校验和断点下载。
- 增加提示词/热词、转写校验、JSON/TXT/SRT/ASS 导出、说话人重命名、环境诊断和显存释放。
- 增加 ComfyUI 进度回传、队列中断、模型缓存和同模型推理锁。
- 提供 UI/API 基础与长音频诊断工作流。
- 通过独立目录导入、Comfy Registry、本地 GPU 2 分钟和 5 分钟冒烟测试。
