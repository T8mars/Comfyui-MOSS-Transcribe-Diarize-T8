# comfyui-MOSS-Transcribe-Diarize-T8

由 T8star-Aix 维护的 MOSS Transcribe Diarize ComfyUI V3 节点包。节点执行本地离线推理，提供转写、说话人编号、句/段级时间戳、结构校验以及 JSON/TXT/SRT/ASS 导出。推理过程支持 ComfyUI 进度回传和队列中断。

## 安装

1. 克隆到 ComfyUI 的 `custom_nodes`：

   ```powershell
   cd ComfyUI/custom_nodes
   git clone https://github.com/T8mars/Comfyui-MOSS-Transcribe-Diarize-T8.git comfyui-MOSS-Transcribe-Diarize-T8
   ```

   也可以下载 GitHub Release ZIP，并解压为同名目录。

2. 在 ComfyUI Python 环境执行 `pip install -r requirements.txt`。依赖清单不会安装或替换 `torch`、`torchaudio`、`torchvision` 或 `transformers`。
3. 执行 `python scripts/check_transformers.py`。已有 Transformers 4.52.1–5.x 会原样保留；仅当版本过旧时，才按提示选择 `requirements-transformers-v4.txt` 修复到 4.57.6。
4. 在节点目录执行 `python scripts/download_models.py`，或手动把固定版本模型放到 `ComfyUI/models/moss_transcribe_diarize/MOSS-Transcribe-Diarize`。
5. 重启 ComfyUI，加载 `example_workflows/ui` 中的可视化工作流；API 调用示例位于 `example_workflows/api`。基础示例覆盖热词和字幕导出，长音频示例增加转写校验与环境诊断。

模型固定到 Hugging Face revision `e8681d68e7042738ffca8ac8212bc8fcb1131ab8`，代码基线固定到 OpenMOSS revision `0e3d1403fd8f1f1c674e883ece96b9f630794ebe`。加载器默认校验文件大小，可选全量 SHA-256。

## 六个 V3 节点

- 模型加载器：扫描标准模型目录和 `extra_model_paths.yaml` 注册路径，惰性加载、精度选择、哈希校验。
- 提示词与热词：构建严格格式提示、语言提示和专有名词热词。
- 转写与说话人分离：接收标准 `AUDIO`，安全下混/重采样至 16kHz，回传生成进度并响应 ComfyUI 中断，输出透传音频、原文、JSON、SRT、ASS、`T8_MOSS_TRANSCRIPT`。
- 转写解析与校验：检查输出格式、时间戳顺序/越界和 token 上限。
- 字幕导出：说话人重命名并导出 JSON/TXT/SRT/ASS，可写入 ComfyUI `output` 目录。
- 环境诊断与模型释放：报告 Transformers/PyTorch/CUDA/显存，只释放本节点包缓存的模型。

## 兼容与限制

- 支持 Transformers `>=4.52.1,<6`，共享兼容层避免强制把现有 ComfyUI 升级到 5.x；已通过 4.52.1、4.57.6、5.6.0、5.15.1 兼容测试。
- Windows 10/11 x64 + NVIDIA 10GB 以上为正式目标。8GB 只保证经过实测的短音频；CPU FP32 仅作功能兜底，不承诺速度。
- 输出是句/段级时间戳，不是逐词时间戳。长音频优先一次推理；将来若启用分片，各分片的 `S01/S02` 不能自动视为跨分片同一人。
- 静音、长静音、复杂长音频可能出现幻觉、提前结束或重复。节点会输出诊断，但不会自动把可疑结果伪装成可靠结果。
- 模型权重不放入 `custom_nodes`，由固定 revision 下载或外置共享。

本项目不是 OpenMOSS/MOSI 官方发行物。许可证与第三方声明见 `LICENSE`、`DISCLAIMER`、`THIRD_PARTY_NOTICES.md`。

## 已验证环境与正常测试

- Windows 11 x64、Python 3.12.13、PyTorch 2.8.0+cu128、Transformers 5.15.1。
- ComfyUI revision `5ab2f7a2d676c1fb7b410c22e82e2ed8f217b56c`，六个 V3 节点独立注册成功。
- RTX 5090 Laptop 24GB，BF16，本地固定模型 revision。

| 输入时长 | 耗时 | 峰值显存 | 生成 token | 结果 |
|---:|---:|---:|---:|---|
| 2 分钟 | 32.2 秒 | 2.33GB | 782 / 2944 | 31 段，无截断、无诊断错误 |
| 5 分钟 | 96.9 秒 | 4.53GB | 2073 / 5824 | 79 段，无截断、无诊断错误 |

测试音频为重复的清晰英语语音，用于确认普通时长的完整执行链路，不代表真实会议质量基准；不同音频、显卡和提示词会产生不同结果。

## 更新与卸载

更新时在节点目录执行 `git pull`，然后重启 ComfyUI。卸载时删除本节点目录即可；模型默认位于 `ComfyUI/models/moss_transcribe_diarize`，是否保留由用户自行决定。
