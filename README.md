# comfyui-MOSS-Transcribe-Diarize-T8

**简体中文** | [English](README_EN.md)

由 T8star-Aix 维护的 MOSS Transcribe Diarize ComfyUI V3 节点包。节点执行本地离线推理，提供单段与可恢复长音频转写、说话人编号、句/段级时间戳、真实 VAD、质量门以及 JSON/TXT/SRT/ASS 导出。推理过程使用 ComfyUI V3 原生进度接口并支持队列中断；节点界面可随 ComfyUI 在中文和英文之间切换，默认定义仍为中文。

## 安装

1. 克隆到 ComfyUI 的 `custom_nodes`：

   ```powershell
   cd ComfyUI/custom_nodes
   git clone https://github.com/T8mars/Comfyui-MOSS-Transcribe-Diarize-T8.git comfyui-MOSS-Transcribe-Diarize-T8
   ```

   也可以下载 GitHub Release ZIP，并解压为同名目录。

2. 在 ComfyUI Python 环境执行 `pip install -r requirements.txt`。依赖清单不会安装或替换 `torch`、`torchaudio`、`torchvision` 或 `transformers`。Windows Portable 应使用 `..\..\python_embeded\python.exe -m pip install -r requirements.txt`，不要误装到系统 Python。
3. 执行 `python scripts/check_transformers.py`；Windows Portable 对应 `..\..\python_embeded\python.exe scripts\check_transformers.py`。已有 Transformers 4.52.1–5.x 会原样保留；仅当版本过旧时，才按提示选择 `requirements-transformers-v4.txt` 修复到 4.57.6。
4. 在节点目录执行 `python scripts/download_models.py --comfyui-root ..\..`；Windows Portable 对应 `..\..\python_embeded\python.exe scripts\download_models.py --comfyui-root ..\..`。也可以用 `--target` 指定绝对目录，或手动把固定版本模型放到 `ComfyUI/models/moss_transcribe_diarize/MOSS-Transcribe-Diarize`。脚本启动下载前会打印最终目标目录，无法确认 ComfyUI 根目录时会直接报错，不会把权重静默放进 `custom_nodes`。
5. 重启 ComfyUI，加载 `example_workflows/ui` 中的可视化工作流；API 调用示例位于 `example_workflows/api`。基础示例覆盖热词、严格重试和字幕样式，长音频示例覆盖 VAD 切分、断点续跑、质量门与环境诊断。

模型固定到 Hugging Face revision `e8681d68e7042738ffca8ac8212bc8fcb1131ab8`，代码审计基线固定到 OpenMOSS revision `cde3c13af82c3001a21cf085d37ebc7d81e8981d`。该上游提交撤销了显式 Attention 选择，因此本包的 `auto` 不传覆盖参数、跟随上游/Transformers 默认并记录实际结果；用户显式指定的后端失败时会直接报错，不会静默换成另一实现。加载器默认校验文件大小，可选全量 SHA-256；下载脚本会在模型目录写入仅供本机诊断的 `.t8-download-report.json`，不发送远程统计。

## 九个 V3 节点

- 模型加载器：扫描标准模型目录和 `extra_model_paths.yaml` 注册路径，惰性加载、精度/Attention 选择、哈希校验，并提供常驻、显存压力释放、每次释放三档策略。
- 提示词与热词：构建严格格式提示、场景预设、语言提示和专有名词热词。
- 转写与说话人分离：接收标准 `AUDIO`，安全下混/重采样至 16kHz，模型加载前执行 WebRTC VAD/能量预检，回传原生生成进度并响应 ComfyUI 中断；格式异常或质量风险可用更严格提示自动重试一次。
- 智能长音频：优先在 VAD 静音边界附近分片，支持重叠边界去重、全局时间轴、分片说话人命名空间、原子检查点和中断后续跑；不会把不同分片的局部 `S01/S02` 擅自合并。
- 转写解析与校验：检查输出格式、时间戳顺序/越界和 token 上限；缺失说话人标签的有效片段以未知 `S00` 保留并告警。
- 转写质量门：按尾部覆盖、未知说话人比例、重复循环、截断和格式错误输出可用性布尔值及 JSON 报告。
- 字幕样式：集中设置视频分辨率、自动/固定字号、字体、对齐、边距、描边、阴影和说话人配色。
- 字幕导出：接受可选字幕样式，支持当前分片重命名和显式的跨分片人工说话人映射，导出 JSON/TXT/SRT/ASS，可写入 ComfyUI `output` 目录。
- 环境诊断与模型释放：报告 Transformers/PyTorch/CUDA/显存，只释放本节点包缓存的模型。

## 兼容与限制

- 支持 Transformers `>=4.52.1,<6`，共享兼容层避免强制把现有 ComfyUI 升级到 5.x；已通过 4.52.1、4.57.6、5.6.0、5.15.1 兼容测试。
- Windows 10/11 x64 + NVIDIA 12GB 以上为正式目标。8-10GB 仅作为短音频兼容档；CPU FP32 仅作功能兜底，不承诺速度。
- 输出是句/段级时间戳，不是逐词时间戳。智能长音频节点会把说话人编号按分片隔离；跨分片同一人仍需通过导出节点显式人工映射。
- 静音、音乐、噪声和复杂长音频仍可能出现幻觉、提前结束或重复。VAD、严格重试和质量门用于暴露并拦截风险，不等同于人工审核或绝对准确性保证。
- 模型权重不放入 `custom_nodes`，由固定 revision 下载或外置共享。

本项目不是 OpenMOSS/MOSI 官方发行物。许可证与第三方声明见 `LICENSE`、`DISCLAIMER`、`THIRD_PARTY_NOTICES.md`。

## 已验证环境与正常测试

- Windows 11 x64、Python 3.12.13、PyTorch 2.8.0+cu128、Transformers 5.15.1。
- ComfyUI revision `5ab2f7a2d676c1fb7b410c22e82e2ed8f217b56c`，九个 V3 节点独立注册成功；31 项自动化测试覆盖新功能与兼容路径。
- RTX 5090 Laptop 24GB，BF16，本地固定模型 revision。

| 输入时长 | 耗时 | 峰值显存 | 生成 token | 结果 |
|---:|---:|---:|---:|---|
| 7.66 秒 | 8.58 秒 | 1.74GB | 34 / 512 | v0.3.0 实机：WebRTC VAD 判定语音，`auto` 解析为 SDPA，质量门可用 |
| 75 秒（智能长音频） | 21.46 秒；续跑 0.044 秒 | 1.879GB；续跑 0GB | 492 / 2048 | 2 片、检查点全量恢复；循环夹具按预期被质量门以 `repeated_text` 拒绝 |
| 2 分钟 | 32.2 秒 | 2.33GB | 782 / 2944 | 31 段，无截断、无诊断错误 |
| 5 分钟 | 96.9 秒 | 4.53GB | 2073 / 5824 | 79 段，无截断、无诊断错误 |
| 10 分钟 | 282.4 秒 | 11.606GB | 4184 / 10624 | 157 段，覆盖至 600 秒、未截断；循环夹具触发重复文本告警 |

测试音频为清晰英语语音或其循环版本，用于确认完整执行链路和风险拦截，不代表真实会议质量基准；不同音频、显卡和提示词会产生不同结果。75 秒循环夹具的“不通过”是质量门正确识别重复内容。10 分钟结果来自 24GB RTX 5090 Laptop，不能证明 10GB 显卡可完成同一任务。

## 更新与卸载

更新时在节点目录执行 `git pull`，然后重启 ComfyUI。卸载时删除本节点目录即可；模型默认位于 `ComfyUI/models/moss_transcribe_diarize`，是否保留由用户自行决定。
