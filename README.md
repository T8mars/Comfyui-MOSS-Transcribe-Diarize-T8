# comfyui-MOSS-Transcribe-Diarize-T8

**简体中文** | [English](README_EN.md)

由 T8star-Aix 维护的 MOSS Transcribe Diarize ComfyUI V3 节点包。支持本地离线模型以及显式授权的 SGLang Omni/vLLM OpenAI 兼容服务，提供单段与可恢复长音频转写、说话人编号、可选独立模型词级时间戳、跨分块声纹关联、真实 VAD、质量门、字幕可读性后处理以及 JSON/TXT/SRT/ASS/WebVTT/RTTM 导出。节点界面可随 ComfyUI 在中文和英文之间切换，默认仍为中文。

Windows 独立字幕工作台整合包请下载 [`desktop-v0.2.3`](https://github.com/T8mars/Comfyui-MOSS-Transcribe-Diarize-T8/releases/tag/desktop-v0.2.3)。它提供转写、说话人分离、字幕编辑与导出，不包含翻译或 TTS；配套 ComfyUI 节点仍为 `v0.4.0`。

## 安装

1. 推荐先在 ComfyUI Manager 中搜索 `comfyui-moss-transcribe-diarize-t8` 并安装。也可以克隆到 ComfyUI 的 `custom_nodes`：

   ```powershell
   cd ComfyUI/custom_nodes
   git clone https://github.com/T8mars/Comfyui-MOSS-Transcribe-Diarize-T8.git comfyui-MOSS-Transcribe-Diarize-T8
   ```

   也可以下载 GitHub Release ZIP，并解压为同名目录。

2. 在 ComfyUI Python 环境执行 `pip install -r requirements.txt`。依赖清单不会安装或替换 `torch`、`torchaudio`、`torchvision` 或 `transformers`。Windows Portable 应使用 `..\..\python_embeded\python.exe -m pip install -r requirements.txt`，不要误装到系统 Python。
3. 执行 `python scripts/check_transformers.py`；Windows Portable 对应 `..\..\python_embeded\python.exe scripts\check_transformers.py`。本节点安全下限为 Transformers 5.5.0；更旧版本存在已公开安全公告并会被加载器拒绝。需要修复时，在确认其他节点兼容后用 `requirements-transformers-v5.txt` 升级到 5.16.1。
4. 在节点目录执行 `python scripts/download_models.py --comfyui-root ..\..`；Windows Portable 对应 `..\..\python_embeded\python.exe scripts\download_models.py --comfyui-root ..\..`。也可以用 `--target` 指定绝对目录，或手动把固定版本模型放到 `ComfyUI/models/moss_transcribe_diarize/MOSS-Transcribe-Diarize`。脚本启动下载前会打印最终目标目录，无法确认 ComfyUI 根目录时会直接报错，不会把权重静默放进 `custom_nodes`。
5. 重启 ComfyUI，加载 `example_workflows/ui` 中的可视化工作流；API 调用示例位于 `example_workflows/api`。基础示例包含字幕后处理，长音频示例启用保守说话人关联，`03_remote_transcribe.json` 演示远程服务（运行前必须把 `allow_remote_upload` 改为 `true`），`04_word_alignment_voice_link.json` 演示词级对齐与声纹关联。两个辅助加载器首次执行时会按固定 revision 下载各自模型，也可填写本地模型目录。

模型固定到 Hugging Face revision `e8681d68e7042738ffca8ac8212bc8fcb1131ab8`，代码审计基线固定到 OpenMOSS revision `cb765f2b0fe6f7a298aa2002e2281ae693d1f3c3`。该上游提交修复了 Transformers 可能静默选择 eager、导致长音频 Attention 显存平方增长的问题。本包的 `auto` 会显式按 FlashAttention-2、SDPA、eager 顺序尝试并记录每次跳过、失败和最终选择；用户显式指定的后端失败时仍会直接报错，不会静默换成另一实现。加载器使用全文件 SHA-256 作为模型缓存身份；开启“完整 SHA-256 校验”时还会与固定 manifest 的预期摘要逐项比对。下载脚本会在模型目录写入仅供本机诊断的 `.t8-download-report.json`，不发送远程统计。

## 十五个 V3 节点

- 模型加载器：扫描标准模型目录和 `extra_model_paths.yaml` 注册路径，惰性加载、精度/Attention 选择、哈希校验，并提供常驻、显存压力释放、每次释放三档策略。
- 远程推理连接：连接 SGLang Omni/vLLM 的 OpenAI 兼容 `/v1/audio/transcriptions`，强制显式确认音频上传；可选 Bearer 密钥只从 `MOSS_TRANSCRIBE_API_KEY` 读取，不保存进工作流。
- 提示词与热词：构建严格格式提示、场景预设和专有名词热词；常用语言可直接选择，其他语言可填写名称或 BCP-47 标签，覆盖模型宣称的 50+ 语言能力。
- 转写与说话人分离：接收标准 `AUDIO`，安全下混/重采样至 16kHz，模型加载前执行 WebRTC VAD/能量预检，回传原生生成进度并响应 ComfyUI 中断；格式异常或质量风险可用更严格提示自动重试一次。
- 智能长音频：优先在 VAD 静音边界附近分片，支持重叠边界去重、全局时间轴、原子检查点和中断后续跑；默认隔离每片说话人，可选 `overlap_only` 只在重叠区文本和时间同时匹配时建立可审计的保守关联。
- Whisper 词级对齐模型：按固定 revision 配置独立 `openai/whisper-small`，支持设备、精度、语言、分片长度和完成后释放策略。
- 独立模型词级对齐：用 Whisper 生成真实词级时间锚点并映射回 MOSS 原文；报告模型匹配覆盖率，无法匹配的单位显式标记为插值而不是伪装成模型结果。
- WavLM 声纹模型：按固定 revision 配置 `microsoft/wavlm-base-plus-sv` X-Vector 说话人验证模型，并与主转写模型独立缓存。
- 声纹跨分块关联：从每个局部说话人的真实音频提取向量，以余弦阈值保守聚类；禁止同一分块内两个不同说话人互相合并，并输出链接、拒绝和失败报告。
- 转写解析与校验：检查输出格式、时间戳顺序/越界和 token 上限；缺失说话人标签的有效片段以未知 `S00` 保留并告警。
- 转写质量门：按尾部覆盖、未知说话人比例、重复循环、截断和格式错误输出可用性布尔值及 JSON 报告。
- 字幕后处理：按持续时间和字符数合并/切分，限制每行字符与行数，并报告字符每秒（CPS）超限段；不扩张原始总时间范围。
- 字幕样式：集中设置视频分辨率、自动/固定字号、字体、对齐、边距、描边、阴影和说话人配色。
- 字幕导出：接受可选字幕样式，支持当前分片重命名和显式的跨分片人工说话人映射，导出 JSON/TXT/SRT/ASS/WebVTT/RTTM，可写入 ComfyUI `output` 目录；JSON 保留稳定的 `speaker` ID，并在配置重命名时附加 `speaker_name`。
- 环境诊断与模型释放：报告 Transformers/PyTorch/CUDA/显存，只释放本节点包缓存的模型。

跨分片映射键使用“实际分片 ID:局部说话人编号”，例如 `{"part001:S01":"主持人"}`。字幕导出节点会自动将它应用到长音频合并后的命名空间说话人编号。

远程推理会把音频发送到所填服务。非本机 HTTP 地址会被拒绝，外网服务必须使用 HTTPS；URL 不能携带用户名、密码、查询令牌或片段。服务端进度和请求中的即时取消能力取决于服务实现。

## 兼容与限制

- 运行时要求 Transformers `>=5.5.0,<6`，兼容矩阵覆盖 5.6.0 与 5.16.1；安全修复文件固定到 5.16.1。自动化同时覆盖 Python 3.10、3.12、3.13 和当前 ComfyUI。
- Windows 10/11 x64 + NVIDIA 12GB 以上为正式目标。8-10GB 仅作为短音频兼容档；CPU FP32 仅作功能兜底，不承诺速度。
- MOSS 原生输出仍是句/段级时间戳；逐词时间戳必须显式连接独立 Whisper 对齐节点，并检查覆盖率与插值标记。`overlap_only` 只使用重叠文本证据；WavLM 节点可进一步按声音关联跨分块说话人，但阈值依素材而异，重要结果仍需人工复核。
- 静音、音乐、噪声和复杂长音频仍可能出现幻觉、提前结束或重复。VAD、严格重试和质量门用于暴露并拦截风险，不等同于人工审核或绝对准确性保证。
- 模型权重不放入 `custom_nodes`，由固定 revision 下载或外置共享。

本项目不是 OpenMOSS/MOSI 官方发行物。许可证与第三方声明见 `LICENSE`、`DISCLAIMER`、`THIRD_PARTY_NOTICES.md`。

## 已验证环境与正常测试

- Windows 11 x64、Python 3.12.13、PyTorch 2.8.0+cu128、Transformers 5.15.1（既有 GPU 实测基线）；CI 新增 Python 3.13、PyTorch 2.13.0 CPU 与 Transformers 5.16.1。
- ComfyUI revision `5ab2f7a2d676c1fb7b410c22e82e2ed8f217b56c` 的既有实测基线；十五个 V3 节点可独立注册，自动化测试覆盖本地/远程、辅助模型、安全和兼容路径。
- RTX 5090 Laptop 24GB，BF16，本地固定模型 revision。

| 输入时长 | 耗时 | 峰值显存 | 生成 token | 结果 |
|---:|---:|---:|---:|---|
| 7.66 秒 | 8.58 秒 | 1.74GB | 34 / 512 | v0.3.0 实机：WebRTC VAD 判定语音，`auto` 解析为 SDPA，质量门可用 |
| 75 秒（智能长音频） | 21.46 秒；续跑 0.044 秒 | 1.879GB；续跑 0GB | 492 / 2048 | 2 片、检查点全量恢复；循环夹具按预期被质量门以 `repeated_text` 拒绝 |
| 2 分钟 | 32.2 秒 | 2.33GB | 782 / 2944 | 31 段，无截断、无诊断错误 |
| 5 分钟 | 96.9 秒 | 4.53GB | 2073 / 5824 | 79 段，无截断、无诊断错误 |
| 10 分钟 | 282.4 秒 | 11.606GB | 4184 / 10624 | 157 段，覆盖至 600 秒、未截断；循环夹具触发重复文本告警 |

测试音频为清晰英语语音或其循环版本，用于确认完整执行链路和风险拦截，不代表真实会议质量基准；不同音频、显卡和提示词会产生不同结果。75 秒循环夹具的“不通过”是质量门正确识别重复内容。10 分钟结果来自 24GB RTX 5090 Laptop，不能证明 10GB 显卡可完成同一任务。

v0.4.0 还在同一台 RTX 5090 Laptop 上完成了固定 FLEURS revision 的公开真实人声回归；耗时包含启用的辅助模型：

| 公开用例 | 音频/耗时 | 峰值显存 | 质量与辅助指标 |
|---|---:|---:|---|
| 中文双性别短样本 | 34.28 秒 / 21.200 秒 | 3.762GB | CER 5.31%，词级模型匹配覆盖率 64.602% |
| 英文热词与大小写 | 20.97 秒 / 5.344 秒 | 3.477GB | WER 20.93%，`Lakkha Singh` 命中，词级覆盖率 71.698% |
| 中英日多语言 | 26.10 秒 / 5.220 秒 | 3.557GB | CER 2.098%，词级覆盖率 55.914% |
| 30 分钟、18dB SNR、4 分块 | 1800 秒 / 270.323 秒（RTF 0.15018） | 3.331GB | 127 段、尾部覆盖率 98.632%、7 个声纹链接、0 个声纹失败，全部门槛通过 |

新的可重复基准框架见 `benchmarks/README.md`：仓库提供确定性非语音守卫样本，以及固定 FLEURS revision、逐文件 SHA-256 来源记录和 30 分钟带噪真实人声用例生成器；不会提交生成的音频。报告包含 WER/CER、实时率、峰值显存、质量门、词级对齐覆盖率、声纹链接/失败数和回归阈值。

## 更新与卸载

更新时在节点目录执行 `git pull`，然后重启 ComfyUI。卸载时删除本节点目录即可；模型默认位于 `ComfyUI/models/moss_transcribe_diarize`，是否保留由用户自行决定。
