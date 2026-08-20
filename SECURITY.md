# Security Policy

请不要在公开 issue 中粘贴敏感录音、访问令牌、私人路径或完整环境变量。

安全问题请通过 GitHub 仓库的 **Security → Report a vulnerability** 私下提交。普通兼容性和功能问题可以使用公开 issue，并附上已脱敏的 ComfyUI、Python、PyTorch、Transformers、CUDA 和显卡版本。

本节点默认只访问本地模型。只有用户主动运行 `scripts/download_models.py` 时才会连接 Hugging Face，并且下载固定 revision 后执行清单校验。
