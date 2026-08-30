# Third-party notices

This ComfyUI custom-node source package redistributes integration code derived from
[OpenMOSS/MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize),
reviewed at commit `cb765f2b0fe6f7a298aa2002e2281ae693d1f3c3`, under the Apache License 2.0.
The license text is included in `LICENSE`. Files changed by this integration carry an explicit
modification notice; the integration changes are summarized in `CHANGELOG.md`.

The release ZIP does **not** bundle model weights, Python, PyTorch, CUDA/cuDNN, FFmpeg, Electron,
Chromium, or third-party Python wheels. The model downloader retrieves the pinned OpenMOSS model
snapshot separately from Hugging Face. Python dependencies listed in `requirements.txt` and the
optional `requirements-transformers-v5.txt` are installed separately by the user's environment and
remain subject to their own upstream licenses and notices.

No third-party project endorses the T8star-Aix integration unless explicitly stated by that project.
