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

The optional word-alignment workflow downloads the pinned
[`openai/whisper-small`](https://huggingface.co/openai/whisper-small) snapshot, whose model card
declares Apache-2.0. The optional voice-link workflow downloads the pinned
[`microsoft/wavlm-base-plus-sv`](https://huggingface.co/microsoft/wavlm-base-plus-sv) snapshot.
Neither auxiliary model is included in this repository or release ZIP; each remains subject to its
upstream model card, license, and usage terms.

The public real-audio benchmark generator downloads selected rows from the pinned
[`google/fleurs`](https://huggingface.co/datasets/google/fleurs) revision under CC-BY-4.0 and writes
source IDs, hashes, and attribution into a local provenance file. Downloaded/generated benchmark
audio is ignored by Git and is not distributed in the release ZIP.

No third-party project endorses the T8star-Aix integration unless explicitly stated by that project.
