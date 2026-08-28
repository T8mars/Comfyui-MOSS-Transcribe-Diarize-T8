# comfyui-MOSS-Transcribe-Diarize-T8

[简体中文](README.md) | **English**

A MOSS Transcribe Diarize ComfyUI V3 custom node pack maintained by T8star-Aix. It runs locally and offline, providing single-pass and resumable long-audio transcription, speaker IDs, sentence/segment-level timestamps, real VAD, quality gating, and JSON/TXT/SRT/ASS export. Inference uses ComfyUI's native V3 progress API and supports queue interruption. The node UI follows ComfyUI's Chinese/English language setting; the backend definitions remain Chinese by default.

## Installation

1. Clone the repository into ComfyUI's `custom_nodes` directory:

   ```powershell
   cd ComfyUI/custom_nodes
   git clone https://github.com/T8mars/Comfyui-MOSS-Transcribe-Diarize-T8.git comfyui-MOSS-Transcribe-Diarize-T8
   ```

   Alternatively, download the GitHub Release ZIP and extract it to a directory with the same name.

2. Run `pip install -r requirements.txt` in the ComfyUI Python environment. The dependency list will not install or replace `torch`, `torchaudio`, `torchvision`, or `transformers`. For Windows Portable, use `..\..\python_embeded\python.exe -m pip install -r requirements.txt` to avoid installing packages into the system Python environment by mistake.
3. Run `python scripts/check_transformers.py`; for Windows Portable, use `..\..\python_embeded\python.exe scripts/check_transformers.py`. Transformers 5.5.0 through 5.x are kept unchanged. Versions 4.52.1 through 5.4 remain runtime-compatible with this node but are affected by published security advisories; after checking compatibility with other nodes, follow the prompt to upgrade to 5.15.1 with `requirements-transformers-v5.txt`.
4. From the node directory, run `python scripts/download_models.py --comfyui-root ..\..`; for Windows Portable, use `..\..\python_embeded\python.exe scripts/download_models.py --comfyui-root ..\..`. You can also specify an absolute path with `--target`, or manually place the pinned model in `ComfyUI/models/moss_transcribe_diarize/MOSS-Transcribe-Diarize`. The script prints the final destination before starting the download. If it cannot identify the ComfyUI root, it exits with an error instead of silently placing model weights under `custom_nodes`.
5. Restart ComfyUI and load a visual workflow from `example_workflows/ui`. API examples are available in `example_workflows/api`. The basic example covers hotwords, strict retry, and subtitle styling; the long-audio example covers VAD splitting, resumable checkpoints, quality gating, and environment diagnostics.

The model is pinned to Hugging Face revision `e8681d68e7042738ffca8ac8212bc8fcb1131ab8`, while the audited OpenMOSS code baseline is pinned to revision `cb765f2b0fe6f7a298aa2002e2281ae693d1f3c3`. That upstream commit fixes a Transformers path that could silently select eager Attention and cause quadratic memory growth on long audio. This pack's `auto` mode now explicitly tries FlashAttention-2, SDPA, and eager in that order and records every skipped, failed, and selected backend. An explicitly requested backend still fails clearly instead of silently changing to another implementation. The loader validates file sizes by default and optionally performs full SHA-256 verification. The download script writes `.t8-download-report.json` to the model directory for local diagnostics only and sends no remote telemetry.

## Nine V3 Nodes

- Model Loader: scans standard model directories and paths registered in `extra_model_paths.yaml`; supports lazy loading, precision/Attention selection, hash verification, and three cache policies: keep resident, release under VRAM pressure, or release after every run.
- Prompt and Hotwords: builds strictly formatted prompts, scene presets, language hints, and proper-noun hotwords.
- Transcription and Speaker Diarization: accepts standard `AUDIO`, safely downmixes and resamples to 16 kHz, performs WebRTC VAD or energy preflight before loading the model, reports native generation progress, responds to interruption, and can retry once with stricter formatting after a format or quality failure.
- Smart Long Audio: splits near VAD silence boundaries, removes overlap duplicates, keeps a global timeline, namespaces per-chunk speakers, writes atomic checkpoints, and resumes after interruption without pretending local `S01/S02` labels identify the same person across chunks.
- Transcript Parsing and Validation: checks output structure, timestamp ordering and bounds, and token limits. Valid segments without speaker labels are retained as unknown speaker `S00` and reported as warnings.
- Transcript Quality Gate: returns a usability decision and JSON report from end coverage, unknown-speaker ratio, repetition loops, truncation, and structural errors.
- Subtitle Style: configures video resolution, automatic or fixed font size, font, alignment, margins, outline, shadow, and per-speaker colors.
- Subtitle Export: accepts an optional subtitle style, supports speaker renaming and explicit manual mapping across chunks, exports JSON/TXT/SRT/ASS, and can write to the ComfyUI `output` directory.
- Environment Diagnostics and Model Release: reports Transformers/PyTorch/CUDA/VRAM information and releases only models cached by this node pack.

## Compatibility and Limitations

- Runtime compatibility covers Transformers `>=4.52.1,<6` and has been tested with 4.52.1, 4.57.6, 5.6.0, and 5.15.1. Because of published security advisories, `>=5.5.0,<6` is recommended and the security-repair file pins 5.15.1.
- The supported production target is Windows 10/11 x64 with an NVIDIA GPU and at least 12 GB of VRAM. GPUs with 8–10 GB are supported only as a short-audio compatibility tier. CPU FP32 is a functional fallback with no performance guarantee.
- Timestamps are generated at the sentence/segment level, not per word. Smart Long Audio namespaces speaker IDs by chunk; matching the same person across chunks still requires explicit manual mapping in the export node.
- Silence, music, noise, and complex long-form audio may still cause hallucinations, early termination, or repetition. VAD, strict retry, and the quality gate expose and block risk; they are not a substitute for human review or a guarantee of accuracy.
- Model weights are not stored in `custom_nodes`; they are downloaded from the pinned revision or loaded from an external shared location.

This project is not an official OpenMOSS/MOSI distribution. See `LICENSE`, `DISCLAIMER`, and `THIRD_PARTY_NOTICES.md` for license and third-party notices.

## Verified Environment and Normal-Path Tests

- Windows 11 x64, Python 3.12.13, PyTorch 2.8.0+cu128, and Transformers 5.15.1.
- ComfyUI revision `5ab2f7a2d676c1fb7b410c22e82e2ed8f217b56c`; all nine V3 nodes registered independently, with 36 automated tests covering the new features and compatibility paths.
- RTX 5090 Laptop 24 GB, BF16, with the locally pinned model revision.

| Input length | Runtime | Peak VRAM | Generated tokens | Result |
|---:|---:|---:|---:|---|
| 7.66 seconds | 8.58 seconds | 1.74 GB | 34 / 512 | v0.3.0 hardware run: WebRTC VAD detected speech, `auto` resolved to SDPA, and the quality gate passed |
| 75 seconds (Smart Long Audio) | 21.46 seconds; 0.044-second resume | 1.879 GB; 0 GB on resume | 492 / 2048 | 2 chunks and full checkpoint restore; the looping fixture was correctly rejected as `repeated_text` |
| 2 minutes | 32.2 seconds | 2.33 GB | 782 / 2944 | 31 segments; no truncation or diagnostic errors |
| 5 minutes | 96.9 seconds | 4.53 GB | 2073 / 5824 | 79 segments; no truncation or diagnostic errors |
| 10 minutes | 282.4 seconds | 11.606 GB | 4184 / 10624 | 157 segments covering 600 seconds without truncation; the looping fixture triggered a repeated-text warning |

The test inputs contain clear English speech or a looped version of it and are intended to validate the complete execution path and risk controls. They are not a benchmark for real meeting quality; results vary with audio, GPU, and prompting. The 75-second loop's failed quality decision is expected and demonstrates repetition detection. The 10-minute result was produced on an RTX 5090 Laptop with 24 GB of VRAM and does not demonstrate that a 10 GB GPU can complete the same workload.

## Updating and Uninstalling

To update, run `git pull` from the node directory and restart ComfyUI. To uninstall, delete the node directory. Models are stored in `ComfyUI/models/moss_transcribe_diarize` by default; you can choose whether to retain them.
