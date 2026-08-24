# comfyui-MOSS-Transcribe-Diarize-T8

[简体中文](README.md) | **English**

A MOSS Transcribe Diarize ComfyUI V3 custom node pack maintained by T8star-Aix. It runs inference locally and offline, providing transcription, speaker IDs, sentence/segment-level timestamps, structural validation, and JSON/TXT/SRT/ASS export. Inference reports progress to ComfyUI and supports queue interruption.

## Installation

1. Clone the repository into ComfyUI's `custom_nodes` directory:

   ```powershell
   cd ComfyUI/custom_nodes
   git clone https://github.com/T8mars/Comfyui-MOSS-Transcribe-Diarize-T8.git comfyui-MOSS-Transcribe-Diarize-T8
   ```

   Alternatively, download the GitHub Release ZIP and extract it to a directory with the same name.

2. Run `pip install -r requirements.txt` in the ComfyUI Python environment. The dependency list will not install or replace `torch`, `torchaudio`, `torchvision`, or `transformers`. For Windows Portable, use `..\..\python_embeded\python.exe -m pip install -r requirements.txt` to avoid installing packages into the system Python environment by mistake.
3. Run `python scripts/check_transformers.py`; for Windows Portable, use `..\..\python_embeded\python.exe scripts/check_transformers.py`. Existing Transformers versions from 4.52.1 through 5.x are kept unchanged. Only when the installed version is too old should you follow the prompt and use `requirements-transformers-v4.txt` to repair it to 4.57.6.
4. From the node directory, run `python scripts/download_models.py --comfyui-root ..\..`; for Windows Portable, use `..\..\python_embeded\python.exe scripts/download_models.py --comfyui-root ..\..`. You can also specify an absolute path with `--target`, or manually place the pinned model in `ComfyUI/models/moss_transcribe_diarize/MOSS-Transcribe-Diarize`. The script prints the final destination before starting the download. If it cannot identify the ComfyUI root, it exits with an error instead of silently placing model weights under `custom_nodes`.
5. Restart ComfyUI and load a visual workflow from `example_workflows/ui`. API examples are available in `example_workflows/api`. The basic example covers hotwords and subtitle export; the long-audio example adds transcript validation and environment diagnostics.

The model is pinned to Hugging Face revision `e8681d68e7042738ffca8ac8212bc8fcb1131ab8`, while the audited OpenMOSS code baseline is pinned to revision `e607537b1b870475e7898969d40b864de8b691b6`. The loader validates file sizes by default and optionally performs full SHA-256 verification. The download script writes `.t8-download-report.json` to the model directory for local diagnostics only and sends no remote telemetry.

## Six V3 Nodes

- Model Loader: scans standard model directories and paths registered in `extra_model_paths.yaml`; supports lazy loading, precision/Attention selection, hash verification, and three cache policies: keep resident, release under VRAM pressure, or release after every run.
- Prompt and Hotwords: builds strictly formatted prompts, scene presets, language hints, and proper-noun hotwords.
- Transcription and Speaker Diarization: accepts standard `AUDIO`, safely downmixes and resamples to 16 kHz, performs a configurable silence preflight before loading the model, reports generation progress, responds to ComfyUI interruption, and outputs pass-through audio, raw text, JSON, SRT, ASS, and `T8_MOSS_TRANSCRIPT`.
- Transcript Parsing and Validation: checks output structure, timestamp ordering and bounds, and token limits. Valid segments without speaker labels are retained as unknown speaker `S00` and reported as warnings.
- Subtitle Export: supports renaming speakers in the current chunk and explicit manual speaker mapping across chunks; exports JSON/TXT/SRT/ASS and can write to the ComfyUI `output` directory.
- Environment Diagnostics and Model Release: reports Transformers/PyTorch/CUDA/VRAM information and releases only models cached by this node pack.

## Compatibility and Limitations

- Supports Transformers `>=4.52.1,<6`. A shared compatibility layer avoids forcing an existing ComfyUI installation to upgrade to Transformers 5.x. Compatibility has been tested with 4.52.1, 4.57.6, 5.6.0, and 5.15.1.
- The supported production target is Windows 10/11 x64 with an NVIDIA GPU and at least 12 GB of VRAM. GPUs with 8–10 GB are supported only as a short-audio compatibility tier. CPU FP32 is a functional fallback with no performance guarantee.
- Timestamps are generated at the sentence/segment level, not per word. Long audio is preferably processed in one inference pass. `S01/S02` from separate chunks cannot automatically be treated as the same person across chunks; use explicit manual mapping in the export node.
- Silence, long silent regions, and complex long-form audio may cause hallucinations, early termination, or repetition. The nodes report diagnostics but do not present suspicious output as reliable transcription.
- Model weights are not stored in `custom_nodes`; they are downloaded from the pinned revision or loaded from an external shared location.

This project is not an official OpenMOSS/MOSI distribution. See `LICENSE`, `DISCLAIMER`, and `THIRD_PARTY_NOTICES.md` for license and third-party notices.

## Verified Environment and Normal-Path Tests

- Windows 11 x64, Python 3.12.13, PyTorch 2.8.0+cu128, and Transformers 5.15.1.
- ComfyUI revision `5ab2f7a2d676c1fb7b410c22e82e2ed8f217b56c`; all six V3 nodes registered independently and successfully.
- RTX 5090 Laptop 24 GB, BF16, with the locally pinned model revision.

| Input length | Runtime | Peak VRAM | Generated tokens | Result |
|---:|---:|---:|---:|---|
| 2 minutes | 32.2 seconds | 2.33 GB | 782 / 2944 | 31 segments; no truncation or diagnostic errors |
| 5 minutes | 96.9 seconds | 4.53 GB | 2073 / 5824 | 79 segments; no truncation or diagnostic errors |
| 10 minutes | 282.4 seconds | 11.606 GB | 4184 / 10624 | 157 segments covering 600 seconds without truncation; the looping fixture triggered a repeated-text warning |

The test audio consists of repeated, clear English speech and is intended to validate the complete normal-duration execution path. It is not a benchmark for real meeting quality; results vary with audio, GPU, and prompting. The 10-minute result was produced on an RTX 5090 Laptop with 24 GB of VRAM and does not demonstrate that a 10 GB GPU can complete the same workload.

## Updating and Uninstalling

To update, run `git pull` from the node directory and restart ComfyUI. To uninstall, delete the node directory. Models are stored in `ComfyUI/models/moss_transcribe_diarize` by default; you can choose whether to retain them.
