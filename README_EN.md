# comfyui-MOSS-Transcribe-Diarize-T8

[简体中文](README.md) | **English**

A MOSS Transcribe Diarize ComfyUI V3 custom node pack maintained by T8star-Aix. It supports both local offline inference and explicitly authorized SGLang Omni/vLLM OpenAI-compatible services, with single-pass and resumable long-audio transcription, speaker IDs, optional independent-model word timestamps, cross-chunk voice-embedding links, real VAD, quality gating, subtitle readability postprocessing, and JSON/TXT/SRT/ASS/WebVTT/RTTM export. The node UI follows ComfyUI's Chinese/English language setting; Chinese remains the default.

For the standalone Windows Subtitle Studio bundle, download [`desktop-v0.2.3`](https://github.com/T8mars/Comfyui-MOSS-Transcribe-Diarize-T8/releases/tag/desktop-v0.2.3). It provides transcription, diarization, subtitle editing, and export; translation and TTS are not included. The companion ComfyUI node remains `v0.4.0`.

## Installation

1. Recommended: search for `comfyui-moss-transcribe-diarize-t8` in ComfyUI Manager and install it. You can also clone the repository into ComfyUI's `custom_nodes` directory:

   ```powershell
   cd ComfyUI/custom_nodes
   git clone https://github.com/T8mars/Comfyui-MOSS-Transcribe-Diarize-T8.git comfyui-MOSS-Transcribe-Diarize-T8
   ```

   Alternatively, download the GitHub Release ZIP and extract it to a directory with the same name.

2. Run `pip install -r requirements.txt` in the ComfyUI Python environment. The dependency list will not install or replace `torch`, `torchaudio`, `torchvision`, or `transformers`. For Windows Portable, use `..\..\python_embeded\python.exe -m pip install -r requirements.txt` to avoid installing packages into the system Python environment by mistake.
3. Run `python scripts/check_transformers.py`; for Windows Portable, use `..\..\python_embeded\python.exe scripts/check_transformers.py`. This node enforces Transformers 5.5.0 as its security minimum; older versions have published advisories and are rejected by the loader. After checking compatibility with other nodes, use `requirements-transformers-v5.txt` to repair the environment to 5.16.1.
4. From the node directory, run `python scripts/download_models.py --comfyui-root ..\..`; for Windows Portable, use `..\..\python_embeded\python.exe scripts/download_models.py --comfyui-root ..\..`. You can also specify an absolute path with `--target`, or manually place the pinned model in `ComfyUI/models/moss_transcribe_diarize/MOSS-Transcribe-Diarize`. The script prints the final destination before starting the download. If it cannot identify the ComfyUI root, it exits with an error instead of silently placing model weights under `custom_nodes`.
5. Restart ComfyUI and load a visual workflow from `example_workflows/ui`. API examples are in `example_workflows/api`. The basic example includes subtitle postprocessing, the long-audio example enables conservative speaker linking, `03_remote_transcribe.json` demonstrates a remote service (set `allow_remote_upload` to `true` only after reviewing the endpoint), and `04_word_alignment_voice_link.json` demonstrates word alignment plus voice linking. The two auxiliary loaders download their pinned model revisions on first execution, or they can use local model directories.

The model is pinned to Hugging Face revision `e8681d68e7042738ffca8ac8212bc8fcb1131ab8`, while the audited OpenMOSS code baseline is pinned to revision `cb765f2b0fe6f7a298aa2002e2281ae693d1f3c3`. That upstream commit fixes a Transformers path that could silently select eager Attention and cause quadratic memory growth on long audio. This pack's `auto` mode explicitly tries FlashAttention-2, SDPA, and eager in that order and records every skipped, failed, and selected backend. An explicitly requested backend still fails clearly instead of silently changing to another implementation. The loader uses full-file SHA-256 values for model cache identity; enabling full verification also compares each file with the pinned manifest digest. The download script writes `.t8-download-report.json` to the model directory for local diagnostics only and sends no remote telemetry.

## Fifteen V3 Nodes

- Model Loader: scans standard model directories and paths registered in `extra_model_paths.yaml`; supports lazy loading, precision/Attention selection, hash verification, and three cache policies: keep resident, release under VRAM pressure, or release after every run.
- Remote Inference Connection: connects to an SGLang Omni/vLLM OpenAI-compatible `/v1/audio/transcriptions` endpoint and requires explicit upload consent. An optional Bearer token is read only from `MOSS_TRANSCRIBE_API_KEY`, never stored in the workflow.
- Prompt and Hotwords: builds strict-format prompts, scene presets, common language selections, arbitrary language/BCP-47 hints, and proper-noun hotwords for the model's 50+ supported languages.
- Transcription and Speaker Diarization: accepts standard `AUDIO`, safely downmixes and resamples to 16 kHz, performs WebRTC VAD or energy preflight before loading the model, reports native generation progress, responds to interruption, and can retry once with stricter formatting after a format or quality failure.
- Smart Long Audio: splits near VAD silence boundaries, removes overlap duplicates, keeps a global timeline, writes atomic checkpoints, and resumes after interruption. Speakers remain chunk-isolated by default; optional `overlap_only` linking requires matching text and time evidence in the actual overlap and reports its evidence.
- Whisper Word-Alignment Model: configures an independent pinned `openai/whisper-small` revision with device, precision, language, chunk length, and post-run release controls.
- Independent-Model Word Alignment: derives real Whisper word anchors and maps them back to the MOSS transcript. It reports model-match coverage and explicitly labels unmatched units as interpolated instead of presenting them as model output.
- WavLM Speaker Model: configures a pinned `microsoft/wavlm-base-plus-sv` X-Vector speaker-verification model with a cache independent of the main transcription model.
- Voice-Embedding Cross-Chunk Link: extracts embeddings from each local speaker's actual audio and clusters conservatively by cosine threshold. It forbids merging two distinct speakers from the same chunk and reports links, rejections, and failures.
- Transcript Parsing and Validation: checks output structure, timestamp ordering and bounds, and token limits. Valid segments without speaker labels are retained as unknown speaker `S00` and reported as warnings.
- Transcript Quality Gate: returns a usability decision and JSON report from end coverage, unknown-speaker ratio, repetition loops, truncation, and structural errors.
- Subtitle Postprocess: merges/splits cues by duration and length, constrains lines and characters per line, and reports characters-per-second violations without expanding the original timeline.
- Subtitle Style: configures video resolution, automatic or fixed font size, font, alignment, margins, outline, shadow, and per-speaker colors.
- Subtitle Export: accepts an optional subtitle style, supports speaker renaming and explicit manual mapping across chunks, exports JSON/TXT/SRT/ASS/WebVTT/RTTM, and can write to the ComfyUI `output` directory. JSON keeps the stable `speaker` ID and adds `speaker_name` when a rename is configured.
- Environment Diagnostics and Model Release: reports Transformers/PyTorch/CUDA/VRAM information and releases only models cached by this node pack.

Cross-chunk mapping keys use the actual chunk ID and local speaker ID, for example `{"part001:S01":"Host"}`. Subtitle Export automatically applies that mapping to the namespaced speaker ID in the merged long-audio transcript.

Remote inference sends audio to the configured service. Non-loopback HTTP is rejected, external services must use HTTPS, and endpoint URLs cannot carry usernames, passwords, query tokens, or fragments. Server-side progress and cancellation during an in-flight request depend on the server implementation.

## Compatibility and Limitations

- Runtime support requires Transformers `>=5.5.0,<6`; the compatibility matrix covers 5.6.0 and 5.16.1. The security-repair file pins 5.16.1, and automation covers Python 3.10, 3.12, 3.13, plus current ComfyUI.
- The supported production target is Windows 10/11 x64 with an NVIDIA GPU and at least 12 GB of VRAM. GPUs with 8–10 GB are supported only as a short-audio compatibility tier. CPU FP32 is a functional fallback with no performance guarantee.
- Native MOSS timestamps remain sentence/segment level. Word timestamps require the explicit Whisper alignment nodes and review of coverage/interpolation labels. `overlap_only` uses duplicate overlap evidence; the WavLM node can additionally link voices across chunks, but its threshold is recording-dependent and consequential output still needs review.
- Silence, music, noise, and complex long-form audio may still cause hallucinations, early termination, or repetition. VAD, strict retry, and the quality gate expose and block risk; they are not a substitute for human review or a guarantee of accuracy.
- Model weights are not stored in `custom_nodes`; they are downloaded from the pinned revision or loaded from an external shared location.

This project is not an official OpenMOSS/MOSI distribution. See `LICENSE`, `DISCLAIMER`, and `THIRD_PARTY_NOTICES.md` for license and third-party notices.

## Verified Environment and Normal-Path Tests

- Windows 11 x64, Python 3.12.13, PyTorch 2.8.0+cu128, and Transformers 5.15.1 are the existing GPU-tested baseline; CI now adds Python 3.13, PyTorch 2.13.0 CPU, and Transformers 5.16.1.
- Existing hardware baseline at ComfyUI revision `5ab2f7a2d676c1fb7b410c22e82e2ed8f217b56c`; all fifteen V3 nodes register independently, with automation covering local/remote, auxiliary-model, security, and compatibility paths.
- RTX 5090 Laptop 24 GB, BF16, with the locally pinned model revision.

| Input length | Runtime | Peak VRAM | Generated tokens | Result |
|---:|---:|---:|---:|---|
| 7.66 seconds | 8.58 seconds | 1.74 GB | 34 / 512 | v0.3.0 hardware run: WebRTC VAD detected speech, `auto` resolved to SDPA, and the quality gate passed |
| 75 seconds (Smart Long Audio) | 21.46 seconds; 0.044-second resume | 1.879 GB; 0 GB on resume | 492 / 2048 | 2 chunks and full checkpoint restore; the looping fixture was correctly rejected as `repeated_text` |
| 2 minutes | 32.2 seconds | 2.33 GB | 782 / 2944 | 31 segments; no truncation or diagnostic errors |
| 5 minutes | 96.9 seconds | 4.53 GB | 2073 / 5824 | 79 segments; no truncation or diagnostic errors |
| 10 minutes | 282.4 seconds | 11.606 GB | 4184 / 10624 | 157 segments covering 600 seconds without truncation; the looping fixture triggered a repeated-text warning |

The test inputs contain clear English speech or a looped version of it and are intended to validate the complete execution path and risk controls. They are not a benchmark for real meeting quality; results vary with audio, GPU, and prompting. The 75-second loop's failed quality decision is expected and demonstrates repetition detection. The 10-minute result was produced on an RTX 5090 Laptop with 24 GB of VRAM and does not demonstrate that a 10 GB GPU can complete the same workload.

v0.4.0 also completed the pinned FLEURS public real-human-speech regression on the same RTX 5090 Laptop. Runtime includes the enabled auxiliary model:

| Public case | Audio / runtime | Peak VRAM | Quality and auxiliary metrics |
|---|---:|---:|---|
| Chinese mixed-gender short speech | 34.28 s / 21.200 s | 3.762 GB | 5.31% CER; 64.602% model-matched word-alignment coverage |
| English hotwords and capitalization | 20.97 s / 5.344 s | 3.477 GB | 20.93% WER; `Lakkha Singh` present; 71.698% word coverage |
| Chinese/English/Japanese | 26.10 s / 5.220 s | 3.557 GB | 2.098% CER; 55.914% word coverage |
| 30 min, 18 dB SNR, 4 chunks | 1800 s / 270.323 s (RTF 0.15018) | 3.331 GB | 127 segments, 98.632% end coverage, 7 voice links, 0 embedding failures; all gates passed |

The reproducible benchmark framework is documented in `benchmarks/README.md`. It provides deterministic non-speech guardrails plus a pinned FLEURS revision, per-file SHA-256 provenance, and a 30-minute noisy real-human-speech generator; generated audio is not committed. Reports include WER/CER, real-time factor, peak VRAM, quality output, word-alignment coverage, speaker-link/failure counts, diagnostics, and regression thresholds.

## Updating and Uninstalling

To update, run `git pull` from the node directory and restart ComfyUI. To uninstall, delete the node directory. Models are stored in `ComfyUI/models/moss_transcribe_diarize` by default; you can choose whether to retain them.
