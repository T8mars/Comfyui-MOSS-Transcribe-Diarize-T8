# Third-party notices

This integration redistributes or can download components maintained by independent projects.
The final release must include the exact license texts produced by the release SBOM and license audit.

- OpenMOSS MOSS-Transcribe-Diarize code and model: Apache License 2.0.
- Electron and Chromium: their respective open-source licenses; packaged Chromium notices must be retained.
- Python: Python Software Foundation License.
- PyTorch and Torchaudio: BSD-style licenses and bundled third-party notices.
- Transformers, Hugging Face Hub, Tokenizers, Safetensors: their respective repository licenses.
- NumPy, SciPy, Librosa, Numba, PyAV, SoundFile, SoXR, FastAPI, Uvicorn and transitive dependencies:
  their respective licenses recorded in the release SBOM.
- FFmpeg/ffprobe: pinned BtbN Windows x64 `lgpl-shared` build from 2026-08-19, FFmpeg commit
  `e1e325235ee2f9f81b39d47ac2f9fe529257589e`; binary and source archive hashes are recorded in
  `manifests/ffmpeg_windows_x64.json`. The package keeps `LICENSE.txt`, matching FFmpeg source and
  BtbN build scripts under `resources/ffmpeg/source`. Video burn-in uses FFmpeg's native MPEG-4 and
  AAC encoders instead of GPL-only libx264.
- NVIDIA CUDA runtime libraries: redistribution is subject to the NVIDIA CUDA Toolkit EULA and must be audited for the exact wheel contents.

No third-party project endorses the T8star-Aix integration unless explicitly stated by that project.
