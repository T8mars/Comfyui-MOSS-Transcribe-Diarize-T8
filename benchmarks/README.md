# Real-audio regression benchmarks

`scripts/run_benchmarks.py` executes the real local model or an explicitly authorized SGLang Omni/vLLM endpoint. It records elapsed time, real-time factor, peak CUDA memory, segment/speaker counts, optional WER/CER against a reviewed reference transcript, case-sensitive hotword checks, quality-gate output, diagnostics, long-audio reports, and expectation failures in one JSON report. Transcript text is not copied into the report.

The repository does not redistribute meeting recordings. Copy reviewed, properly licensed audio into `benchmarks/audio/`, copy `cases.example.json`, and adjust only corpus-specific thresholds after a human review. The example matrix covers Chinese two-speaker speech, English hotwords/capitalization, multilingual switching, and noisy long-form meetings.

For a fully reproducible public corpus run, `prepare_public_benchmarks.py` downloads selected real-human-speech rows from the pinned FLEURS revision in `public_fleurs.json`, verifies every downloaded file with SHA-256, writes an attribution/provenance record, and deterministically composes a noisy 30-minute case. Generated audio, references, provenance, and reports remain Git-ignored:

```powershell
python scripts/prepare_public_benchmarks.py --long-minutes 30
python scripts/run_benchmarks.py benchmarks/generated/public-real/cases.public-real.json `
  --model D:\path\to\MOSS-Transcribe-Diarize `
  --comfyui-root D:\path\to\ComfyUI `
  --word-alignment-model openai/whisper-small `
  --speaker-embedding-model microsoft/wavlm-base-plus-sv `
  --output benchmarks/results/public-real.json `
  --fail-on-regression
```

Deterministic silence, noise, music-like, and long non-speech guardrails can be generated locally:

```powershell
python scripts/generate_benchmark_fixtures.py
python scripts/run_benchmarks.py benchmarks/generated/cases.synthetic.json `
  --model D:\path\to\MOSS-Transcribe-Diarize `
  --output benchmarks/results/local.json `
  --fail-on-regression
```

When the node checkout is not under `ComfyUI/custom_nodes`, add `--comfyui-root D:\path\to\ComfyUI`. Always run the command with the same Python environment as ComfyUI.

Remote execution requires explicit consent. The optional Bearer token is read only from `MOSS_TRANSCRIBE_API_KEY`:

```powershell
python scripts/run_benchmarks.py benchmarks/cases.local.json `
  --endpoint http://127.0.0.1:8000 `
  --allow-remote-upload `
  --output benchmarks/results/remote.json `
  --fail-on-regression
```

Each case may provide `reference_text` or a UTF-8 `reference_text_file`, then gate `max_word_error_rate` and/or `max_character_error_rate`. Use `required_text` for spelling/capitalization-sensitive hotwords and `min_speakers` for diarization structure. Auxiliary cases can additionally gate `min_word_alignment_coverage`, `min_speaker_embedding_links`, and `max_speaker_embedding_failures`. Do not commit private recordings, reference transcripts, API keys, downloaded corpus files, or generated reports. Performance thresholds should be separated by GPU and backend; exact wording and consequential speaker identities still require human review.
