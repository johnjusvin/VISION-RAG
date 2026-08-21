# VisionRAG Paper Evaluation

This repository contains the reproducibility artifacts for the nine-video,
45-query VisionRAG pilot reported in the IEEE conference manuscript. The
official completed experiment is `paper_pilot_cuda_final_v3`: three runs of
six configurations, totaling 810 successful generations.

## Repository layout

- `scripts/evaluate_pipeline.py`: evaluation harness.
- `scripts/run_evaluation_cuda.sh`: CUDA runner with persistent logging.
- `scripts/download_videos.py`: downloads missing source videos from the
  manifest URLs without committing them to Git.
- `scripts/test_evaluate_pipeline.py`: unit tests for metrics, fusion, output
  validation, and experiment bookkeeping.
- `data/videos.csv`: video manifest, source URLs, and local filename mapping.
- `data/queries.csv`: questions, reference answers, and ground-truth intervals.
- `results/paper_pilot_cuda_final_v3/`: raw and aggregate evidence used by the
  paper.
- `docs/EVALUATION.md`: detailed execution and annotation protocol.
- `SHA256SUMS`: integrity hashes for every published artifact file.

The source videos and extracted keyframes are not redistributed here. Obtain
each video from the URL in `data/videos.csv`, confirm that your use complies
with its current terms, and save it under the corresponding `file_path`.
Machine-specific paths in the archived result copies have been replaced with
portable artifact paths; referenced `frames/` files are intentionally absent.

## Reproduce the CUDA experiment

Requirements include Linux, Python 3.12, FFmpeg, MediaInfo, an NVIDIA CUDA GPU,
and Ollama serving `llava:7b` at `http://localhost:11434`.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
ollama pull llava:7b
./scripts/download_videos.py --dry-run
./scripts/download_videos.py
./scripts/run_evaluation_cuda.sh paper_pilot_cuda_reproduction
```

The script refuses to overwrite an existing experiment or log. New results
are written to `results/<experiment_id>/`, and the complete console stream is
written to `logs/<experiment_id>.log`.

To validate the dataset and runtime without launching the full experiment:

```bash
./scripts/run_evaluation_cuda.sh preflight_check --preflight-only
```

## Evidence files

The archived completed result includes unedited generated answers, ranked
retrieval outputs, indexed chunks and ASR transcripts, prompts, per-query and
configuration metrics, timings, latency summaries, local token/call accounting,
the recorded environment, the experiment manifest, ground-truth intervals,
the aggregate report, and the final console log. `experiment_manifest.json`
records the input hashes and completion status.

Run the deterministic unit tests from the repository root with:

```bash
.venv/bin/python scripts/test_evaluate_pipeline.py
```

The blinded annotation packet and blank scoring template are included under
`results/paper_pilot_cuda_final_v3/annotation/`. The private mapping from
blinded answer IDs to configurations is deliberately excluded while the
two-annotator assessment remains incomplete. No human correctness,
groundedness, completeness, hallucination, refusal, or agreement result should
be inferred from the automated metrics.
