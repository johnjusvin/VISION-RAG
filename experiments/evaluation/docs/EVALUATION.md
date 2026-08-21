# VisionRAG nine-video evaluation

`evaluate_pipeline.py` runs the fixed 9-video, 45-question pilot corpus and
writes raw, auditable evidence for the paper. Existing aggregate reports are
legacy artifacts and are not used as inputs.

## Prerequisites

Install `requirements.txt`, FFmpeg/FFprobe, and a compatible PyTorch
build. Start Ollama and pull the selected vision model.

Validate the dataset without loading any models:

```bash
.venv/bin/python scripts/evaluate_pipeline.py --validate-only
```

Run the complete six-configuration, three-repetition experiment:

```bash
.venv/bin/python scripts/evaluate_pipeline.py \
  --generator-model llava:7b \
  --embedder local-clip \
  --runs 3 \
  --compute-bertscore
```

For the CUDA paper run, use the checked launcher from the repository root. It
performs dataset/runtime preflight checks, supplies the CUDA 12 library paths,
refuses to overwrite evidence, and stores the full console log:

```bash
./scripts/run_evaluation_cuda.sh paper_pilot_cuda_reproduction
```

Follow progress from another terminal with:

```bash
tail -f logs/paper_pilot_cuda_reproduction.log
```

On Linux, Faster-Whisper/CTranslate2 requires the CUDA 12 cuBLAS and cuDNN
libraries listed in `requirements.txt`. Expose the virtual environment's
copies before starting a CUDA run:

```bash
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:$PWD/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Adjust `python3.12` if the virtual environment uses a different Python minor
version. Without this setting, CTranslate2 reports that `libcublas.so.12`
cannot be loaded.

On GPUs with approximately 8 GB VRAM, the evaluator automatically shares one
CLIP model instance and stages `whisper-medium` separately from Ollama. The
Ollama runner is unloaded before medium-ASR processing, and the ASR model is
released before embedding and generation. This prevents CUDA OOM without
falling back to CPU or changing the evaluated model.

To check the complete runtime without creating an experiment, add
`--preflight-only` to the Python command. `--validate-only` checks only the
CSV dataset and deliberately does not load CUDA or Ollama.

For a retrieval-only smoke test:

```bash
.venv/bin/python scripts/evaluate_pipeline.py \
  --skip-generation \
  --runs 1 \
  --configs full_rrf_chrono
```

Use a new `--experiment-id` for every execution. The runner refuses to
overwrite an existing experiment directory. Use a new `--frames-cache-dir`
when a guaranteed cold-frame-cache measurement is required; subsequent runs
against that same directory are recorded as warm-cache measurements.

## Outputs

Each run is written to `results/<experiment_id>/` and includes:

- normalized input copies, hashes, environment, configuration, and prompt;
- all indexed chunks and ASR transcripts;
- text, image, and final retrieval rankings with raw and RRF scores;
- exact unedited answers and the evidence supplied to Ollama;
- per-query metrics, raw stage timings, API usage, and failures;
- per-run/across-run configuration summaries plus mean, median, standard
  deviation, and p95 latency summaries;
- a blinded annotation packet and a private answer/configuration key.

`Token F1` and model-based `BERTScore F1` remain separate columns. Transient
Ollama failures are retried with the exact same prompt and evidence; the model
runner is reset between attempts. Any final failure is retained, marks the run
incomplete by default, and is never replaced with altered evidence.

## Human annotation

After generation, give `annotation/annotation_packet.csv` to at least two
independent annotators. Do not give them `private/annotation_key.csv`. Each
annotator completes one row per `answer_id` in
`annotation/answer_annotations.csv`. Human measurements remain pending until
those rows exist; the runner never creates ratings itself.

## Scope

This is a nine-video pilot, not an external benchmark. The six internal
configurations cover modality, RRF/raw-score fusion, chronological ordering,
and Whisper model-size ablations. An external Video RAG baseline is not
included because no specific runnable baseline has been selected.
