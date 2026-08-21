#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing evaluation Python interpreter: $PYTHON_BIN" >&2
  echo "Create .venv and install requirements.txt first." >&2
  exit 2
fi

EXPERIMENT_ID="${1:-paper_pilot_cuda_$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ $# -gt 0 ]]; then
  shift
fi

OUTPUT_DIR="$PROJECT_ROOT/results/$EXPERIMENT_ID"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/$EXPERIMENT_ID.log"
if [[ -e "$OUTPUT_DIR" || -e "$LOG_FILE" ]]; then
  echo "Refusing to overwrite an existing run or log for: $EXPERIMENT_ID" >&2
  exit 2
fi

SITE_PACKAGES="$($PYTHON_BIN -c 'import site; print(site.getsitepackages()[0])')"
CUBLAS_DIR="$SITE_PACKAGES/nvidia/cublas/lib"
CUDNN_DIR="$SITE_PACKAGES/nvidia/cudnn/lib"
for directory in "$CUBLAS_DIR" "$CUDNN_DIR"; do
  if [[ ! -d "$directory" ]]; then
    echo "Missing CUDA runtime directory: $directory" >&2
    echo "Install requirements.txt before running the evaluation." >&2
    exit 2
  fi
done

mkdir -p "$LOG_DIR" "/tmp/visionrag-mpl-cache-$UID"
export LD_LIBRARY_PATH="$CUBLAS_DIR:$CUDNN_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NO_PROXY="localhost,127.0.0.1${NO_PROXY:+,$NO_PROXY}"
export no_proxy="localhost,127.0.0.1${no_proxy:+,$no_proxy}"
export MPLCONFIGDIR="/tmp/visionrag-mpl-cache-$UID"
export PYTHONUNBUFFERED=1

COMMON_ARGS=(
  --generator-model llava:7b
  --embedder local-clip
  --runs 3
  --compute-bertscore
  --asr-device cuda
  --embedding-device cuda
  --bertscore-device cuda
  --bertscore-model roberta-large
  --bertscore-batch-size 8
  --generation-retries 3
  --generation-retry-delay 2
)

cd "$PROJECT_ROOT"
{
  echo "VisionRAG CUDA evaluation: $EXPERIMENT_ID"
  echo "Log: $LOG_FILE"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  "$PYTHON_BIN" "$SCRIPT_DIR/evaluate_pipeline.py" "${COMMON_ARGS[@]}" --preflight-only
  "$PYTHON_BIN" -u "$SCRIPT_DIR/evaluate_pipeline.py" \
    "${COMMON_ARGS[@]}" \
    --videos "$PROJECT_ROOT/data/videos.csv" \
    --queries "$PROJECT_ROOT/data/queries.csv" \
    --output-dir "$PROJECT_ROOT/results" \
    --frames-cache-dir "$PROJECT_ROOT/.cache/evaluation_frames" \
    --experiment-id "$EXPERIMENT_ID" \
    "$@"
} 2>&1 | tee "$LOG_FILE"
