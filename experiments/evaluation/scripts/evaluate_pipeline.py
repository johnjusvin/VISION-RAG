#!/usr/bin/env python3
"""Reproducible nine-video evaluation runner for the VisionRAG paper.

The runner writes raw evidence before aggregate summaries and never fabricates
chunks, answers, annotations, timings, costs, or failures. The default matrix
contains the paper's internal ablations and expects 9 videos / 45 questions.

Full experiment from the evaluation repository root::

    python scripts/evaluate_pipeline.py --generator-model llava:7b --runs 3

Retrieval-only smoke test (not complete paper evidence)::

    python scripts/evaluate_pipeline.py --skip-generation --runs 1 \
        --configs full_rrf_chrono

Input validation only::

    python scripts/evaluate_pipeline.py --validate-only

Each execution creates a new, non-overwriting directory under ``results`` and
records output hashes.
The human annotation CSV is header-only until independent annotators complete
the blinded packet.
"""

from __future__ import annotations

import argparse
import ctypes
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTING_DIR = Path(__file__).resolve().parent
# Support either an installed ``vision-rag`` package or a neighboring/local
# source checkout while keeping this artifact repository self-contained.
LOCAL_VISION_RAG_DIRS = (
    PROJECT_ROOT / "VISION-RAG",
    PROJECT_ROOT.parent / "VISION-RAG",
)
for import_path in (*LOCAL_VISION_RAG_DIRS, TESTING_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from vision_rag.embedding import (  # noqa: E402
    BaseImageEmbedder,
    BaseTextEmbedder,
    EmbeddedChunk,
    EmbeddingBuilder,
)
from vision_rag.generator import BaseGenerator  # noqa: E402
from vision_rag.vectorstores import FAISS, SearchResult  # noqa: E402
from vision_rag.video_chunker import Chunk, Chunker, WhisperLocalASR  # noqa: E402
from vision_rag.video_ingestion import VideoLoader  # noqa: E402

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(TESTING_DIR / ".env")
except ImportError:
    pass


SCHEMA_VERSION = "visionrag-evaluation-v1"
QUESTION_TYPES = {"transcript_dependent", "visual_dependent", "multimodal"}


class EvaluationConfigError(RuntimeError):
    """Raised when a run cannot produce defensible evidence."""


@dataclass(frozen=True)
class ExperimentConfiguration:
    config_id: str
    use_text: bool
    use_image: bool
    fusion: str
    chronological: bool
    whisper_model: str
    purpose: str


CONFIGURATIONS: dict[str, ExperimentConfiguration] = {
    "full_rrf_chrono": ExperimentConfiguration(
        "full_rrf_chrono", True, True, "rrf", True, "base",
        "Main text+image pipeline with RRF and chronological generation.",
    ),
    "text_only": ExperimentConfiguration(
        "text_only", True, False, "single", True, "base",
        "Transcript-modality retrieval baseline.",
    ),
    "image_only": ExperimentConfiguration(
        "image_only", False, True, "single", True, "base",
        "Keyframe-modality retrieval baseline.",
    ),
    "raw_score_fusion": ExperimentConfiguration(
        "raw_score_fusion", True, True, "raw_score", True, "base",
        "Naive cross-modal raw-score sorting ablation.",
    ),
    "rrf_relevance_order": ExperimentConfiguration(
        "rrf_relevance_order", True, True, "rrf", False, "base",
        "RRF retrieval with relevance-ordered generation context.",
    ),
    "whisper_medium": ExperimentConfiguration(
        "whisper_medium", True, True, "rrf", True, "medium",
        "Whisper-medium ASR granularity and latency ablation.",
    ),
}


RETRIEVAL_FIELDS = [
    "experiment_id", "run_id", "config_id", "query_id", "video_id",
    "result_list", "rank", "final_rank", "chunk_id",
    "chunk_start_seconds", "chunk_end_seconds", "modality", "raw_score",
    "text_rank", "image_rank", "text_raw_score", "image_raw_score",
    "rrf_score", "retrieved_text", "frame_path", "is_relevant",
    "matched_intervals",
]
CHUNK_FIELDS = [
    "experiment_id", "run_id", "config_id", "video_id", "chunk_id",
    "start_seconds", "end_seconds", "duration_seconds", "transcript",
    "frame_path", "text_vector_dimension", "image_vector_dimension",
    "asr_provider", "asr_model", "frame_cache_state",
]
ANSWER_FIELDS = [
    "experiment_id", "run_id", "config_id", "query_id", "video_id",
    "generated_answer", "reference_answer", "source_chunk_ids",
    "source_timestamps", "source_evidence", "generator_provider",
    "generator_model", "prompt_version", "temperature", "seed",
    "generation_error", "bleu1", "token_f1", "bertscore_f1",
]
QUERY_METRIC_FIELDS = [
    "experiment_id", "run_id", "config_id", "query_id", "video_id",
    "domain", "question_type", "recall_at_1", "recall_at_5",
    "recall_at_10", "mrr", "retrieval_success", "generation_success",
    "bleu1", "token_f1", "bertscore_f1", "error",
]
TIMING_FIELDS = [
    "experiment_id", "run_id", "config_id", "query_id", "video_id",
    "scope", "ingestion_ms", "audio_extraction_ms", "asr_ms",
    "frame_extraction_ms", "text_embedding_ms", "image_embedding_ms",
    "indexing_ms", "query_embedding_ms", "retrieval_ms", "generation_ms",
    "total_pipeline_ms", "cache_state", "success", "error",
]
ERROR_FIELDS = [
    "experiment_id", "run_id", "config_id", "query_id", "video_id",
    "failure_type", "description", "expected_behavior",
    "retrieved_evidence", "suspected_cause",
]
COST_FIELDS = [
    "experiment_id", "run_id", "config_id", "query_id", "video_id",
    "provider", "model", "operation", "input_tokens", "output_tokens",
    "image_count", "api_calls", "reported_cost_usd", "pricing_date",
]
CONFIG_RESULT_FIELDS = [
    "experiment_id", "run_id", "config_id", "stratum_type", "stratum_value",
    "queries", "retrieval_failures",
    "generation_attempts", "generation_failures", "recall_at_1",
    "recall_at_5", "recall_at_10", "mrr", "bleu1", "token_f1",
    "bertscore_f1",
]
LATENCY_RESULT_FIELDS = [
    "experiment_id", "run_id", "config_id", "scope", "cache_state",
    "metric", "unit", "observations", "failures", "mean", "median",
    "standard_deviation", "p95",
]
ANNOTATION_FIELDS = [
    "annotator_id", "answer_id", "correctness", "groundedness",
    "completeness", "relevance", "hallucination", "refusal", "notes",
]


class CsvSink:
    """Append-only CSV writer that flushes each row for crash visibility."""

    def __init__(self, path: Path, fieldnames: Sequence[str]):
        self.fieldnames = list(fieldnames)
        self._handle = path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.fieldnames)
        self._writer.writeheader()
        self._handle.flush()

    def write(self, row: dict[str, Any]) -> None:
        self._writer.writerow({key: _csv_value(row.get(key, "")) for key in self.fieldnames})
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


class TimedTextEmbedder(BaseTextEmbedder):
    def __init__(self, wrapped: BaseTextEmbedder):
        self.wrapped = wrapped
        self.elapsed_ms = 0.0
        self.calls = 0

    @property
    def provider_name(self) -> str:
        return self.wrapped.provider_name

    def embed(self, text: str) -> list[float]:
        started = time.perf_counter()
        result = self.wrapped.embed(text)
        self.elapsed_ms += _elapsed_ms(started)
        self.calls += 1
        return result


class TimedImageEmbedder(BaseImageEmbedder):
    def __init__(self, wrapped: BaseImageEmbedder):
        self.wrapped = wrapped
        self.elapsed_ms = 0.0
        self.calls = 0

    @property
    def provider_name(self) -> str:
        return self.wrapped.provider_name

    def embed(self, image_path: str) -> list[float]:
        started = time.perf_counter()
        result = self.wrapped.embed(image_path)
        self.elapsed_ms += _elapsed_ms(started)
        self.calls += 1
        return result


def _is_retryable_generation_error(exc: Exception) -> bool:
    """Return whether a local-provider failure may succeed on a later attempt."""
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        try:
            return int(status_code) >= 500
        except (TypeError, ValueError):
            return False
    return isinstance(exc, (ConnectionError, TimeoutError)) or exc.__class__.__name__ in {
        "ConnectError", "ConnectTimeout", "ReadError", "ReadTimeout",
    }


class EvaluationOllamaGenerator(BaseGenerator):
    """Ollama generator with explicit, recorded generation controls."""

    VISION_MARKERS = (
        "llava", "bakllava", "moondream", "minicpm-v", "qwen2-vl",
        "qwen2.5vl", "qwen2.5-vl", "gemma3", "llama3.2-vision",
    )

    def __init__(
        self,
        model: str,
        host: str,
        temperature: float,
        seed: int,
        max_tokens: int,
        retries: int,
        retry_delay: float,
    ):
        self.model = model
        self.host = host
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.retries = retries
        self.retry_delay = retry_delay
        self._is_vision = any(marker in model.lower() for marker in self.VISION_MARKERS)
        self.last_usage: dict[str, Any] = {}

    def generate(self, query: str, chunks: list[EmbeddedChunk]) -> str:
        try:
            import ollama
        except ImportError as exc:
            raise RuntimeError("Install Ollama's Python client with: pip install ollama") from exc

        context_parts = []
        for index, chunk in enumerate(chunks, start=1):
            if chunk.text:
                context_parts.append(
                    f"[Chunk {index} | {chunk.start:.1f}s – {chunk.end:.1f}s]\n{chunk.text}"
                )
        prompt = (
            f"{self._build_system_prompt()}\n\nContext:\n"
            f"{'\n\n'.join(context_parts)}\n\nQuestion: {query}"
        )
        images: list[str] = []
        if self._is_vision:
            for chunk in chunks:
                if chunk.frame_path and Path(chunk.frame_path).exists():
                    encoded = self._read_image_b64(chunk.frame_path)
                    if encoded:
                        images.append(encoded)
        message: dict[str, Any] = {"role": "user", "content": prompt}
        if images:
            message["images"] = images
        self.last_usage = {"image_count": len(images), "api_calls": 1}
        client = ollama.Client(host=self.host)
        attempts = 0
        reset_calls = 0
        while True:
            attempts += 1
            try:
                response = client.chat(
                    model=self.model,
                    messages=[message],
                    options={
                        "temperature": self.temperature,
                        "seed": self.seed,
                        "num_predict": self.max_tokens,
                    },
                )
                break
            except Exception as exc:
                if attempts > self.retries or not _is_retryable_generation_error(exc):
                    self.last_usage["api_calls"] = attempts + reset_calls
                    raise
                delay = self.retry_delay * (2 ** (attempts - 1))
                print(
                    f"[Ollama] transient generation failure; retry "
                    f"{attempts}/{self.retries} in {delay:.1f}s: {exc!r}",
                    flush=True,
                )
                # Ollama/llama.cpp can leave a multimodal runner in a bad KV-cache
                # state (observed as a repeatable "Chunk not found" 500). Force a
                # clean unload before retrying the exact same evidence payload.
                try:
                    client.generate(model=self.model, keep_alive=0)
                    reset_calls += 1
                except Exception as reset_exc:
                    print(f"[Ollama] model reset failed: {reset_exc!r}", flush=True)
                time.sleep(delay)
        input_tokens = (
            response.get("prompt_eval_count", "")
            if isinstance(response, dict)
            else getattr(response, "prompt_eval_count", "")
        )
        output_tokens = (
            response.get("eval_count", "")
            if isinstance(response, dict)
            else getattr(response, "eval_count", "")
        )
        self.last_usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "image_count": len(images),
            "api_calls": attempts + reset_calls,
        }
        if isinstance(response, dict):
            content = response["message"]["content"]
        else:
            content = response.message.content
        return content.strip()


def _csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(result: SearchResult) -> tuple[str, int]:
    return result.chunk.video_path, result.chunk.chunk_id


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def compute_bleu1(reference: str, hypothesis: str) -> float:
    """Clipped unigram precision with the standard BLEU brevity penalty."""
    reference_tokens = _tokenize(reference)
    hypothesis_tokens = _tokenize(hypothesis)
    if not hypothesis_tokens:
        return 0.0
    reference_counts = Counter(reference_tokens)
    hypothesis_counts = Counter(hypothesis_tokens)
    matches = sum(
        min(count, reference_counts[token])
        for token, count in hypothesis_counts.items()
    )
    precision = matches / len(hypothesis_tokens)
    reference_length, candidate_length = len(reference_tokens), len(hypothesis_tokens)
    brevity_penalty = (
        1.0
        if candidate_length > reference_length
        else math.exp(1.0 - (reference_length / candidate_length))
    )
    return precision * brevity_penalty


def compute_token_f1(reference: str, hypothesis: str) -> float:
    """SQuAD-style bag-of-token overlap F1; this is not BERTScore."""
    reference_counts = Counter(_tokenize(reference))
    hypothesis_counts = Counter(_tokenize(hypothesis))
    if not reference_counts or not hypothesis_counts:
        return 0.0
    overlap = sum((reference_counts & hypothesis_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(hypothesis_counts.values())
    recall = overlap / sum(reference_counts.values())
    return 2.0 * precision * recall / (precision + recall)


def interval_overlap(
    chunk_start: float,
    chunk_end: float,
    interval_start: float,
    interval_end: float,
) -> float:
    return max(0.0, min(chunk_end, interval_end) - max(chunk_start, interval_start))


def matching_intervals(
    chunk_start: float,
    chunk_end: float,
    intervals: Sequence[tuple[float, float]],
    minimum_overlap: float,
) -> list[tuple[float, float]]:
    return [
        (start, end)
        for start, end in intervals
        if interval_overlap(chunk_start, chunk_end, start, end) >= minimum_overlap
    ]


def reciprocal_rank_fusion(
    text_results: Sequence[SearchResult],
    image_results: Sequence[SearchResult],
    rrf_k: int,
) -> list[dict[str, Any]]:
    """Fuse rankings by RRF while retaining per-modality evidence."""
    fused: dict[tuple[str, int], dict[str, Any]] = {}
    for modality, ranked_results in (("text", text_results), ("image", image_results)):
        for rank, result in enumerate(ranked_results, start=1):
            key = _identity(result)
            entry = fused.setdefault(
                key,
                {
                    "chunk": result.chunk,
                    "rrf_score": 0.0,
                    "text_rank": None,
                    "image_rank": None,
                    "text_raw_score": None,
                    "image_raw_score": None,
                    "best_raw_score": result.score,
                },
            )
            entry["rrf_score"] += 1.0 / (rrf_k + rank)
            entry[f"{modality}_rank"] = rank
            entry[f"{modality}_raw_score"] = result.score
            entry["best_raw_score"] = max(entry["best_raw_score"], result.score)
    return sorted(
        fused.values(),
        key=lambda item: (
            item["rrf_score"], item["best_raw_score"], -item["chunk"].chunk_id
        ),
        reverse=True,
    )


def raw_score_fusion(
    text_results: Sequence[SearchResult],
    image_results: Sequence[SearchResult],
) -> list[dict[str, Any]]:
    """Deliberately naive raw-score baseline with deterministic deduplication."""
    entries: dict[tuple[str, int], dict[str, Any]] = {}
    for modality, ranked_results in (("text", text_results), ("image", image_results)):
        for rank, result in enumerate(ranked_results, start=1):
            key = _identity(result)
            entry = entries.setdefault(
                key,
                {
                    "chunk": result.chunk,
                    "rrf_score": None,
                    "text_rank": None,
                    "image_rank": None,
                    "text_raw_score": None,
                    "image_raw_score": None,
                    "best_raw_score": result.score,
                },
            )
            entry[f"{modality}_rank"] = rank
            entry[f"{modality}_raw_score"] = result.score
            entry["best_raw_score"] = max(entry["best_raw_score"], result.score)
    return sorted(
        entries.values(),
        key=lambda item: (item["best_raw_score"], -item["chunk"].chunk_id),
        reverse=True,
    )


def single_modality_ranking(
    results: Sequence[SearchResult], modality: str
) -> list[dict[str, Any]]:
    return [
        {
            "chunk": result.chunk,
            "rrf_score": None,
            "text_rank": rank if modality == "text" else None,
            "image_rank": rank if modality == "image" else None,
            "text_raw_score": result.score if modality == "text" else None,
            "image_raw_score": result.score if modality == "image" else None,
            "best_raw_score": result.score,
        }
        for rank, result in enumerate(results, start=1)
    ]


def _read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.exists():
        raise EvaluationConfigError(f"Required CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = required - columns
        if missing:
            raise EvaluationConfigError(f"{path} is missing columns: {sorted(missing)}")
        return list(reader)


def _nonnegative_float(value: str, field: str, item_id: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise EvaluationConfigError(
            f"{item_id}: {field} must be numeric, got {value!r}."
        ) from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise EvaluationConfigError(f"{item_id}: {field} must be finite and >= 0.")
    return parsed


def _positive_float(value: str, field: str, item_id: str) -> float:
    parsed = _nonnegative_float(value, field, item_id)
    if parsed <= 0:
        raise EvaluationConfigError(f"{item_id}: {field} must be > 0.")
    return parsed


def load_dataset(
    videos_path: Path,
    queries_path: Path,
    intervals_path: Optional[Path],
    expected_videos: int,
    expected_queries: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    video_required = {
        "video_id", "file_path", "title", "domain", "duration_seconds",
        "language", "source_url", "license", "has_audio",
        "has_burned_captions", "notes",
    }
    query_required = {
        "query_id", "video_id", "question", "question_type", "gt_start",
        "gt_end", "reference_answer", "annotator_id",
    }
    videos_rows = _read_csv(videos_path, video_required)
    queries = _read_csv(queries_path, query_required)
    if len(videos_rows) != expected_videos:
        raise EvaluationConfigError(
            f"Expected {expected_videos} videos but found {len(videos_rows)} in {videos_path}."
        )
    if len(queries) != expected_queries:
        raise EvaluationConfigError(
            f"Expected {expected_queries} queries but found {len(queries)} in {queries_path}."
        )

    videos: dict[str, dict[str, Any]] = {}
    for row in videos_rows:
        video_id = row["video_id"].strip()
        if not video_id or video_id in videos:
            raise EvaluationConfigError(f"Missing or duplicate video_id: {video_id!r}")
        row["duration_seconds"] = _positive_float(
            row["duration_seconds"], "duration_seconds", video_id
        )
        for field in ("file_path", "title", "domain", "language"):
            if not row[field].strip():
                raise EvaluationConfigError(f"{video_id}: {field} must not be blank.")
        for field in ("has_audio", "has_burned_captions"):
            value = row[field].strip().lower()
            if value not in {"true", "false"}:
                raise EvaluationConfigError(
                    f"{video_id}: {field} must be true or false, got {row[field]!r}."
                )
        candidate = Path(row["file_path"])
        row["resolved_path"] = str(
            candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
        )
        videos[video_id] = row

    extra_intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    if intervals_path and intervals_path.exists():
        rows = _read_csv(intervals_path, {"query_id", "start_seconds", "end_seconds"})
        for row in rows:
            extra_intervals[row["query_id"]].append(
                (
                    _nonnegative_float(
                        row["start_seconds"], "start_seconds", row["query_id"]
                    ),
                    _positive_float(row["end_seconds"], "end_seconds", row["query_id"]),
                )
            )

    seen_query_ids: set[str] = set()
    for row in queries:
        query_id = row["query_id"].strip()
        if not query_id or query_id in seen_query_ids:
            raise EvaluationConfigError(f"Missing or duplicate query_id: {query_id!r}")
        seen_query_ids.add(query_id)
        video_id = row["video_id"].strip()
        if video_id not in videos:
            raise EvaluationConfigError(
                f"Query {query_id} references unknown video {video_id!r}."
            )
        if row["question_type"] not in QUESTION_TYPES:
            raise EvaluationConfigError(
                f"Query {query_id} has unsupported question_type {row['question_type']!r}."
            )
        for field in ("question", "reference_answer", "annotator_id"):
            if not row[field].strip():
                raise EvaluationConfigError(f"Query {query_id}: {field} must not be blank.")
        intervals = [
            (
                _nonnegative_float(row["gt_start"], "gt_start", query_id),
                _positive_float(row["gt_end"], "gt_end", query_id),
            )
        ]
        alternate_start = (row.get("alternate_start_seconds") or "").strip()
        alternate_end = (row.get("alternate_end_seconds") or "").strip()
        if alternate_start or alternate_end:
            if not alternate_start or not alternate_end:
                raise EvaluationConfigError(
                    f"Query {query_id} has an incomplete alternate interval."
                )
            intervals.append(
                (
                    _nonnegative_float(
                        alternate_start, "alternate_start_seconds", query_id
                    ),
                    _positive_float(alternate_end, "alternate_end_seconds", query_id),
                )
            )
        intervals.extend(extra_intervals.get(query_id, []))
        duration = videos[video_id]["duration_seconds"]
        for start, end in intervals:
            if start >= end or end > duration + 1.0:
                raise EvaluationConfigError(
                    f"Query {query_id} has invalid interval [{start}, {end}] "
                    f"for a {duration}s video."
                )
        row["intervals"] = intervals
    unknown_query_ids = set(extra_intervals) - seen_query_ids
    if unknown_query_ids:
        raise EvaluationConfigError(
            f"relevant_intervals.csv has unknown query IDs: {sorted(unknown_query_ids)}"
        )
    observed_types = {row["question_type"] for row in queries}
    if observed_types != QUESTION_TYPES:
        raise EvaluationConfigError(
            f"Dataset must cover all question types; found {sorted(observed_types)}."
        )
    if expected_videos and expected_queries % expected_videos == 0:
        expected_per_video = expected_queries // expected_videos
        counts = Counter(row["video_id"] for row in queries)
        unbalanced = {
            video_id: counts.get(video_id, 0)
            for video_id in videos
            if counts.get(video_id, 0) != expected_per_video
        }
        if unbalanced:
            raise EvaluationConfigError(
                f"Expected {expected_per_video} queries per video; found {unbalanced}."
            )
    return videos, queries


def select_configurations(value: str) -> list[ExperimentConfiguration]:
    names = (
        list(CONFIGURATIONS)
        if value.strip().lower() == "all"
        else [part.strip() for part in value.split(",") if part.strip()]
    )
    unknown = [name for name in names if name not in CONFIGURATIONS]
    if unknown:
        raise EvaluationConfigError(
            f"Unknown configurations {unknown}. Available: {list(CONFIGURATIONS)}"
        )
    if not names:
        raise EvaluationConfigError("At least one configuration is required.")
    if len(names) != len(set(names)):
        raise EvaluationConfigError("Configuration IDs must not be repeated.")
    return [CONFIGURATIONS[name] for name in names]


def setup_embedders(
    name: str, device: str
) -> tuple[BaseTextEmbedder, BaseImageEmbedder, dict[str, Any]]:
    if name == "local-clip":
        try:
            from modular_example import LocalCLIPImageEmbedder, LocalCLIPTextEmbedder

            selected_device = None if device == "auto" else device
            text_embedder = LocalCLIPTextEmbedder(device=selected_device)
            image_embedder = LocalCLIPImageEmbedder(device=selected_device)
            # Both wrappers use the same multimodal SentenceTransformer. Sharing
            # the loaded instance avoids keeping two identical CLIP weight sets
            # in VRAM throughout the multi-hour evaluation.
            if hasattr(text_embedder, "model") and hasattr(image_embedder, "model"):
                previous_image_model = image_embedder.model
                image_embedder.model = text_embedder.model
                del previous_image_model
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
        except Exception as exc:
            raise EvaluationConfigError(f"Failed to initialize local CLIP: {exc!r}") from exc
        return text_embedder, image_embedder, {
            "provider": "local-clip",
            "model": getattr(text_embedder, "model_name", "clip-ViT-B-32"),
            "device": getattr(text_embedder, "device", device),
            "joint_space": True,
        }
    if name == "jina-v4":
        api_key = os.environ.get("JINA_API_KEY")
        if not api_key:
            raise EvaluationConfigError("JINA_API_KEY is required for --embedder jina-v4.")
        from modular_example import JinaV4ImageEmbedder, JinaV4TextEmbedder

        return (
            JinaV4TextEmbedder(api_key),
            JinaV4ImageEmbedder(api_key),
            {"provider": "jina", "model": "jina-embeddings-v4", "joint_space": True},
        )
    raise EvaluationConfigError(f"Unsupported embedder: {name}")


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def release_asr_model(asr: Optional[WhisperLocalASR]) -> None:
    """Release a lazily loaded Whisper backend and return its CUDA memory."""
    if asr is None or getattr(asr, "_model", None) is None:
        return
    model = asr._model
    asr._model = None
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def prepare_chunks(
    video_path: Path,
    frames_cache_dir: Path,
    asr: Optional[WhisperLocalASR],
    use_asr: bool,
    use_frames: bool,
    chunk_size: float,
    chunk_overlap: float,
    keyframe_strategy: str,
) -> tuple[list[Chunk], dict[str, Any]]:
    """Execute Chunker's operations while timing media stages separately."""
    ingestion_started = time.perf_counter()
    VideoLoader().load(str(video_path))
    ingestion_ms = _elapsed_ms(ingestion_started)

    whisper_model = asr.model_size if asr else None
    chunker = Chunker(
        asr=asr,
        use_asr=use_asr,
        use_frames=use_frames,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        keyframe_strategy=keyframe_strategy,
        frames_dir=str(frames_cache_dir / video_path.stem),
    )
    duration = chunker._get_duration(video_path)
    windows = chunker._build_windows(duration)
    output_dir = Path(chunker.frames_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_frames = (
        [
            output_dir
            / f"chunk_{index:04d}_{chunker._frame_cache_key(video_path, start, end)}.jpg"
            for index, (start, end) in enumerate(windows)
        ]
        if use_frames
        else []
    )
    cached_before = sum(path.exists() for path in expected_frames)
    cache_state = (
        "not_applicable"
        if not use_frames
        else "cold"
        if cached_before == 0
        else "warm"
        if cached_before == len(expected_frames)
        else "mixed"
    )

    audio_extraction_ms = 0.0
    asr_ms = 0.0
    segments: list[dict] = []
    if use_asr:
        audio_path: Optional[str] = None
        try:
            audio_started = time.perf_counter()
            audio_path = chunker._extract_audio(video_path)
            audio_extraction_ms = _elapsed_ms(audio_started)
            asr_started = time.perf_counter()
            segments = asr.transcribe(audio_path)  # type: ignore[union-attr]
            asr_ms = _elapsed_ms(asr_started)
        finally:
            if audio_path and Path(audio_path).exists():
                Path(audio_path).unlink()

    frame_extraction_ms = 0.0
    frame_paths: list[Optional[Path]] = [None] * len(windows)
    if use_frames:
        frame_started = time.perf_counter()
        frame_paths = [
            chunker._extract_keyframe(video_path, start, end, output_dir, index)
            for index, (start, end) in enumerate(windows)
        ]
        frame_extraction_ms = _elapsed_ms(frame_started)
    chunks = [
        Chunk(
            chunk_id=index,
            video_path=str(video_path),
            start=start,
            end=end,
            duration=round(end - start, 4),
            frame_path=str(frame_path) if frame_path else None,
            text=(chunker._slice_transcript(segments, start, end) if use_asr else None),
            metadata={
                "video_filename": video_path.name,
                "chunk_index": index,
                "total_chunks": len(windows),
                "keyframe_strategy": keyframe_strategy,
                "asr_provider": asr.provider_name if asr else None,
                "asr_model": whisper_model,
            },
        )
        for index, ((start, end), frame_path) in enumerate(zip(windows, frame_paths))
    ]
    if use_asr and not any(chunk.text for chunk in chunks):
        raise RuntimeError(f"ASR returned no transcript text for {video_path}.")
    if use_frames and not any(chunk.frame_path for chunk in chunks):
        raise RuntimeError(f"Frame extraction returned no frames for {video_path}.")
    return chunks, {
        "ingestion_ms": ingestion_ms,
        "audio_extraction_ms": audio_extraction_ms,
        "asr_ms": asr_ms,
        "frame_extraction_ms": frame_extraction_ms,
        "cache_state": cache_state,
    }


def embed_chunks(
    chunks: list[Chunk],
    text_embedder: BaseTextEmbedder,
    image_embedder: BaseImageEmbedder,
    use_text: bool = True,
    use_image: bool = True,
) -> tuple[list[EmbeddedChunk], dict[str, Any]]:
    timed_text = TimedTextEmbedder(text_embedder)
    timed_image = TimedImageEmbedder(image_embedder)
    embedded = EmbeddingBuilder(
        text_embedding=timed_text if use_text else None,
        image_embedding=timed_image if use_image else None,
    ).embed(chunks)
    return embedded, {
        "text_embedding_ms": timed_text.elapsed_ms,
        "image_embedding_ms": timed_image.elapsed_ms,
        "text_embedding_calls": timed_text.calls,
        "image_embedding_calls": timed_image.calls,
    }


def chunks_for_configuration(
    chunks: Sequence[EmbeddedChunk], config: ExperimentConfiguration
) -> list[EmbeddedChunk]:
    return [
        replace(
            chunk,
            text_vector=chunk.text_vector if config.use_text else None,
            image_vector=chunk.image_vector if config.use_image else None,
            metadata=chunk.metadata.copy(),
        )
        for chunk in chunks
    ]


def generation_sources(
    final_results: Sequence[dict[str, Any]],
    config: ExperimentConfiguration,
    top_k: int,
) -> list[EmbeddedChunk]:
    """Apply modality ablations to the evidence actually sent to the model."""
    sources = []
    for item in final_results[:top_k]:
        chunk = item["chunk"]
        sources.append(
            replace(
                chunk,
                text=chunk.text if config.use_text else None,
                frame_path=chunk.frame_path if config.use_image else None,
                text_vector=None,
                image_vector=None,
                metadata=chunk.metadata.copy(),
            )
        )
    if config.chronological:
        sources.sort(key=lambda chunk: chunk.start)
    return sources


def search_configuration(
    store: FAISS,
    query_vector: list[float],
    config: ExperimentConfiguration,
    top_k_per_modality: int,
    rrf_k: int,
) -> tuple[list[SearchResult], list[SearchResult], list[dict[str, Any]]]:
    text_results = (
        store.search_text(query_vector, top_k_per_modality) if config.use_text else []
    )
    image_results = (
        store.search_image(query_vector, top_k_per_modality) if config.use_image else []
    )
    if config.fusion == "rrf":
        final = reciprocal_rank_fusion(text_results, image_results, rrf_k)
    elif config.fusion == "raw_score":
        final = raw_score_fusion(text_results, image_results)
    elif config.use_text:
        final = single_modality_ranking(text_results, "text")
    else:
        final = single_modality_ranking(image_results, "image")
    return text_results, image_results, final


def retrieval_rows(
    identifiers: dict[str, str],
    query: dict[str, Any],
    text_results: Sequence[SearchResult],
    image_results: Sequence[SearchResult],
    final_results: Sequence[dict[str, Any]],
    minimum_overlap: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    final_rank_by_key = {
        (item["chunk"].video_path, item["chunk"].chunk_id): rank
        for rank, item in enumerate(final_results, start=1)
    }
    for result_list, modality, results in (
        ("text", "text", text_results),
        ("image", "image", image_results),
    ):
        for rank, result in enumerate(results, start=1):
            matches = matching_intervals(
                result.chunk.start,
                result.chunk.end,
                query["intervals"],
                minimum_overlap,
            )
            rows.append(
                {
                    **identifiers,
                    "result_list": result_list,
                    "rank": rank,
                    "final_rank": final_rank_by_key.get(_identity(result), ""),
                    "chunk_id": result.chunk.chunk_id,
                    "chunk_start_seconds": result.chunk.start,
                    "chunk_end_seconds": result.chunk.end,
                    "modality": modality,
                    "raw_score": result.score,
                    "text_rank": rank if modality == "text" else "",
                    "image_rank": rank if modality == "image" else "",
                    "text_raw_score": result.score if modality == "text" else "",
                    "image_raw_score": result.score if modality == "image" else "",
                    "rrf_score": "",
                    "retrieved_text": result.chunk.text or "",
                    "frame_path": result.chunk.frame_path or "",
                    "is_relevant": bool(matches),
                    "matched_intervals": matches,
                }
            )
    final_modality = (
        "fused" if text_results and image_results else "text" if text_results else "image"
    )
    for rank, item in enumerate(final_results, start=1):
        chunk = item["chunk"]
        matches = matching_intervals(
            chunk.start, chunk.end, query["intervals"], minimum_overlap
        )
        rows.append(
            {
                **identifiers,
                "result_list": "final",
                "rank": rank,
                "final_rank": rank,
                "chunk_id": chunk.chunk_id,
                "chunk_start_seconds": chunk.start,
                "chunk_end_seconds": chunk.end,
                "modality": final_modality,
                "raw_score": item["best_raw_score"],
                "text_rank": item["text_rank"] or "",
                "image_rank": item["image_rank"] or "",
                "text_raw_score": (
                    item["text_raw_score"] if item["text_raw_score"] is not None else ""
                ),
                "image_raw_score": (
                    item["image_raw_score"] if item["image_raw_score"] is not None else ""
                ),
                "rrf_score": item["rrf_score"] if item["rrf_score"] is not None else "",
                "retrieved_text": chunk.text or "",
                "frame_path": chunk.frame_path or "",
                "is_relevant": bool(matches),
                "matched_intervals": matches,
            }
        )
    return rows


def retrieval_metrics(
    final_results: Sequence[dict[str, Any]],
    query: dict[str, Any],
    overlap: float,
) -> dict[str, float]:
    hits = [
        bool(
            matching_intervals(
                item["chunk"].start,
                item["chunk"].end,
                query["intervals"],
                overlap,
            )
        )
        for item in final_results
    ]
    reciprocal_rank = next(
        (1.0 / rank for rank, hit in enumerate(hits, start=1) if hit), 0.0
    )
    return {
        "recall_at_1": float(any(hits[:1])),
        "recall_at_5": float(any(hits[:5])),
        "recall_at_10": float(any(hits[:10])),
        "mrr": reciprocal_rank,
    }


def _git_commit(path: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def _command_version(command: Sequence[str]) -> Optional[str]:
    try:
        result = subprocess.run(
            list(command), capture_output=True, text=True, timeout=10, check=False
        )
        output = result.stdout.strip() or result.stderr.strip()
        return output.splitlines()[0] if result.returncode == 0 and output else None
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def _json_safe(value: Any) -> Any:
    """Recursively convert provider SDK objects into JSON-native values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump(mode="json"))
        except TypeError:
            return _json_safe(value.model_dump())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _ollama_model_info(model: Optional[str], host: str) -> Optional[dict[str, Any]]:
    if not model:
        return None
    try:
        import ollama

        response = ollama.Client(host=host).show(model)
        if hasattr(response, "model_dump"):
            try:
                response = response.model_dump(mode="json")
            except TypeError:
                response = response.model_dump()
        details = response.get("details", {}) if isinstance(response, dict) else {}
        return _json_safe({
            "model": model,
            "modified_at": response.get("modified_at") if isinstance(response, dict) else None,
            "details": details,
        })
    except Exception as exc:
        return {"model": model, "inspection_error": repr(exc)}


def _unload_ollama_model(model: Optional[str], host: str) -> None:
    """Release Ollama VRAM before the separate BERTScore CUDA stage."""
    if not model:
        return
    try:
        import ollama

        ollama.Client(host=host).generate(model=model, keep_alive=0)
    except Exception as exc:
        print(f"[Ollama] warning: unable to unload {model!r}: {exc!r}", flush=True)


def validate_runtime(
    args: argparse.Namespace,
    configurations: Sequence[ExperimentConfiguration],
    videos: Optional[dict[str, dict[str, Any]]] = None,
) -> None:
    """Fail before creating an experiment when required runtime pieces are absent."""
    for command in ("ffmpeg", "ffprobe"):
        if _command_version([command, "-version"]) is None:
            raise EvaluationConfigError(f"Required executable is unavailable: {command}")

    for video_id, video in (videos or {}).items():
        path = Path(video["resolved_path"])
        if not path.is_file():
            raise EvaluationConfigError(f"Video file is unavailable: {video_id} -> {path}")
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        try:
            actual_duration = float(probe.stdout.strip())
        except ValueError as exc:
            raise EvaluationConfigError(
                f"ffprobe could not read duration for {video_id}: {probe.stderr.strip()}"
            ) from exc
        if abs(actual_duration - float(video["duration_seconds"])) > 2.0:
            raise EvaluationConfigError(
                f"{video_id}: manifest duration {video['duration_seconds']}s differs from "
                f"the file duration {actual_duration:.3f}s by more than 2s."
            )
        if str(video["has_audio"]).strip().lower() == "true":
            audio_probe = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-select_streams", "a:0",
                    "-show_entries", "stream=index", "-of", "csv=p=0", str(path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if audio_probe.returncode != 0 or not audio_probe.stdout.strip():
                raise EvaluationConfigError(
                    f"{video_id} is marked has_audio=true but has no readable audio stream."
                )

    requested_devices = {
        str(args.embedding_device).lower(),
        str(args.asr_device).lower(),
        str(args.bertscore_device).lower() if args.compute_bertscore else "auto",
    }
    if any(device == "cuda" or device.startswith("cuda:") for device in requested_devices):
        try:
            import torch
        except ImportError as exc:
            raise EvaluationConfigError("CUDA was requested but PyTorch is not installed.") from exc
        if not torch.cuda.is_available():
            raise EvaluationConfigError(
                "CUDA was requested but torch.cuda.is_available() is false."
            )

    asr_device = str(args.asr_device).lower()
    if asr_device == "cuda" or asr_device.startswith("cuda:"):
        missing_libraries = []
        for library in ("libcublas.so.12", "libcudnn.so.9"):
            try:
                ctypes.CDLL(library)
            except OSError:
                missing_libraries.append(library)
        if missing_libraries:
            raise EvaluationConfigError(
                "CUDA ASR cannot load " + ", ".join(missing_libraries) + ". "
                "Install the CUDA 12 wheels from requirements.txt and set "
                "LD_LIBRARY_PATH as documented in docs/EVALUATION.md."
            )

    if args.compute_bertscore:
        try:
            import bert_score  # noqa: F401
        except ImportError as exc:
            raise EvaluationConfigError(
                "--compute-bertscore requires the bert-score package."
            ) from exc

    if not args.skip_generation:
        model_info = _ollama_model_info(args.generator_model, args.ollama_host)
        if not model_info or model_info.get("inspection_error"):
            detail = model_info.get("inspection_error") if model_info else "no model"
            raise EvaluationConfigError(
                f"Ollama model preflight failed for {args.generator_model!r}: {detail}"
            )
        uses_visual_evidence = any(config.use_image for config in configurations)
        is_vision_model = any(
            marker in str(args.generator_model).lower()
            for marker in EvaluationOllamaGenerator.VISION_MARKERS
        )
        if uses_visual_evidence and not is_vision_model:
            raise EvaluationConfigError(
                f"{args.generator_model!r} is not recognized as a vision model, but "
                "the selected configurations require image evidence."
            )


def write_environment(
    path: Path,
    args: argparse.Namespace,
    configurations: Sequence[ExperimentConfiguration],
    embedder_info: Optional[dict[str, Any]],
) -> None:
    packages = {}
    for package in (
        "vision-rag", "torch", "sentence-transformers", "faiss-cpu",
        "faster-whisper", "ctranslate2", "ollama", "bert-score", "numpy",
        "pymediainfo",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    gpu: dict[str, Any] = {}
    try:
        result = subprocess.run(
            [
                "nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0:
            gpu["nvidia_smi"] = result.stdout.strip().splitlines()
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    cpu_model = platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    memory_bytes = None
    try:
        memory_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        pass
    try:
        import torch

        gpu["torch_cuda_version"] = torch.version.cuda
        gpu["cuda_available"] = torch.cuda.is_available()
        gpu["mps_available"] = bool(
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
        )
        if torch.cuda.is_available():
            gpu["torch_cuda_devices"] = [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
    except ImportError:
        pass
    environment = {
        "schema_version": SCHEMA_VERSION,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "cpu_model": cpu_model,
            "logical_cores": os.cpu_count(),
            "ram_bytes": memory_bytes,
            "accelerator": gpu or None,
        },
        "operating_system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "visionrag": {
            "package_version": packages.get("vision-rag"),
            "git_commit": _git_commit(VISION_RAG_DIR),
        },
        "packages": packages,
        "models": {
            "embedding": embedder_info,
            "generator": _ollama_model_info(args.generator_model, args.ollama_host),
            "asr_models": sorted({config.whisper_model for config in configurations}),
            "gpu_memory_policy": (
                "Share one CLIP model instance; for CUDA whisper-medium, unload "
                "Ollama before ASR and release Whisper before embedding/generation."
            ),
        },
        "system_tools": {
            "ffmpeg": _command_version(["ffmpeg", "-version"]),
            "ffprobe": _command_version(["ffprobe", "-version"]),
        },
        "chunking": {
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "keyframe_strategy": args.keyframe_strategy,
        },
        "retrieval": {
            "top_k_per_modality": args.top_k_per_modality,
            "final_top_k": args.final_top_k,
            "rrf_k": args.rrf_k,
            "minimum_interval_overlap_seconds": args.minimum_overlap,
            "metric_definition": (
                "Query-level evidence hit: a retrieved chunk is relevant when its "
                "overlap with any annotated interval meets the threshold. Recall@K "
                "is the fraction of queries with at least one relevant chunk in the "
                "top K; MRR is truncated at final_top_k (MRR@10 by default)."
            ),
        },
        "generation": {
            "source_top_k": args.generation_top_k,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "host": args.ollama_host,
            "retries": args.generation_retries,
            "retry_delay_seconds": args.generation_retry_delay,
        },
        "bertscore": {
            "enabled": args.compute_bertscore,
            "model_type": args.bertscore_model,
            "device": args.bertscore_device,
            "batch_size": args.bertscore_batch_size,
            "language": "en",
        },
        "arguments": _json_safe(vars(args)),
        "random_seed": args.seed,
        "repetitions": args.runs,
        "configurations": [asdict(config) for config in configurations],
    }
    path.write_text(json.dumps(environment, indent=2), encoding="utf-8")


def normalize_inputs(
    output_dir: Path,
    videos: dict[str, dict[str, Any]],
    queries: list[dict[str, Any]],
) -> None:
    video_fields = [
        "video_id", "file_path", "title", "domain", "duration_seconds", "language",
        "source_url", "license", "has_audio", "has_burned_captions", "notes",
    ]
    with (output_dir / "normalized_videos.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=video_fields)
        writer.writeheader()
        for video in videos.values():
            writer.writerow({field: video.get(field, "") for field in video_fields})
    query_fields = [
        "query_id", "video_id", "question", "question_type", "reference_answer",
        "annotator_id", "answerable", "difficulty", "annotation_notes",
    ]
    with (output_dir / "normalized_queries.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=query_fields)
        writer.writeheader()
        for query in queries:
            writer.writerow({field: query.get(field, "") for field in query_fields})
    with (output_dir / "ground_truth_intervals.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = ["query_id", "interval_id", "start_seconds", "end_seconds"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for query in queries:
            for index, (start, end) in enumerate(query["intervals"], start=1):
                writer.writerow(
                    {
                        "query_id": query["query_id"],
                        "interval_id": index,
                        "start_seconds": start,
                        "end_seconds": end,
                    }
                )


def build_annotation_materials(output_dir: Path, seed: int) -> None:
    with (output_dir / "answers.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        answers = [row for row in csv.DictReader(handle) if row["generated_answer"]]
    random.Random(seed).shuffle(answers)
    annotation_dir = output_dir / "annotation"
    private_dir = output_dir / "private"
    annotation_dir.mkdir(exist_ok=True)
    private_dir.mkdir(exist_ok=True)
    packet_fields = [
        "answer_id", "query_id", "question", "reference_answer",
        "generated_answer", "source_evidence",
    ]
    query_lookup = {
        row["query_id"]: row
        for row in _read_csv(
            output_dir / "normalized_queries.csv", {"query_id", "question"}
        )
    }
    with (annotation_dir / "annotation_packet.csv").open(
        "w", encoding="utf-8", newline=""
    ) as packet_handle, (private_dir / "annotation_key.csv").open(
        "w", encoding="utf-8", newline=""
    ) as key_handle:
        packet_writer = csv.DictWriter(packet_handle, fieldnames=packet_fields)
        key_fields = ["answer_id", "experiment_id", "run_id", "config_id", "query_id"]
        key_writer = csv.DictWriter(key_handle, fieldnames=key_fields)
        packet_writer.writeheader()
        key_writer.writeheader()
        for index, answer in enumerate(answers, start=1):
            answer_id = f"ans_{index:06d}"
            query = query_lookup[answer["query_id"]]
            packet_writer.writerow(
                {
                    "answer_id": answer_id,
                    "query_id": answer["query_id"],
                    "question": query["question"],
                    "reference_answer": answer["reference_answer"],
                    "generated_answer": answer["generated_answer"],
                    "source_evidence": answer["source_evidence"],
                }
            )
            key_writer.writerow(
                {
                    "answer_id": answer_id,
                    "experiment_id": answer["experiment_id"],
                    "run_id": answer["run_id"],
                    "config_id": answer["config_id"],
                    "query_id": answer["query_id"],
                }
            )
    with (annotation_dir / "answer_annotations.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        csv.DictWriter(handle, fieldnames=ANNOTATION_FIELDS).writeheader()
    (annotation_dir / "README.txt").write_text(
        "Use at least two independent annotators. Copy one row per answer_id "
        "into answer_annotations.csv. Score correctness, groundedness, "
        "completeness, and relevance from 0 to 2; hallucination and refusal "
        "from 0 to 1. Do not provide private/annotation_key.csv to annotators.\n",
        encoding="utf-8",
    )


def add_bertscore(
    output_dir: Path,
    device: Optional[str],
    model_type: str,
    batch_size: int,
) -> None:
    try:
        from bert_score import score
    except ImportError as exc:
        raise EvaluationConfigError(
            "--compute-bertscore requires: pip install bert-score"
        ) from exc
    answers_path = output_dir / "answers.csv"
    with answers_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    valid_indices = [index for index, row in enumerate(rows) if row["generated_answer"]]
    if valid_indices:
        candidates = [rows[index]["generated_answer"] for index in valid_indices]
        references = [rows[index]["reference_answer"] for index in valid_indices]
        _, _, f1 = score(
            candidates,
            references,
            lang="en",
            model_type=model_type,
            device=device,
            batch_size=batch_size,
            verbose=True,
        )
        for index, value in zip(valid_indices, f1.tolist()):
            rows[index]["bertscore_f1"] = f"{value:.6f}"
    temporary = answers_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANSWER_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(answers_path)

    metrics_path = output_dir / "query_metrics.csv"
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        metric_rows = list(csv.DictReader(handle))
    scores_by_key = {
        (row["run_id"], row["config_id"], row["query_id"]): row["bertscore_f1"]
        for row in rows
        if row["bertscore_f1"]
    }
    for row in metric_rows:
        row["bertscore_f1"] = scores_by_key.get(
            (row["run_id"], row["config_id"], row["query_id"]), ""
        )
    temporary_metrics = metrics_path.with_suffix(".csv.tmp")
    with temporary_metrics.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUERY_METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(metric_rows)
    temporary_metrics.replace(metrics_path)


def _mean_csv(rows: Sequence[dict[str, str]], field: str) -> Optional[float]:
    values = [float(row[field]) for row in rows if row.get(field) not in (None, "")]
    return sum(values) / len(values) if values else None


def aggregate_results(output_dir: Path, experiment_id: str) -> list[dict[str, Any]]:
    with (output_dir / "query_metrics.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        metric_rows = list(csv.DictReader(handle))
    with (output_dir / "answers.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        answer_rows = list(csv.DictReader(handle))
    answers_by_key = {
        (row["run_id"], row["config_id"], row["query_id"]): row
        for row in answer_rows
    }
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in metric_rows:
        grouped[(row["run_id"], row["config_id"])].append(row)
        grouped[("all_runs", row["config_id"])].append(row)
    results = []
    for (run_id, config_id), rows in sorted(grouped.items()):
        strata: list[tuple[str, str, list[dict[str, str]]]] = [
            ("overall", "all", rows)
        ]
        for question_type in sorted({row["question_type"] for row in rows}):
            strata.append(
                (
                    "question_type",
                    question_type,
                    [row for row in rows if row["question_type"] == question_type],
                )
            )
        for domain in sorted({row["domain"] for row in rows}):
            strata.append(
                ("domain", domain, [row for row in rows if row["domain"] == domain])
            )
        for stratum_type, stratum_value, stratum_rows in strata:
            answers = [
                answers_by_key[
                    (
                        row["run_id"] if run_id == "all_runs" else run_id,
                        config_id,
                        row["query_id"],
                    )
                ]
                for row in stratum_rows
                if (
                    row["run_id"] if run_id == "all_runs" else run_id,
                    config_id,
                    row["query_id"],
                ) in answers_by_key
            ]
            results.append({
                "experiment_id": experiment_id,
                "run_id": run_id,
                "config_id": config_id,
                "stratum_type": stratum_type,
                "stratum_value": stratum_value,
                "queries": len(stratum_rows),
                "retrieval_failures": sum(
                    row["retrieval_success"] != "true" for row in stratum_rows
                ),
                "generation_attempts": sum(
                    row["generator_provider"] != "not_run" for row in answers
                ),
                "generation_failures": sum(
                    bool(row["generation_error"]) for row in answers
                ),
                "recall_at_1": _mean_csv(stratum_rows, "recall_at_1"),
                "recall_at_5": _mean_csv(stratum_rows, "recall_at_5"),
                "recall_at_10": _mean_csv(stratum_rows, "recall_at_10"),
                "mrr": _mean_csv(stratum_rows, "mrr"),
                "bleu1": _mean_csv(answers, "bleu1"),
                "token_f1": _mean_csv(answers, "token_f1"),
                "bertscore_f1": _mean_csv(answers, "bertscore_f1"),
            })
    with (output_dir / "configuration_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CONFIG_RESULT_FIELDS)
        writer.writeheader()
        for row in results:
            writer.writerow(
                {key: _csv_value(row.get(key, "")) for key in CONFIG_RESULT_FIELDS}
            )
    return results


def aggregate_timings(output_dir: Path, experiment_id: str) -> None:
    with (output_dir / "timings.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    timing_fields = (
        "ingestion_ms", "audio_extraction_ms", "asr_ms",
        "frame_extraction_ms", "text_embedding_ms", "image_embedding_ms",
        "indexing_ms", "query_embedding_ms", "retrieval_ms",
        "generation_ms", "total_pipeline_ms",
    )
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row["run_id"], row["config_id"], row["scope"], row["cache_state"])
        grouped[key].append(row)
        grouped[("all_runs", row["config_id"], row["scope"], row["cache_state"])].append(row)

    output_rows = []
    for (run_id, config_id, scope, cache_state), group_rows in sorted(grouped.items()):
        failures = sum(row["success"] != "true" for row in group_rows)
        for field in timing_fields:
            values = [
                float(row[field])
                for row in group_rows
                if row["success"] == "true" and row.get(field) not in (None, "")
            ]
            if not values:
                continue
            ordered = sorted(values)
            p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
            output_rows.append(
                {
                    "experiment_id": experiment_id,
                    "run_id": run_id,
                    "config_id": config_id,
                    "scope": scope,
                    "cache_state": cache_state,
                    "metric": field,
                    "unit": "milliseconds",
                    "observations": len(values),
                    "failures": failures,
                    "mean": statistics.fmean(values),
                    "median": statistics.median(values),
                    "standard_deviation": statistics.pstdev(values),
                    "p95": p95,
                }
            )
    with (output_dir / "latency_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=LATENCY_RESULT_FIELDS)
        writer.writeheader()
        for row in output_rows:
            writer.writerow(
                {key: _csv_value(row.get(key, "")) for key in LATENCY_RESULT_FIELDS}
            )


def _format_metric(value: Optional[float]) -> str:
    return "not measured" if value is None else f"{value:.3f}"


def write_markdown_report(
    output_dir: Path,
    experiment_id: str,
    results: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    lines = [
        "# VisionRAG Nine-Video Pilot Evidence Report",
        "",
        f"Experiment ID: `{experiment_id}`",
        "",
        "> Generated from the raw CSV evidence in this directory. This is a "
        "pilot evaluation, not an external benchmark.",
        "",
        "| Run | Configuration | N | R@1 | R@5 | R@10 | MRR@10 | BLEU-1 | Token F1 | BERTScore F1 | Gen. failures |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        if row["stratum_type"] != "overall":
            continue
        lines.append(
            f"| {row['run_id']} | `{row['config_id']}` | {row['queries']} | "
            f"{_format_metric(row['recall_at_1'])} | "
            f"{_format_metric(row['recall_at_5'])} | "
            f"{_format_metric(row['recall_at_10'])} | "
            f"{_format_metric(row['mrr'])} | {_format_metric(row['bleu1'])} | "
            f"{_format_metric(row['token_f1'])} | "
            f"{_format_metric(row['bertscore_f1'])} | "
            f"{row['generation_failures']} |"
        )
    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            f"- Corpus: 9 videos and 45 questions; {args.runs} repetition(s).",
            "- Token F1 and BERTScore are separate metrics.",
            "- Human answer quality is pending until two independent annotators complete the blinded packet.",
            "- Hosted cost is unmeasured unless an actual billed value was returned; no price is inferred.",
            "- Video licenses must be verified before redistribution.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _source_evidence(chunks: Sequence[EmbeddedChunk]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": chunk.chunk_id,
            "start": chunk.start,
            "end": chunk.end,
            "text": chunk.text or "",
            "frame_path": chunk.frame_path or "",
        }
        for chunk in chunks
    ]


def _record_video_failure(
    sinks: dict[str, CsvSink],
    experiment_id: str,
    run_id: str,
    config: ExperimentConfiguration,
    video: dict[str, Any],
    queries: Sequence[dict[str, Any]],
    failure_type: str,
    description: str,
) -> None:
    base = {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "config_id": config.config_id,
        "query_id": "",
        "video_id": video["video_id"],
    }
    sinks["errors"].write(
        {
            **base,
            "failure_type": failure_type,
            "description": description,
            "expected_behavior": "Build real transcript and frame evidence from the video.",
            "retrieved_evidence": "",
            "suspected_cause": "Unclassified input/preprocessing failure; inspect the exact exception.",
        }
    )
    sinks["timings"].write(
        {
            **base,
            "scope": "video_preprocessing",
            "total_pipeline_ms": 0.0,
            "cache_state": "unknown",
            "success": False,
            "error": description,
        }
    )
    for query in queries:
        sinks["metrics"].write(
            {
                **base,
                "query_id": query["query_id"],
                "domain": video["domain"],
                "question_type": query["question_type"],
                "recall_at_1": 0.0,
                "recall_at_5": 0.0,
                "recall_at_10": 0.0,
                "mrr": 0.0,
                "retrieval_success": False,
                "generation_success": "",
                "error": description,
            }
        )


def run_experiment(
    args: argparse.Namespace,
    videos: dict[str, dict[str, Any]],
    queries: list[dict[str, Any]],
    configurations: Sequence[ExperimentConfiguration],
) -> Path:
    if not args.skip_generation and not args.generator_model:
        raise EvaluationConfigError(
            "--generator-model is required unless --skip-generation is set."
        )
    if args.top_k_per_modality < 10 or args.final_top_k < 10:
        raise EvaluationConfigError(
            "top-k values must be at least 10 to calculate Recall@10."
        )
    if args.chunk_overlap >= args.chunk_size:
        raise EvaluationConfigError(
            "--chunk-overlap must be smaller than --chunk-size."
        )

    experiment_id = args.experiment_id or datetime.now(timezone.utc).strftime(
        "vrag_%Y%m%dT%H%M%SZ"
    )
    output_dir = Path(args.output_dir).resolve() / experiment_id
    if output_dir.exists():
        raise EvaluationConfigError(
            f"Experiment directory already exists: {output_dir}. "
            "Use a new --experiment-id."
        )
    output_dir.mkdir(parents=True)
    normalize_inputs(output_dir, videos, queries)
    prompt_dir = output_dir / "prompt"
    prompt_dir.mkdir()
    (prompt_dir / "system_prompt.txt").write_text(
        BaseGenerator()._build_system_prompt() + "\n", encoding="utf-8"
    )
    (prompt_dir / "message_template.txt").write_text(
        "[Chunk N | STARTs – ENDs]\nTRANSCRIPT\n\nQuestion: QUERY\n",
        encoding="utf-8",
    )

    manifest_path = output_dir / "experiment_manifest.json"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_files": {
            "videos_csv": str(Path(args.videos).resolve()),
            "videos_csv_sha256": _sha256_file(Path(args.videos).resolve()),
            "queries_csv": str(Path(args.queries).resolve()),
            "queries_csv_sha256": _sha256_file(Path(args.queries).resolve()),
        },
        "video_count": len(videos),
        "query_count": len(queries),
        "runs": args.runs,
        "seed": args.seed,
        "configurations": [asdict(config) for config in configurations],
        "video_assets": [
            {
                "video_id": video_id,
                "path": video["resolved_path"],
                "size_bytes": (
                    Path(video["resolved_path"]).stat().st_size
                    if Path(video["resolved_path"]).exists()
                    else None
                ),
                "sha256": (
                    _sha256_file(Path(video["resolved_path"]))
                    if Path(video["resolved_path"]).exists()
                    and not args.skip_video_hashes
                    else None
                ),
            }
            for video_id, video in videos.items()
        ],
    }
    intervals_path = Path(args.relevant_intervals).resolve()
    if intervals_path.exists():
        manifest["input_files"]["relevant_intervals_csv"] = str(intervals_path)
        manifest["input_files"]["relevant_intervals_csv_sha256"] = _sha256_file(
            intervals_path
        )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    text_embedder, image_embedder, embedder_info = setup_embedders(
        args.embedder, args.embedding_device
    )
    asr_instances = {
        model: WhisperLocalASR(model_size=model, device=args.asr_device)
        for model in {config.whisper_model for config in configurations}
    }
    write_environment(
        output_dir / "environment.json", args, configurations, embedder_info
    )
    sinks = {
        "retrieval": CsvSink(output_dir / "retrieval_results.csv", RETRIEVAL_FIELDS),
        "chunks": CsvSink(output_dir / "chunks.csv", CHUNK_FIELDS),
        "answers": CsvSink(output_dir / "answers.csv", ANSWER_FIELDS),
        "metrics": CsvSink(output_dir / "query_metrics.csv", QUERY_METRIC_FIELDS),
        "timings": CsvSink(output_dir / "timings.csv", TIMING_FIELDS),
        "errors": CsvSink(output_dir / "errors.csv", ERROR_FIELDS),
        "costs": CsvSink(output_dir / "costs.csv", COST_FIELDS),
    }
    queries_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for query in queries:
        queries_by_video[query["video_id"]].append(query)
    frames_cache = Path(args.frames_cache_dir).resolve()
    frames_cache.mkdir(parents=True, exist_ok=True)

    try:
        for run_number in range(1, args.runs + 1):
            run_id = f"run_{run_number:02d}"
            run_seed = args.seed + run_number - 1
            set_random_seeds(run_seed)
            print(f"\n=== {run_id}/{args.runs} (seed={run_seed}) ===")
            for config in configurations:
                print(f"\n--- {config.config_id} ---")
                evict_medium_asr = (
                    config.use_text
                    and config.whisper_model == "medium"
                    and str(args.asr_device).lower().startswith("cuda")
                )
                if evict_medium_asr:
                    for asr_instance in asr_instances.values():
                        release_asr_model(asr_instance)
                    _unload_ollama_model(args.generator_model, args.ollama_host)
                    print(
                        "[GPU memory] Enabled staged eviction for whisper-medium.",
                        flush=True,
                    )
                generator = (
                    None
                    if args.skip_generation
                    else EvaluationOllamaGenerator(
                        model=args.generator_model,
                        host=args.ollama_host,
                        temperature=args.temperature,
                        seed=run_seed,
                        max_tokens=args.max_tokens,
                        retries=args.generation_retries,
                        retry_delay=args.generation_retry_delay,
                    )
                )
                prompt_version = (
                    "not_applicable"
                    if generator is None
                    else hashlib.sha256(
                        generator._build_system_prompt().encode("utf-8")
                    ).hexdigest()[:16]
                )
                for video_id, video in videos.items():
                    video_path = Path(video["resolved_path"])
                    video_queries = queries_by_video.get(video_id, [])
                    if not video_path.exists():
                        description = f"Video file not found: {video_path}"
                        _record_video_failure(
                            sinks, experiment_id, run_id, config, video,
                            video_queries, "input", description,
                        )
                        if not args.allow_missing_as_failure:
                            raise EvaluationConfigError(description)
                        continue
                    print(
                        f"[{run_id}/{config.config_id}] {video_id}: {video['title']}"
                    )
                    preprocessing_started = time.perf_counter()
                    active_asr = (
                        asr_instances[config.whisper_model] if config.use_text else None
                    )
                    if evict_medium_asr:
                        _unload_ollama_model(args.generator_model, args.ollama_host)
                    try:
                        chunks, media_timing = prepare_chunks(
                            video_path,
                            frames_cache,
                            active_asr,
                            config.use_text,
                            config.use_image,
                            args.chunk_size,
                            args.chunk_overlap,
                            args.keyframe_strategy,
                        )
                        if evict_medium_asr:
                            release_asr_model(active_asr)
                        embedded, embedding_timing = embed_chunks(
                            chunks,
                            text_embedder,
                            image_embedder,
                            use_text=config.use_text,
                            use_image=config.use_image,
                        )
                        configured_chunks = chunks_for_configuration(embedded, config)
                        for chunk in configured_chunks:
                            sinks["chunks"].write(
                                {
                                    "experiment_id": experiment_id,
                                    "run_id": run_id,
                                    "config_id": config.config_id,
                                    "video_id": video_id,
                                    "chunk_id": chunk.chunk_id,
                                    "start_seconds": chunk.start,
                                    "end_seconds": chunk.end,
                                    "duration_seconds": chunk.duration,
                                    "transcript": chunk.text or "",
                                    "frame_path": chunk.frame_path or "",
                                    "text_vector_dimension": (
                                        len(chunk.text_vector) if chunk.text_vector else ""
                                    ),
                                    "image_vector_dimension": (
                                        len(chunk.image_vector) if chunk.image_vector else ""
                                    ),
                                    "asr_provider": chunk.metadata.get("asr_provider", ""),
                                    "asr_model": chunk.metadata.get("asr_model", ""),
                                    "frame_cache_state": media_timing["cache_state"],
                                }
                            )
                        indexing_started = time.perf_counter()
                        store = FAISS()
                        store.index(configured_chunks)
                        indexing_ms = _elapsed_ms(indexing_started)
                        sinks["timings"].write(
                            {
                                "experiment_id": experiment_id,
                                "run_id": run_id,
                                "config_id": config.config_id,
                                "query_id": "",
                                "video_id": video_id,
                                "scope": "video_preprocessing",
                                **media_timing,
                                **embedding_timing,
                                "indexing_ms": indexing_ms,
                                "total_pipeline_ms": _elapsed_ms(preprocessing_started),
                                "cache_state": media_timing["cache_state"],
                                "success": True,
                                "error": "",
                            }
                        )
                        if args.embedder == "jina-v4":
                            sinks["costs"].write(
                                {
                                    "experiment_id": experiment_id,
                                    "run_id": run_id,
                                    "config_id": config.config_id,
                                    "query_id": "",
                                    "video_id": video_id,
                                    "provider": "Jina",
                                    "model": "jina-embeddings-v4",
                                    "operation": "corpus_embedding",
                                    "image_count": embedding_timing["image_embedding_calls"],
                                    "api_calls": (
                                        embedding_timing["text_embedding_calls"]
                                        + embedding_timing["image_embedding_calls"]
                                    ),
                                    "reported_cost_usd": "",
                                    "pricing_date": "",
                                }
                            )
                    except Exception as exc:
                        if evict_medium_asr:
                            release_asr_model(active_asr)
                        _record_video_failure(
                            sinks, experiment_id, run_id, config, video,
                            video_queries, "preprocessing", repr(exc),
                        )
                        if not args.allow_runtime_failures:
                            raise EvaluationConfigError(
                                f"Preprocessing failed for {run_id}/{config.config_id}/"
                                f"{video_id}: {exc!r}"
                            ) from exc
                        continue

                    for query in video_queries:
                        identifiers = {
                            "experiment_id": experiment_id,
                            "run_id": run_id,
                            "config_id": config.config_id,
                            "query_id": query["query_id"],
                            "video_id": video_id,
                        }
                        query_started = time.perf_counter()
                        query_embedding_ms = 0.0
                        retrieval_ms = 0.0
                        generation_ms = 0.0
                        query_error = ""
                        try:
                            embedding_started = time.perf_counter()
                            query_vector = text_embedder.embed(query["question"])
                            query_embedding_ms = _elapsed_ms(embedding_started)
                            retrieval_started = time.perf_counter()
                            text_results, image_results, final_results = search_configuration(
                                store,
                                query_vector,
                                config,
                                args.top_k_per_modality,
                                args.rrf_k,
                            )
                            final_results = final_results[: args.final_top_k]
                            retrieval_ms = _elapsed_ms(retrieval_started)
                            for row in retrieval_rows(
                                identifiers,
                                query,
                                text_results,
                                image_results,
                                final_results,
                                args.minimum_overlap,
                            ):
                                sinks["retrieval"].write(row)
                            metrics = retrieval_metrics(
                                final_results, query, args.minimum_overlap
                            )
                            sources = generation_sources(
                                final_results, config, args.generation_top_k
                            )
                            generated_answer = ""
                            generation_error = ""
                            if generator is not None:
                                generation_started = time.perf_counter()
                                try:
                                    generated_answer = generator.generate(
                                        query["question"], sources
                                    )
                                except Exception as exc:
                                    generation_error = repr(exc)
                                    sinks["errors"].write(
                                        {
                                            **identifiers,
                                            "failure_type": "generation",
                                            "description": generation_error,
                                            "expected_behavior": "Generate one answer from the recorded source evidence.",
                                            "retrieved_evidence": _source_evidence(sources),
                                            "suspected_cause": "Unclassified runtime/provider failure; inspect the exception.",
                                        }
                                    )
                                generation_ms = _elapsed_ms(generation_started)
                                query_error = generation_error
                            bleu1 = (
                                compute_bleu1(
                                    query["reference_answer"], generated_answer
                                )
                                if generated_answer
                                else None
                            )
                            token_f1 = (
                                compute_token_f1(
                                    query["reference_answer"], generated_answer
                                )
                                if generated_answer
                                else None
                            )
                            sinks["answers"].write(
                                {
                                    **identifiers,
                                    "generated_answer": generated_answer,
                                    "reference_answer": query["reference_answer"],
                                    "source_chunk_ids": [
                                        chunk.chunk_id for chunk in sources
                                    ],
                                    "source_timestamps": [
                                        [chunk.start, chunk.end] for chunk in sources
                                    ],
                                    "source_evidence": _source_evidence(sources),
                                    "generator_provider": (
                                        "Ollama" if generator else "not_run"
                                    ),
                                    "generator_model": args.generator_model or "",
                                    "prompt_version": prompt_version,
                                    "temperature": (
                                        args.temperature if generator else ""
                                    ),
                                    "seed": run_seed,
                                    "generation_error": generation_error,
                                    "bleu1": bleu1 if bleu1 is not None else "",
                                    "token_f1": (
                                        token_f1 if token_f1 is not None else ""
                                    ),
                                    "bertscore_f1": "",
                                }
                            )
                            if generator is not None:
                                sinks["costs"].write(
                                    {
                                        **identifiers,
                                        "provider": "Ollama (local)",
                                        "model": args.generator_model,
                                        "operation": "generation",
                                        "input_tokens": generator.last_usage.get(
                                            "input_tokens", ""
                                        ),
                                        "output_tokens": generator.last_usage.get(
                                            "output_tokens", ""
                                        ),
                                        "image_count": generator.last_usage.get(
                                            "image_count", 0
                                        ),
                                        "api_calls": generator.last_usage.get(
                                            "api_calls", 1
                                        ),
                                        "reported_cost_usd": 0.0,
                                        "pricing_date": "not_applicable_local_execution",
                                    }
                                )
                            sinks["metrics"].write(
                                {
                                    **identifiers,
                                    "domain": video["domain"],
                                    "question_type": query["question_type"],
                                    **metrics,
                                    "retrieval_success": True,
                                    "generation_success": (
                                        bool(generated_answer) if generator else ""
                                    ),
                                    "bleu1": bleu1 if bleu1 is not None else "",
                                    "token_f1": (
                                        token_f1 if token_f1 is not None else ""
                                    ),
                                    "bertscore_f1": "",
                                    "error": generation_error,
                                }
                            )
                            if args.embedder == "jina-v4":
                                sinks["costs"].write(
                                    {
                                        **identifiers,
                                        "provider": "Jina",
                                        "model": "jina-embeddings-v4",
                                        "operation": "query_embedding",
                                        "image_count": 0,
                                        "api_calls": 1,
                                        "reported_cost_usd": "",
                                        "pricing_date": "",
                                    }
                                )
                        except Exception as exc:
                            query_error = repr(exc)
                            sinks["errors"].write(
                                {
                                    **identifiers,
                                    "failure_type": "retrieval",
                                    "description": query_error,
                                    "expected_behavior": "Return a ranked list from the configured indexes.",
                                    "retrieved_evidence": "",
                                    "suspected_cause": "Unclassified retrieval/runtime failure; inspect the exception.",
                                }
                            )
                            sinks["metrics"].write(
                                {
                                    **identifiers,
                                    "domain": video["domain"],
                                    "question_type": query["question_type"],
                                    "recall_at_1": 0.0,
                                    "recall_at_5": 0.0,
                                    "recall_at_10": 0.0,
                                    "mrr": 0.0,
                                    "retrieval_success": False,
                                    "generation_success": "",
                                    "error": query_error,
                                }
                            )
                        finally:
                            sinks["timings"].write(
                                {
                                    **identifiers,
                                    "scope": "query",
                                    "query_embedding_ms": query_embedding_ms,
                                    "retrieval_ms": retrieval_ms,
                                    "generation_ms": generation_ms,
                                    "total_pipeline_ms": _elapsed_ms(query_started),
                                    "cache_state": "not_applicable",
                                    "success": not bool(query_error),
                                    "error": query_error,
                                }
                            )
    except BaseException:
        manifest["status"] = "incomplete"
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        raise
    finally:
        for asr_instance in asr_instances.values():
            release_asr_model(asr_instance)
        for sink in sinks.values():
            sink.close()

    with (output_dir / "errors.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        runtime_error_count = sum(1 for _ in csv.DictReader(handle))
    if runtime_error_count and not args.allow_runtime_failures:
        raise EvaluationConfigError(
            f"Experiment recorded {runtime_error_count} runtime failure(s); "
            "the manifest is incomplete. Inspect errors.csv and the run log."
        )

    if args.compute_bertscore:
        _unload_ollama_model(args.generator_model, args.ollama_host)
        add_bertscore(
            output_dir,
            None if args.bertscore_device == "auto" else args.bertscore_device,
            args.bertscore_model,
            args.bertscore_batch_size,
        )
    results = aggregate_results(output_dir, experiment_id)
    aggregate_timings(output_dir, experiment_id)
    build_annotation_materials(output_dir, args.seed)
    write_markdown_report(output_dir, experiment_id, results, args)
    manifest["status"] = "complete"
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["outputs"] = sorted(path.name for path in output_dir.iterdir())
    manifest["output_sha256"] = {
        str(path.relative_to(output_dir)): _sha256_file(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path != manifest_path
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a reproducible nine-video VisionRAG paper evaluation."
    )
    parser.add_argument(
        "--videos", default=str(PROJECT_ROOT / "data" / "videos.csv")
    )
    parser.add_argument(
        "--queries", default=str(PROJECT_ROOT / "data" / "queries.csv")
    )
    parser.add_argument(
        "--relevant-intervals",
        default=str(PROJECT_ROOT / "data" / "relevant_intervals.csv"),
        help="Optional CSV containing additional valid intervals.",
    )
    parser.add_argument(
        "--output-dir", default=str(PROJECT_ROOT / "results")
    )
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument(
        "--frames-cache-dir",
        default=str(PROJECT_ROOT / ".cache" / "evaluation_frames"),
    )
    parser.add_argument("--configs", default="all", help="Comma-separated IDs or 'all'.")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--expected-videos", type=int, default=9)
    parser.add_argument("--expected-queries", type=int, default=45)
    parser.add_argument(
        "--embedder", choices=("local-clip", "jina-v4"), default="local-clip"
    )
    parser.add_argument("--embedding-device", default="auto")
    parser.add_argument("--asr-device", default="auto")
    parser.add_argument("--generator-model", default=None)
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--generation-retries", type=int, default=3)
    parser.add_argument("--generation-retry-delay", type=float, default=2.0)
    parser.add_argument("--chunk-size", type=float, default=5.0)
    parser.add_argument("--chunk-overlap", type=float, default=1.0)
    parser.add_argument(
        "--keyframe-strategy", choices=("middle", "first"), default="middle"
    )
    parser.add_argument("--top-k-per-modality", type=int, default=10)
    parser.add_argument("--final-top-k", type=int, default=10)
    parser.add_argument("--generation-top-k", type=int, default=5)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--minimum-overlap", type=float, default=0.5)
    parser.add_argument("--compute-bertscore", action="store_true")
    parser.add_argument("--bertscore-device", default="auto")
    parser.add_argument("--bertscore-model", default="roberta-large")
    parser.add_argument("--bertscore-batch-size", type=int, default=8)
    parser.add_argument(
        "--allow-runtime-failures",
        action="store_true",
        help="Finish diagnostic runs even when preprocessing/generation rows fail.",
    )
    parser.add_argument(
        "--allow-missing-as-failure",
        action="store_true",
        help="Record missing videos as failures and continue instead of stopping.",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate the dataset, CUDA libraries, dependencies, and Ollama model, then exit.",
    )
    parser.add_argument(
        "--skip-video-hashes",
        action="store_true",
        help="Skip SHA-256 video identity hashes (faster, but less reproducible).",
    )
    return parser


def _mark_manifest_incomplete(args: argparse.Namespace, exc: BaseException) -> None:
    if not args.experiment_id:
        return
    manifest_path = (
        Path(args.output_dir).resolve() / args.experiment_id / "experiment_manifest.json"
    )
    if not manifest_path.exists():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "incomplete"
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["fatal_error_type"] = exc.__class__.__name__
        manifest["fatal_error"] = str(exc)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except Exception as manifest_exc:
        print(
            f"[WARNING] Could not update incomplete manifest: {manifest_exc!r}",
            file=sys.stderr,
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.runs < 1:
            raise EvaluationConfigError("--runs must be at least 1.")
        if args.generation_retries < 0 or args.generation_retry_delay < 0:
            raise EvaluationConfigError(
                "Generation retries and retry delay must be non-negative."
            )
        if args.bertscore_batch_size < 1:
            raise EvaluationConfigError("--bertscore-batch-size must be at least 1.")
        if args.chunk_overlap < 0:
            raise EvaluationConfigError("--chunk-overlap must be >= 0.")
        for name in (
            "chunk_size", "top_k_per_modality", "final_top_k",
            "generation_top_k", "rrf_k", "minimum_overlap", "max_tokens",
        ):
            if getattr(args, name) <= 0:
                raise EvaluationConfigError(f"--{name.replace('_', '-')} must be > 0.")
        if args.generation_top_k > args.final_top_k:
            raise EvaluationConfigError(
                "--generation-top-k cannot exceed --final-top-k."
            )
        if args.minimum_overlap > args.chunk_size:
            raise EvaluationConfigError(
                "--minimum-overlap cannot exceed --chunk-size."
            )
        if args.experiment_id and not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", args.experiment_id
        ):
            raise EvaluationConfigError(
                "--experiment-id may contain only letters, digits, dot, underscore, "
                "and hyphen (maximum 128 characters)."
            )
        configurations = select_configurations(args.configs)
        intervals_path = Path(args.relevant_intervals).resolve()
        videos, queries = load_dataset(
            Path(args.videos).resolve(),
            Path(args.queries).resolve(),
            intervals_path if intervals_path.exists() else None,
            args.expected_videos,
            args.expected_queries,
        )
        distribution = Counter(query["question_type"] for query in queries)
        print(
            f"Validated {len(videos)} videos and {len(queries)} queries; "
            f"question types: {dict(distribution)}"
        )
        if args.validate_only:
            return 0
        if args.preflight_only:
            validate_runtime(args, configurations, videos)
            print("Runtime preflight passed.")
            return 0
        if not args.experiment_id:
            args.experiment_id = datetime.now(timezone.utc).strftime(
                "vrag_%Y%m%dT%H%M%SZ"
            )
        validate_runtime(args, configurations, videos)
        output_dir = run_experiment(args, videos, queries, configurations)
        print(f"\nExperiment complete: {output_dir}")
        print(
            "Human annotation remains pending; use annotation/annotation_packet.csv."
        )
        return 0
    except EvaluationConfigError as exc:
        _mark_manifest_incomplete(args, exc)
        print(f"\n[CONFIG ERROR] {exc}\n", file=sys.stderr)
        return 2
    except KeyboardInterrupt as exc:
        _mark_manifest_incomplete(args, exc)
        print("\n[INTERRUPTED] Experiment stopped; partial evidence was preserved.\n", file=sys.stderr)
        return 130
    except BaseException as exc:
        _mark_manifest_incomplete(args, exc)
        print(f"\n[FATAL ERROR] {exc!r}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
