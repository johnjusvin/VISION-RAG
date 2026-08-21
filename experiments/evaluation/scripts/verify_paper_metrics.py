#!/usr/bin/env python3
"""Verify the VisionRAG paper's reported values from the raw final-v3 CSVs."""

from __future__ import annotations

import csv
import math
import statistics
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "paper_pilot_cuda_final_v3"


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean(rows: list[dict[str, str]], field: str) -> float:
    return statistics.fmean(float(row[field]) for row in rows)


def require(actual: str, expected: str, label: str) -> None:
    if actual != expected:
        raise SystemExit(f"{label}: paper={expected}, recomputed={actual}")


def rounded(value: float, places: int) -> str:
    quantum = Decimal(1).scaleb(-places)
    return format(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP), f".{places}f")


def latency_values(
    rows: list[dict[str, str]], config: str, scope: str, field: str
) -> list[float]:
    return [
        float(row[field])
        for row in rows
        if row["config_id"] == config
        and row["scope"] == scope
        and row["success"] == "true"
        and row.get(field) not in (None, "")
    ]


def main() -> int:
    query_rows = read_csv("query_metrics.csv")
    answer_rows = read_csv("answers.csv")
    timing_rows = read_csv("timings.csv")
    answer_by_key = {
        (row["run_id"], row["config_id"], row["query_id"]): row
        for row in answer_rows
    }

    require(str(len(query_rows)), "810", "query trials")
    require(str(len(answer_rows)), "810", "generated answers")
    require(
        str(sum(bool(row["generation_error"]) for row in answer_rows)),
        "0",
        "generation failures",
    )

    expected_table = {
        "full_rrf_chrono": ("0.459", "0.756", "0.844", "0.592", "0.201", "0.862"),
        "image_only": ("0.289", "0.444", "0.622", "0.376", "0.104", "0.834"),
        "raw_score_fusion": ("0.519", "0.726", "0.911", "0.635", "0.186", "0.860"),
        "rrf_relevance_order": ("0.467", "0.756", "0.844", "0.598", "0.221", "0.865"),
        "text_only": ("0.556", "0.733", "0.911", "0.656", "0.292", "0.879"),
        "whisper_medium": ("0.422", "0.822", "0.933", "0.596", "0.183", "0.859"),
    }
    fields = ("recall_at_1", "recall_at_5", "recall_at_10", "mrr")
    for config, expected in expected_table.items():
        rows = [row for row in query_rows if row["config_id"] == config]
        answers = [
            answer_by_key[(row["run_id"], config, row["query_id"])] for row in rows
        ]
        actual = tuple(
            rounded(mean(rows, field), 3) for field in fields
        ) + (
            rounded(mean(answers, "token_f1"), 3),
            rounded(mean(answers, "bertscore_f1"), 3),
        )
        require(str(len(rows)), "135", f"{config} trials")
        require(str(actual), str(expected), f"{config} table row")

    image_type_expected = {
        "visual_dependent": "0.556",
        "transcript_dependent": "0.063",
    }
    for question_type, expected in image_type_expected.items():
        rows = [
            row
            for row in query_rows
            if row["config_id"] == "image_only"
            and row["question_type"] == question_type
        ]
        require(rounded(mean(rows, "recall_at_1"), 3), expected, f"image-only R@1/{question_type}")

    rrf_type_expected = {
        "transcript_dependent": "0.938",
        "visual_dependent": "0.833",
        "multimodal": "0.727",
    }
    for question_type, expected in rrf_type_expected.items():
        rows = [
            row
            for row in query_rows
            if row["config_id"] == "full_rrf_chrono"
            and row["question_type"] == question_type
        ]
        require(rounded(mean(rows, "recall_at_10"), 3), expected, f"RRF R@10/{question_type}")

    run_mrr = []
    run_recall = []
    for run_id in ("run_01", "run_02", "run_03"):
        rows = [
            row
            for row in query_rows
            if row["config_id"] == "full_rrf_chrono" and row["run_id"] == run_id
        ]
        run_mrr.append(mean(rows, "mrr"))
        run_recall.append(mean(rows, "recall_at_10"))
    require(rounded(statistics.pstdev(run_recall), 3), "0.000", "RRF R@10 population SD")
    require(rounded(statistics.stdev(run_mrr), 3), "0.011", "RRF MRR sample SD")

    latency_checks = (
        ("full_rrf_chrono", "query", "total_pipeline_ms", "median", "5.87", 1000.0, 2),
        ("full_rrf_chrono", "query", "total_pipeline_ms", "p95", "14.50", 1000.0, 2),
        ("full_rrf_chrono", "query", "generation_ms", "median", "5.86", 1000.0, 2),
        ("full_rrf_chrono", "query", "retrieval_ms", "median", "0.33", 1.0, 2),
        ("text_only", "query", "total_pipeline_ms", "median", "1.52", 1000.0, 2),
        ("full_rrf_chrono", "video_preprocessing", "asr_ms", "median", "3.17", 1000.0, 2),
        ("whisper_medium", "video_preprocessing", "asr_ms", "median", "11.66", 1000.0, 2),
    )
    for config, scope, field, statistic, expected, divisor, places in latency_checks:
        values = latency_values(timing_rows, config, scope, field)
        if statistic == "median":
            value = statistics.median(values)
        else:
            ordered = sorted(values)
            value = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
        require(rounded(value / divisor, places), expected, f"{config} {field} {statistic}")

    print("PASS: all paper evaluation numbers match the raw final-v3 artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
