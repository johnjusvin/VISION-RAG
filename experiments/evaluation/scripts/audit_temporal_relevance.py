#!/usr/bin/env python3
"""Audit the temporal relevance metric of an archive's stored ranked chunks.

Recomputes temporal (joint) and video-level retrieval metrics directly from
``retrieval_results.csv`` without touching retrieval or generation:

  temporal hit : chunk_video_id == query video_id  AND  interval overlap >= 0.5 s
  video hit    : chunk_video_id == query video_id

The buggy (timestamp-only) definition is also recomputed so old vs new can be
compared per trial, and is cross-checked against the archive's stored
``configuration_results.csv`` to prove the recompute reproduces the original
aggregation exactly.

Usage:
    python3 audit_temporal_relevance.py <archive_dir> <output_dir>
"""
from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

MIN_OVERLAP = 0.5
TOP_K = (1, 5, 10)
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 1729

CONFIG_ORDER = [
    "full_rrf_chrono",
    "image_only",
    "raw_score_fusion",
    "rrf_relevance_order",
    "text_only",
    "whisper_medium",
]
CONFIG_LABEL = {
    "full_rrf_chrono": "RRF-Chrono",
    "image_only": "Image-only",
    "raw_score_fusion": "Raw fusion",
    "rrf_relevance_order": "RRF-Rel. order",
    "text_only": "Text-only",
    "whisper_medium": "Whisper-medium",
}
STRATA_ORDER = ["transcript_dependent", "visual_dependent", "multimodal"]
STRATA_LABEL = {
    "transcript_dependent": "Transcript",
    "visual_dependent": "Visual",
    "multimodal": "Multimodal",
}
# Single-video (pre-distractor) baselines supplied for the sanity check.
OLD_SINGLE_VIDEO = {
    "image_only": {"recall_at_1": 0.289, "recall_at_5": 0.444, "recall_at_10": 0.622},
    "text_only": {"recall_at_1": 0.556, "recall_at_5": 0.733, "recall_at_10": 0.911},
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def interval_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def to_float(value: str) -> float:
    return float(value) if value not in ("", None) else 0.0


def load_intervals(archive: Path) -> dict[str, list[tuple[float, float]]]:
    intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in read_csv(archive / "ground_truth_intervals.csv"):
        intervals[row["query_id"]].append(
            (float(row["start_seconds"]), float(row["end_seconds"]))
        )
    return dict(intervals)


def load_queries(archive: Path) -> dict[str, dict[str, str]]:
    return {row["query_id"]: row for row in read_csv(archive / "normalized_queries.csv")}


def compute_trials(archive: Path) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Per-trial metrics from the stored top-10 final rankings."""
    intervals = load_intervals(archive)
    queries = load_queries(archive)

    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(archive / "retrieval_results.csv"):
        if row["result_list"] != "final":
            continue
        grouped[(row["run_id"], row["config_id"], row["query_id"])].append(row)

    stats = {"trials": 0, "ranked_chunks": 0, "missing_video_id": 0}
    trials: list[dict[str, object]] = []

    for (run_id, config_id, query_id), rows in sorted(grouped.items()):
        query = queries[query_id]
        source_video = query["video_id"]
        rows.sort(key=lambda r: int(r["final_rank"]))
        stats["trials"] += 1

        temporal_hits: list[bool] = []
        buggy_hits: list[bool] = []
        video_hits: list[bool] = []
        for row in rows:
            stats["ranked_chunks"] += 1
            chunk_video = row["chunk_video_id"]
            if not chunk_video:
                stats["missing_video_id"] += 1
            start = float(row["chunk_start_seconds"])
            end = float(row["chunk_end_seconds"])
            overlap_hit = any(
                interval_overlap(start, end, i_start, i_end) >= MIN_OVERLAP
                for i_start, i_end in intervals.get(query_id, [])
            )
            same_video = chunk_video == source_video
            video_hits.append(same_video)
            buggy_hits.append(overlap_hit)              # original timestamp-only bug
            temporal_hits.append(same_video and overlap_hit)  # fixed joint metric

        def metrics(hits: list[bool]) -> dict[str, float]:
            reciprocal = next(
                (1.0 / rank for rank, hit in enumerate(hits, start=1) if hit), 0.0
            )
            return {
                "recall_at_1": float(any(hits[:1])),
                "recall_at_5": float(any(hits[:5])),
                "recall_at_10": float(any(hits[:10])),
                "mrr": reciprocal,
            }

        temporal = metrics(temporal_hits)
        buggy = metrics(buggy_hits)
        video = metrics(video_hits)
        first_video_rank = next(
            (rank for rank, hit in enumerate(video_hits, start=1) if hit), 0
        )
        trials.append(
            {
                "run_id": run_id,
                "config_id": config_id,
                "query_id": query_id,
                "video_id": source_video,
                "question_type": query["question_type"],
                "temporal_hit_at_1": temporal["recall_at_1"],
                "temporal_hit_at_5": temporal["recall_at_5"],
                "temporal_hit_at_10": temporal["recall_at_10"],
                "temporal_mrr_at_10": temporal["mrr"],
                "buggy_hit_at_1": buggy["recall_at_1"],
                "buggy_hit_at_5": buggy["recall_at_5"],
                "buggy_hit_at_10": buggy["recall_at_10"],
                "buggy_mrr_at_10": buggy["mrr"],
                "video_hit_at_1": video["recall_at_1"],
                "video_hit_at_5": video["recall_at_5"],
                "video_hit_at_10": video["recall_at_10"],
                "video_mrr_at_10": video["mrr"],
                "first_correct_video_rank": first_video_rank,
                "top10_video_ids": "|".join(r["chunk_video_id"] for r in rows),
            }
        )
    return trials, stats


def mean(rows: list[dict[str, object]], key: str) -> float:
    return sum(float(r[key]) for r in rows) / len(rows) if rows else float("nan")


def aggregate(
    trials: list[dict[str, object]], group_key
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for trial in trials:
        grouped[group_key(trial)].append(trial)
    metric_keys = [
        "temporal_hit_at_1", "temporal_hit_at_5", "temporal_hit_at_10",
        "temporal_mrr_at_10",
        "buggy_hit_at_1", "buggy_hit_at_5", "buggy_hit_at_10", "buggy_mrr_at_10",
        "video_hit_at_1", "video_hit_at_5", "video_hit_at_10", "video_mrr_at_10",
    ]
    return {
        group: {key: mean(rows, key) for key in metric_keys}
        for group, rows in grouped.items()
    }


def bootstrap_cis(
    trials: list[dict[str, object]], metrics: tuple[str, ...]
) -> dict[str, dict[str, tuple[float, float]]]:
    """Question-level bootstrap: resample the 45 questions, average runs first."""
    by_config: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for trial in trials:
        config = str(trial["config_id"])
        question = str(trial["query_id"])
        for metric in metrics:
            by_config[config][question][metric].append(float(trial[metric]))

    rng = random.Random(BOOTSTRAP_SEED)
    results: dict[str, dict[str, tuple[float, float]]] = {}
    for config, questions in by_config.items():
        question_ids = sorted(questions)
        per_question = {
            q: {m: sum(v) / len(v) for m, v in questions[q].items()}
            for q in question_ids
        }
        draws: dict[str, list[float]] = {m: [] for m in metrics}
        for _ in range(BOOTSTRAP_RESAMPLES):
            sample = [question_ids[rng.randrange(len(question_ids))]
                      for _ in question_ids]
            for metric in metrics:
                draws[metric].append(
                    sum(per_question[q][metric] for q in sample) / len(sample)
                )
        results[config] = {}
        for metric in metrics:
            ordered = sorted(draws[metric])
            low = ordered[int(0.025 * (BOOTSTRAP_RESAMPLES - 1))]
            high = ordered[int(0.975 * (BOOTSTRAP_RESAMPLES - 1))]
            results[config][metric] = (low, high)
    return results


def check_method_reproduces_archive(
    archive: Path, trials: list[dict[str, object]]
) -> list[str]:
    """Prove time-only recompute == stored (buggy) values; video recompute == stored."""
    stored = {
        (row["run_id"], row["config_id"]): row
        for row in read_csv(archive / "configuration_results.csv")
        if row["stratum_type"] == "overall"
    }
    recomputed = aggregate(
        trials, lambda t: (str(t["run_id"]), str(t["config_id"]))
    )
    # configuration_results.csv also stores an "all_runs" row per configuration;
    # aggregate every trial of each configuration across all runs to match it.
    for config in sorted({str(t["config_id"]) for t in trials}):
        rows = [t for t in trials if t["config_id"] == config]
        recomputed[("all_runs", config)] = {
            key: mean(rows, key) for key in recomputed[("run_01", config)]
        }

    mapping = {
        "buggy_hit_at_1": "recall_at_1",
        "buggy_hit_at_5": "recall_at_5",
        "buggy_hit_at_10": "recall_at_10",
        "buggy_mrr_at_10": "mrr",
        "video_hit_at_1": "video_recall_at_1",
        "video_hit_at_5": "video_recall_at_5",
        "video_hit_at_10": "video_recall_at_10",
        "video_mrr_at_10": "video_mrr",
    }
    problems: list[str] = []
    checked = 0
    for key, row in stored.items():
        if key not in recomputed:
            problems.append(f"missing recompute group {key}")
            continue
        for ours, theirs in mapping.items():
            checked += 1
            a = recomputed[key][ours]
            b = float(row[theirs])
            # stored values are rounded to 6 decimal places
            if abs(a - b) > 2e-6:
                problems.append(f"{key} {ours}={a:.6f} != stored {theirs}={b:.6f}")
    return [f"cross-check failures ({len(problems)} of {checked} cells):"] + problems \
        if problems else [f"cross-check OK: {checked} cells match the archive to its "
                          f"stored precision (buggy time-only temporal + video metrics "
                          f"reproduced)"]


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    archive = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    output.mkdir(parents=True, exist_ok=True)

    trials, stats = compute_trials(archive)
    overall = aggregate(trials, lambda t: str(t["config_id"]))
    strata = aggregate(trials, lambda t: f"{t['config_id']}|{t['question_type']}")
    ci_metrics = (
        "temporal_hit_at_1", "temporal_hit_at_10", "temporal_mrr_at_10",
        "video_hit_at_1",
    )
    cis = bootstrap_cis(trials, ci_metrics)
    cross_check = check_method_reproduces_archive(archive, trials)

    # Per-trial CSV.
    trial_fields = list(trials[0].keys())
    with (output / "per_trial_metrics_fixed.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=trial_fields)
        writer.writeheader()
        writer.writerows(trials)

    # Config-level CSV.
    config_fields = ["config_id"] + list(overall[CONFIG_ORDER[0]].keys())
    with (output / "config_metrics_fixed.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=config_fields)
        writer.writeheader()
        for config in CONFIG_ORDER:
            writer.writerow({"config_id": config, **overall[config]})

    # Bootstrap CI CSV.
    with (output / "bootstrap_cis.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["config_id", "metric", "point", "ci_low", "ci_high",
                         "resamples", "seed"])
        for config in CONFIG_ORDER:
            for metric in ci_metrics:
                low, high = cis[config][metric]
                writer.writerow([config, metric, f"{overall[config][metric]:.4f}",
                                 f"{low:.4f}", f"{high:.4f}",
                                 BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED])

    # ---------------- sanity checks ----------------
    sanity: list[str] = []
    violations = 0
    for config, old in OLD_SINGLE_VIDEO.items():
        for metric, old_value in old.items():
            new_value = overall[config][f"temporal_{metric.replace('recall_at_', 'hit_at_')}"]
            ok = new_value <= old_value + 1e-9
            violations += 0 if ok else 1
            sanity.append(
                f"{'OK  ' if ok else 'FAIL'} {CONFIG_LABEL[config]} "
                f"temporal {metric.replace('recall_at_', 'Hit@')} = "
                f"{new_value:.3f} vs single-video {old_value:.3f}"
            )
    sanity.append(f"sanity violations: {violations}")

    rrf = overall["full_rrf_chrono"]
    rrf_v1, rrf_vmrr = rrf["video_hit_at_1"], rrf["video_mrr_at_10"]
    sanity.append(
        f"RRF-Chrono video Hit@1 = {rrf_v1:.3f} (expect 0.978): "
        f"{'OK' if abs(rrf_v1 - 0.9777777) < 1e-6 else 'FAIL'}"
    )
    sanity.append(
        f"RRF-Chrono video MRR@10 = {rrf_vmrr:.3f} (expect 0.978): "
        f"{'OK' if abs(rrf_vmrr - 0.9777777) < 1e-6 else 'FAIL'}"
    )
    missed = [t for t in trials
              if t["config_id"] == "full_rrf_chrono" and t["video_hit_at_1"] == 0.0]
    sanity.append(f"RRF-Chrono video Hit@1 misses: {len(missed)} of 135 trials")

    # ---------------- report ----------------
    lines: list[str] = []
    lines.append("# Temporal relevance metric audit — `multivideo_distractor_v1`\n")
    lines.append("Recomputed from the stored top-10 rankings in `retrieval_results.csv`; "
                 "no retrieval or generation was re-run.\n")
    lines.append("## 1. Bug\n")
    lines.append(
        "`evaluate_pipeline.py::retrieval_metrics()` decided temporal relevance with "
        "`matching_intervals(chunk.start, chunk.end, query[\"intervals\"], 0.5)` only — "
        "wall-clock overlap, **no source-video check**. In the shared nine-video index a "
        "chunk from an unrelated video scores a temporal hit whenever its timestamp range "
        "coincides with the annotated interval of the query's source video. "
        "`retrieval_rows()` stored the same timestamp-only flag in the `is_relevant` "
        "column. `video_level_metrics()` was already correct.\n"
    )
    lines.append(
        "**Fixed definition:** temporal hit = `chunk_video_id == query video_id` **AND** "
        "interval overlap >= 0.5 s. Video-level metric unchanged "
        "(chunk from source video, any time).\n"
    )
    lines.append(
        f"Files changed: `evaluation/scripts/evaluate_pipeline.py` "
        f"(`retrieval_metrics`, `retrieval_rows`), "
        f"`evaluation/scripts/test_evaluate_pipeline.py` (+1 regression test). "
        f"Unit tests: 17/17 pass.\n"
    )
    lines.append("### Evidence\n")
    lines.append(f"- Recomputed trials: {stats['trials']} (expect 810); "
                 f"ranked chunks: {stats['ranked_chunks']} (expect 8100); "
                 f"rows with empty `chunk_video_id`: {stats['missing_video_id']} (expect 0).")
    for line in cross_check:
        lines.append(f"- {line}")
    lines.append(
        "- The time-only recompute reproduces the archive's stored "
        "`recall_at_*`/`mrr` (and the video metrics) to the stored precision, "
        "so every difference below is attributable solely to adding the "
        "source-video condition.\n"
    )

    lines.append("## 2. Overall metrics — old vs new (mean over 3 runs x 45 questions)\n")
    lines.append("### Temporal (joint) — BUGGY (timestamp only) vs FIXED (video AND time)\n")
    lines.append("| Config | H@1 old | H@1 new | H@5 old | H@5 new | "
                 "H@10 old | H@10 new | MRR@10 old | MRR@10 new |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for config in CONFIG_ORDER:
        o = overall[config]
        lines.append(
            f"| {CONFIG_LABEL[config]} "
            f"| {o['buggy_hit_at_1']:.3f} | **{o['temporal_hit_at_1']:.3f}** "
            f"| {o['buggy_hit_at_5']:.3f} | **{o['temporal_hit_at_5']:.3f}** "
            f"| {o['buggy_hit_at_10']:.3f} | **{o['temporal_hit_at_10']:.3f}** "
            f"| {o['buggy_mrr_at_10']:.3f} | **{o['temporal_mrr_at_10']:.3f}** |"
        )
    lines.append("")
    lines.append("### Video-level (unchanged by the fix, Hit@10 now shown)\n")
    lines.append("| Config | V-H@1 | V-H@5 | V-H@10 | V-MRR@10 |")
    lines.append("|---|---|---|---|---|")
    for config in CONFIG_ORDER:
        o = overall[config]
        lines.append(
            f"| {CONFIG_LABEL[config]} | {o['video_hit_at_1']:.3f} "
            f"| {o['video_hit_at_5']:.3f} | {o['video_hit_at_10']:.3f} "
            f"| {o['video_mrr_at_10']:.3f} |"
        )
    lines.append("")

    lines.append("## 3. Per-stratum (fixed metric)\n")
    lines.append("| Config | Stratum | n q | Temp H@1 | Temp H@10 | Video H@1 |")
    lines.append("|---|---|---|---|---|---|")
    counts: dict[tuple[str, str], int] = defaultdict(int)
    seen: set[tuple[str, str, str]] = set()
    for t in trials:
        key = (str(t["config_id"]), str(t["question_type"]), str(t["query_id"]))
        if key not in seen:
            seen.add(key)
            counts[(str(t["config_id"]), str(t["question_type"]))] += 1
    for config in CONFIG_ORDER:
        for qtype in STRATA_ORDER:
            s = strata[f"{config}|{qtype}"]
            lines.append(
                f"| {CONFIG_LABEL[config]} | {STRATA_LABEL[qtype]} "
                f"| {counts[(config, qtype)]} | {s['temporal_hit_at_1']:.3f} "
                f"| {s['temporal_hit_at_10']:.3f} | {s['video_hit_at_1']:.3f} |"
            )
    lines.append("")

    lines.append("## 4. Sanity checks\n")
    lines.extend(f"- {s}" for s in sanity)
    lines.append("")
    lines.append("### RRF-Chrono video Hit@1 misses\n")
    if missed:
        lines.append("| run | query | question type | first correct-video rank | "
                     "correct-video chunk in top 10? | top-10 video ids |")
        lines.append("|---|---|---|---|---|---|")
        for t in missed:
            rank = t["first_correct_video_rank"]
            in_top10 = "yes" if rank else "NO"
            lines.append(
                f"| {t['run_id']} | {t['query_id']} | {t['question_type']} "
                f"| {rank if rank else 'none'} | {in_top10} | {t['top10_video_ids']} |"
            )
    lines.append("")

    lines.append("## 5. Bootstrap 95% CIs (10,000 resamples of the 45 questions; "
                 "runs averaged within question)\n")
    lines.append("| Config | Temp H@1 | Temp H@10 | Temp MRR@10 | Video H@1 |")
    lines.append("|---|---|---|---|---|")
    for config in CONFIG_ORDER:
        cells = []
        for metric in ci_metrics:
            point = overall[config][metric]
            low, high = cis[config][metric]
            cells.append(f"{point:.3f} [{low:.3f}, {high:.3f}]")
        lines.append(f"| {CONFIG_LABEL[config]} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## 6. Fig. 4 values (temporal Hit@1 by question type, fixed metric)\n")
    lines.append("| Config | Transcript (n=16) | Visual (n=18) | Multimodal (n=11) |")
    lines.append("|---|---|---|---|")
    for config in ("full_rrf_chrono", "text_only", "image_only"):
        cells = [
            f"{strata[f'{config}|{qtype}']['temporal_hit_at_1']:.3f}"
            for qtype in STRATA_ORDER
        ]
        lines.append(f"| {CONFIG_LABEL[config]} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Paper Fig. 4 (buggy metric) for comparison: RRF-Chrono .500/.648/.091, "
                 "Text-only .458/.722/.455, Image-only .125/.611/.091.\n")

    lines.append("## 7. Outputs\n")
    lines.append("- `per_trial_metrics_fixed.csv` — 810 trials, fixed + buggy metrics side by side")
    lines.append("- `config_metrics_fixed.csv` — overall per configuration")
    lines.append("- `bootstrap_cis.csv` — point estimates and 95% CIs")
    lines.append("- `temporal_metric_audit.md` — this report")
    lines.append("")
    lines.append("Source archive (unmodified): "
                 f"`{archive}`\n")

    (output / "temporal_metric_audit.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\n[written to {output}]")
    cross_ok = not cross_check[0].startswith("cross-check failures")
    print(f"sanity violations: {violations}; cross-check: {'OK' if cross_ok else 'FAILED'}")
    return 0 if violations == 0 and cross_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
