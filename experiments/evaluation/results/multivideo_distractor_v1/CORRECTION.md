# Correction note — temporal relevance metrics in `multivideo_distractor_v1/`

This directory contains the raw evidence for the VisionRAG nine-video shared-index
experiment. Four of its summary files report temporal relevance numbers that were
computed **before** a bug fix in the evaluation code. Those temporal numbers are
superseded. Corrected values are in [`corrected_metrics/`](corrected_metrics/).

Nothing in this directory has been modified or deleted; the corrected files were added
alongside the existing ones.

## 1. What was wrong

In the original evaluation code (commit `54b2d75`,
`experiments/evaluation/scripts/evaluate_pipeline.py`), both
`retrieval_metrics()` (then at line 1084) and `retrieval_rows()` (then at line 1001)
decided relevance with **timestamp overlap only**:

```python
matches = matching_intervals(
    chunk.start, chunk.end, query["intervals"], minimum_overlap
)
```

A chunk therefore counted as a temporal hit whenever it overlapped an annotated support
interval **by the required number of seconds, even if the chunk came from a different
video**. Because the nine videos were indexed together and eight of them are distractors,
a coincidental overlap in another video could be scored as a hit. `retrieval_rows()` wrote
the same overlap-only test into the `is_relevant` column of `retrieval_results.csv`.

## 2. The fix

Commit `1da653a` ("Publish corrected temporal-relevance evaluation code and metric audit")
changed the criterion to require **both**:

1. the chunk's source video must be the question's source video, and
2. the chunk must overlap an annotated support interval of that video by at least
   **0.5 seconds** (the `--minimum-overlap` default,
   `evaluate_pipeline.py` line 2656: `parser.add_argument("--minimum-overlap", type=float, default=0.5)`).

Current code (`evaluate_pipeline.py` at `HEAD = 1da653a`):

- `retrieval_metrics()` — line 1115; docstring lines 1124-1128 state that a chunk counts
  only when it "comes from the source video *and* overlaps an annotated support interval by
  at least `overlap` seconds"; source-video test at line 1131:
  `path_to_video_id.get(item["chunk"].video_path, "") == query_video_id`.
- `retrieval_rows()` — line 1021; the relevance decision now passes through the same joint
  test (line 1047 → line 1083).
- Unit tests: `experiments/evaluation/scripts/test_evaluate_pipeline.py` — 17 tests,
  including `test_temporal_relevance_requires_source_video` (line 62).

## 3. Which files are superseded

The temporal metric columns of these files **in this directory** were produced by the
pre-fix code and are superseded by `corrected_metrics/`:

| File | Superseded columns |
|---|---|
| `configuration_results.csv` | `recall_at_1`, `recall_at_5`, `recall_at_10`, `mrr` |
| `query_metrics.csv` | `recall_at_1`, `recall_at_5`, `recall_at_10`, `mrr` |
| `report.md` | `R@1`, `R@5`, `R@10`, `MRR@10` |
| `retrieval_results.csv` | the `is_relevant` / `matched_intervals` columns (overlap-only test) |

Example of the difference for `full_rrf_chrono` (all runs): the pre-fix report gives
`R@10 = 0.867`, while the corrected temporal Hit@10 is `0.800`
(`corrected_metrics/config_metrics_fixed.csv`).

## 4. What the paper uses

Tables III and IV and Figs. 3 and 4 of *VisionRAG: A Modular, Pluggable Python Library for
Video Retrieval-Augmented Generation* report the **corrected** values, taken from
`corrected_metrics/`:

- `corrected_metrics/per_trial_metrics_fixed.csv` — per-run trial metrics (standard
  deviations, Fig. 4 question-type strata);
- `corrected_metrics/config_metrics_fixed.csv` — the three-run means in Table III;
- `corrected_metrics/bootstrap_cis.csv` — the 95 % percentile intervals in Table IV
  (10 000 question resamples, seed 1729);
- `corrected_metrics/temporal_metric_audit.md` — the audit narrative and the tables as
  published.

## 5. What was NOT affected

- **Video-level metrics** (source-video Hit@k / MRR@10, i.e. distractor rejection) never
  depended on the timestamp test; `video_recall_at_*` / `video_mrr` in
  `configuration_results.csv` and `query_metrics.csv` are unchanged and still valid.
- **Ranked chunks** — the retrieval order itself (which chunk is at which rank) was never
  changed by the relevance test; only the `is_relevant` labels derived from it were wrong.
- **Generated answers** — answer text and every answer-similarity score (BLEU-1, token F1,
  BERTScore F1) are independent of the relevance test and are unchanged.

## 6. Provenance of `corrected_metrics/`

The four files are byte-identical copies of
`../multivideo_distractor_v1_metric_audit/`, which was produced by
`experiments/evaluation/scripts/audit_temporal_relevance.py` (commit `1da653a`):

```
bootstrap_cis.csv
config_metrics_fixed.csv
per_trial_metrics_fixed.csv
temporal_metric_audit.md
```
