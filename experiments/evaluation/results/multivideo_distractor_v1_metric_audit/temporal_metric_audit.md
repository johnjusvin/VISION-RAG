# Temporal relevance metric audit — `multivideo_distractor_v1`

Recomputed from the stored top-10 rankings in `retrieval_results.csv`; no retrieval or generation was re-run.

## 1. Bug

`evaluate_pipeline.py::retrieval_metrics()` decided temporal relevance with `matching_intervals(chunk.start, chunk.end, query["intervals"], 0.5)` only — wall-clock overlap, **no source-video check**. In the shared nine-video index a chunk from an unrelated video scores a temporal hit whenever its timestamp range coincides with the annotated interval of the query's source video. `retrieval_rows()` stored the same timestamp-only flag in the `is_relevant` column. `video_level_metrics()` was already correct.

**Fixed definition:** temporal hit = `chunk_video_id == query video_id` **AND** interval overlap >= 0.5 s. Video-level metric unchanged (chunk from source video, any time).

Files changed: `evaluation/scripts/evaluate_pipeline.py` (`retrieval_metrics`, `retrieval_rows`), `evaluation/scripts/test_evaluate_pipeline.py` (+1 regression test). Unit tests: 17/17 pass.

### Evidence

- Recomputed trials: 810 (expect 810); ranked chunks: 8100 (expect 8100); rows with empty `chunk_video_id`: 0 (expect 0).
- cross-check OK: 192 cells match the archive to its stored precision (buggy time-only temporal + video metrics reproduced)
- The time-only recompute reproduces the archive's stored `recall_at_*`/`mrr` (and the video metrics) to the stored precision, so every difference below is attributable solely to adding the source-video condition.

## 2. Overall metrics — old vs new (mean over 3 runs x 45 questions)

### Temporal (joint) — BUGGY (timestamp only) vs FIXED (video AND time)

| Config | H@1 old | H@1 new | H@5 old | H@5 new | H@10 old | H@10 new | MRR@10 old | MRR@10 new |
|---|---|---|---|---|---|---|---|---|
| RRF-Chrono | 0.459 | **0.444** | 0.822 | **0.756** | 0.867 | **0.800** | 0.612 | **0.572** |
| Image-only | 0.311 | **0.244** | 0.511 | **0.422** | 0.689 | **0.578** | 0.414 | **0.337** |
| Raw fusion | 0.548 | **0.526** | 0.681 | **0.659** | 0.889 | **0.844** | 0.639 | **0.614** |
| RRF-Rel. order | 0.474 | **0.452** | 0.815 | **0.748** | 0.859 | **0.793** | 0.618 | **0.573** |
| Text-only | 0.563 | **0.541** | 0.674 | **0.652** | 0.881 | **0.837** | 0.643 | **0.618** |
| Whisper-medium | 0.400 | **0.400** | 0.844 | **0.778** | 0.933 | **0.889** | 0.600 | **0.568** |

### Video-level (unchanged by the fix, Hit@10 now shown)

| Config | V-H@1 | V-H@5 | V-H@10 | V-MRR@10 |
|---|---|---|---|---|
| RRF-Chrono | 0.978 | 0.978 | 0.978 | 0.978 |
| Image-only | 0.778 | 0.911 | 0.911 | 0.830 |
| Raw fusion | 0.933 | 0.978 | 0.978 | 0.945 |
| RRF-Rel. order | 0.978 | 0.978 | 0.978 | 0.978 |
| Text-only | 0.933 | 0.978 | 0.978 | 0.945 |
| Whisper-medium | 0.933 | 0.956 | 0.978 | 0.948 |

## 3. Per-stratum (fixed metric)

| Config | Stratum | n q | Temp H@1 | Temp H@10 | Video H@1 |
|---|---|---|---|---|---|
| RRF-Chrono | Transcript | 16 | 0.500 | 0.938 | 1.000 |
| RRF-Chrono | Visual | 18 | 0.611 | 0.778 | 0.944 |
| RRF-Chrono | Multimodal | 11 | 0.091 | 0.636 | 1.000 |
| Image-only | Transcript | 16 | 0.062 | 0.500 | 0.750 |
| Image-only | Visual | 18 | 0.500 | 0.667 | 0.778 |
| Image-only | Multimodal | 11 | 0.091 | 0.545 | 0.818 |
| Raw fusion | Transcript | 16 | 0.479 | 0.938 | 1.000 |
| Raw fusion | Visual | 18 | 0.667 | 0.778 | 0.944 |
| Raw fusion | Multimodal | 11 | 0.364 | 0.818 | 0.818 |
| RRF-Rel. order | Transcript | 16 | 0.521 | 0.917 | 1.000 |
| RRF-Rel. order | Visual | 18 | 0.611 | 0.778 | 0.944 |
| RRF-Rel. order | Multimodal | 11 | 0.091 | 0.636 | 1.000 |
| Text-only | Transcript | 16 | 0.458 | 0.917 | 1.000 |
| Text-only | Visual | 18 | 0.722 | 0.778 | 0.944 |
| Text-only | Multimodal | 11 | 0.364 | 0.818 | 0.818 |
| Whisper-medium | Transcript | 16 | 0.500 | 1.000 | 1.000 |
| Whisper-medium | Visual | 18 | 0.556 | 0.889 | 0.944 |
| Whisper-medium | Multimodal | 11 | 0.000 | 0.727 | 0.818 |

## 4. Sanity checks

- OK   Image-only temporal Hit@1 = 0.244 vs single-video 0.289
- OK   Image-only temporal Hit@5 = 0.422 vs single-video 0.444
- OK   Image-only temporal Hit@10 = 0.578 vs single-video 0.622
- OK   Text-only temporal Hit@1 = 0.541 vs single-video 0.556
- OK   Text-only temporal Hit@5 = 0.652 vs single-video 0.733
- OK   Text-only temporal Hit@10 = 0.837 vs single-video 0.911
- sanity violations: 0
- RRF-Chrono video Hit@1 = 0.978 (expect 0.978): OK
- RRF-Chrono video MRR@10 = 0.978 (expect 0.978): OK
- RRF-Chrono video Hit@1 misses: 3 of 135 trials

### RRF-Chrono video Hit@1 misses

| run | query | question type | first correct-video rank | correct-video chunk in top 10? | top-10 video ids |
|---|---|---|---|---|---|
| run_01 | q029 | visual_dependent | none | NO | vid_003|vid_001|vid_003|vid_001|vid_008|vid_004|vid_001|vid_003|vid_002|vid_003 |
| run_02 | q029 | visual_dependent | none | NO | vid_001|vid_003|vid_001|vid_008|vid_004|vid_003|vid_001|vid_003|vid_002|vid_003 |
| run_03 | q029 | visual_dependent | none | NO | vid_003|vid_001|vid_003|vid_001|vid_008|vid_004|vid_001|vid_003|vid_002|vid_003 |

## 5. Bootstrap 95% CIs (10,000 resamples of the 45 questions; runs averaged within question)

| Config | Temp H@1 | Temp H@10 | Temp MRR@10 | Video H@1 |
|---|---|---|---|---|
| RRF-Chrono | 0.444 [0.311, 0.600] | 0.800 [0.667, 0.911] | 0.572 [0.450, 0.692] | 0.978 [0.933, 1.000] |
| Image-only | 0.244 [0.133, 0.378] | 0.578 [0.422, 0.711] | 0.337 [0.222, 0.460] | 0.778 [0.644, 0.889] |
| Raw fusion | 0.526 [0.378, 0.667] | 0.844 [0.733, 0.933] | 0.614 [0.488, 0.739] | 0.933 [0.844, 1.000] |
| RRF-Rel. order | 0.452 [0.311, 0.600] | 0.793 [0.667, 0.904] | 0.573 [0.453, 0.694] | 0.978 [0.933, 1.000] |
| Text-only | 0.541 [0.400, 0.681] | 0.837 [0.726, 0.933] | 0.618 [0.491, 0.738] | 0.933 [0.844, 1.000] |
| Whisper-medium | 0.400 [0.267, 0.533] | 0.889 [0.800, 0.978] | 0.568 [0.453, 0.678] | 0.933 [0.844, 1.000] |

## 6. Fig. 4 values (temporal Hit@1 by question type, fixed metric)

| Config | Transcript (n=16) | Visual (n=18) | Multimodal (n=11) |
|---|---|---|---|
| RRF-Chrono | 0.500 | 0.611 | 0.091 |
| Text-only | 0.458 | 0.722 | 0.364 |
| Image-only | 0.062 | 0.500 | 0.091 |

Paper Fig. 4 (buggy metric) for comparison: RRF-Chrono .500/.648/.091, Text-only .458/.722/.455, Image-only .125/.611/.091.

## 7. Outputs

- `per_trial_metrics_fixed.csv` — 810 trials, fixed + buggy metrics side by side
- `config_metrics_fixed.csv` — overall per configuration
- `bootstrap_cis.csv` — point estimates and 95% CIs
- `temporal_metric_audit.md` — this report

Source archive (unmodified): `/home/jusvin-britto/V-RAG/evaluation/results/multivideo_distractor_v1`
