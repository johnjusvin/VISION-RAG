# VisionRAG Nine-Video Pilot Evidence Report

Experiment ID: `multivideo_distractor_v1`

> Generated from the raw CSV evidence in this directory. This is a pilot evaluation, not an external benchmark.

| Run | Configuration | N | R@1 | R@5 | R@10 | MRR@10 | vR@1 | vR@5 | vR@10 | vMRR | BLEU-1 | Token F1 | BERTScore F1 | Gen. failures |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all_runs | `full_rrf_chrono` | 135 | 0.459 | 0.822 | 0.867 | 0.612 | 0.978 | 0.978 | 0.978 | 0.978 | 0.136 | 0.204 | 0.862 | 0 |
| all_runs | `image_only` | 135 | 0.311 | 0.511 | 0.689 | 0.414 | 0.778 | 0.911 | 0.911 | 0.830 | 0.063 | 0.109 | 0.837 | 0 |
| all_runs | `raw_score_fusion` | 135 | 0.548 | 0.681 | 0.889 | 0.639 | 0.933 | 0.978 | 0.978 | 0.945 | 0.124 | 0.190 | 0.860 | 0 |
| all_runs | `rrf_relevance_order` | 135 | 0.474 | 0.815 | 0.859 | 0.618 | 0.978 | 0.978 | 0.978 | 0.978 | 0.148 | 0.216 | 0.864 | 0 |
| all_runs | `text_only` | 135 | 0.563 | 0.674 | 0.881 | 0.643 | 0.933 | 0.978 | 0.978 | 0.945 | 0.215 | 0.294 | 0.878 | 0 |
| all_runs | `whisper_medium` | 135 | 0.400 | 0.844 | 0.933 | 0.600 | 0.933 | 0.956 | 0.978 | 0.948 | 0.121 | 0.182 | 0.858 | 0 |
| run_01 | `full_rrf_chrono` | 45 | 0.467 | 0.822 | 0.867 | 0.616 | 0.978 | 0.978 | 0.978 | 0.978 | 0.136 | 0.204 | 0.863 | 0 |
| run_01 | `image_only` | 45 | 0.311 | 0.511 | 0.689 | 0.414 | 0.778 | 0.911 | 0.911 | 0.830 | 0.063 | 0.109 | 0.837 | 0 |
| run_01 | `raw_score_fusion` | 45 | 0.556 | 0.689 | 0.889 | 0.645 | 0.933 | 0.978 | 0.978 | 0.945 | 0.124 | 0.189 | 0.860 | 0 |
| run_01 | `rrf_relevance_order` | 45 | 0.467 | 0.800 | 0.844 | 0.609 | 0.978 | 0.978 | 0.978 | 0.978 | 0.148 | 0.217 | 0.864 | 0 |
| run_01 | `text_only` | 45 | 0.556 | 0.667 | 0.889 | 0.636 | 0.933 | 0.978 | 0.978 | 0.945 | 0.211 | 0.288 | 0.876 | 0 |
| run_01 | `whisper_medium` | 45 | 0.400 | 0.844 | 0.933 | 0.600 | 0.933 | 0.956 | 0.978 | 0.948 | 0.121 | 0.182 | 0.858 | 0 |
| run_02 | `full_rrf_chrono` | 45 | 0.444 | 0.822 | 0.867 | 0.605 | 0.978 | 0.978 | 0.978 | 0.978 | 0.135 | 0.204 | 0.862 | 0 |
| run_02 | `image_only` | 45 | 0.311 | 0.511 | 0.689 | 0.414 | 0.778 | 0.911 | 0.911 | 0.830 | 0.063 | 0.109 | 0.837 | 0 |
| run_02 | `raw_score_fusion` | 45 | 0.533 | 0.667 | 0.889 | 0.625 | 0.933 | 0.978 | 0.978 | 0.945 | 0.124 | 0.188 | 0.858 | 0 |
| run_02 | `rrf_relevance_order` | 45 | 0.467 | 0.822 | 0.867 | 0.616 | 0.978 | 0.978 | 0.978 | 0.978 | 0.149 | 0.218 | 0.864 | 0 |
| run_02 | `text_only` | 45 | 0.578 | 0.689 | 0.889 | 0.657 | 0.933 | 0.978 | 0.978 | 0.945 | 0.220 | 0.300 | 0.879 | 0 |
| run_02 | `whisper_medium` | 45 | 0.400 | 0.844 | 0.933 | 0.600 | 0.933 | 0.956 | 0.978 | 0.948 | 0.121 | 0.182 | 0.858 | 0 |
| run_03 | `full_rrf_chrono` | 45 | 0.467 | 0.822 | 0.867 | 0.616 | 0.978 | 0.978 | 0.978 | 0.978 | 0.135 | 0.203 | 0.862 | 0 |
| run_03 | `image_only` | 45 | 0.311 | 0.511 | 0.689 | 0.414 | 0.778 | 0.911 | 0.911 | 0.830 | 0.063 | 0.109 | 0.837 | 0 |
| run_03 | `raw_score_fusion` | 45 | 0.556 | 0.689 | 0.889 | 0.645 | 0.933 | 0.978 | 0.978 | 0.945 | 0.125 | 0.192 | 0.861 | 0 |
| run_03 | `rrf_relevance_order` | 45 | 0.489 | 0.822 | 0.867 | 0.629 | 0.978 | 0.978 | 0.978 | 0.978 | 0.145 | 0.213 | 0.863 | 0 |
| run_03 | `text_only` | 45 | 0.556 | 0.667 | 0.867 | 0.634 | 0.933 | 0.978 | 0.978 | 0.945 | 0.215 | 0.292 | 0.878 | 0 |
| run_03 | `whisper_medium` | 45 | 0.400 | 0.844 | 0.933 | 0.600 | 0.933 | 0.956 | 0.978 | 0.948 | 0.121 | 0.182 | 0.858 | 0 |

## Evidence boundary

- Corpus: 9 videos and 45 questions; 3 repetition(s).
- Token F1 and BERTScore are separate metrics.
- Human answer quality is pending until two independent annotators complete the blinded packet.
- Hosted cost is unmeasured unless an actual billed value was returned; no price is inferred.
- Video licenses must be verified before redistribution.
