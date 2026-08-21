# VisionRAG Nine-Video Pilot Evidence Report

Experiment ID: `paper_pilot_cuda_final_v3`

> Generated from the raw CSV evidence in this directory. This is a pilot evaluation, not an external benchmark.

| Run | Configuration | N | R@1 | R@5 | R@10 | MRR@10 | BLEU-1 | Token F1 | BERTScore F1 | Gen. failures |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all_runs | `full_rrf_chrono` | 135 | 0.459 | 0.756 | 0.844 | 0.592 | 0.133 | 0.201 | 0.862 | 0 |
| all_runs | `image_only` | 135 | 0.289 | 0.444 | 0.622 | 0.376 | 0.060 | 0.104 | 0.834 | 0 |
| all_runs | `raw_score_fusion` | 135 | 0.519 | 0.726 | 0.911 | 0.635 | 0.120 | 0.186 | 0.860 | 0 |
| all_runs | `rrf_relevance_order` | 135 | 0.467 | 0.756 | 0.844 | 0.598 | 0.152 | 0.221 | 0.865 | 0 |
| all_runs | `text_only` | 135 | 0.556 | 0.733 | 0.911 | 0.656 | 0.209 | 0.292 | 0.879 | 0 |
| all_runs | `whisper_medium` | 135 | 0.422 | 0.822 | 0.933 | 0.596 | 0.121 | 0.183 | 0.859 | 0 |
| run_01 | `full_rrf_chrono` | 45 | 0.444 | 0.756 | 0.844 | 0.580 | 0.132 | 0.201 | 0.862 | 0 |
| run_01 | `image_only` | 45 | 0.289 | 0.444 | 0.622 | 0.376 | 0.060 | 0.104 | 0.834 | 0 |
| run_01 | `raw_score_fusion` | 45 | 0.533 | 0.733 | 0.911 | 0.645 | 0.121 | 0.187 | 0.861 | 0 |
| run_01 | `rrf_relevance_order` | 45 | 0.467 | 0.756 | 0.844 | 0.598 | 0.152 | 0.220 | 0.864 | 0 |
| run_01 | `text_only` | 45 | 0.556 | 0.733 | 0.911 | 0.656 | 0.202 | 0.282 | 0.877 | 0 |
| run_01 | `whisper_medium` | 45 | 0.422 | 0.822 | 0.933 | 0.596 | 0.121 | 0.183 | 0.859 | 0 |
| run_02 | `full_rrf_chrono` | 45 | 0.467 | 0.756 | 0.844 | 0.598 | 0.132 | 0.201 | 0.862 | 0 |
| run_02 | `image_only` | 45 | 0.289 | 0.444 | 0.622 | 0.376 | 0.060 | 0.104 | 0.834 | 0 |
| run_02 | `raw_score_fusion` | 45 | 0.511 | 0.733 | 0.911 | 0.634 | 0.118 | 0.183 | 0.859 | 0 |
| run_02 | `rrf_relevance_order` | 45 | 0.467 | 0.756 | 0.844 | 0.598 | 0.152 | 0.221 | 0.865 | 0 |
| run_02 | `text_only` | 45 | 0.556 | 0.733 | 0.911 | 0.656 | 0.213 | 0.297 | 0.879 | 0 |
| run_02 | `whisper_medium` | 45 | 0.422 | 0.822 | 0.933 | 0.596 | 0.121 | 0.183 | 0.859 | 0 |
| run_03 | `full_rrf_chrono` | 45 | 0.467 | 0.756 | 0.844 | 0.598 | 0.134 | 0.203 | 0.862 | 0 |
| run_03 | `image_only` | 45 | 0.289 | 0.444 | 0.622 | 0.376 | 0.060 | 0.104 | 0.834 | 0 |
| run_03 | `raw_score_fusion` | 45 | 0.511 | 0.711 | 0.911 | 0.625 | 0.121 | 0.188 | 0.860 | 0 |
| run_03 | `rrf_relevance_order` | 45 | 0.467 | 0.756 | 0.844 | 0.598 | 0.152 | 0.220 | 0.864 | 0 |
| run_03 | `text_only` | 45 | 0.556 | 0.733 | 0.911 | 0.656 | 0.212 | 0.296 | 0.880 | 0 |
| run_03 | `whisper_medium` | 45 | 0.422 | 0.822 | 0.933 | 0.596 | 0.121 | 0.183 | 0.859 | 0 |

## Evidence boundary

- Corpus: 9 videos and 45 questions; 3 repetition(s).
- Token F1 and BERTScore are separate metrics.
- Human answer quality is pending until two independent annotators complete the blinded packet.
- Hosted cost is unmeasured unless an actual billed value was returned; no price is inferred.
- Video licenses must be verified before redistribution.
