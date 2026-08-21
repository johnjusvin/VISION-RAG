from __future__ import annotations

import tempfile
import unittest
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import evaluate_pipeline as evaluation
from vision_rag.embedding import EmbeddedChunk
from vision_rag.vectorstores import SearchResult


def result(chunk_id: int, score: float) -> SearchResult:
    chunk = EmbeddedChunk(
        chunk_id=chunk_id,
        video_path="video.mp4",
        start=float(chunk_id * 5),
        end=float(chunk_id * 5 + 5),
        duration=5.0,
        frame_path=f"frame_{chunk_id}.jpg",
        text=f"chunk {chunk_id}",
    )
    return SearchResult(chunk, score)


class MetricTests(unittest.TestCase):
    def test_token_f1_is_bag_of_tokens(self):
        self.assertAlmostEqual(
            evaluation.compute_token_f1("red red ball", "red ball"), 0.8
        )

    def test_bleu1_exact_match(self):
        self.assertAlmostEqual(evaluation.compute_bleu1("a useful answer", "a useful answer"), 1.0)

    def test_interval_threshold(self):
        intervals = [(10.0, 12.0), (30.0, 35.0)]
        self.assertEqual(
            evaluation.matching_intervals(9.5, 10.5, intervals, 0.5),
            [(10.0, 12.0)],
        )
        self.assertEqual(
            evaluation.matching_intervals(9.6, 10.4, intervals, 0.5), []
        )

    def test_retrieval_metrics_at_requested_cutoffs(self):
        ranked = [
            {"chunk": result(index, 1.0).chunk}
            for index in range(10)
        ]
        query = {"intervals": [(35.0, 40.0)]}
        metrics = evaluation.retrieval_metrics(ranked, query, 0.5)
        self.assertEqual(metrics["recall_at_1"], 0.0)
        self.assertEqual(metrics["recall_at_5"], 0.0)
        self.assertEqual(metrics["recall_at_10"], 1.0)
        self.assertAlmostEqual(metrics["mrr"], 1.0 / 8.0)


class FusionTests(unittest.TestCase):
    def test_rrf_rewards_chunk_present_in_both_rankings(self):
        text = [result(1, 0.9), result(2, 0.8)]
        image = [result(2, 0.2), result(3, 0.99)]
        fused = evaluation.reciprocal_rank_fusion(text, image, rrf_k=60)
        self.assertEqual(fused[0]["chunk"].chunk_id, 2)
        self.assertEqual(fused[0]["text_rank"], 2)
        self.assertEqual(fused[0]["image_rank"], 1)

    def test_raw_score_baseline_uses_highest_unscaled_score(self):
        text = [result(1, 0.7)]
        image = [result(2, 0.95)]
        fused = evaluation.raw_score_fusion(text, image)
        self.assertEqual([item["chunk"].chunk_id for item in fused], [2, 1])

    def test_text_only_generation_does_not_receive_frames(self):
        ranked = evaluation.single_modality_ranking([result(1, 0.9)], "text")
        sources = evaluation.generation_sources(
            ranked, evaluation.CONFIGURATIONS["text_only"], top_k=1
        )
        self.assertIsNotNone(sources[0].text)
        self.assertIsNone(sources[0].frame_path)

    def test_image_only_generation_does_not_receive_transcript(self):
        ranked = evaluation.single_modality_ranking([result(1, 0.9)], "image")
        sources = evaluation.generation_sources(
            ranked, evaluation.CONFIGURATIONS["image_only"], top_k=1
        )
        self.assertIsNone(sources[0].text)
        self.assertIsNotNone(sources[0].frame_path)


class DatasetTests(unittest.TestCase):
    def test_repository_pilot_dataset(self):
        videos, queries = evaluation.load_dataset(
            evaluation.PROJECT_ROOT / "data" / "videos.csv",
            evaluation.PROJECT_ROOT / "data" / "queries.csv",
            None,
            expected_videos=9,
            expected_queries=45,
        )
        self.assertEqual(len(videos), 9)
        self.assertEqual(len(queries), 45)
        self.assertEqual(
            {query["question_type"] for query in queries},
            evaluation.QUESTION_TYPES,
        )

    def test_duplicate_query_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos_path = root / "videos.csv"
            queries_path = root / "queries.csv"
            videos_path.write_text(
                "video_id,file_path,title,domain,duration_seconds,language,source_url,license,has_audio,has_burned_captions,notes\n"
                "v1,x.mp4,T,d,10,en,u,l,true,false,n\n",
                encoding="utf-8",
            )
            queries_path.write_text(
                "query_id,video_id,question,question_type,gt_start,gt_end,reference_answer,annotator_id\n"
                "q1,v1,Q,visual_dependent,0,2,A,a1\n"
                "q1,v1,Q2,visual_dependent,2,4,A2,a1\n",
                encoding="utf-8",
            )
            with self.assertRaises(evaluation.EvaluationConfigError):
                evaluation.load_dataset(
                    videos_path,
                    queries_path,
                    None,
                    expected_videos=1,
                    expected_queries=2,
                )


class ArtifactTests(unittest.TestCase):
    def test_release_asr_model_clears_loaded_backend(self):
        class FakeASR:
            _model = object()

        asr = FakeASR()
        evaluation.release_asr_model(asr)
        self.assertIsNone(asr._model)

    def test_only_transient_generation_errors_are_retried(self):
        class ProviderError(RuntimeError):
            def __init__(self, status_code):
                self.status_code = status_code

        self.assertTrue(evaluation._is_retryable_generation_error(ProviderError(500)))
        self.assertTrue(evaluation._is_retryable_generation_error(ProviderError(503)))
        self.assertFalse(evaluation._is_retryable_generation_error(ProviderError(400)))

    def test_provider_metadata_is_json_serializable(self):
        value = evaluation._json_safe(
            {"modified_at": datetime(2026, 8, 21, tzinfo=timezone.utc)}
        )
        self.assertEqual(value["modified_at"], "2026-08-21T00:00:00+00:00")

    def test_fatal_error_marks_existing_manifest_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "exp"
            run.mkdir()
            manifest = run / "experiment_manifest.json"
            manifest.write_text('{"status":"running"}', encoding="utf-8")
            args = argparse.Namespace(output_dir=str(root), experiment_id="exp")
            evaluation._mark_manifest_incomplete(args, RuntimeError("failure"))
            result = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(result["fatal_error_type"], "RuntimeError")

    def test_latency_aggregation_includes_all_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sink = evaluation.CsvSink(root / "timings.csv", evaluation.TIMING_FIELDS)
            for run_id, duration in (("run_01", 10.0), ("run_02", 20.0)):
                sink.write(
                    {
                        "experiment_id": "exp", "run_id": run_id,
                        "config_id": "full_rrf_chrono", "query_id": "q1",
                        "video_id": "v1", "scope": "query",
                        "total_pipeline_ms": duration,
                        "cache_state": "not_applicable", "success": True,
                    }
                )
            sink.close()
            evaluation.aggregate_timings(root, "exp")
            with (root / "latency_results.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            aggregate = next(
                row for row in rows
                if row["run_id"] == "all_runs"
                and row["metric"] == "total_pipeline_ms"
            )
            self.assertEqual(aggregate["observations"], "2")
            self.assertEqual(aggregate["mean"], "15.000000")

    def test_aggregate_results_keeps_overall_type_and_domain_strata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = evaluation.CsvSink(
                root / "query_metrics.csv", evaluation.QUERY_METRIC_FIELDS
            )
            answers = evaluation.CsvSink(
                root / "answers.csv", evaluation.ANSWER_FIELDS
            )
            for query_id, question_type, domain, hit in (
                ("q1", "visual_dependent", "lecture", 1.0),
                ("q2", "multimodal", "tutorial", 0.0),
            ):
                identity = {
                    "experiment_id": "exp",
                    "run_id": "run_01",
                    "config_id": "full_rrf_chrono",
                    "query_id": query_id,
                    "video_id": "v1",
                }
                metrics.write(
                    {
                        **identity,
                        "domain": domain,
                        "question_type": question_type,
                        "recall_at_1": hit,
                        "recall_at_5": hit,
                        "recall_at_10": hit,
                        "mrr": hit,
                        "retrieval_success": True,
                        "generation_success": True,
                        "bleu1": hit,
                        "token_f1": hit,
                    }
                )
                answers.write(
                    {
                        **identity,
                        "generated_answer": "answer",
                        "reference_answer": "answer",
                        "generator_provider": "Ollama",
                        "bleu1": hit,
                        "token_f1": hit,
                    }
                )
            metrics.close()
            answers.close()
            rows = evaluation.aggregate_results(root, "exp")
            self.assertEqual(len(rows), 10)
            overall = next(
                row for row in rows
                if row["run_id"] == "run_01" and row["stratum_type"] == "overall"
            )
            self.assertEqual(overall["queries"], 2)
            self.assertAlmostEqual(overall["recall_at_1"], 0.5)
            all_runs = next(
                row for row in rows
                if row["run_id"] == "all_runs" and row["stratum_type"] == "overall"
            )
            self.assertEqual(all_runs["queries"], 2)


if __name__ == "__main__":
    unittest.main()
