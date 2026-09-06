"""Offline regressions for authorization, candidate reranking and experiment isolation."""

import os
import asyncio
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from weaviate.classes.query import Filter, HybridFusion

from rag_chatbot.chat_pipeline import retrieve_documents_with_resources
from rag_chatbot.config import load_settings
from rag_chatbot.reranking import rerank_documents, get_reranker, _cached_reranker
from evaluation.langsmith_eval import variant_settings, experiment_metadata


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"LANGSMITH_TRACING": "false"})
        self.env.start()
        with patch.dict(os.environ, {}, clear=True):
            self.settings = load_settings()
        self.client = Mock()
        self.query = self.client.collections.use.return_value.query
        self.embeddings = Mock()
        self.embeddings.embed_query.return_value = [0.1, 0.2]
        self.reranker = Mock()
        self.reranker.predict.return_value = [0.2, 0.9, 0.4]
        self.objects = [SimpleNamespace(properties={
            "text": f"Policy passage {i}", "document_id": f"HR-00{i}",
            "page_number": i, "source_file": f"policy-{i}.pdf",
        }, metadata=SimpleNamespace(score=1 / i)) for i in (1, 2, 3)]
        self.query.hybrid.return_value = SimpleNamespace(objects=self.objects)
        self.query.near_vector.return_value = SimpleNamespace(objects=self.objects)

    def tearDown(self):
        self.env.stop()

    def retrieve(self, **kwargs):
        options = dict(question="leave policy", user_groups=["All-Employees"],
                       client=self.client, embeddings=self.embeddings,
                       collection_name="Policies", settings=self.settings,
                       reranker=self.reranker, k=2)
        options.update(kwargs)
        return retrieve_documents_with_resources(**options)

    def test_hybrid_filters_before_reranking_and_preserves_citations(self):
        docs = self.retrieve()
        args = self.query.hybrid.call_args.kwargs
        self.assertEqual(args["query"], "leave policy")
        self.assertEqual(args["vector"], [0.1, 0.2])
        self.assertEqual(args["limit"], 20)
        self.assertEqual(args["alpha"], 0.5)
        self.assertEqual(args["fusion_type"], HybridFusion.RELATIVE_SCORE)
        self.assertEqual(args["filters"].model_dump(),
                         Filter.by_property("access_groups").contains_any(["All-Employees"]).model_dump())
        pairs = self.reranker.predict.call_args.args[0]
        self.assertEqual(len(pairs), 3)
        self.assertEqual([doc.metadata["document_id"] for doc in docs], ["HR-002", "HR-003"])
        self.assertEqual(docs[0].metadata["page_number"], 2)
        self.assertEqual(docs[0].metadata["source_file"], "policy-2.pdf")
        self.assertEqual(docs[0].metadata["retrieval_rank"], 2)
        self.assertEqual(docs[0].metadata["rerank_score"], 0.9)
        self.query.near_vector.assert_not_called()

    def test_baseline_uses_original_top_k_vector_path(self):
        docs = self.retrieve(settings=variant_settings(self.settings, "baseline"))
        self.assertEqual(self.query.near_vector.call_args.kwargs["limit"], 2)
        self.assertIn("filters", self.query.near_vector.call_args.kwargs)
        self.assertEqual(docs[0].metadata["document_id"], "HR-001")
        self.query.hybrid.assert_not_called()
        self.reranker.predict.assert_not_called()

    def test_hybrid_only_skips_reranker(self):
        self.retrieve(settings=variant_settings(self.settings, "hybrid"))
        self.assertEqual(self.query.hybrid.call_args.kwargs["limit"], 2)
        self.reranker.predict.assert_not_called()

    def test_invalid_requests_do_not_embed_or_search(self):
        for kwargs, error in [({"question": "  "}, ValueError),
                              ({"user_groups": []}, PermissionError),
                              ({"k": 0}, ValueError)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(error):
                self.retrieve(**kwargs)
        self.embeddings.embed_query.assert_not_called()
        self.client.collections.use.assert_not_called()

    def test_empty_candidates_do_not_load_or_call_reranker(self):
        self.query.hybrid.return_value = SimpleNamespace(objects=[])
        with patch("rag_chatbot.reranking.get_reranker") as load:
            self.assertEqual(self.retrieve(), [])
            load.assert_not_called()
        self.reranker.predict.assert_not_called()

    def test_k_larger_than_candidate_setting_is_supported(self):
        self.retrieve(k=25)
        self.assertEqual(self.query.hybrid.call_args.kwargs["limit"], 25)

    def test_reranker_failure_is_not_silently_ignored(self):
        self.reranker.predict.side_effect = RuntimeError("model unavailable")
        with self.assertRaisesRegex(RuntimeError, "model unavailable"):
            self.retrieve()

    def test_bad_scores_rejected_and_ties_stable(self):
        docs = [Document(page_content=str(i), metadata={"page_number": i}) for i in range(3)]
        for scores in ([1.0], [1, float("nan"), 0], [1, float("inf"), 0]):
            self.reranker.predict.return_value = scores
            with self.subTest(scores=scores), self.assertRaises(RuntimeError):
                rerank_documents("q", docs, self.settings, k=2, reranker=self.reranker)
        self.reranker.predict.return_value = [1, 1, 0]
        result = rerank_documents("q", docs, self.settings, k=2, reranker=self.reranker)
        self.assertEqual([doc.page_content for doc in result], ["0", "1"])
        self.assertNotIn("rerank_score", docs[0].metadata)

    def test_settings_validation(self):
        for update in ({"hybrid_alpha": -0.1}, {"hybrid_alpha": float("nan")},
                       {"retrieval_candidates": 0}, {"reranker_batch_size": 0},
                       {"retrieval_mode": "typo"}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                replace(self.settings, **update)
        with patch.dict(os.environ, {"RAG_RERANK_ENABLED": "typo"}):
            with self.assertRaises(ValueError):
                load_settings()

    def test_model_reused_across_requests(self):
        _cached_reranker.cache_clear()
        try:
            with patch("rag_chatbot.reranking.LocalCrossEncoder") as constructor:
                self.assertIs(get_reranker(self.settings), get_reranker(self.settings))
                constructor.assert_called_once()
        finally:
            _cached_reranker.cache_clear()

    def test_experiment_profiles_keep_models_and_collection_fixed(self):
        for variant in ("baseline", "hybrid", "hybrid-rerank"):
            config = variant_settings(self.settings, variant)
            self.assertEqual(config.collection_name, self.settings.collection_name)
            self.assertEqual(config.chat_model, self.settings.chat_model)
            self.assertEqual(config.embedding_model, self.settings.embedding_model)
            metadata = experiment_metadata(config, variant)
            self.assertEqual(metadata["variant"], variant)
            self.assertEqual(metadata["rerank_enabled"], variant == "hybrid-rerank")

    def test_api_loads_model_at_startup_only_when_enabled(self):
        from rag_chatbot import api

        async def start():
            async with api.lifespan(api.app):
                pass

        for enabled in (True, False):
            with self.subTest(enabled=enabled), patch.object(
                api, "settings", replace(self.settings, rerank_enabled=enabled)
            ), patch.object(api, "get_reranker") as load:
                asyncio.run(start())
                self.assertEqual(load.call_count, int(enabled))

    def test_api_does_not_accept_startup_if_model_cannot_load(self):
        from rag_chatbot import api

        async def start():
            async with api.lifespan(api.app):
                self.fail("Startup must not complete with a missing model")

        with patch.object(api, "settings", self.settings), patch.object(
            api, "get_reranker", side_effect=OSError("missing model")
        ), self.assertRaisesRegex(OSError, "missing model"):
            asyncio.run(start())


if __name__ == "__main__":
    unittest.main()
