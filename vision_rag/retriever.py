"""
vision_rag/retriever.py

Stage 5 of the vision_rag pipeline — Retrieval.
Takes a text query, searches both text and image indexes simultaneously,
and returns ranked results.

Usage:
    from vision_rag.retriever import Retriever

    retriever = Retriever(
        store=store,                   # FAISS or Chroma or any BaseVectorStore
        text_embedder=text_embedder,   # same embedder used during indexing
        top_k_text=5,
        top_k_image=5,
    )
    results = retriever.retrieve("What is shown in the diagram?")

    results.text_results    # top matches from text index
    results.image_results   # top matches from image index
    results.all             # everything combined and ranked by score
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from vision_rag.embedding import BaseTextEmbedder
from vision_rag.vectorstores import BaseVectorStore, SearchResult


# ──────────────────────────────────────────────────────────────
# RetrievalResult — what comes back from retrieve()
# ──────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """
    Holds results from both text and image searches.
    Returned by Retriever.retrieve().
    """
    query:         str
    text_results:  list[SearchResult] = field(default_factory=list)
    image_results: list[SearchResult] = field(default_factory=list)
    rrf_k:         int = 60   # RRF damping constant (Cormack et al., 2009)

    @property
    def all(self) -> list[SearchResult]:
        """
        All results from both text and image searches, combined into a
        single ranking via Reciprocal Rank Fusion (RRF).

        Why not just sort by raw score? Raw similarity scores from
        different embedding models are not guaranteed to be on comparable
        scales — e.g. a SentenceTransformer text embedder and a CLIP image
        embedder produce similarity distributions with different shapes and
        ranges, so interleaving and sorting them by raw score is not
        meaningful in general. It only happens to work when text and image
        vectors come from the same joint embedding space (e.g. Jina v4).

        RRF sidesteps this by fusing the two *rankings* instead of the raw
        scores, which only assumes each individual list is internally
        well-ordered — a much weaker and more broadly valid assumption:

            rrf(chunk) = sum over lists L containing chunk of 1 / (k + rank_in_L)

        where rank_in_L is the chunk's 1-indexed position within that list.
        Deduplicates by chunk_id. Each result's original per-modality
        similarity score is preserved on `.score` for inspection — only the
        ordering is determined by the fused rank.
        """
        k = self.rrf_k
        fused: dict[int, dict] = {}

        for ranked_list in (self.text_results, self.image_results):
            for rank, result in enumerate(ranked_list, start=1):
                cid = result.chunk.chunk_id
                entry = fused.setdefault(
                    cid, {"result": result, "rrf_score": 0.0, "best_raw_score": result.score}
                )
                entry["rrf_score"] += 1.0 / (k + rank)
                # keep the SearchResult carrying the higher raw score for display,
                # and use raw score only to break exact RRF ties deterministically
                if result.score > entry["best_raw_score"]:
                    entry["best_raw_score"] = result.score
                    entry["result"] = result

        ordered = sorted(
            fused.values(),
            key=lambda e: (e["rrf_score"], e["best_raw_score"]),
            reverse=True,
        )
        return [e["result"] for e in ordered]

    @property
    def by_time(self) -> list[SearchResult]:
        """All results sorted by timestamp (earliest first)."""
        return sorted(self.all, key=lambda r: r.chunk.start)

    def __repr__(self) -> str:
        return (
            f"RetrievalResult("
            f"query={self.query!r}, "
            f"text_results={len(self.text_results)}, "
            f"image_results={len(self.image_results)}, "
            f"unique_chunks={len(self.all)}"
            f")"
        )


# ──────────────────────────────────────────────────────────────
# Retriever
# ──────────────────────────────────────────────────────────────

class Retriever:
    """
    Retrieves relevant chunks from the vector store for a given query.
    Searches both text and image indexes simultaneously.

    Parameters
    ----------
    store : BaseVectorStore
        The indexed vector store (FAISS, Chroma, or custom).
    text_embedder : BaseTextEmbedder
        Same text embedder used during indexing — must produce
        vectors in the same space.
    top_k_text : int
        Number of text results to return. Default = 5.
    top_k_image : int
        Number of image results to return. Default = 5.

    Usage:
        retriever = Retriever(
            store=store,
            text_embedder=JinaV4TextEmbedder(api_key="..."),
            top_k_text=5,
            top_k_image=5,
        )
        results = retriever.retrieve("What is shown in the diagram?")
    """

    def __init__(
        self,
        store: BaseVectorStore,
        text_embedder: BaseTextEmbedder,
        top_k_text: int = 5,
        top_k_image: int = 5,
    ):
        self.store         = store
        self.text_embedder = text_embedder
        self.top_k_text    = top_k_text
        self.top_k_image   = top_k_image

    def retrieve(self, query: str) -> RetrievalResult:
        """
        Embed the query and search both text and image indexes at once.

        Parameters
        ----------
        query : str
            Natural language question or description.

        Returns
        -------
        RetrievalResult with text_results, image_results, and all combined.
        """
        # embed the query once — used for both text and image search
        # works because Jina v4 (and similar multimodal models) share
        # the same vector space for text and images
        query_vector = self.text_embedder.embed(query)

        # search both indexes simultaneously
        text_results  = self._search_text(query_vector)
        image_results = self._search_image(query_vector)

        return RetrievalResult(
            query         = query,
            text_results  = text_results,
            image_results = image_results,
        )

    def retrieve_by_time(self, start: float, end: float) -> list:
        """
        Retrieve chunks by timestamp range instead of semantic search.
        Useful for queries like 'what happened at 12 seconds?'

        Parameters
        ----------
        start : float — start time in seconds
        end   : float — end time in seconds
        """
        return self.store.search_by_time(start, end)

    # ── private ──────────────────────────────────────────────

    def _search_text(self, query_vector: list[float]) -> list[SearchResult]:
        try:
            return self.store.search_text(query_vector, top_k=self.top_k_text)
        except RuntimeError:
            # no text index — skip silently
            return []

    def _search_image(self, query_vector: list[float]) -> list[SearchResult]:
        try:
            return self.store.search_image(query_vector, top_k=self.top_k_image)
        except RuntimeError:
            # no image index — skip silently
            return []