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

    @property
    def all(self) -> list[SearchResult]:
        """
        All results from both text and image searches,
        combined and ranked by score (highest first).
        Deduplicates by chunk_id — keeps the highest score per chunk.
        """
        seen = {}
        for result in self.text_results + self.image_results:
            cid = result.chunk.chunk_id
            if cid not in seen or result.score > seen[cid].score:
                seen[cid] = result
        return sorted(seen.values(), key=lambda r: r.score, reverse=True)

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
