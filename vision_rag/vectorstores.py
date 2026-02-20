"""
vision_rag/vectorstores.py

Stage 4 of the vision_rag pipeline — Vector Store.
Indexes embedded chunks into a vector database for retrieval.

Built-in stores:
  - FAISS  (local, no server needed)
  - Chroma (local, no server needed)

Usage:
    from vision_rag.vectorstores import FAISS, Chroma

    store = FAISS()
    store.index(embedded_chunks)
    store.save("my_index")

    # later
    store.load("my_index")
    results = store.search_text(query_vector, top_k=5)
    results = store.search_image(query_vector, top_k=5)
"""

from __future__ import annotations

import os
import json
import pickle
from pathlib import Path
from typing import Optional

from vision_rag.embedding import EmbeddedChunk


# ──────────────────────────────────────────────────────────────
# SearchResult — what comes back from search
# ──────────────────────────────────────────────────────────────

class SearchResult:
    """
    A single result returned from a vector store search.
    """
    def __init__(self, chunk: EmbeddedChunk, score: float):
        self.chunk  = chunk    # full EmbeddedChunk with all fields
        self.score  = score    # similarity score (higher = more similar)

    def __repr__(self) -> str:
        return (
            f"SearchResult("
            f"score={self.score:.4f}, "
            f"chunk_id={self.chunk.chunk_id}, "
            f"start={self.chunk.start:.2f}s, "
            f"end={self.chunk.end:.2f}s, "
            f"text={repr(self.chunk.text[:50] + '...') if self.chunk.text and len(self.chunk.text) > 50 else repr(self.chunk.text)}"
            f")"
        )


# ──────────────────────────────────────────────────────────────
# BaseVectorStore — plug in anything
# ──────────────────────────────────────────────────────────────

class BaseVectorStore:
    """
    Base class for all vector stores.
    Subclass and implement these methods to plug in any vector database.

    Example:
        class MyVectorStore(BaseVectorStore):
            def index(self, embedded_chunks): ...
            def search_text(self, vector, top_k): ...
            def search_image(self, vector, top_k): ...
            def save(self, path): ...
            def load(self, path): ...
    """

    def index(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        raise NotImplementedError

    def search_text(self, query_vector: list[float], top_k: int = 5) -> list[SearchResult]:
        raise NotImplementedError

    def search_image(self, query_vector: list[float], top_k: int = 5) -> list[SearchResult]:
        raise NotImplementedError

    def search_by_time(self, start: float, end: float) -> list[EmbeddedChunk]:
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    def load(self, path: str) -> None:
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────
# FAISS — local, fast, no server needed
# ──────────────────────────────────────────────────────────────

class FAISS(BaseVectorStore):
    """
    Vector store using Facebook FAISS.
    Runs entirely locally. No server needed.

    Usage:
        store = FAISS()
        store.index(embedded_chunks)
        store.save("my_index")

        store.load("my_index")
        results = store.search_text(query_vector, top_k=5)
        results = store.search_image(query_vector, top_k=5)

    pip install faiss-cpu
    """

    def __init__(self):
        self._text_index   = None
        self._image_index  = None
        self._chunks: list[EmbeddedChunk] = []   # all chunks stored for lookup

    def index(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        """Index all embedded chunks into FAISS."""
        try:
            import faiss
            import numpy as np
        except ImportError:
            raise RuntimeError("pip install faiss-cpu")

        self._chunks = embedded_chunks

        # -- text index --
        text_vecs = [
            ec.text_vector for ec in embedded_chunks
            if ec.text_vector is not None
        ]
        if text_vecs:
            dim = len(text_vecs[0])
            self._text_index = faiss.IndexFlatIP(dim)   # Inner Product = cosine on normalized vecs
            matrix = np.array(text_vecs, dtype="float32")
            faiss.normalize_L2(matrix)
            self._text_index.add(matrix)
            print(f"[FAISS] Indexed {len(text_vecs)} text vectors (dim={dim})")

        # -- image index --
        image_vecs = [
            ec.image_vector for ec in embedded_chunks
            if ec.image_vector is not None
        ]
        if image_vecs:
            dim = len(image_vecs[0])
            self._image_index = faiss.IndexFlatIP(dim)
            matrix = np.array(image_vecs, dtype="float32")
            faiss.normalize_L2(matrix)
            self._image_index.add(matrix)
            print(f"[FAISS] Indexed {len(image_vecs)} image vectors (dim={dim})")

    def search_text(self, query_vector: list[float], top_k: int = 5) -> list[SearchResult]:
        """Search by text vector. Returns top_k most similar chunks."""
        if self._text_index is None:
            raise RuntimeError("No text index found. Run index() first.")
        import faiss
        import numpy as np

        query = np.array([query_vector], dtype="float32")
        faiss.normalize_L2(query)
        scores, indices = self._text_index.search(query, top_k)

        # map indices back to chunks that have text vectors
        text_chunks = [ec for ec in self._chunks if ec.text_vector is not None]
        return [
            SearchResult(chunk=text_chunks[i], score=float(scores[0][rank]))
            for rank, i in enumerate(indices[0])
            if i < len(text_chunks)
        ]

    def search_image(self, query_vector: list[float], top_k: int = 5) -> list[SearchResult]:
        """Search by image vector. Returns top_k most similar chunks."""
        if self._image_index is None:
            raise RuntimeError("No image index found. Run index() first.")
        import faiss
        import numpy as np

        query = np.array([query_vector], dtype="float32")
        faiss.normalize_L2(query)
        scores, indices = self._image_index.search(query, top_k)

        image_chunks = [ec for ec in self._chunks if ec.image_vector is not None]
        return [
            SearchResult(chunk=image_chunks[i], score=float(scores[0][rank]))
            for rank, i in enumerate(indices[0])
            if i < len(image_chunks)
        ]

    def search_by_time(self, start: float, end: float) -> list[EmbeddedChunk]:
        """Return all chunks that overlap with the given time window."""
        return [
            ec for ec in self._chunks
            if ec.end > start and ec.start < end
        ]

    def save(self, path: str) -> None:
        """Save the FAISS index and chunks to disk."""
        try:
            import faiss
        except ImportError:
            raise RuntimeError("pip install faiss-cpu")

        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)

        if self._text_index:
            faiss.write_index(self._text_index, str(out / "text.index"))
        if self._image_index:
            faiss.write_index(self._image_index, str(out / "image.index"))

        with open(out / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)

        print(f"[FAISS] Saved index to {out}/")

    def load(self, path: str) -> None:
        """Load a previously saved FAISS index from disk."""
        try:
            import faiss
        except ImportError:
            raise RuntimeError("pip install faiss-cpu")

        out = Path(path)
        text_path  = out / "text.index"
        image_path = out / "image.index"
        chunks_path = out / "chunks.pkl"

        if text_path.exists():
            self._text_index = faiss.read_index(str(text_path))
        if image_path.exists():
            self._image_index = faiss.read_index(str(image_path))
        if chunks_path.exists():
            with open(chunks_path, "rb") as f:
                self._chunks = pickle.load(f)

        print(f"[FAISS] Loaded index from {out}/")


# ──────────────────────────────────────────────────────────────
# Chroma — local, persistent, no server needed
# ──────────────────────────────────────────────────────────────

class Chroma(BaseVectorStore):
    """
    Vector store using ChromaDB.
    Runs entirely locally with persistent storage.

    Usage:
        store = Chroma(path="my_chroma_db")
        store.index(embedded_chunks)
        results = store.search_text(query_vector, top_k=5)
        results = store.search_image(query_vector, top_k=5)

    pip install chromadb
    """

    def __init__(self, path: str = ".vision_rag_cache/chroma"):
        self.path = path
        self._client      = None
        self._text_col    = None
        self._image_col   = None
        self._chunks: list[EmbeddedChunk] = []

    def _get_client(self):
        try:
            import chromadb
        except ImportError:
            raise RuntimeError("pip install chromadb")
        if self._client is None:
            self._client = chromadb.PersistentClient(path=self.path)
        return self._client

    def index(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        """Index all embedded chunks into Chroma."""
        client = self._get_client()
        self._chunks = embedded_chunks

        self._text_col  = client.get_or_create_collection("text_vectors")
        self._image_col = client.get_or_create_collection("image_vectors")

        # -- text --
        text_ids, text_vecs, text_metas = [], [], []
        for ec in embedded_chunks:
            if ec.text_vector:
                text_ids.append(str(ec.chunk_id))
                text_vecs.append(ec.text_vector)
                text_metas.append({
                    "chunk_id":   ec.chunk_id,
                    "start":      ec.start,
                    "end":        ec.end,
                    "video_path": ec.video_path,
                    "text":       ec.text or "",
                    "frame_path": ec.frame_path or "",
                })

        if text_ids:
            self._text_col.add(ids=text_ids, embeddings=text_vecs, metadatas=text_metas)
            print(f"[Chroma] Indexed {len(text_ids)} text vectors")

        # -- image --
        image_ids, image_vecs, image_metas = [], [], []
        for ec in embedded_chunks:
            if ec.image_vector:
                image_ids.append(str(ec.chunk_id))
                image_vecs.append(ec.image_vector)
                image_metas.append({
                    "chunk_id":   ec.chunk_id,
                    "start":      ec.start,
                    "end":        ec.end,
                    "video_path": ec.video_path,
                    "text":       ec.text or "",
                    "frame_path": ec.frame_path or "",
                })

        if image_ids:
            self._image_col.add(ids=image_ids, embeddings=image_vecs, metadatas=image_metas)
            print(f"[Chroma] Indexed {len(image_ids)} image vectors")

    def _results_to_search(self, results: dict) -> list[SearchResult]:
        """Convert Chroma query results to SearchResult list."""
        out = []
        for i, meta in enumerate(results["metadatas"][0]):
            # find the matching EmbeddedChunk
            chunk = next(
                (ec for ec in self._chunks if ec.chunk_id == meta["chunk_id"]),
                None
            )
            if chunk:
                score = 1 - results["distances"][0][i]   # chroma returns distance, convert to similarity
                out.append(SearchResult(chunk=chunk, score=score))
        return out

    def search_text(self, query_vector: list[float], top_k: int = 5) -> list[SearchResult]:
        if self._text_col is None:
            client = self._get_client()
            self._text_col = client.get_or_create_collection("text_vectors")
        results = self._text_col.query(query_embeddings=[query_vector], n_results=top_k)
        return self._results_to_search(results)

    def search_image(self, query_vector: list[float], top_k: int = 5) -> list[SearchResult]:
        if self._image_col is None:
            client = self._get_client()
            self._image_col = client.get_or_create_collection("image_vectors")
        results = self._image_col.query(query_embeddings=[query_vector], n_results=top_k)
        return self._results_to_search(results)

    def search_by_time(self, start: float, end: float) -> list[EmbeddedChunk]:
        return [
            ec for ec in self._chunks
            if ec.end > start and ec.start < end
        ]

    def save(self, path: str) -> None:
        # Chroma persists automatically — just save chunks for time search
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)
        print(f"[Chroma] Persisted to {self.path}/")

    def load(self, path: str) -> None:
        chunks_path = Path(path) / "chunks.pkl"
        if chunks_path.exists():
            with open(chunks_path, "rb") as f:
                self._chunks = pickle.load(f)
        print(f"[Chroma] Loaded from {self.path}/")
