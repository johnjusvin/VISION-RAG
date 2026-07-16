"""
vision_rag/embedding.py

Stage 3 of the vision_rag pipeline — Embedding.
Converts chunk.text and chunk.frame_path into vectors.

Usage:
    from vision_rag.embedding import EmbeddingBuilder, CLIPImageEmbedder, OpenAITextEmbedder

    embedder = EmbeddingBuilder(
        text_embedding=OpenAITextEmbedder(),
        image_embedding=CLIPImageEmbedder()
    )
    embedded_chunks = embedder.embed(chunks)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from vision_rag.video_chunker import Chunk
from vision_rag._device import resolve_device


# ──────────────────────────────────────────────────────────────
# EmbeddedChunk — Chunk + vectors
# ──────────────────────────────────────────────────────────────

@dataclass
class EmbeddedChunk:
    """
    A Chunk with text and image vectors attached.
    This is what flows into the indexing stage.
    """
    # -- original chunk fields (unchanged) --
    chunk_id:     int
    video_path:   str
    start:        float
    end:          float
    duration:     float
    frame_path:   Optional[str]
    text:         Optional[str]
    metadata:     dict = field(default_factory=dict)

    # -- new: vectors --
    text_vector:  Optional[list[float]] = None   # None if text=None or text_embedding not set
    image_vector: Optional[list[float]] = None   # None if frame_path=None or image_embedding not set

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> "EmbeddedChunk":
        """Create an EmbeddedChunk from a Chunk (vectors are None until embed() is called)."""
        return cls(
            chunk_id   = chunk.chunk_id,
            video_path = chunk.video_path,
            start      = chunk.start,
            end        = chunk.end,
            duration   = chunk.duration,
            frame_path = chunk.frame_path,
            text       = chunk.text,
            metadata   = chunk.metadata.copy(),
        )

    def __repr__(self) -> str:
        tv = f"dim={len(self.text_vector)}"  if self.text_vector  else "None"
        iv = f"dim={len(self.image_vector)}" if self.image_vector else "None"
        return (
            f"EmbeddedChunk("
            f"id={self.chunk_id}, "
            f"start={self.start:.2f}s, "
            f"end={self.end:.2f}s, "
            f"text_vector={tv}, "
            f"image_vector={iv}"
            f")"
        )


# ──────────────────────────────────────────────────────────────
# Base classes — plug in anything
# ──────────────────────────────────────────────────────────────

class BaseTextEmbedder:
    """
    Base class for all text embedding providers.
    Subclass and implement embed() to use any model or API.

    Example:
        class MyTextEmbedder(BaseTextEmbedder):
            def embed(self, text: str) -> list[float]:
                # your model or API here
                return [...]
    """

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError("Implement embed() in your TextEmbedder subclass.")

    @property
    def provider_name(self) -> str:
        return self.__class__.__name__


class BaseImageEmbedder:
    """
    Base class for all image embedding providers.
    Subclass and implement embed() to use any model or API.

    Example:
        class MyImageEmbedder(BaseImageEmbedder):
            def embed(self, image_path: str) -> list[float]:
                # your model or API here
                return [...]
    """

    def embed(self, image_path: str) -> list[float]:
        raise NotImplementedError("Implement embed() in your ImageEmbedder subclass.")

    @property
    def provider_name(self) -> str:
        return self.__class__.__name__


# ──────────────────────────────────────────────────────────────
# Built-in Text Embedders
# ──────────────────────────────────────────────────────────────

class CLIPTextEmbedder(BaseTextEmbedder):
    """
    Text embeddings using OpenAI CLIP's text encoder. No API key needed.

    IMPORTANT: pair this with CLIPImageEmbedder (same model= string) when
    you need cross-modal retrieval -- i.e. text queries searching the
    image index, or image-derived content ranked against text queries.
    Text and image vectors only land in the same space when both come
    from CLIP's joint encoder. Pairing CLIPImageEmbedder with an
    unrelated text embedder (e.g. SentenceTransformerTextEmbedder)
    produces vectors from two unconnected spaces -- searching the image
    index with such a query is not meaningful even when the dimensions
    happen to match, and Retriever will reject it outright when they don't.

    Usage:
        text_embedder  = CLIPTextEmbedder(model="ViT-B/32")                # auto device
        text_embedder  = CLIPTextEmbedder(model="ViT-B/32", device="cuda")  # explicit

    pip install git+https://github.com/openai/CLIP.git torch
    """

    def __init__(self, model: str = "ViT-B/32", device: str = "auto"):
        self.model_name = model
        self._requested_device = device
        self.device: Optional[str] = None   # resolved lazily -- see _resolve_device()
        self._model = None

    def _resolve_device(self) -> str:
        # Resolved lazily (not in __init__) so simply constructing this
        # class never requires torch to be installed -- only embed() does.
        if self.device is None:
            self.device = resolve_device(self._requested_device)
        return self.device

    def embed(self, text: str) -> list[float]:
        try:
            import clip
            import torch
        except ImportError:
            raise RuntimeError(
                "pip install git+https://github.com/openai/CLIP.git torch"
            )

        device = self._resolve_device()

        if self._model is None:
            self._model, _ = clip.load(self.model_name, device=device)

        tokens = clip.tokenize([text], truncate=True).to(device)
        with torch.no_grad():
            vector = self._model.encode_text(tokens)
        return vector.squeeze().tolist()


class OpenAITextEmbedder(BaseTextEmbedder):
    """
    OpenAI text embeddings (text-embedding-3-small by default).

    Usage:
        embedder = OpenAITextEmbedder(api_key="sk-...")
        # or set OPENAI_API_KEY env var and just do OpenAITextEmbedder()
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Pass api_key= or set OPENAI_API_KEY env var."
            )
        self.model = model
        self._client = None

    def embed(self, text: str) -> list[float]:
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("pip install openai")

        if self._client is None:
            self._client = OpenAI(api_key=self.api_key)

        response = self._client.embeddings.create(input=text, model=self.model)
        return response.data[0].embedding


class SentenceTransformerTextEmbedder(BaseTextEmbedder):
    """
    Local text embeddings using SentenceTransformers. No API key needed.

    Usage:
        embedder = SentenceTransformerTextEmbedder()
        embedder = SentenceTransformerTextEmbedder(model="all-mpnet-base-v2")
        embedder = SentenceTransformerTextEmbedder(device="cuda")   # force GPU

    pip install sentence-transformers
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2", device: str = "auto"):
        self.model_name = model
        self._requested_device = device
        self.device: Optional[str] = None   # resolved lazily -- see _resolve_device()
        self._model = None

    def _resolve_device(self) -> str:
        if self.device is None:
            self.device = resolve_device(self._requested_device)
        return self.device

    def embed(self, text: str) -> list[float]:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError("pip install sentence-transformers")

        device = self._resolve_device()

        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=device)

        return self._model.encode(text).tolist()


# ──────────────────────────────────────────────────────────────
# Built-in Image Embedders
# ──────────────────────────────────────────────────────────────

class CLIPImageEmbedder(BaseImageEmbedder):
    """
    Local image embeddings using OpenAI CLIP. No API key needed.
    Industry standard for image embeddings.

    Usage:
        embedder = CLIPImageEmbedder()                        # auto device
        embedder = CLIPImageEmbedder(model="ViT-B/32")
        embedder = CLIPImageEmbedder(device="cuda")            # explicit

    pip install git+https://github.com/openai/CLIP.git Pillow torch
    """

    def __init__(self, model: str = "ViT-B/32", device: str = "auto"):
        self.model_name = model
        self._requested_device = device
        self.device: Optional[str] = None   # resolved lazily -- see _resolve_device()
        self._model = None
        self._preprocess = None

    def _resolve_device(self) -> str:
        if self.device is None:
            self.device = resolve_device(self._requested_device)
        return self.device

    def embed(self, image_path: str) -> list[float]:
        try:
            import clip
            import torch
            from PIL import Image
        except ImportError:
            raise RuntimeError(
                "pip install git+https://github.com/openai/CLIP.git Pillow torch"
            )

        # Resolve the actual device before loading the CLIP image model
        device = self._resolve_device()

        if self._model is None:
            self._model, self._preprocess = clip.load(self.model_name, device=device)

        image = self._preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            vector = self._model.encode_image(image)
        return vector.squeeze().tolist()


class OpenAIImageEmbedder(BaseImageEmbedder):
    """
    Image embeddings via OpenAI vision model.
    Encodes the image as base64 and gets an embedding via the API.

    Usage:
        embedder = OpenAIImageEmbedder(api_key="sk-...")
        # or set OPENAI_API_KEY env var

    pip install openai
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Pass api_key= or set OPENAI_API_KEY env var."
            )
        self.model = model
        self._client = None

    def embed(self, image_path: str) -> list[float]:
        import base64
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("pip install openai")

        if self._client is None:
            self._client = OpenAI(api_key=self.api_key)

        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        response = self._client.embeddings.create(
            input=f"data:image/jpeg;base64,{b64}",
            model=self.model,
        )
        return response.data[0].embedding


# ──────────────────────────────────────────────────────────────
# EmbeddingBuilder — main entry point
# ──────────────────────────────────────────────────────────────

class EmbeddingBuilder:
    """
    Embeds a list of Chunk objects into EmbeddedChunk objects.

    Parameters
    ----------
    text_embedding : BaseTextEmbedder | None
        Any text embedding provider. None = skip text embedding.
    image_embedding : BaseImageEmbedder | None
        Any image embedding provider. None = skip image embedding.

    Usage:
        embedder = EmbeddingBuilder(
            text_embedding=OpenAITextEmbedder(),
            image_embedding=CLIPImageEmbedder()
        )
        embedded_chunks = embedder.embed(chunks)
    """

    def __init__(
        self,
        text_embedding:  Optional[BaseTextEmbedder]  = None,
        image_embedding: Optional[BaseImageEmbedder] = None,
    ):
        if text_embedding is None and image_embedding is None:
            raise ValueError(
                "At least one of text_embedding or image_embedding must be provided."
            )
        self.text_embedding  = text_embedding
        self.image_embedding = image_embedding

    def embed(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """
        Embed a list of chunks. Returns a list of EmbeddedChunk objects.

        Parameters
        ----------
        chunks : list[Chunk]
            Output from Chunker.chunk()
        """
        embedded = []
        for chunk in chunks:
            ec = EmbeddedChunk.from_chunk(chunk)

            # -- text vector --
            if self.text_embedding and chunk.text:
                ec.text_vector = self.text_embedding.embed(chunk.text)
                ec.metadata["text_embedder"] = self.text_embedding.provider_name

            # -- image vector --
            if self.image_embedding and chunk.frame_path:
                if Path(chunk.frame_path).exists():
                    ec.image_vector = self.image_embedding.embed(chunk.frame_path)
                    ec.metadata["image_embedder"] = self.image_embedding.provider_name

            embedded.append(ec)

        return embedded