"""
vision_rag/generator.py

Stage 6 of the vision_rag pipeline — Generation.
Takes retrieved chunks and a query, sends them to a VLM/LLM,
and returns a generated answer.

Built-in generators:
  - OpenAIGenerator    (GPT-4o — VLM, text + image)
  - AnthropicGenerator (Claude — VLM, text + image)
  - GeminiGenerator    (Gemini — VLM, text + image)
  - OllamaGenerator    (local models — text only)

Usage:
    from vision_rag.generator import Generator, OpenAIGenerator

    generator = Generator(llm=OpenAIGenerator(api_key="sk-..."))
    answer = generator.generate(
        query="What did they say about frozen yogurt?",
        results=results
    )
    print(answer.text)
    print(answer.sources)
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from vision_rag.retriever import RetrievalResult
from vision_rag.embedding import EmbeddedChunk


# ──────────────────────────────────────────────────────────────
# GeneratorAnswer — what comes back from generate()
# ──────────────────────────────────────────────────────────────

@dataclass
class GeneratorAnswer:
    """
    The final answer returned by Generator.generate().
    """
    query:    str                          # the original question
    text:     str                          # the generated answer
    sources:  list[EmbeddedChunk] = field(default_factory=list)  # chunks used to answer

    def __repr__(self) -> str:
        return (
            f"GeneratorAnswer("
            f"query={self.query!r}, "
            f"answer={self.text[:80]!r}{'...' if len(self.text) > 80 else ''}, "
            f"sources={len(self.sources)} chunks"
            f")"
        )


# ──────────────────────────────────────────────────────────────
# BaseGenerator — plug in anything
# ──────────────────────────────────────────────────────────────

class BaseGenerator:
    """
    Base class for all LLM/VLM generators.
    Subclass and implement generate() to use any model or API.

    Example — custom generator:
        class MyGenerator(BaseGenerator):
            def generate(self, query: str, chunks: list[EmbeddedChunk]) -> str:
                # your model/API logic here
                return "the answer..."
    """

    def generate(self, query: str, chunks: list[EmbeddedChunk]) -> str:
        """
        Generate an answer for the query using the provided chunks.

        Parameters
        ----------
        query : str
            The user's question.
        chunks : list[EmbeddedChunk]
            Retrieved chunks — each has .text and .frame_path.

        Returns
        -------
        str — the generated answer.
        """
        raise NotImplementedError("Implement generate() in your Generator subclass.")

    @property
    def provider_name(self) -> str:
        return self.__class__.__name__

    def _read_image_b64(self, image_path: str) -> Optional[str]:
        """Read an image file and return base64 encoded string."""
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return None

    def _build_system_prompt(self) -> str:
        return (
            "You are a Retrieval-Augmented Generation (RAG) assistant for video "
            "understanding. The chunks provided below are your ONLY knowledge "
            "source -- do not use outside knowledge about shows, people, places, "
            "or events, even if you recognize them.\n\n"
            "Each chunk covers a short time window and includes:\n"
            "  - a TRANSCRIPT segment: what was said, from automatic speech recognition\n"
            "  - a KEYFRAME image: what was visible on screen at that moment, which may "
            "itself contain burned-in captions matching the transcript -- that overlap "
            "is expected, not a discrepancy worth noting.\n\n"
            "Rules:\n"
            "1. Use the TRANSCRIPT for what was said, discussed, named, or explained.\n"
            "2. Use the KEYFRAME only for what is visibly shown (people, setting, "
            "on-screen text) -- never as a substitute for the transcript.\n"
            "3. Never state a detail (in text or image) that is not clearly supported "
            "by the provided chunks. Do not guess, complete missing information, or "
            "identify shows/people/places unless explicitly stated or clearly visible.\n"
            "4. Chunks are ordered chronologically below; synthesize across all of "
            "them rather than relying on just one.\n"
            "5. If the provided chunks do not contain enough information to answer, "
            "say so directly instead of speculating.\n"
            "6. When possible, cite the timestamp(s) (in seconds) your answer draws on."
        )


# ──────────────────────────────────────────────────────────────
# Built-in Generators
# ──────────────────────────────────────────────────────────────

class OpenAIGenerator(BaseGenerator):
    """
    GPT-4o — VLM, supports both text and images.

    Usage:
        gen = OpenAIGenerator(api_key="sk-...")
        # or set OPENAI_API_KEY env var

    pip install openai
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Pass api_key= or set OPENAI_API_KEY env var.")
        self.model   = model
        self._client = None

    def generate(self, query: str, chunks: list[EmbeddedChunk]) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("pip install openai")

        if self._client is None:
            self._client = OpenAI(api_key=self.api_key)

        # build content — interleave text + images per chunk
        content = []
        for i, chunk in enumerate(chunks):
            # text segment
            if chunk.text:
                content.append({
                    "type": "text",
                    "text": f"[Chunk {i+1} | {chunk.start:.1f}s – {chunk.end:.1f}s]\n{chunk.text}"
                })
            # frame image
            if chunk.frame_path and Path(chunk.frame_path).exists():
                b64 = self._read_image_b64(chunk.frame_path)
                if b64:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                    })

        content.append({"type": "text", "text": f"\nQuestion: {query}"})

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user",   "content": content},
            ],
        )
        return response.choices[0].message.content.strip()


class AnthropicGenerator(BaseGenerator):
    """
    Claude — VLM, supports both text and images.

    Usage:
        gen = AnthropicGenerator(api_key="sk-ant-...")
        # or set ANTHROPIC_API_KEY env var

    pip install anthropic
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-6"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key required. Pass api_key= or set ANTHROPIC_API_KEY env var.")
        self.model   = model
        self._client = None

    def generate(self, query: str, chunks: list[EmbeddedChunk]) -> str:
        try:
            import anthropic
        except ImportError:
            raise RuntimeError("pip install anthropic")

        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.api_key)

        content = []
        for i, chunk in enumerate(chunks):
            if chunk.text:
                content.append({
                    "type": "text",
                    "text": f"[Chunk {i+1} | {chunk.start:.1f}s – {chunk.end:.1f}s]\n{chunk.text}"
                })
            if chunk.frame_path and Path(chunk.frame_path).exists():
                b64 = self._read_image_b64(chunk.frame_path)
                if b64:
                    content.append({
                        "type": "image",
                        "source": {
                            "type":       "base64",
                            "media_type": "image/jpeg",
                            "data":       b64,
                        }
                    })

        content.append({"type": "text", "text": f"\nQuestion: {query}"})

        response = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=self._build_system_prompt(),
            messages=[{"role": "user", "content": content}],
        )
        return response.content[0].text.strip()


class GeminiGenerator(BaseGenerator):
    """
    Gemini — VLM, supports both text and images.

    Usage:
        gen = GeminiGenerator(api_key="...")
        # or set GEMINI_API_KEY env var

    pip install google-genai
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key required. Pass api_key= or set GEMINI_API_KEY env var.")
        self.model   = model
        self._client = None

    def generate(self, query: str, chunks: list[EmbeddedChunk]) -> str:
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise RuntimeError("pip install google-genai")

        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)

        parts = [self._build_system_prompt(), "\n\n"]
        for i, chunk in enumerate(chunks):
            if chunk.text:
                parts.append(f"[Chunk {i+1} | {chunk.start:.1f}s – {chunk.end:.1f}s]\n{chunk.text}\n")
            if chunk.frame_path and Path(chunk.frame_path).exists():
                b64 = self._read_image_b64(chunk.frame_path)
                if b64:
                    parts.append(
                        types.Part.from_bytes(
                            data=base64.b64decode(b64),
                            mime_type="image/jpeg",
                        )
                    )

        parts.append(f"\nQuestion: {query}")

        response = self._client.models.generate_content(
            model=self.model,
            contents=parts,
        )
        return response.text.strip()


class OllamaGenerator(BaseGenerator):
    """
    Ollama — local models, text only or vision (e.g. llava:7b).
    Runs entirely on your machine. No API key needed.

    Usage:
        gen = OllamaGenerator(model="llava:7b")   # VLM — text + images
        gen = OllamaGenerator(model="llama3.2")   # text only

    Install Ollama: https://ollama.com
    Then pull the model: ollama pull llava:7b
    pip install ollama
    """

    # vision models that support image input
    VISION_MODELS = {"llava", "llava:7b", "llava:13b", "llava:34b", "bakllava", "moondream"}

    def __init__(self, model: str = "llava:7b", host: str = "http://localhost:11434"):
        self.model = model
        self.host  = host
        self._is_vision = any(vm in model.lower() for vm in ["llava", "bakllava", "moondream"])

    def generate(self, query: str, chunks: list[EmbeddedChunk]) -> str:
        try:
            import ollama
        except ImportError:
            raise RuntimeError("pip install ollama")

        context_parts = []
        for i, chunk in enumerate(chunks):
            if chunk.text:
                context_parts.append(
                    f"[Chunk {i+1} | {chunk.start:.1f}s – {chunk.end:.1f}s]\n{chunk.text}"
                )

        context = "\n\n".join(context_parts)
        prompt  = f"{self._build_system_prompt()}\n\nContext:\n{context}\n\nQuestion: {query}"

        # collect frame images if vision model
        images = []
        if self._is_vision:
            for chunk in chunks:
                if chunk.frame_path and Path(chunk.frame_path).exists():
                    b64 = self._read_image_b64(chunk.frame_path)
                    if b64:
                        images.append(b64)

        message = {"role": "user", "content": prompt}
        if images:
            message["images"] = images

        response = ollama.chat(
            model=self.model,
            messages=[message],
        )
        return response["message"]["content"].strip()


# ──────────────────────────────────────────────────────────────
# Generator — main entry point
# ──────────────────────────────────────────────────────────────

class Generator:
    """
    Takes retrieved chunks and generates an answer using a VLM/LLM.

    Parameters
    ----------
    llm : BaseGenerator
        Any generator — OpenAI, Anthropic, Gemini, Ollama, or custom.
    top_k : int
        How many chunks to pass to the LLM. Default = 5.
        Uses results.all (combined text + image, ranked by score).
    chronological : bool
        If True (default), the top_k chunks are selected by relevance but
        then reordered by timestamp before being sent to the LLM, so it
        reads a coherent timeline. Set False to preserve relevance order.

    Usage:
        generator = Generator(llm=OpenAIGenerator(api_key="sk-..."))
        answer = generator.generate(
            query="What did they say about frozen yogurt?",
            results=results   # RetrievalResult from Stage 5
        )
        print(answer.text)
        print(answer.sources)
    """

    def __init__(self, llm: BaseGenerator, top_k: int = 5, chronological: bool = True):
        self.llm           = llm
        self.top_k         = top_k
        self.chronological = chronological

    def generate(self, query: str, results: RetrievalResult) -> GeneratorAnswer:
        """
        Generate an answer for the query using retrieved chunks.

        Parameters
        ----------
        query : str
            The user's question.
        results : RetrievalResult
            Output from Retriever.retrieve().
        """
        # select top_k chunks by relevance (RRF-fused rank)...
        top_chunks = [r.chunk for r in results.all[:self.top_k]]

        # ...then reorder chronologically before generation. The model reads
        # a coherent timeline this way instead of a relevance-ranked jumble
        # (e.g. 40s, 8s, 32s, 0s), which is harder for it to synthesize even
        # though relevance ranking is exactly what you want for *selecting*
        # which chunks to include.
        if self.chronological:
            top_chunks = sorted(top_chunks, key=lambda c: c.start)

        # generate answer
        answer_text = self.llm.generate(query=query, chunks=top_chunks)

        return GeneratorAnswer(
            query   = query,
            text    = answer_text,
            sources = top_chunks,
        )