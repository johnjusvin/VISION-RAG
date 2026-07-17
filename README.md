# vision-rag

<p align="center">
  <strong>Retrieval-Augmented Generation over video — in pure Python.</strong><br>
  Ask any question about a video. Get a grounded answer from transcript + frames.
</p>

<p align="center">
  <a href="https://pypi.org/project/vision-rag/"><img src="https://img.shields.io/pypi/v/vision-rag?color=blue&label=PyPI" alt="PyPI version"></a>
  <a href="https://pypi.org/project/vision-rag/"><img src="https://img.shields.io/pypi/pyversions/vision-rag" alt="Python versions"></a>
  <a href="https://pypi.org/project/vision-rag/"><img src="https://img.shields.io/pypi/dm/vision-rag?color=green" alt="Monthly downloads"></a>
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="License">
</p>

---

## What is vision-rag?

`vision-rag` is a modular Python library that lets you build **video question-answering pipelines** using any combination of:

- **ASR** — Whisper (local), OpenAI, Deepgram, or your own
- **Embeddings** — OpenAI, CLIP, SentenceTransformers, Jina, or your own
- **Vector stores** — FAISS, Chroma, or your own
- **Generators (VLMs)** — GPT-4o, Claude, Gemini, Ollama, or your own

Every component has a clean base class. Plug in any model, any API — the rest of the pipeline stays the same.

---

## Pipeline

```
Video
  │
  ▼
Stage 1 ── VideoLoader       reads video metadata
  │
  ▼
Stage 2 ── Chunker           splits into overlapping time windows
              ├── keyframe extraction  (ffmpeg)
              └── transcription        (ASR of your choice)
  │
  ▼
Stage 3 ── EmbeddingBuilder  converts text + frames → vectors
  │
  ▼
Stage 4 ── FAISS / Chroma    indexes vectors for fast retrieval
  │
  ▼
Stage 5 ── Retriever         searches text + image indexes, fuses with RRF
  │
  ▼
Stage 6 ── Generator         sends chunks to a VLM → grounded answer
```

---

## Install

```bash
pip install vision-rag
```

Only one hard dependency is installed: `pymediainfo`. Everything else is optional — install only what you need (see [Dependencies](#dependencies)).

---

## Quick Start

```python
import requests, base64
from vision_rag import (
    VideoLoader, Chunker, WhisperLocalASR,
    EmbeddingBuilder, BaseTextEmbedder, BaseImageEmbedder,
    FAISS, Retriever, Generator, OllamaGenerator,
)

# ── Bring your own embedder (example: Jina v4 multimodal) ──────────────────

class JinaTextEmbedder(BaseTextEmbedder):
    def __init__(self, api_key):
        self.api_key = api_key
    def embed(self, text: str) -> list[float]:
        r = requests.post(
            "https://api.jina.ai/v1/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": "jina-embeddings-v4",
                  "input": [{"text": text}], "task": "retrieval.passage"},
        )
        return r.json()["data"][0]["embedding"]

class JinaImageEmbedder(BaseImageEmbedder):
    def __init__(self, api_key):
        self.api_key = api_key
    def embed(self, image_path: str) -> list[float]:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        r = requests.post(
            "https://api.jina.ai/v1/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": "jina-embeddings-v4",
                  "input": [{"image": b64}], "task": "retrieval.passage"},
        )
        return r.json()["data"][0]["embedding"]

# ── Pipeline ────────────────────────────────────────────────────────────────

# Stage 1 — Ingest
video = VideoLoader().load("video.mp4")
print(video)  # VideoDocument(file='video.mp4', duration=120.0s, ...)

# Stage 2 — Chunk  (5-second windows, 1s overlap, extract frames + transcript)
chunks = Chunker(
    asr=WhisperLocalASR(model_size="base"),
    use_asr=True,
    use_frames=True,
    chunk_size=5.0,
    chunk_overlap=1.0,
).chunk("video.mp4")

# Stage 3 — Embed
text_embedder  = JinaTextEmbedder(api_key="your_jina_key")
image_embedder = JinaImageEmbedder(api_key="your_jina_key")
embedded_chunks = EmbeddingBuilder(
    text_embedding=text_embedder,
    image_embedding=image_embedder,
).embed(chunks)

# Stage 4 — Index
store = FAISS()
store.index(embedded_chunks)

# Stage 5 + 6 — Retrieve and Generate
query   = "What was shown on the whiteboard?"
results = Retriever(store=store, text_embedder=text_embedder).retrieve(query)
answer  = Generator(llm=OllamaGenerator(model="llava:7b")).generate(
    query=query, results=results
)

print(answer.text)
print(answer.sources)  # list of EmbeddedChunk used to produce the answer
```

---

## Components

### Stage 2 — Chunker

```python
from vision_rag import Chunker, WhisperLocalASR, OpenAIASR, DeepgramASR

chunker = Chunker(
    asr=WhisperLocalASR(model_size="medium"),  # or OpenAIASR(), DeepgramASR()
    use_asr=True,       # transcribe audio
    use_frames=True,    # extract a keyframe per chunk
    chunk_size=5.0,     # seconds per chunk
    chunk_overlap=1.0,  # overlap between consecutive chunks
)
chunks = chunker.chunk("video.mp4")
```

Each `Chunk` contains:

| Field | Type | Description |
|---|---|---|
| `chunk_id` | `int` | 0-indexed chunk number |
| `start` | `float` | Start time in seconds |
| `end` | `float` | End time in seconds |
| `duration` | `float` | `end - start` |
| `text` | `str \| None` | ASR transcript for this window |
| `frame_path` | `str \| None` | Path to the extracted keyframe `.jpg` |
| `metadata` | `dict` | Source video, ASR provider, chunk count, etc. |

**Bring your own ASR:**

```python
from vision_rag import BaseASR

class MyASR(BaseASR):
    def transcribe(self, audio_path: str) -> list[dict]:
        # return list of {"start": float, "end": float, "text": str}
        return [{"start": 0.0, "end": 5.0, "text": "..."}]
```

---

### Stage 3 — Embedding

Built-in providers:

```python
from vision_rag import (
    OpenAITextEmbedder,              # text-embedding-3-small
    SentenceTransformerTextEmbedder, # all-MiniLM-L6-v2 (local)
    CLIPTextEmbedder,                # ViT-B/32 text encoder (local)
    CLIPImageEmbedder,               # ViT-B/32 image encoder (local)
    OpenAIImageEmbedder,             # via OpenAI API
)
```

**Bring your own embedder:**

```python
from vision_rag import BaseTextEmbedder, BaseImageEmbedder

class MyTextEmbedder(BaseTextEmbedder):
    def embed(self, text: str) -> list[float]:
        return [...]  # call your model or API

class MyImageEmbedder(BaseImageEmbedder):
    def embed(self, image_path: str) -> list[float]:
        return [...]  # call your model or API
```

> **Note on CLIP:** pair `CLIPTextEmbedder` with `CLIPImageEmbedder` (same `model=` string) for cross-modal search. Mixing CLIP image vectors with an unrelated text embedder produces vectors in incompatible spaces.

---

### Stage 4 — Vector Stores

```python
from vision_rag import FAISS, Chroma

# FAISS — fast, local, no server
store = FAISS()
store.index(embedded_chunks)
store.save("my_index")
store.load("my_index")

# Chroma — persistent local DB
store = Chroma(path="my_chroma_db")
store.index(embedded_chunks)
```

**Bring your own:**

```python
from vision_rag import BaseVectorStore

class MyVectorStore(BaseVectorStore):
    def index(self, embedded_chunks): ...
    def search_text(self, vector, top_k): ...
    def search_image(self, vector, top_k): ...
    def save(self, path): ...
    def load(self, path): ...
```

---

### Stage 5 — Retriever

```python
from vision_rag import Retriever

retriever = Retriever(
    store=store,
    text_embedder=text_embedder,  # same embedder used at indexing time
    top_k_text=5,
    top_k_image=5,
)

results = retriever.retrieve("What did they say about the product launch?")

results.text_results   # top chunks from the text index
results.image_results  # top chunks from the image index
results.all            # both, fused and ranked via Reciprocal Rank Fusion (RRF)
results.by_time        # same chunks, sorted chronologically

# Time-based retrieval (no embedding needed)
chunks = retriever.retrieve_by_time(start=30.0, end=60.0)
```

Results from text and image searches are fused using **Reciprocal Rank Fusion (RRF)** — a rank-based method that works even when text and image similarity scores are on different scales.

---

### Stage 6 — Generator

Built-in providers:

```python
from vision_rag import Generator, OpenAIGenerator, AnthropicGenerator, GeminiGenerator, OllamaGenerator

# GPT-4o  (VLM — text + images)
answer = Generator(llm=OpenAIGenerator(api_key="sk-...")).generate(query, results)

# Claude  (VLM — text + images)
answer = Generator(llm=AnthropicGenerator(api_key="sk-ant-...")).generate(query, results)

# Gemini  (VLM — text + images)
answer = Generator(llm=GeminiGenerator(api_key="...")).generate(query, results)

# Ollama  (local — text + images for vision models)
answer = Generator(llm=OllamaGenerator(model="llava:7b")).generate(query, results)
```

**Bring your own:**

```python
from vision_rag import BaseGenerator

class MyGenerator(BaseGenerator):
    def generate(self, query: str, chunks) -> str:
        # call your model
        return "the answer..."

answer = Generator(llm=MyGenerator()).generate(query, results)
```

`GeneratorAnswer` fields:

| Field | Description |
|---|---|
| `answer.text` | The generated answer string |
| `answer.query` | The original question |
| `answer.sources` | List of `EmbeddedChunk` objects used to produce the answer |

---

## Dependencies

`vision-rag` ships with **one hard dependency**: `pymediainfo`. All others are optional — install only what your use case needs.

| Feature | Install |
|---|---|
| Video metadata | `pip install pymediainfo` *(included)* |
| Frame & audio extraction | `apt install ffmpeg` / `brew install ffmpeg` |
| ASR — local Whisper | `pip install faster-whisper` |
| ASR — OpenAI Whisper API | `pip install openai` |
| ASR — Deepgram | `pip install deepgram-sdk` |
| FAISS vector store | `pip install faiss-cpu` |
| Chroma vector store | `pip install chromadb` |
| OpenAI embeddings / generation | `pip install openai` |
| SentenceTransformers text embedding | `pip install sentence-transformers` |
| CLIP image + text embedding | `pip install git+https://github.com/openai/CLIP.git torch Pillow` |
| Ollama generation (local) | `pip install ollama` |
| Anthropic generation | `pip install anthropic` |
| Gemini generation | `pip install google-genai` |

---

## License

MIT © [JOHNJUSVIN](https://pypi.org/user/JOHNJUSVIN/)
