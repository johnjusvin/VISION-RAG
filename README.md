# vision-rag

**Video RAG** — A Python library for Retrieval-Augmented Generation over video.

Ask questions about any video and get answers using the transcript and visual frames.

---

## Install

```bash
pip install vision-rag
```

---

## How it works

```
Video → Chunks → Embeddings → Vector Store → Retrieval → Answer
```

1. **Ingest** — reads video metadata
2. **Chunk** — splits video into time-based overlapping chunks with frames and transcript
3. **Embed** — converts text and frames into vectors (your choice of model)
4. **Index** — stores vectors in FAISS or Chroma
5. **Retrieve** — searches both text and image indexes for a query
6. **Generate** — passes retrieved chunks to a VLM to generate the answer

---

## Quick Start

```python
from vision_rag.video_ingestion import VideoLoader
from vision_rag.video_chunker import Chunker, WhisperLocalASR
from vision_rag.embedding import EmbeddingBuilder, BaseTextEmbedder, BaseImageEmbedder
from vision_rag.vectorstores import FAISS
from vision_rag.retriever import Retriever
from vision_rag.generator import Generator, OllamaGenerator
import requests, base64

# --- your choice of embedder (example: Jina v4) ---
class JinaTextEmbedder(BaseTextEmbedder):
    def __init__(self, api_key):
        self.api_key = api_key
    def embed(self, text):
        r = requests.post("https://api.jina.ai/v1/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": "jina-embeddings-v4", "input": [{"text": text}], "task": "retrieval.passage"})
        return r.json()["data"][0]["embedding"]

class JinaImageEmbedder(BaseImageEmbedder):
    def __init__(self, api_key):
        self.api_key = api_key
    def embed(self, image_path):
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        r = requests.post("https://api.jina.ai/v1/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": "jina-embeddings-v4", "input": [{"image": b64}], "task": "retrieval.passage"})
        return r.json()["data"][0]["embedding"]

# Stage 1 — Ingest
video_doc = VideoLoader().load("video.mp4")

# Stage 2 — Chunk
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
query   = input("Ask a question: ")
results = Retriever(store=store, text_embedder=text_embedder).retrieve(query)
answer  = Generator(llm=OllamaGenerator(model="llava:7b")).generate(query=query, results=results)

print(answer.text)
```

---

## Chunker

```python
from vision_rag.video_chunker import Chunker, WhisperLocalASR

chunker = Chunker(
    asr=WhisperLocalASR(model_size="medium"),  # or OpenAIASR, DeepgramASR, or your own
    use_asr=True,
    use_frames=True,
    chunk_size=5.0,      # seconds per chunk
    chunk_overlap=1.0,   # overlap between chunks
)
chunks = chunker.chunk("video.mp4")
```

Each chunk contains:

| Field | Description |
|---|---|
| `chunk.chunk_id` | chunk index |
| `chunk.start` | start time in seconds |
| `chunk.end` | end time in seconds |
| `chunk.duration` | duration in seconds |
| `chunk.text` | transcript for this chunk |
| `chunk.frame_path` | path to keyframe image |
| `chunk.metadata` | source info, asr provider, etc. |

---

## ASR — Bring Your Own

vision_rag ships with built-in ASR providers but you can plug in anything:

```python
from vision_rag.video_chunker import BaseASR

# built-in
from vision_rag.video_chunker import WhisperLocalASR, OpenAIASR, DeepgramASR

# your own — any model, any API
class MyASR(BaseASR):
    def transcribe(self, audio_path: str) -> list[dict]:
        return [{"start": 0.0, "end": 5.0, "text": "..."}]

chunker = Chunker(asr=MyASR(), use_asr=True)
```

---

## Embedding — Bring Your Own

```python
from vision_rag.embedding import BaseTextEmbedder, BaseImageEmbedder

# your own text embedder
class MyTextEmbedder(BaseTextEmbedder):
    def embed(self, text: str) -> list[float]:
        return [...]  # your model or API

# your own image embedder
class MyImageEmbedder(BaseImageEmbedder):
    def embed(self, image_path: str) -> list[float]:
        return [...]  # your model or API

embedder = EmbeddingBuilder(
    text_embedding=MyTextEmbedder(),
    image_embedding=MyImageEmbedder(),
)
```

Built-in providers: `OpenAITextEmbedder`, `SentenceTransformerTextEmbedder`, `CLIPImageEmbedder`, `OpenAIImageEmbedder`

---

## Vector Stores

```python
from vision_rag.vectorstores import FAISS, Chroma

# FAISS — fast local search
store = FAISS()
store.index(embedded_chunks)
store.save("my_index")
store.load("my_index")

# Chroma — persistent local DB
store = Chroma(path="my_chroma_db")
store.index(embedded_chunks)
```

Plug in your own:

```python
from vision_rag.vectorstores import BaseVectorStore

class MyVectorStore(BaseVectorStore):
    def index(self, embedded_chunks): ...
    def search_text(self, vector, top_k): ...
    def search_image(self, vector, top_k): ...
```

---

## Retrieval

```python
from vision_rag.retriever import Retriever

retriever = Retriever(
    store=store,
    text_embedder=text_embedder,
    top_k_text=5,
    top_k_image=5,
)

# semantic search
results = retriever.retrieve("What did they say about frozen yogurt?")
results.text_results    # top text matches
results.image_results   # top image matches
results.all             # combined, ranked by score

# time-based search
chunks = retriever.retrieve_by_time(start=10.0, end=20.0)
```

---

## Generation — Bring Your Own VLM

```python
from vision_rag.generator import Generator, OpenAIGenerator, AnthropicGenerator, GeminiGenerator, OllamaGenerator

# GPT-4o
generator = Generator(llm=OpenAIGenerator(api_key="sk-..."))

# Claude
generator = Generator(llm=AnthropicGenerator(api_key="sk-ant-..."))

# Gemini
generator = Generator(llm=GeminiGenerator(api_key="..."))

# Ollama (local)
generator = Generator(llm=OllamaGenerator(model="llava:7b"))

# your own
from vision_rag.generator import BaseGenerator

class MyGenerator(BaseGenerator):
    def generate(self, query: str, chunks) -> str:
        return "answer..."

generator = Generator(llm=MyGenerator())
```

---

## Dependencies

vision_rag ships with only one hard dependency — `pymediainfo`. Everything else is installed based on what you use:

| Feature | Install |
|---|---|
| ASR (local Whisper) | `pip install faster-whisper` |
| ASR (OpenAI) | `pip install openai` |
| ASR (Deepgram) | `pip install deepgram-sdk` |
| Frames + Audio | `brew install ffmpeg` |
| FAISS vector store | `pip install faiss-cpu` |
| Chroma vector store | `pip install chromadb` |
| OpenAI embedding | `pip install openai` |
| Sentence Transformers | `pip install sentence-transformers` |
| CLIP image embedding | `pip install git+https://github.com/openai/CLIP.git torch Pillow` |
| Ollama generation | `pip install ollama` |
| Anthropic generation | `pip install anthropic` |
| Gemini generation | `pip install google-genai` |

---

## License

MIT
