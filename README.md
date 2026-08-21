# vision-rag

Build question-answering pipelines for video with Python. `vision-rag` splits a
video into timestamped chunks, transcribes its audio, extracts frames, retrieves
relevant moments, and generates a grounded answer.

<p>
  <a href="https://pypi.org/project/vision-rag/"><img src="https://img.shields.io/pypi/v/vision-rag?color=blue&label=PyPI" alt="PyPI version"></a>
  <a href="https://pypi.org/project/vision-rag/"><img src="https://img.shields.io/pypi/pyversions/vision-rag" alt="Python versions"></a>
  <img src="https://img.shields.io/badge/license-Apache--2.0-lightgrey" alt="Apache License 2.0">
</p>

## Features

- Transcript and frame-based video retrieval
- Local and hosted model providers
- FAISS and Chroma vector stores
- Replaceable ASR, embedding, storage, and generation components
- Python 3.9+

## Install

```bash
pip install vision-rag
```

Install only the optional packages needed by your pipeline. This local example
uses FFmpeg, Whisper, SentenceTransformers, FAISS, and Ollama:

```bash
pip install faster-whisper sentence-transformers faiss-cpu ollama
ollama pull llava:7b
```

FFmpeg must also be installed and available on your system `PATH`.

## Quick start

```python
from vision_rag import (
    Chunker,
    EmbeddingBuilder,
    FAISS,
    Generator,
    OllamaGenerator,
    Retriever,
    SentenceTransformerTextEmbedder,
    WhisperLocalASR,
)

# Split the video, transcribe its audio, and extract one frame per chunk.
chunks = Chunker(
    asr=WhisperLocalASR(model_size="base"),
    use_asr=True,
    use_frames=True,
    chunk_size=5.0,
    chunk_overlap=1.0,
).chunk("video.mp4")

# Embed and index the transcripts.
text_embedder = SentenceTransformerTextEmbedder()
embedded_chunks = EmbeddingBuilder(
    text_embedding=text_embedder,
).embed(chunks)

store = FAISS()
store.index(embedded_chunks)

# Retrieve relevant moments and answer using their transcripts and frames.
question = "What did the speaker say about the product launch?"
results = Retriever(
    store=store,
    text_embedder=text_embedder,
).retrieve(question)

answer = Generator(
    llm=OllamaGenerator(model="llava:7b"),
).generate(question, results)

print(answer.text)
print(answer.sources)
```

## Providers

| Component | Built-in options |
|---|---|
| Speech recognition | Local Whisper, OpenAI, Deepgram |
| Text embeddings | OpenAI, SentenceTransformers, CLIP |
| Image embeddings | CLIP, OpenAI |
| Vector stores | FAISS, Chroma |
| Generation | OpenAI, Anthropic, Gemini, Ollama |

Each component has a base class, so you can connect another model or service
without changing the rest of the pipeline.

## License

Licensed under the [Apache License 2.0](LICENSE) © [JOHNJUSVIN](https://pypi.org/user/JOHNJUSVIN/).
