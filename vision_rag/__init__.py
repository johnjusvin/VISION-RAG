from vision_rag.video_ingestion import VideoLoader, VideoDocument
from vision_rag.video_chunker import Chunker, Chunk, BaseASR, WhisperLocalASR, OpenAIASR, DeepgramASR
from vision_rag.embedding import (
    EmbeddingBuilder, EmbeddedChunk,
    BaseTextEmbedder, BaseImageEmbedder,
    OpenAITextEmbedder, SentenceTransformerTextEmbedder, CLIPTextEmbedder,
    CLIPImageEmbedder, OpenAIImageEmbedder,
)
from vision_rag.vectorstores import BaseVectorStore, SearchResult, FAISS, Chroma
from vision_rag.retriever import Retriever, RetrievalResult
from vision_rag.generator import (
    Generator, GeneratorAnswer,
    BaseGenerator,
    OpenAIGenerator, AnthropicGenerator, GeminiGenerator, OllamaGenerator,
)

__version__ = "0.1.2"
__all__ = [
    # Stage 1
    "VideoLoader", "VideoDocument",
    # Stage 2
    "Chunker", "Chunk",
    "BaseASR", "WhisperLocalASR", "OpenAIASR", "DeepgramASR",
    # Stage 3
    "EmbeddingBuilder", "EmbeddedChunk",
    "BaseTextEmbedder", "BaseImageEmbedder",
    "OpenAITextEmbedder", "SentenceTransformerTextEmbedder", "CLIPTextEmbedder",
    "CLIPImageEmbedder", "OpenAIImageEmbedder",
    # Stage 4
    "BaseVectorStore", "SearchResult",
    "FAISS", "Chroma",
    # Stage 5
    "Retriever", "RetrievalResult",
    # Stage 6
    "Generator", "GeneratorAnswer",
    "BaseGenerator",
    "OpenAIGenerator", "AnthropicGenerator", "GeminiGenerator", "OllamaGenerator",
]