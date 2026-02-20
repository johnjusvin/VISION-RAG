"""
vision_rag/video_chunker.py

Stage 2 of the vision_rag pipeline — Video Chunking.
Splits a video into time-based overlapping chunks.

Each chunk contains:
  - timestamps (start, end, duration)
  - keyframe image path (if use_frames=True)
  - transcript text (if use_asr=True)
  - metadata dict (source info, asr provider, etc.)

Usage:
    from vision_rag.video_chunker import Chunker
    from vision_rag.asr import WhisperLocalASR

    chunker = Chunker(
        asr=WhisperLocalASR(),
        use_asr=True,
        use_frames=True,
        chunk_size=5.0,
        chunk_overlap=1.0,
    )
    chunks = chunker.chunk("video.mp4")
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ──────────────────────────────────────────────────────────────
# Chunk — the core data structure that flows through the pipeline
# ──────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """
    A single video chunk. Produced by Chunker, consumed by every
    downstream stage (embedding, indexing, retrieval, generation).
    """
    chunk_id:   int             # 0, 1, 2 ...
    video_path: str             # absolute path to source video
    start:      float           # start time in seconds
    end:        float           # end time in seconds
    duration:   float           # end - start (convenience)

    frame_path: Optional[str]   # path to keyframe .jpg  (None if use_frames=False)
    text:       Optional[str]   # ASR transcript for this chunk (None if use_asr=False)

    metadata:   dict = field(default_factory=dict)
    # metadata keys set by Chunker:
    #   video_filename   — basename of the video
    #   chunk_index      — same as chunk_id (explicit label)
    #   total_chunks     — how many chunks in total
    #   keyframe_strategy — "middle" / "first"
    #   asr_provider     — class name of ASR used e.g. "WhisperLocalASR"

    def __repr__(self) -> str:
        return (
            f"Chunk("
            f"id={self.chunk_id}, "
            f"start={self.start:.2f}s, "
            f"end={self.end:.2f}s, "
            f"text={repr(self.text[:40] + '...') if self.text and len(self.text) > 40 else repr(self.text)}, "
            f"frame={'yes' if self.frame_path else 'no'}"
            f")"
        )


# ──────────────────────────────────────────────────────────────
# ASR Base — anyone can plug in their own
# ──────────────────────────────────────────────────────────────

class BaseASR:
    """
    Base class for all ASR providers.
    Subclass this and implement transcribe() to plug in any ASR.

    Example — custom ASR:
        class MyASR(BaseASR):
            def transcribe(self, audio_path: str) -> list[dict]:
                # your logic here
                return [{"start": 0.0, "end": 5.0, "text": "hello world"}]
    """

    def transcribe(self, audio_path: str) -> list[dict]:
        """
        Transcribe an audio file.

        Parameters
        ----------
        audio_path : str
            Path to a .wav audio file extracted from the video.

        Returns
        -------
        list of dicts with keys:
            start : float  — segment start in seconds
            end   : float  — segment end in seconds
            text  : str    — transcript text
        """
        raise NotImplementedError("Implement transcribe() in your ASR subclass.")

    @property
    def provider_name(self) -> str:
        return self.__class__.__name__


# ──────────────────────────────────────────────────────────────
# Built-in ASR providers
# ──────────────────────────────────────────────────────────────

class WhisperLocalASR(BaseASR):
    """
    Local Whisper model via faster-whisper (preferred) or openai-whisper (fallback).
    No API key needed. Runs entirely on your machine.

    pip install faster-whisper
    OR
    pip install openai-whisper
    """

    def __init__(self, model_size: str = "base", device: str = "cpu"):
        self.model_size = model_size
        self.device = device
        self._model = None

    def transcribe(self, audio_path: str) -> list[dict]:
        try:
            return self._transcribe_faster_whisper(audio_path)
        except ImportError:
            pass
        try:
            return self._transcribe_openai_whisper(audio_path)
        except ImportError:
            raise RuntimeError(
                "No Whisper backend found. Install one:\n"
                "  pip install faster-whisper\n"
                "  pip install openai-whisper"
            )

    def _transcribe_faster_whisper(self, audio_path: str) -> list[dict]:
        from faster_whisper import WhisperModel
        if self._model is None:
            self._model = WhisperModel(self.model_size, device=self.device, compute_type="int8")
        segments, _ = self._model.transcribe(audio_path, beam_size=1)
        return [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]

    def _transcribe_openai_whisper(self, audio_path: str) -> list[dict]:
        import whisper
        if self._model is None:
            self._model = whisper.load_model(self.model_size)
        result = self._model.transcribe(audio_path)
        return [
            {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
            for s in result.get("segments", [])
        ]


class OpenAIASR(BaseASR):
    """
    OpenAI Whisper API.

    Usage:
        asr = OpenAIASR(api_key="sk-...")
        # or set OPENAI_API_KEY env var and just do OpenAIASR()
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Pass api_key= or set OPENAI_API_KEY env var."
            )

    def transcribe(self, audio_path: str) -> list[dict]:
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("pip install openai")

        client = OpenAI(api_key=self.api_key)
        with open(audio_path, "rb") as f:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
            )
        return [
            {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
            for s in response.segments
        ]


class DeepgramASR(BaseASR):
    """
    Deepgram API.

    Usage:
        asr = DeepgramASR(api_key="...")
        # or set DEEPGRAM_API_KEY env var and just do DeepgramASR()
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Deepgram API key required. Pass api_key= or set DEEPGRAM_API_KEY env var."
            )

    def transcribe(self, audio_path: str) -> list[dict]:
        try:
            from deepgram import DeepgramClient, PrerecordedOptions
        except ImportError:
            raise RuntimeError("pip install deepgram-sdk")

        client = DeepgramClient(self.api_key)
        with open(audio_path, "rb") as f:
            response = client.listen.prerecorded.v("1").transcribe_file(
                {"buffer": f},
                PrerecordedOptions(model="nova-2", utterances=True),
            )
        words = response.results.channels[0].alternatives[0].words
        # Group words into sentence-level segments
        segments, buf, t0 = [], [], None
        for w in words:
            if t0 is None:
                t0 = w.start
            buf.append(w.punctuated_word)
            if w.punctuated_word.endswith((".", "?", "!")):
                segments.append({"start": t0, "end": w.end, "text": " ".join(buf)})
                buf, t0 = [], None
        if buf:
            segments.append({"start": t0, "end": words[-1].end, "text": " ".join(buf)})
        return segments


# ──────────────────────────────────────────────────────────────
# Chunker
# ──────────────────────────────────────────────────────────────

class Chunker:
    """
    Splits a video into overlapping time-based chunks.

    Parameters
    ----------
    asr : BaseASR | None
        Any ASR provider. Pass None or set use_asr=False to skip transcription.
    use_asr : bool
        Whether to transcribe audio. Default = True.
    use_frames : bool
        Whether to extract a keyframe per chunk. Default = True.
    chunk_size : float
        Duration of each chunk in seconds. Default = 5.0.
    chunk_overlap : float
        Overlap between consecutive chunks in seconds. Default = 1.0.
    keyframe_strategy : str
        Where to sample the keyframe from each chunk: "middle" or "first".
    frames_dir : str | None
        Directory to save keyframe images. Defaults to .vision_rag_cache/frames/.
    """

    def __init__(
        self,
        asr: Optional[BaseASR] = None,
        use_asr: bool = True,
        use_frames: bool = True,
        chunk_size: float = 5.0,
        chunk_overlap: float = 1.0,
        keyframe_strategy: str = "middle",
        frames_dir: Optional[str] = None,
    ):
        if use_asr and asr is None:
            raise ValueError(
                "use_asr=True but no asr= provided. "
                "Pass an ASR provider e.g. asr=WhisperLocalASR()"
            )
        self.asr = asr
        self.use_asr = use_asr
        self.use_frames = use_frames
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.keyframe_strategy = keyframe_strategy
        self.frames_dir = frames_dir

    def chunk(self, video_path: str) -> list[Chunk]:
        """
        Main entry point. Returns a list of Chunk objects.

        Parameters
        ----------
        video_path : str
            Path to the video file.
        """
        path = Path(video_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")

        duration = self._get_duration(path)
        windows  = self._build_windows(duration)
        total    = len(windows)

        # Extract full audio once, transcribe once, slice per chunk
        segments = []
        if self.use_asr:
            audio_path = self._extract_audio(path)
            segments   = self.asr.transcribe(audio_path)
            os.remove(audio_path)   # cleanup temp audio

        # Set up frames output dir
        frames_dir = None
        if self.use_frames:
            frames_dir = Path(self.frames_dir) if self.frames_dir else \
                         Path(".vision_rag_cache") / "frames" / path.stem
            frames_dir.mkdir(parents=True, exist_ok=True)

        chunks = []
        for i, (start, end) in enumerate(windows):
            frame_path = None
            if self.use_frames:
                frame_path = self._extract_keyframe(path, start, end, frames_dir, i)

            text = None
            if self.use_asr:
                text = self._slice_transcript(segments, start, end)

            chunks.append(Chunk(
                chunk_id   = i,
                video_path = str(path),
                start      = start,
                end        = end,
                duration   = round(end - start, 4),
                frame_path = str(frame_path) if frame_path else None,
                text       = text,
                metadata   = {
                    "video_filename":    path.name,
                    "chunk_index":       i,
                    "total_chunks":      total,
                    "keyframe_strategy": self.keyframe_strategy if self.use_frames else None,
                    "asr_provider":      self.asr.provider_name if self.use_asr else None,
                },
            ))

        return chunks

    # ── window generation ────────────────────────────────────────

    def _build_windows(self, duration: float) -> list[tuple[float, float]]:
        step     = self.chunk_size - self.chunk_overlap
        windows  = []
        start    = 0.0
        while start < duration:
            end = min(start + self.chunk_size, duration)
            windows.append((round(start, 4), round(end, 4)))
            if end >= duration:
                break
            start += step
        return windows

    # ── keyframe extraction ──────────────────────────────────────

    def _extract_keyframe(
        self,
        video_path: Path,
        start: float,
        end: float,
        output_dir: Path,
        chunk_id: int,
    ) -> Optional[Path]:
        out_path = output_dir / f"chunk_{chunk_id:04d}.jpg"
        if out_path.exists():
            return out_path     # use cached

        t = (start + end) / 2 if self.keyframe_strategy == "middle" else start + 0.5

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(t),
            "-i", str(video_path),
            "-vframes", "1",
            "-q:v", "2",
            str(out_path),
        ]
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if result.returncode != 0 or not out_path.exists():
            return None     # frame extraction failed, return None cleanly

        return out_path

    # ── audio extraction ─────────────────────────────────────────

    def _extract_audio(self, video_path: Path) -> str:
        """Extract full audio from video to a temp .wav file."""
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-ar", "16000",
            "-ac", "1",
            "-f", "wav",
            tmp.name,
        ]
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg audio extraction failed for: {video_path}")

        return tmp.name

    # ── transcript slicing ───────────────────────────────────────

    def _slice_transcript(self, segments: list[dict], start: float, end: float) -> str:
        """Collect transcript text from segments overlapping with [start, end]."""
        texts = [
            seg["text"] for seg in segments
            if seg.get("end", 0) > start and seg.get("start", 0) < end
        ]
        return " ".join(texts).strip() or None

    # ── duration ─────────────────────────────────────────────────

    def _get_duration(self, video_path: Path) -> float:
        """Get video duration in seconds via ffprobe."""
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            return float(result.stdout.strip())
        except (ValueError, subprocess.SubprocessError):
            raise RuntimeError(
                f"Could not read duration for {video_path}. "
                "Make sure ffprobe is installed: brew install ffmpeg"
            )
