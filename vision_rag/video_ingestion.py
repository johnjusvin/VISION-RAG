"""
vision_rag/video_ingestion.py

Stage 1 of the vision_rag pipeline — Video Ingestion.
Reads a video file and returns its metadata as a VideoDocument.

Usage:
    from vision_rag.video_ingestion import VideoLoader
    loader = VideoLoader()
    video_doc = loader.load("sample.mp4")
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from pymediainfo import MediaInfo  # pip install pymediainfo


@dataclass
class VideoDocument:
    """
    Returned by VideoLoader.load().
    Contains only raw video metadata — no frames, no audio.
    Passed into the next stage (chunking) for further processing.
    """
    # Source
    source_path: str
    filename: str
    format: str               # e.g. "mp4", "avi"
    file_size_bytes: int

    # Timing
    duration_seconds: float
    fps: float
    total_frames: int

    # Dimensions
    width: int
    height: int

    # Codec
    codec: str                # e.g. "avc1", "HEVC"

    def __repr__(self) -> str:
        return (
            f"VideoDocument("
            f"file={self.filename!r}, "
            f"duration={self.duration_seconds:.2f}s, "
            f"fps={self.fps}, "
            f"resolution={self.width}x{self.height}, "
            f"frames={self.total_frames}, "
            f"codec={self.codec!r}"
            f")"
        )


class VideoLoader:
    """
    Loads a video file and extracts its metadata.

    Usage:
        loader = VideoLoader()
        video_doc = loader.load("sample.mp4")
    """

    def load(self, video_path: str) -> VideoDocument:
        path = Path(video_path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        info = MediaInfo.parse(str(path))

        if not info.video_tracks:
            raise ValueError(f"No video track found in: {path}")

        track = info.video_tracks[0]

        fps      = float(track.frame_rate or 0)
        width    = int(track.width or 0)
        height   = int(track.height or 0)
        duration = float(track.duration or 0) / 1000      # ms → seconds
        frames   = int(track.frame_count or round(duration * fps))
        codec    = track.codec_id or track.format or "unknown"

        return VideoDocument(
            source_path      = str(path),
            filename         = path.name,
            format           = path.suffix.lstrip(".").lower(),
            file_size_bytes  = os.path.getsize(path),
            duration_seconds = round(duration, 4),
            fps              = fps,
            total_frames     = frames,
            width            = width,
            height           = height,
            codec            = codec,
        )
