#!/usr/bin/env python3
"""
Download videos listed in data/videos.csv that are missing locally.

The script reads the repository's videos.csv, checks each file_path, and
downloads only the missing files from source_url using yt-dlp.

Usage:
    python scripts/download_videos.py
    python scripts/download_videos.py --csv data/videos.csv --root .
    python scripts/download_videos.py --dry-run

Notes:
    - Requires: pip install yt-dlp
    - Uses the file_path column from videos.csv as the final local path.
    - Skips rows whose source_url is missing, empty, or set to "unknown".
    - Does not overwrite existing files unless --force is passed.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


@dataclass
class DownloadItem:
    video_id: str
    title: str
    file_path: Path
    source_url: str


def load_download_items(videos_csv: Path, root: Path) -> list[DownloadItem]:
    items: list[DownloadItem] = []
    with videos_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source_url = (row.get("source_url") or "").strip()
            file_path_value = (row.get("file_path") or "").strip()
            if not file_path_value:
                continue
            items.append(
                DownloadItem(
                    video_id=(row.get("video_id") or "").strip(),
                    title=(row.get("title") or "").strip(),
                    file_path=(root / file_path_value).resolve(),
                    source_url=source_url,
                )
            )
    return items


def download_with_ytdlp(url: str, output_path: Path, force: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure Deno JS runtime is in PATH if installed in user home
    env = os.environ.copy()
    deno_bin = Path.home() / ".deno" / "bin"
    if deno_bin.exists():
        env["PATH"] = f"{deno_bin}:{env.get('PATH', '')}"

    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        url,
        "-o",
        str(output_path),
        "--extractor-args",
        "youtube:player_client=web_embedded",
        "--remote-components",
        "ejs:github",
        "-f",
        "bv*[ext=mp4]+ba*[ext=m4a]/b[ext=mp4]/best",
        "--merge-output-format",
        "mp4",
        "--no-playlist",
        "--newline",
        "--concurrent-fragments",
        "4",
    ]

    if force:
        command.append("--force-overwrites")

    subprocess.run(command, check=True, env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download missing video assets listed in videos.csv.")
    parser.add_argument("--csv", default=str(REPO_ROOT / "data" / "videos.csv"), help="Path to videos.csv")
    parser.add_argument("--root", default=str(REPO_ROOT), help="Repository root used to resolve relative file paths")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files if they are present")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be downloaded without downloading")
    args = parser.parse_args()

    repo_root = Path(args.root).resolve()
    videos_csv_arg = Path(args.csv)
    videos_csv = videos_csv_arg if videos_csv_arg.is_absolute() else (repo_root / videos_csv_arg).resolve()

    if not videos_csv.exists():
        raise SystemExit(f"videos.csv not found: {videos_csv}")

    items = load_download_items(videos_csv, repo_root)
    if not items:
        print("No rows found in videos.csv.")
        return 0

    # Filter items to download: if --force is set, re-download all items; otherwise only missing files.
    to_process = [item for item in items if args.force or not item.file_path.exists()]
    print(f"Found {len(items)} video rows; {len(to_process)} to process (missing locally or forced).")

    downloaded = 0
    skipped = 0
    failed = 0

    for item in to_process:
        if not item.source_url or item.source_url.lower() == "unknown":
            print(f"[SKIP] {item.video_id}: no usable source_url ({item.source_url!r})")
            skipped += 1
            continue

        print(f"[MISSING] {item.video_id} -> {item.file_path}")
        print(f"          source: {item.source_url}")

        if args.dry_run:
            print("          dry-run: not downloading")
            skipped += 1
            continue

        try:
            download_with_ytdlp(item.source_url, item.file_path, force=args.force)
            if item.file_path.exists():
                print("          downloaded successfully")
                downloaded += 1
            else:
                print("          download finished but file still missing")
                failed += 1
        except subprocess.CalledProcessError as exc:
            print(f"          download failed ({exc.returncode})")
            failed += 1
        except FileNotFoundError:
            print("          yt-dlp is not installed. Install it with: pip install yt-dlp")
            return 2

    print("")
    print(f"Summary: downloaded={downloaded}, skipped={skipped}, failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
