"""Download example audio files from the GitHub release.

Usage:
    python download_audio.py

The audio files are not included in the repository to keep clone sizes small.
This script downloads them once from the GitHub release assets.
"""

import sys
import urllib.request
import zipfile
from pathlib import Path

RELEASE_URL = (
    "https://github.com/jmg049/Spectrograms/releases/download/"
    "example-audio-v1/example_audio.zip"
)

DEST = Path(__file__).parent / "audio"


def download_audio(force: bool = False) -> None:
    if DEST.exists() and not force:
        print(f"Audio already present at {DEST}. Pass --force to re-download.")
        return

    zip_path = DEST.parent / "example_audio.zip"

    print("Downloading example audio from GitHub release...")

    def reporthook(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            bar = int(pct // 2)
            sys.stdout.write(f"\r  [{'#' * bar}{' ' * (50 - bar)}] {pct:.1f}%")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(RELEASE_URL, zip_path, reporthook)
        print()  # newline after progress bar
    except Exception as e:
        zip_path.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed: {e}") from e

    print(f"Extracting to {DEST.parent}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DEST.parent)

    zip_path.unlink()
    print("Done.")


if __name__ == "__main__":
    force = "--force" in sys.argv
    download_audio(force=force)
