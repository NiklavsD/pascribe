"""Save per-user audio recordings to disk as WAV files."""

from __future__ import annotations

import io
import logging
import wave
from datetime import datetime, timezone
from pathlib import Path

from config import RECORDINGS_DIR, SAMPLE_RATE, CHANNELS, PCM_SAMPLE_WIDTH

log = logging.getLogger(__name__)


def _user_dir(username: str, date: datetime | None = None) -> Path:
    """Return (and create) the directory for a user's recordings on a given date."""
    date = date or datetime.now(timezone.utc)
    day_str = date.strftime("%Y-%m-%d")
    d = RECORDINGS_DIR / day_str / username
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_pcm_as_wav(
    pcm_data: bytes,
    username: str,
    *,
    label: str = "",
    date: datetime | None = None,
    sample_rate: int = SAMPLE_RATE,
    channels: int = CHANNELS,
    sample_width: int = PCM_SAMPLE_WIDTH,
) -> Path:
    """Write raw PCM bytes to a timestamped WAV file. Returns the file path."""
    now = datetime.now(timezone.utc)
    date = date or now
    d = _user_dir(username, date)
    ts = now.strftime("%H%M%S_%f")
    suffix = f"_{label}" if label else ""
    filename = f"{ts}{suffix}.wav"
    filepath = d / filename

    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)

    log.debug("Saved %d bytes → %s", len(pcm_data), filepath)
    return filepath


def get_user_recordings(username: str, date: datetime | None = None) -> list[Path]:
    """Return all WAV files for a user on a given date, sorted by name."""
    d = _user_dir(username, date)
    return sorted(d.glob("*.wav"))


def get_all_users_for_date(date: datetime | None = None) -> list[str]:
    """Return all usernames that have recordings on a given date."""
    date = date or datetime.now(timezone.utc)
    day_str = date.strftime("%Y-%m-%d")
    day_dir = RECORDINGS_DIR / day_str
    if not day_dir.exists():
        return []
    return [d.name for d in sorted(day_dir.iterdir()) if d.is_dir()]


def get_processed_marker_path(date: datetime | None = None) -> Path:
    """Path to the file tracking which segments have been transcribed."""
    date = date or datetime.now(timezone.utc)
    day_str = date.strftime("%Y-%m-%d")
    return RECORDINGS_DIR / day_str / "_processed.txt"


def get_processed_files(date: datetime | None = None) -> set[str]:
    """Return set of filenames already transcribed."""
    marker = get_processed_marker_path(date)
    if not marker.exists():
        return set()
    return set(marker.read_text().strip().splitlines())


def mark_files_processed(filenames: list[str], date: datetime | None = None) -> None:
    """Append filenames to the processed marker."""
    marker = get_processed_marker_path(date)
    with open(marker, "a") as f:
        for name in filenames:
            f.write(name + "\n")


def get_all_speech_segments_chronological(date: datetime | None = None) -> list[tuple[str, Path]]:
    """Get all speech segments across all users, sorted chronologically.

    Returns list of (username, filepath) tuples sorted by timestamp in filename.
    """
    date = date or datetime.now(timezone.utc)
    day_str = date.strftime("%Y-%m-%d")
    day_dir = RECORDINGS_DIR / day_str
    if not day_dir.exists():
        return []

    segments = []
    for user_dir in day_dir.iterdir():
        if not user_dir.is_dir():
            continue
        username = user_dir.name
        for f in user_dir.glob("*_speech*.wav"):
            segments.append((username, f))

    # Sort by filename (which starts with HHMMSS_microseconds)
    segments.sort(key=lambda x: x[1].name)
    return segments


def get_new_speech_segments(date: datetime | None = None) -> list[tuple[str, Path]]:
    """Get only unprocessed speech segments, sorted chronologically."""
    all_segs = get_all_speech_segments_chronological(date)
    processed = get_processed_files(date)
    return [(u, p) for u, p in all_segs if p.name not in processed]


def concatenate_wav_files(paths: list[Path]) -> Path | None:
    """Concatenate multiple WAV files into a single temporary WAV. Returns path."""
    if not paths:
        return None
    buf = io.BytesIO()
    with wave.open(str(paths[0]), "rb") as first:
        params = first.getparams()

    with wave.open(buf, "wb") as out:
        out.setparams(params)
        for p in paths:
            with wave.open(str(p), "rb") as wf:
                out.writeframes(wf.readframes(wf.getnframes()))

    # Write to a temp file next to the first recording
    combined = paths[0].parent / "_combined.wav"
    combined.write_bytes(buf.getvalue())
    return combined
