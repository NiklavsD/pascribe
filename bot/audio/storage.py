"""Save per-user audio recordings to disk as Ogg Opus files."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import wave
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from config import RECORDINGS_DIR, SAMPLE_RATE, CHANNELS, PCM_SAMPLE_WIDTH

log = logging.getLogger(__name__)

# Speech segments are stored as Ogg Opus. 64 kb/s Opus is ~12x smaller than
# the 48 kHz 16-bit WAV it replaces, decodes to the exact same sample count,
# and in a Whisper A/B on real segments changed no more words than inaudible
# dither did (32 kb/s VOIP mode did, so don't go lower). Days recorded before
# the switch still hold .wav files; compress_recordings.py converts those with
# the same settings.
AUDIO_EXTS = (".opus", ".wav")
OPUS_ENCODE_ARGS = ["-c:a", "libopus", "-b:a", "64k", "-application", "audio", "-f", "ogg"]
# Segments shorter than this are not worth an API call. Matches the old
# 10 KB cutoff on 48 kHz mono 16-bit WAV, which no longer works on Opus sizes.
MIN_TRANSCRIBE_S = 0.1

# Encoding runs off the voice socket thread so a slow ffmpeg never stalls
# packet handling. One worker keeps segments landing in order.
_encoder = ThreadPoolExecutor(max_workers=1, thread_name_prefix="opus-encode")


# ---------------------------------------------------------------------------
# Participants sidecar — maps username → Discord user_id per recording day.
# Audio files are keyed by username for legacy reasons; this sidecar carries
# the Discord ID through to pascribe-server so downstream (idea-gen) can
# ping the speakers an idea drew from.
# ---------------------------------------------------------------------------

def _participants_sidecar(date: datetime | None = None) -> Path:
    date = date or datetime.now(timezone.utc)
    day_str = date.strftime("%Y-%m-%d")
    return RECORDINGS_DIR / day_str / "_participants.json"


def record_speaker_id(username: str, user_id: int,
                      date: datetime | None = None) -> None:
    """Idempotently record (username, user_id) in today's sidecar. Safe to
    call once per save_pcm_as_wav."""
    path = _participants_sidecar(date)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = json.loads(path.read_text()) if path.exists() else {}
    except (json.JSONDecodeError, OSError):
        existing = {}
    if existing.get(username) == user_id:
        return
    existing[username] = int(user_id)
    path.write_text(json.dumps(existing, indent=2))


def load_speaker_ids(date: datetime | None = None) -> dict[str, int]:
    path = _participants_sidecar(date)
    if not path.exists():
        return {}
    try:
        return {str(k): int(v) for k, v in json.loads(path.read_text()).items()}
    except (json.JSONDecodeError, OSError, ValueError):
        return {}


def _user_dir(username: str, date: datetime | None = None) -> Path:
    """Return (and create) the directory for a user's recordings on a given date."""
    date = date or datetime.now(timezone.utc)
    day_str = date.strftime("%Y-%m-%d")
    d = RECORDINGS_DIR / day_str / username
    d.mkdir(parents=True, exist_ok=True)
    return d


def encode_pcm_to_opus(
    pcm_data: bytes,
    filepath: Path,
    *,
    sample_rate: int = SAMPLE_RATE,
    channels: int = CHANNELS,
) -> None:
    """Encode raw s16le PCM to an Ogg Opus file. The file only appears under
    its final name once complete, so readers never see a half-written one."""
    tmp = filepath.with_name(filepath.name + ".part")
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y",
         "-f", "s16le", "-ar", str(sample_rate), "-ac", str(channels), "-i", "pipe:0",
         *OPUS_ENCODE_ARGS, str(tmp)],
        input=pcm_data, capture_output=True, check=True, timeout=300,
    )
    os.replace(tmp, filepath)


def _write_wav(pcm_data: bytes, filepath: Path, sample_rate: int,
               channels: int, sample_width: int) -> None:
    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)


def _save_opus_or_wav(pcm_data: bytes, filepath: Path, sample_rate: int,
                      channels: int, sample_width: int) -> None:
    try:
        encode_pcm_to_opus(pcm_data, filepath, sample_rate=sample_rate, channels=channels)
        log.debug("Saved %d PCM bytes → %s", len(pcm_data), filepath)
    except Exception as e:  # noqa: BLE001 — never lose a segment over the encoder
        filepath.with_name(filepath.name + ".part").unlink(missing_ok=True)
        wav_path = filepath.with_suffix(".wav")
        err = getattr(e, "stderr", b"") or b""
        log.warning("Opus encode failed for %s (%s %s); saving WAV instead",
                    filepath.name, e, err.decode(errors="replace").strip()[:200])
        _write_wav(pcm_data, wav_path, sample_rate, channels, sample_width)


def save_pcm_audio(
    pcm_data: bytes,
    username: str,
    *,
    label: str = "",
    date: datetime | None = None,
    sample_rate: int = SAMPLE_RATE,
    channels: int = CHANNELS,
    sample_width: int = PCM_SAMPLE_WIDTH,
) -> Path:
    """Queue raw PCM bytes to be written as a timestamped .opus file and return
    its path. The file appears once the background encode finishes; if ffmpeg
    fails, the segment is written as .wav under the same stem instead."""
    now = datetime.now(timezone.utc)
    date = date or now
    d = _user_dir(username, date)
    ts = now.strftime("%H%M%S_%f")
    suffix = f"_{label}" if label else ""
    filepath = d / f"{ts}{suffix}.opus"
    if sample_width != 2:
        # ffmpeg is fed s16le; anything else stays WAV.
        _write_wav(pcm_data, filepath.with_suffix(".wav"), sample_rate, channels, sample_width)
        return filepath.with_suffix(".wav")
    _encoder.submit(_save_opus_or_wav, pcm_data, filepath, sample_rate, channels, sample_width)
    return filepath


# Older callers (audio/capture.py) still use the WAV-era name.
save_pcm_as_wav = save_pcm_audio


def flush_pending_saves(timeout: float | None = 30) -> None:
    """Wait for queued encodes to land on disk."""
    _encoder.submit(lambda: None).result(timeout=timeout)


def _audio_files(directory: Path, pattern: str = "*") -> list[Path]:
    """Audio files in `directory` matching `pattern` (without extension). When
    a stem exists as both .opus and .wav (mid-migration), the .opus wins."""
    by_stem: dict[str, Path] = {}
    for ext in reversed(AUDIO_EXTS):  # .wav first so .opus overwrites it
        for f in directory.glob(pattern + ext):
            by_stem[f.stem] = f
    return list(by_stem.values())


def audio_seconds(path: Path) -> float:
    """Duration of a recorded segment, 0.0 if unreadable."""
    try:
        if path.suffix == ".wav":
            with wave.open(str(path), "rb") as w:
                return w.getnframes() / float(w.getframerate())
        return _ogg_opus_seconds(path)
    except Exception:  # noqa: BLE001
        return 0.0


def _ogg_opus_seconds(path: Path) -> float:
    """Read an Ogg Opus duration from its headers without decoding: the last
    page's granule position minus the OpusHead pre-skip, at Opus's fixed
    48 kHz granule rate."""
    with open(path, "rb") as f:
        head = f.read(4096)
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - 65536))
        tail = f.read()
    i = head.find(b"OpusHead")
    j = tail.rfind(b"OggS")
    if i < 0 or j < 0:
        return 0.0
    pre_skip = int.from_bytes(head[i + 10:i + 12], "little")
    granule = int.from_bytes(tail[j + 6:j + 14], "little", signed=True)
    return max(0, granule - pre_skip) / 48000.0


def decode_to_pcm(path: Path, sample_rate: int = SAMPLE_RATE, channels: int = 1) -> bytes:
    """Decode any recorded segment to raw s16le PCM."""
    return subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
         "-f", "s16le", "-ar", str(sample_rate), "-ac", str(channels), "pipe:1"],
        capture_output=True, check=True, timeout=300,
    ).stdout


def get_user_recordings(username: str, date: datetime | None = None) -> list[Path]:
    """Return all audio files for a user on a given date, sorted by name."""
    d = _user_dir(username, date)
    return sorted(_audio_files(d))


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
        for f in _audio_files(user_dir, "*_speech*"):
            # _merged_* are the pipeline's temp concatenations; a crash
            # mid-run leaves them behind and they'd be re-sent as duplicates.
            if f.name.startswith("_"):
                continue
            segments.append((username, f))

    # Sort by filename (which starts with HHMMSS_microseconds)
    segments.sort(key=lambda x: x[1].name)
    return segments


def get_new_speech_segments(date: datetime | None = None) -> list[tuple[str, Path]]:
    """Get only unprocessed speech segments, sorted chronologically.

    Matched on the stem: _processed.txt lists WAV names from before the
    switch to Opus, and those segments now live on as .opus files."""
    all_segs = get_all_speech_segments_chronological(date)
    processed = {Path(n).stem for n in get_processed_files(date)}
    return [(u, p) for u, p in all_segs if p.stem not in processed]


def concatenate_audio_files(paths: list[Path]) -> Path | None:
    """Decode and join segments into one temporary 48 kHz mono WAV next to the
    first one. WAV, not Opus, so the merged upload adds no second lossy pass."""
    if not paths:
        return None
    pcm = b"".join(decode_to_pcm(p) for p in paths)
    combined = paths[0].parent / "_combined.wav"
    _write_wav(pcm, combined, SAMPLE_RATE, 1, PCM_SAMPLE_WIDTH)
    return combined
