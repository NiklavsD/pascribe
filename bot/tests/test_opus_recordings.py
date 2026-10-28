"""Opus segment storage and the WAV → Opus migration. Needs ffmpeg and numpy:

    cd bot && python3 -m pytest tests/test_opus_recordings.py
"""

from __future__ import annotations

import math
import os
import struct
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _var in ("DISCORD_TOKEN", "ASSEMBLYAI_API_KEY"):  # config.py requires these at import
    os.environ.setdefault(_var, "test")

from audio import storage  # noqa: E402
import compress_recordings  # noqa: E402

RATE = 48000
DAY = datetime(2026, 3, 3, tzinfo=timezone.utc)


def speech_like_pcm(seconds: float, seed: int = 0) -> bytes:
    """A 220 Hz tone with a syllable-rate amplitude envelope, so the envelope
    check has something to correlate."""
    n = int(seconds * RATE)
    return b"".join(
        struct.pack("<h", int(12000 * (0.55 + 0.45 * math.sin(2 * math.pi * (3 + seed) * i / RATE))
                              * math.sin(2 * math.pi * 220 * i / RATE)))
        for i in range(n)
    )


def write_wav(path: Path, pcm: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)
    old = time.time() - 3600
    os.utime(path, (old, old))
    return path


@pytest.fixture
def rec(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "RECORDINGS_DIR", tmp_path)
    return tmp_path


def test_save_writes_opus_with_exact_duration(rec):
    path = storage.save_pcm_audio(speech_like_pcm(2.5), "alice", label="speech",
                                  date=DAY, sample_rate=RATE, channels=1)
    storage.flush_pending_saves()
    assert path.suffix == ".opus" and path.exists()
    assert not list(path.parent.glob("*.part"))
    assert storage.audio_seconds(path) == pytest.approx(2.5, abs=0.001)
    assert len(storage.decode_to_pcm(path)) == 2.5 * RATE * 2
    assert path.stat().st_size < 2.5 * RATE * 2 / 8  # well over 8x smaller than WAV


def test_save_falls_back_to_wav_when_ffmpeg_fails(rec, monkeypatch):
    def boom(*a, **k):
        raise OSError("ffmpeg missing")
    monkeypatch.setattr(storage, "encode_pcm_to_opus", boom)
    path = storage.save_pcm_audio(speech_like_pcm(0.5), "alice", label="speech",
                                  date=DAY, sample_rate=RATE, channels=1)
    storage.flush_pending_saves()
    assert not path.exists()
    assert storage.audio_seconds(path.with_suffix(".wav")) == pytest.approx(0.5)


def test_segments_mix_formats_and_match_processed_wav_names(rec):
    day = rec / "2026-03-03"
    old = write_wav(day / "alice" / "100000_000001_speech.wav", speech_like_pcm(0.3))
    # Mid-migration: the same stem as both formats is listed once, as .opus.
    both = write_wav(day / "bob" / "100001_000001_speech.wav", speech_like_pcm(0.3))
    storage.encode_pcm_to_opus(speech_like_pcm(0.3), both.with_suffix(".opus"), sample_rate=RATE, channels=1)
    new = day / "alice" / "100002_000001_speech.opus"
    storage.encode_pcm_to_opus(speech_like_pcm(0.3), new, sample_rate=RATE, channels=1)
    (day / "alice" / "_merged_100000_000001_speech.wav").write_bytes(old.read_bytes())

    segs = storage.get_all_speech_segments_chronological(DAY)
    assert [p.name for _, p in segs] == [old.name, both.with_suffix(".opus").name, new.name]

    # _processed.txt written before the switch names WAVs; their Opus copies
    # must still count as processed.
    storage.mark_files_processed([old.name, both.name], DAY)
    assert [p.name for _, p in storage.get_new_speech_segments(DAY)] == [new.name]


def test_concatenate_decodes_both_formats_to_wav(rec):
    d = rec / "2026-03-03" / "alice"
    a = write_wav(d / "100000_000001_speech.wav", speech_like_pcm(0.4))
    b = d / "100001_000001_speech.opus"
    storage.encode_pcm_to_opus(speech_like_pcm(0.6), b, sample_rate=RATE, channels=1)
    combined = storage.concatenate_audio_files([a, b])
    assert combined.suffix == ".wav"
    assert storage.audio_seconds(combined) == pytest.approx(1.0, abs=0.001)


def test_migration_verifies_then_deletes(rec):
    wav = write_wav(rec / "2026-03-03" / "alice" / "100000_000001_speech.wav", speech_like_pcm(3.0))
    mtime = wav.stat().st_mtime

    status, *_ , detail = compress_recordings.convert(wav, delete=False)
    assert status == "verified", detail
    opus = wav.with_suffix(".opus")
    assert wav.exists() and opus.exists()
    assert opus.stat().st_mtime == pytest.approx(mtime)

    status, *_ = compress_recordings.convert(wav, delete=True)  # reuses the existing .opus
    assert status == "deleted"
    assert not wav.exists() and storage.audio_seconds(opus) == pytest.approx(3.0, abs=0.001)


def test_migration_keeps_wav_when_opus_does_not_match(rec):
    wav = write_wav(rec / "2026-03-03" / "alice" / "100000_000001_speech.wav", speech_like_pcm(3.0))
    storage.encode_pcm_to_opus(speech_like_pcm(2.0), wav.with_suffix(".opus"), sample_rate=RATE, channels=1)
    status, *_, detail = compress_recordings.convert(wav, delete=True)
    assert status == "failed" and "length" in detail
    assert wav.exists()


def test_migration_skips_today_and_fresh_files(rec):
    old = write_wav(rec / "2026-03-03" / "alice" / "100000_000001_speech.wav", speech_like_pcm(0.2))
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    write_wav(rec / today / "alice" / "100000_000001_speech.wav", speech_like_pcm(0.2))
    fresh = rec / "2026-03-04" / "alice" / "100000_000001_speech.wav"
    write_wav(fresh, speech_like_pcm(0.2))
    os.utime(fresh)
    assert compress_recordings.find_wavs(rec, None, 600) == [old]
