"""Disk-backed access to full-day desktop recordings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np


class RawAudioSource:
    """Read normalized float32 chunks from a fixed-size raw PCM snapshot."""

    def __init__(self, path: Path, dtype: np.dtype, sample_count: int) -> None:
        self.path = Path(path)
        self.dtype = np.dtype(dtype)
        self.sample_count = max(0, int(sample_count))
        self._mapping = (
            np.memmap(self.path, dtype=self.dtype, mode="r", shape=(self.sample_count,))
            if self.sample_count
            else None
        )

    def read(self, start: int, end: int) -> np.ndarray:
        start = max(0, min(int(start), self.sample_count))
        end = max(start, min(int(end), self.sample_count))
        if self._mapping is None or end <= start:
            return np.empty(0, dtype=np.float32)
        chunk = np.asarray(self._mapping[start:end])
        if self.dtype == np.dtype("int16"):
            return chunk.astype(np.float32) / 32767.0
        return chunk.astype(np.float32, copy=False)

    def close(self) -> None:
        mapping = self._mapping
        self._mapping = None
        if mapping is not None:
            mmap_object = getattr(mapping, "_mmap", None)
            if mmap_object is not None:
                mmap_object.close()


@dataclass
class DailyRecordingSnapshot:
    day_dir: Path
    started: datetime
    sample_rate: int
    mic: RawAudioSource | None
    system: RawAudioSource | None

    @property
    def has_both(self) -> bool:
        return self.mic is not None and self.system is not None

    @property
    def total_samples(self) -> int:
        if self.has_both:
            return min(self.mic.sample_count, self.system.sample_count)  # type: ignore[union-attr]
        source = self.mic if self.mic is not None else self.system
        return source.sample_count if source is not None else 0

    def mixed_reader(self) -> Callable[[int, int], np.ndarray]:
        if self.has_both:
            def read(start: int, end: int) -> np.ndarray:
                mic = self.mic.read(start, end)  # type: ignore[union-attr]
                system = self.system.read(start, end)  # type: ignore[union-attr]
                length = min(len(mic), len(system))
                return mic[:length] * 0.5 + system[:length] * 0.5
            return read
        source = self.mic if self.mic is not None else self.system
        if source is None:
            return lambda _start, _end: np.empty(0, dtype=np.float32)
        return source.read

    def close(self) -> None:
        if self.mic is not None:
            self.mic.close()
        if self.system is not None:
            self.system.close()


def _open_source(path: Path, dtype: np.dtype, sample_limit: int | None) -> RawAudioSource | None:
    if not path.exists():
        return None
    itemsize = dtype.itemsize
    available = path.stat().st_size // itemsize
    count = min(available, sample_limit) if sample_limit is not None else available
    if count <= 0:
        return None
    return RawAudioSource(path, dtype, count)


def open_daily_snapshot(
    recording_path: Path,
    day: str,
    *,
    sample_limits: dict[str, int] | None = None,
) -> DailyRecordingSnapshot | None:
    day_dir = Path(recording_path) / day
    meta_path = day_dir / "meta.json"
    if not meta_path.exists():
        return None
    with meta_path.open(encoding="utf-8") as handle:
        meta = json.load(handle)
    sample_rate = int(meta.get("sample_rate", 16000))
    started = datetime.fromisoformat(meta["started"])
    dtype = np.dtype(meta.get("dtype", "float32"))
    limits = sample_limits or {}
    mic = _open_source(day_dir / "mic.raw", dtype, limits.get("mic"))
    system = _open_source(day_dir / "sys.raw", dtype, limits.get("sys"))
    if mic is None and system is None:
        return None
    return DailyRecordingSnapshot(day_dir, started, sample_rate, mic, system)


def assign_speaker(
    start_seconds: float,
    end_seconds: float,
    mic: RawAudioSource,
    system: RawAudioSource,
    sample_rate: int,
) -> str:
    start = int(start_seconds * sample_rate)
    end = max(start + 1, int(end_seconds * sample_rate))
    mic_audio = mic.read(start, end)
    system_audio = system.read(start, end)
    mic_rms = float(np.sqrt(np.mean(np.square(mic_audio)))) if len(mic_audio) else 0.0
    system_rms = float(np.sqrt(np.mean(np.square(system_audio)))) if len(system_audio) else 0.0
    return "you" if mic_rms >= system_rms else "discord"
