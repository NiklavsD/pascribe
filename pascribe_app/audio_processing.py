"""Pure and streaming audio-processing helpers for Pascribe."""

from __future__ import annotations

import threading
import wave
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


AudioReader = Callable[[int, int], np.ndarray]
ProgressCallback = Callable[[float, str], None]


class ProcessingCancelled(RuntimeError):
    pass


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample a mono array using linear interpolation."""
    audio = np.asarray(audio)
    if src_rate == dst_rate:
        return audio
    if len(audio) == 0:
        return np.empty(0, dtype=np.float32)
    new_len = max(1, int(round(len(audio) * dst_rate / src_rate)))
    indices = np.linspace(0, len(audio) - 1, new_len)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def format_ssmd(
    segments: list[tuple[float, float, str]], pause_threshold: float = 2.0
) -> str:
    if not segments:
        return ""
    lines: list[str] = []
    previous_end: float | None = None
    for start, end, text in segments:
        if previous_end is not None and start - previous_end > pause_threshold:
            lines.append("")
        minutes = int(start) // 60
        seconds = int(start) % 60
        lines.append(f"[{minutes:02d}:{seconds:02d}] {text}")
        previous_end = end
    return "\n".join(lines)


def _segments_from_rms(
    frames_rms: np.ndarray,
    *,
    total_samples: int,
    sample_rate: int,
    frame_ms: int,
    min_speech_s: float,
    pad_s: float,
) -> list[tuple[float, float]]:
    if len(frames_rms) == 0 or float(frames_rms.max()) == 0:
        return []

    noise_floor = float(np.percentile(frames_rms, 20))
    speech_level = float(np.percentile(frames_rms, 90))
    dynamic_range = speech_level - noise_floor
    if dynamic_range <= max(1e-4, noise_floor * 0.25):
        # A nearly flat recording is steady background noise, not speech.
        threshold = max(speech_level * 1.1, 1e-4)
    else:
        threshold = max(
            noise_floor * 1.8,
            noise_floor + dynamic_range * 0.25,
            1e-4,
        )
    is_speech = frames_rms > threshold

    gap_fill = max(1, int(500 / frame_ms))
    speech_indices = np.flatnonzero(is_speech)
    for left, right in zip(speech_indices, speech_indices[1:]):
        if right - left <= gap_fill:
            is_speech[left:right + 1] = True

    pad_frames = int(pad_s * 1000 / frame_ms)
    transitions = np.diff(np.pad(is_speech.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)

    padded: list[tuple[int, int]] = []
    frame_count = len(frames_rms)
    for start, end in zip(starts, ends):
        padded_start = max(0, int(start) - pad_frames)
        padded_end = min(frame_count, int(end) + pad_frames)
        if (padded_end - padded_start) * frame_ms / 1000 >= min_speech_s:
            if padded and padded_start <= padded[-1][1]:
                padded[-1] = (padded[-1][0], max(padded[-1][1], padded_end))
            else:
                padded.append((padded_start, padded_end))

    max_seconds = total_samples / sample_rate
    return [
        (
            start * frame_ms / 1000,
            min(end * frame_ms / 1000, max_seconds),
        )
        for start, end in padded
    ]


def energy_vad(
    audio: np.ndarray,
    sample_rate: int,
    frame_ms: int = 30,
    min_speech_s: float = 0.4,
    pad_s: float = 0.3,
) -> list[tuple[float, float]]:
    frame_samples = int(sample_rate * frame_ms / 1000)
    audio = np.asarray(audio, dtype=np.float32)
    frame_count = len(audio) // frame_samples
    if frame_count == 0:
        return []
    frames = audio[:frame_count * frame_samples].reshape(frame_count, frame_samples)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float32), axis=1))
    return _segments_from_rms(
        rms,
        total_samples=len(audio),
        sample_rate=sample_rate,
        frame_ms=frame_ms,
        min_speech_s=min_speech_s,
        pad_s=pad_s,
    )


def energy_vad_from_reader(
    reader: AudioReader,
    total_samples: int,
    sample_rate: int,
    *,
    frame_ms: int = 30,
    min_speech_s: float = 0.4,
    pad_s: float = 0.3,
    chunk_seconds: int = 60,
    cancel_event: threading.Event | None = None,
    progress: ProgressCallback | None = None,
) -> list[tuple[float, float]]:
    """Detect speech without loading the source recording into memory."""
    frame_samples = int(sample_rate * frame_ms / 1000)
    frame_count = total_samples // frame_samples
    if frame_count == 0:
        return []

    frames_per_chunk = max(1, int(chunk_seconds * 1000 / frame_ms))
    rms_parts: list[np.ndarray] = []
    for first_frame in range(0, frame_count, frames_per_chunk):
        if cancel_event is not None and cancel_event.is_set():
            raise ProcessingCancelled("Audio processing cancelled")
        count = min(frames_per_chunk, frame_count - first_frame)
        start = first_frame * frame_samples
        end = start + count * frame_samples
        audio = np.asarray(reader(start, end), dtype=np.float32)
        usable_count = min(count, len(audio) // frame_samples)
        if usable_count:
            frames = audio[:usable_count * frame_samples].reshape(usable_count, frame_samples)
            rms_parts.append(np.sqrt(np.mean(np.square(frames, dtype=np.float32), axis=1)))
        if usable_count < count:
            rms_parts.append(np.zeros(count - usable_count, dtype=np.float32))
        if progress is not None:
            progress(min(1.0, (first_frame + count) / frame_count), "Scanning for speech")

    frames_rms = np.concatenate(rms_parts) if rms_parts else np.empty(0, dtype=np.float32)
    return _segments_from_rms(
        frames_rms,
        total_samples=total_samples,
        sample_rate=sample_rate,
        frame_ms=frame_ms,
        min_speech_s=min_speech_s,
        pad_s=pad_s,
    )


def build_segment_map(
    vad_segments: Iterable[tuple[float, float]],
) -> list[tuple[float, float, float]]:
    segment_map: list[tuple[float, float, float]] = []
    stripped_time = 0.0
    for original_start, original_end in vad_segments:
        duration = original_end - original_start
        segment_map.append((stripped_time, stripped_time + duration, original_start))
        stripped_time += duration
    return segment_map


def remap_to_original(
    timestamp: float,
    segment_map: list[tuple[float, float, float]],
    *,
    prefer_next_at_boundary: bool = False,
) -> float:
    for index, (stripped_start, stripped_end, original_start) in enumerate(segment_map):
        if (
            prefer_next_at_boundary
            and index + 1 < len(segment_map)
            and timestamp == stripped_end
        ):
            continue
        if stripped_start <= timestamp <= stripped_end:
            return original_start + (timestamp - stripped_start)
    if segment_map and timestamp > segment_map[-1][1]:
        stripped_start, stripped_end, original_start = segment_map[-1]
        return original_start + (stripped_end - stripped_start)
    return timestamp


def write_wav_segments(
    output_path: Path,
    reader: AudioReader,
    segments: list[tuple[float, float]],
    sample_rate: int,
    *,
    chunk_seconds: int = 30,
    cancel_event: threading.Event | None = None,
    progress: ProgressCallback | None = None,
) -> float:
    """Write VAD-selected ranges to a normalized WAV using bounded memory."""
    output_path = Path(output_path)
    chunk_samples = max(1, chunk_seconds * sample_rate)
    ranges = [
        (int(start * sample_rate), int(end * sample_rate))
        for start, end in segments
        if end > start
    ]
    total = sum(end - start for start, end in ranges)
    if total <= 0:
        raise ValueError("No audio ranges to write")

    peak = 0.0
    scanned = 0
    for start, end in ranges:
        for position in range(start, end, chunk_samples):
            if cancel_event is not None and cancel_event.is_set():
                raise ProcessingCancelled("Audio processing cancelled")
            audio = np.asarray(reader(position, min(position + chunk_samples, end)), dtype=np.float32)
            if len(audio):
                peak = max(peak, float(np.max(np.abs(audio))))
            scanned += len(audio)
            if progress is not None:
                progress(min(0.5, 0.5 * scanned / total), "Measuring speech audio")

    scale = 0.95 / peak if peak > 0 else 1.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            for start, end in ranges:
                for position in range(start, end, chunk_samples):
                    if cancel_event is not None and cancel_event.is_set():
                        raise ProcessingCancelled("Audio processing cancelled")
                    audio = np.asarray(
                        reader(position, min(position + chunk_samples, end)),
                        dtype=np.float32,
                    )
                    pcm = (audio * scale * 32767).clip(-32768, 32767).astype(np.int16)
                    wav_file.writeframesraw(pcm.tobytes())
                    written += len(audio)
                    if progress is not None:
                        progress(0.5 + min(0.5, 0.5 * written / total), "Writing speech audio")
    except Exception:
        try:
            output_path.unlink()
        except OSError:
            pass
        raise
    return total / sample_rate
