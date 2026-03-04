"""Voice Activity Detection using webrtcvad.

webrtcvad requires:
  - 16-bit mono PCM
  - sample rates: 8000, 16000, 32000, 48000
  - frame durations: 10, 20, or 30 ms
"""

from __future__ import annotations

import logging
import struct
from collections import deque

import webrtcvad

from config import VAD_AGGRESSIVENESS, VAD_SILENCE_TIMEOUT, SAMPLE_RATE

log = logging.getLogger(__name__)

# We operate on 20ms frames at 48kHz mono (960 samples × 2 bytes = 1920 bytes)
FRAME_DURATION_MS = 20
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # 960 samples
FRAME_BYTES = FRAME_SIZE * 2  # 16-bit mono

# Number of silent frames before we declare end-of-speech
SILENCE_FRAMES = int(VAD_SILENCE_TIMEOUT * 1000 / FRAME_DURATION_MS)


def stereo_to_mono(pcm_stereo: bytes) -> bytes:
    """Convert 16-bit stereo PCM to mono by averaging channels."""
    samples = struct.unpack(f"<{len(pcm_stereo) // 2}h", pcm_stereo)
    mono = []
    for i in range(0, len(samples), 2):
        if i + 1 < len(samples):
            mono.append((samples[i] + samples[i + 1]) // 2)
        else:
            mono.append(samples[i])
    return struct.pack(f"<{len(mono)}h", *mono)


class SpeechSegmenter:
    """Accumulates PCM audio and yields speech segments via VAD."""

    def __init__(self, aggressiveness: int = VAD_AGGRESSIVENESS):
        self.vad = webrtcvad.Vad(aggressiveness)
        self._speech_buf: bytearray = bytearray()
        self._silent_count: int = 0
        self._in_speech: bool = False
        # Ring buffer of recent frames for pre-roll
        self._ring: deque[bytes] = deque(maxlen=5)
        # Accumulation buffer for partial frames
        self._mono_buf: bytearray = bytearray()

    def process_chunk(self, pcm_stereo: bytes) -> list[bytes]:
        """Feed a chunk of stereo PCM. Returns list of completed speech segments (mono PCM)."""
        mono = stereo_to_mono(pcm_stereo)
        self._mono_buf.extend(mono)
        segments: list[bytes] = []

        offset = 0
        while offset + FRAME_BYTES <= len(self._mono_buf):
            frame = bytes(self._mono_buf[offset : offset + FRAME_BYTES])
            offset += FRAME_BYTES

            try:
                is_speech = self.vad.is_speech(frame, SAMPLE_RATE)
            except Exception:
                is_speech = False

            if is_speech:
                if not self._in_speech:
                    # Start of speech — include pre-roll
                    self._in_speech = True
                    for pre in self._ring:
                        self._speech_buf.extend(pre)
                self._speech_buf.extend(frame)
                self._silent_count = 0
            else:
                self._ring.append(frame)
                if self._in_speech:
                    self._speech_buf.extend(frame)
                    self._silent_count += 1
                    if self._silent_count >= SILENCE_FRAMES:
                        # End of speech segment
                        segments.append(bytes(self._speech_buf))
                        self._speech_buf.clear()
                        self._in_speech = False
                        self._silent_count = 0

        # Remove consumed bytes from buffer
        if offset > 0:
            del self._mono_buf[:offset]

        return segments

    def flush(self) -> bytes | None:
        """Flush any remaining speech buffer."""
        if self._speech_buf:
            data = bytes(self._speech_buf)
            self._speech_buf.clear()
            self._in_speech = False
            self._silent_count = 0
            return data
        return None
