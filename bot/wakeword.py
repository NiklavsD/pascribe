"""Local wake word detection using Vosk (offline, no network dependency).

Runs a per-user Vosk recognizer on 16kHz mono PCM audio.
Checks partial results for 'benjamin' keyword — instant detection.
"""

from __future__ import annotations

import json
import logging
import struct
from pathlib import Path
from typing import Callable

from vosk import Model, KaldiRecognizer, SetLogLevel

log = logging.getLogger(__name__)

# Suppress Vosk's verbose LOG output
SetLogLevel(-1)

MODEL_PATH = Path(__file__).parent / "vosk-model-small-en-us-0.15"
SAMPLE_RATE = 16000  # Vosk needs 16kHz
KEYWORD = "benjamin"

# Downsample 48kHz stereo to 16kHz mono
def _downsample_48k_stereo_to_16k_mono(pcm_48k_stereo: bytes) -> bytes:
    """Convert 48kHz stereo PCM to 16kHz mono PCM (take every 6th sample from left channel)."""
    samples = struct.unpack(f"<{len(pcm_48k_stereo)//2}h", pcm_48k_stereo)
    # Stereo: L R L R L R ... take every 3rd left sample (indices 0, 6, 12...)
    mono_16k = samples[::6]  # every 6th = left channel every 3rd frame = 48k/3 = 16k
    return struct.pack(f"<{len(mono_16k)}h", *mono_16k)


def _downsample_48k_mono_to_16k(pcm_48k_mono: bytes) -> bytes:
    """Convert 48kHz mono PCM to 16kHz mono (take every 3rd sample)."""
    samples = struct.unpack(f"<{len(pcm_48k_mono)//2}h", pcm_48k_mono)
    mono_16k = samples[::3]
    return struct.pack(f"<{len(mono_16k)}h", *mono_16k)


class LocalWakeWordDetector:
    """Per-user local wake word detection via Vosk."""
    
    def __init__(self, on_wake: Callable[[int, str, str], None] | None = None):
        """
        Args:
            on_wake: callback(user_id, username, transcript) called when wake word detected.
                     Called from the audio thread — must be thread-safe.
        """
        self._model = Model(str(MODEL_PATH))
        self._recognizers: dict[int, KaldiRecognizer] = {}
        self._usernames: dict[int, str] = {}
        self._on_wake = on_wake
        log.info("Local wake word detector loaded (Vosk, model=%s)", MODEL_PATH.name)
    
    def add_user(self, user_id: int, username: str) -> None:
        if user_id not in self._recognizers:
            self._recognizers[user_id] = KaldiRecognizer(self._model, SAMPLE_RATE)
            self._usernames[user_id] = username
            log.debug("Vosk recognizer created for %s (%d)", username, user_id)
    
    def remove_user(self, user_id: int) -> None:
        self._recognizers.pop(user_id, None)
        self._usernames.pop(user_id, None)
    
    def feed_audio(self, user_id: int, pcm_mono_48k: bytes) -> None:
        """Feed 48kHz mono PCM audio for a user. Checks for wake word."""
        rec = self._recognizers.get(user_id)
        if not rec:
            return
        
        # Downsample to 16kHz mono for Vosk
        pcm_16k = _downsample_48k_mono_to_16k(pcm_mono_48k)
        
        if not hasattr(self, '_feed_count'):
            self._feed_count = 0
        self._feed_count += 1
        if self._feed_count % 500 == 1:
            log.info("Vosk feed_audio #%d for %s (%d bytes → %d bytes)", 
                     self._feed_count, self._usernames.get(user_id, "?"), len(pcm_mono_48k), len(pcm_16k))
        
        # Feed to recognizer — returns True if there's a final result
        if rec.AcceptWaveform(pcm_16k):
            result = json.loads(rec.Result())
            text = result.get("text", "")
            if text:
                log.info("Vosk final [%s]: %s", self._usernames.get(user_id, "?"), text)
                if self._is_valid_trigger(text):
                    log.info("🔊 Vosk WAKE WORD [%s]: %s", self._usernames.get(user_id, "?"), text)
                    self._check_keyword(user_id, text)
        else:
            # Check partials too for speed, but with stricter validation
            partial = json.loads(rec.PartialResult())
            text = partial.get("partial", "")
            if KEYWORD in text.lower() and self._is_valid_trigger(text):
                log.info("🔊 Vosk WAKE WORD (partial) [%s]: %s", 
                        self._usernames.get(user_id, "?"), text)
                self._check_keyword(user_id, text)
                rec.Reset()
    
    def _is_valid_trigger(self, text: str) -> bool:
        """Filter out false positives — require 'benjamin' near the start or as a standalone call."""
        words = text.lower().split()
        if KEYWORD not in words:
            return False
        idx = words.index(KEYWORD)
        # Benjamin must be within first 3 words (e.g. "hey benjamin", "benjamin can you")
        # OR the total phrase is short (≤5 words = likely a direct call)
        if idx <= 2 or len(words) <= 5:
            return True
        # Long garbled text with "benjamin" buried deep = probably noise
        log.debug("Vosk rejected (keyword at position %d in %d words): %s", idx, len(words), text)
        return False

    def _check_keyword(self, user_id: int, text: str) -> None:
        if self._on_wake:
            username = self._usernames.get(user_id, "unknown")
            self._on_wake(user_id, username, text)
