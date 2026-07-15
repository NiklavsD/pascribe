"""GCP Speech-to-Text V2 streaming transcription — per-user streams.

Replaces AssemblyAI streaming with Google Cloud Speech-to-Text.
Each user gets their own streaming session. Sessions auto-reconnect
on the 5-minute limit.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import queue
import struct
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

STREAMING_SAMPLE_RATE = 16000
SOURCE_SAMPLE_RATE = 48000
# GCP streaming has a ~5 min limit; reconnect before that
MAX_SESSION_SECONDS = 280  # reconnect at 4m40s to be safe
KEYWORD = "benjamin"


def _downsample_48k_to_16k(pcm_mono_48k: bytes) -> bytes:
    """Downsample 48kHz mono PCM16 to 16kHz by taking every 3rd sample."""
    n_samples = len(pcm_mono_48k) // 2
    samples = struct.unpack(f"<{n_samples}h", pcm_mono_48k)
    downsampled = samples[::3]
    return struct.pack(f"<{len(downsampled)}h", *downsampled)


class GCPStreamingTranscriber:
    """Manages a single GCP Speech-to-Text V2 streaming session for one user."""

    def __init__(
        self,
        project_id: str,
        user_id: int,
        username: str,
        on_transcript=None,
        on_wake_word=None,
        recognizer_id: str = "_",
        location: str = "global",
    ):
        self._project_id = project_id
        self._user_id = user_id
        self._username = username
        self._on_transcript = on_transcript  # async callback(user_id, username, text, is_final)
        self._on_wake_word = on_wake_word    # async callback(transcript, recent, user_id, username)
        self._recognizer_id = recognizer_id
        self._location = location

        self._audio_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=500)
        self._running = False
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        self._recent_text: list[str] = []
        self._recent_max = 50
        self._session_start: float = 0
        self._connected = False
        self._consecutive_fails = 0
        self._last_final_text = ""
        self._last_wake_fire_text = ""  # dedupe wake fires from incremental partials

        # Drain the queue of any stale audio before starting a new stream
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Start the streaming thread."""
        if self._running:
            return
        self._loop = loop
        self._running = True
        self._thread = threading.Thread(
            target=self._stream_loop, daemon=True, name=f"gcp-stt-{self._username}"
        )
        self._thread.start()
        log.info("GCP STT stream started for %s (%d)", self._username, self._user_id)

    def stop(self) -> None:
        """Stop the streaming thread."""
        self._running = False
        # Poison pill to unblock the queue
        try:
            self._audio_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._connected = False
        log.info("GCP STT stream stopped for %s (%d)", self._username, self._user_id)

    def feed_audio(self, pcm_mono_48k: bytes) -> None:
        """Feed 48kHz mono PCM audio. Non-blocking, drops if queue full."""
        downsampled = _downsample_48k_to_16k(pcm_mono_48k)
        try:
            self._audio_queue.put_nowait(downsampled)
        except queue.Full:
            pass  # drop audio rather than block the capture thread

    def _audio_generator(self):
        """Yields StreamingRecognizeRequest with audio chunks."""
        recognizer = f"projects/{self._project_id}/locations/{self._location}/recognizers/{self._recognizer_id}"

        # First request: config — explicit LINEAR16 at 16kHz mono (raw PCM, no headers)
        config = cloud_speech.StreamingRecognitionConfig(
            config=cloud_speech.RecognitionConfig(
                explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                    encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=STREAMING_SAMPLE_RATE,
                    audio_channel_count=1,
                ),
                language_codes=["en-US"],
                model="long",
                features=cloud_speech.RecognitionFeatures(
                    enable_automatic_punctuation=True,
                ),
            ),
            streaming_features=cloud_speech.StreamingRecognitionFeatures(
                interim_results=True,
            ),
        )
        yield cloud_speech.StreamingRecognizeRequest(
            recognizer=recognizer,
            streaming_config=config,
        )

        # Subsequent requests: audio data
        self._session_start = time.time()
        while self._running:
            elapsed = time.time() - self._session_start
            if elapsed >= MAX_SESSION_SECONDS:
                log.info("GCP STT session for %s hit %ds limit, reconnecting", self._username, int(elapsed))
                break

            try:
                data = self._audio_queue.get(timeout=0.5)
            except queue.Empty:
                # Send a 20ms frame of silence to keep the stream alive
                data = b'\x00' * 640
                log.debug("Sending silence frame to GCP for %s", self._username)
                yield cloud_speech.StreamingRecognizeRequest(audio=data)
                continue # Continue to the next iteration after sending silence

            if data is None:
                break

            yield cloud_speech.StreamingRecognizeRequest(audio=data)

    def _stream_loop(self) -> None:
        """Main streaming loop — runs in a dedicated thread."""
        client = SpeechClient()

        while self._running:
            try:
                self._connected = True
                self._session_start = time.time()

                responses = client.streaming_recognize(
                    requests=self._audio_generator()
                )

                for response in responses:
                    if not self._running:
                        break
                    self._handle_response(response)

                # Clean exit (session limit or shutdown)
                self._consecutive_fails = 0

            except Exception as e:
                self._consecutive_fails += 1
                err_str = str(e)
                if "exceeded maximum allowed stream duration" in err_str.lower():
                    log.info("GCP STT session for %s expired (normal), reconnecting", self._username)
                    self._consecutive_fails = 0
                else:
                    backoff = min(30, 2 ** self._consecutive_fails)
                    log.warning(
                        "GCP STT error for %s (fail #%d, backoff %ds): %s",
                        self._username, self._consecutive_fails, backoff, err_str[:200]
                    )
                    if self._running:
                        time.sleep(backoff)
            finally:
                self._connected = False

            # Drain stale audio from queue before reconnecting
            drained = 0
            while not self._audio_queue.empty():
                try:
                    self._audio_queue.get_nowait()
                    drained += 1
                except queue.Empty:
                    break
            if drained > 0:
                log.debug("Drained %d stale audio chunks for %s", drained, self._username)

        self._connected = False

    def _handle_response(self, response: cloud_speech.StreamingRecognizeResponse) -> None:
        """Process a streaming response."""
        for result in response.results:
            if not result.alternatives:
                continue

            transcript = result.alternatives[0].transcript.strip()
            if not transcript:
                continue

            is_final = result.is_final

            if is_final:
                log.info("GCP STT FINAL [%s]: %s", self._username, transcript[:150])
                self._last_final_text = transcript
                self._recent_text.append(transcript)
                if len(self._recent_text) > self._recent_max:
                    self._recent_text.pop(0)

                # Fire transcript callback
                if self._on_transcript and self._loop:
                    asyncio.run_coroutine_threadsafe(
                        self._on_transcript(self._user_id, self._username, transcript, True),
                        self._loop,
                    )

                # Check for wake word in finals (always counts as a fresh utterance)
                if KEYWORD in transcript.lower():
                    self._last_wake_fire_text = ""  # reset dedupe on final
                    self._fire_wake_word(transcript)
            else:
                # Partials: only fire ONCE per utterance even as the partial grows
                # The partial keeps expanding word-by-word; we want only the FIRST fire.
                if KEYWORD in transcript.lower():
                    # Dedupe: skip if this partial is a continuation of one we already fired
                    if self._last_wake_fire_text and transcript.startswith(self._last_wake_fire_text[:30]):
                        return
                    self._last_wake_fire_text = transcript
                    log.info("GCP STT PARTIAL wake word [%s]: %s", self._username, transcript[:100])
                    self._fire_wake_word(transcript)

    def _fire_wake_word(self, transcript: str) -> None:
        """Fire the wake word callback."""
        if not self._on_wake_word or not self._loop:
            return
        recent = "\n".join(self._recent_text[-10:])
        asyncio.run_coroutine_threadsafe(
            self._on_wake_word(transcript, recent, self._user_id, self._username),
            self._loop,
        )

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_recent_text(self) -> str:
        return "\n".join(self._recent_text)


class GCPPerUserStreamManager:
    """Manages one GCPStreamingTranscriber per user. Drop-in replacement for PerUserStreamManager."""

    def __init__(
        self,
        project_id: str = "",
        on_wake_word=None,
        on_transcript=None,
        transcript_dir: Path | None = None,
    ):
        self._project_id = project_id
        self._on_wake_word = on_wake_word
        self._on_transcript = on_transcript
        self._transcript_dir = transcript_dir
        self._streams: dict[int, GCPStreamingTranscriber] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    async def _default_transcript_callback(self, user_id: int, username: str, text: str, is_final: bool):
        """Default handler: append final transcripts to _transcript.txt."""
        if not is_final or not self._transcript_dir or not text:
            return
        try:
            day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            day_dir = self._transcript_dir / day
            day_dir.mkdir(parents=True, exist_ok=True)
            transcript_path = day_dir / "_transcript.txt"

            timestamp = datetime.now(timezone.utc).strftime("%H:%M")

            # Check if we need a gap marker (read last line timestamp)
            needs_gap = False
            if transcript_path.exists():
                content = transcript_path.read_text(encoding="utf-8")
                if content.strip():
                    last_lines = content.strip().split("\n")
                    # Simple gap detection: if file hasn't been written to in 2+ min
                    # (handled by checking file mtime)
                    import os
                    mtime = os.path.getmtime(transcript_path)
                    if time.time() - mtime > 120:
                        needs_gap = True

            with open(transcript_path, "a", encoding="utf-8") as f:
                if needs_gap:
                    f.write(f"\n\n--- {timestamp} ---\n\n")
                f.write(f"{username}: {text}\n")

            log.debug("Transcript appended for %s: %s", username, text[:80])
        except Exception:
            log.exception("Failed to write transcript for %s", username)

    async def _wake_word_wrapper(self, transcript: str, recent: str, user_id: int, username: str):
        """Pass through to wake word callback."""
        log.info("🎤 GCP STREAM WAKE WORD by %s: %s", username, transcript[:200])
        if self._on_wake_word:
            await self._on_wake_word(transcript, recent, user_id, username)

    async def add_user(self, user_id: int, username: str) -> None:
        """Create and start a streaming transcriber for a user."""
        if user_id in self._streams:
            return
        if not self._loop:
            self._loop = asyncio.get_running_loop()

        transcript_cb = self._on_transcript or self._default_transcript_callback

        transcriber = GCPStreamingTranscriber(
            project_id=self._project_id,
            user_id=user_id,
            username=username,
            on_transcript=transcript_cb,
            on_wake_word=self._wake_word_wrapper,
        )
        self._streams[user_id] = transcriber
        transcriber.start(self._loop)
        log.info("GCP per-user stream started for %s (%d)", username, user_id)

    async def remove_user(self, user_id: int) -> None:
        """Stop and remove a user's streaming transcriber."""
        transcriber = self._streams.pop(user_id, None)
        if transcriber:
            transcriber.stop()

    def feed_audio(self, user_id: int, pcm_mono_48k: bytes) -> None:
        """Feed audio to a specific user's stream. Non-blocking."""
        transcriber = self._streams.get(user_id)
        if transcriber:
            transcriber.feed_audio(pcm_mono_48k)

    async def disconnect_all(self) -> None:
        """Stop all user streams."""
        for uid in list(self._streams.keys()):
            await self.remove_user(uid)

    def has_user(self, user_id: int) -> bool:
        return user_id in self._streams

    @property
    def is_connected(self) -> bool:
        return any(t.is_connected for t in self._streams.values())

    def get_recent_text(self) -> str:
        parts = []
        for uid, t in self._streams.items():
            text = t.get_recent_text()
            if text:
                parts.append(f"[{t._username}]:\n{text}")
        return "\n\n".join(parts)
