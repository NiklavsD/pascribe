"""AssemblyAI real-time streaming transcription — per-user streams for wake word detection."""

from __future__ import annotations

import asyncio
import json
import logging
import struct
import time
from datetime import datetime, timezone
from urllib.parse import urlencode

import aiohttp

from config import ASSEMBLYAI_STREAMING_KEY
from triggers import check_cooldown, mark_triggered

log = logging.getLogger(__name__)

STREAMING_URL = "wss://streaming.assemblyai.com/v3/ws"
STREAMING_SAMPLE_RATE = 16000
SOURCE_SAMPLE_RATE = 48000


def _downsample_48k_to_16k(pcm_mono_48k: bytes) -> bytes:
    """Downsample 48kHz mono PCM16 to 16kHz by taking every 3rd sample."""
    n_samples = len(pcm_mono_48k) // 2
    samples = struct.unpack(f"<{n_samples}h", pcm_mono_48k)
    downsampled = samples[::3]
    return struct.pack(f"<{len(downsampled)}h", *downsampled)


class StreamingTranscriber:
    """Manages a single AssemblyAI streaming WebSocket session for one user."""

    def __init__(self, api_key: str, user_id: int, username: str, on_wake_word=None):
        self._api_key = api_key
        self._user_id = user_id
        self._username = username
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._session: aiohttp.ClientSession | None = None
        self._running = False
        self._recv_task: asyncio.Task | None = None
        self._on_wake_word = on_wake_word

        self._recent_text: list[str] = []
        self._recent_max = 50

        self._audio_buf = bytearray()
        self._send_interval = 0.1
        self._send_task: asyncio.Task | None = None

        self._session_id: str | None = None
        self._connected = False
        self._reconnecting = False

    async def connect(self) -> bool:
        if self._connected:
            return True
        try:
            params = {
                "speech_model": "u3-rt-pro",
                "sample_rate": STREAMING_SAMPLE_RATE,
                "format_turns": "true",
            }
            url = f"{STREAMING_URL}?{urlencode(params)}"
            self._session = aiohttp.ClientSession()
            self._ws = await self._session.ws_connect(
                url,
                headers={"Authorization": self._api_key},
                heartbeat=30,
            )
            self._running = True
            self._connected = True
            self._recv_task = asyncio.create_task(self._receive_loop())
            self._send_task = asyncio.create_task(self._send_loop())
            log.info("Streaming session started for user %s (%d)", self._username, self._user_id)
            return True
        except Exception:
            log.exception("Failed to connect streaming for user %s", self._username)
            await self._cleanup()
            return False

    async def disconnect(self):
        self._running = False
        if self._ws and not self._ws.closed:
            try:
                await self._ws.send_json({"type": "Terminate"})
                await asyncio.sleep(0.5)
                await self._ws.close()
            except Exception:
                pass
        for task in (self._recv_task, self._send_task):
            if task:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        await self._cleanup()
        log.info("Streaming session ended for user %s (%d)", self._username, self._user_id)

    async def _cleanup(self):
        self._connected = False
        self._ws = None
        if self._session:
            await self._session.close()
            self._session = None

    async def _lazy_reconnect(self):
        """Reconnect only when audio data is available (called from feed_audio)."""
        try:
            await self.connect()
        except Exception:
            log.exception("Lazy reconnect failed for %s", self._username)
        finally:
            self._reconnecting = False

    def feed_audio(self, pcm_mono_48k: bytes):
        downsampled = _downsample_48k_to_16k(pcm_mono_48k)
        self._audio_buf.extend(downsampled)

    async def _send_loop(self):
        SILENCE = b"\x00" * 3200
        idle_count = 0
        try:
            while self._running:
                await asyncio.sleep(self._send_interval)
                if not self._ws or self._ws.closed:
                    break
                if self._audio_buf:
                    data = bytes(self._audio_buf)
                    self._audio_buf.clear()
                    idle_count = 0
                else:
                    idle_count += 1
                    # Send silence every frame to keep WebSocket alive
                    data = SILENCE
                try:
                    await self._ws.send_bytes(data)
                except Exception as e:
                    log.warning("Failed to send audio for %s: %s", self._username, e)
                    break
        except asyncio.CancelledError:
            pass

    async def _receive_loop(self):
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self._handle_event(data)
                    except json.JSONDecodeError:
                        pass
                elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    log.warning("Streaming WS for %s: %s", self._username, msg.type)
                    break
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("Error in streaming receive loop for %s", self._username)
        finally:
            self._connected = False
            if self._running:
                # Reconnect quickly if there's buffered audio, otherwise wait longer
                if self._audio_buf:
                    delay = 0.5
                else:
                    delay = 10
                log.debug("Streaming for %s disconnected — reconnect in %.0fs", self._username, delay)
                await asyncio.sleep(delay)
                if self._running:
                    await self._cleanup()
                    await self.connect()

    async def _handle_event(self, data: dict):
        event_type = data.get("type", "")

        if event_type == "Begin":
            self._session_id = data.get("id")
            log.info("Streaming session started: %s (user: %s)", self._session_id, self._username)

        elif event_type == "Turn":
            transcript = data.get("transcript", "")
            end_of_turn = data.get("end_of_turn", False)

            if transcript:
                if not end_of_turn:
                    log.debug("STREAM PARTIAL [%s]: %s", self._username, transcript[:100])
                if end_of_turn:
                    log.info("STREAM FINAL [%s]: %s", self._username, transcript[:150])
                    self._recent_text.append(transcript)
                    if len(self._recent_text) > self._recent_max:
                        self._recent_text.pop(0)

                if "benjamin" in transcript.lower():
                    await self._check_trigger(transcript)

        elif event_type == "Termination":
            duration = data.get("audio_duration_seconds", 0)
            log.info("Streaming session for %s terminated after %.1fs", self._username, duration)

        elif event_type == "Error":
            log.error("Streaming error for %s: %s", self._username, data.get("error", data))

    async def _check_trigger(self, transcript: str):
        if self._on_wake_word:
            recent = "\n".join(self._recent_text[-10:])
            try:
                await self._on_wake_word(transcript, recent, self._user_id, self._username)
            except Exception:
                log.exception("Error in wake word callback for %s", self._username)

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_recent_text(self) -> str:
        return "\n".join(self._recent_text)


class PerUserStreamManager:
    """Manages one StreamingTranscriber per user."""

    def __init__(self, api_key: str = ASSEMBLYAI_STREAMING_KEY, on_wake_word=None):
        self._api_key = api_key
        self._on_wake_word = on_wake_word  # async callback(transcript, recent, user_id, username)
        self._streams: dict[int, StreamingTranscriber] = {}  # user_id → transcriber

    async def _wake_word_wrapper(self, transcript: str, recent: str, user_id: int, username: str):
        """Pass through to wake word callback — cooldown is handled in the callback itself."""
        log.info("🎤 STREAM WAKE WORD by %s: %s", username, transcript[:200])
        if self._on_wake_word:
            await self._on_wake_word(transcript, recent, user_id, username)

    async def add_user(self, user_id: int, username: str) -> None:
        """Create and connect a streaming transcriber for a user."""
        if user_id in self._streams:
            return
        transcriber = StreamingTranscriber(
            api_key=self._api_key,
            user_id=user_id,
            username=username,
            on_wake_word=self._wake_word_wrapper,
        )
        self._streams[user_id] = transcriber
        connected = await transcriber.connect()
        if connected:
            log.info("Per-user stream connected for %s (%d)", username, user_id)
        else:
            log.warning("Per-user stream failed to connect for %s (%d)", username, user_id)

    async def remove_user(self, user_id: int) -> None:
        """Disconnect and remove a user's streaming transcriber."""
        transcriber = self._streams.pop(user_id, None)
        if transcriber:
            await transcriber.disconnect()

    def feed_audio(self, user_id: int, pcm_mono_48k: bytes) -> None:
        """Feed audio to a specific user's stream. Non-blocking."""
        transcriber = self._streams.get(user_id)
        if transcriber:
            transcriber.feed_audio(pcm_mono_48k)

    async def disconnect_all(self) -> None:
        """Disconnect all user streams."""
        for uid in list(self._streams.keys()):
            await self.remove_user(uid)

    def has_user(self, user_id: int) -> bool:
        return user_id in self._streams

    @property
    def is_connected(self) -> bool:
        """True if at least one stream is connected."""
        return any(t.is_connected for t in self._streams.values())

    def get_recent_text(self) -> str:
        """Get combined recent text from all user streams."""
        parts = []
        for uid, t in self._streams.items():
            text = t.get_recent_text()
            if text:
                parts.append(f"[{t._username}]:\n{text}")
        return "\n\n".join(parts)
