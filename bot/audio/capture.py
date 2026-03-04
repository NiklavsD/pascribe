"""Per-user audio capture for discord.py with DAVE E2EE support.

Hooks into the VoiceClient's UDP socket reader to receive raw RTP packets,
decrypts them (transport + DAVE), decodes Opus → PCM, runs VAD, and saves
speech segments to disk.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import struct
from typing import TYPE_CHECKING

import nacl.secret

from audio.storage import save_pcm_as_wav
from audio.vad import SpeechSegmenter
from config import MIN_SPEECH_DURATION, SAMPLE_RATE, PCM_SAMPLE_WIDTH

if TYPE_CHECKING:
    import discord
    from commands.voice import VoiceCommandDetector

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Opus decoder (we decode incoming audio ourselves since discord.py only has encoder)
# ---------------------------------------------------------------------------
class OpusDecoder:
    """Minimal ctypes wrapper around libopus for decoding."""

    SAMPLE_RATE = 48000
    CHANNELS = 2
    FRAME_SIZE = 960  # 20ms at 48kHz

    def __init__(self):
        lib_path = ctypes.util.find_library("opus")
        if lib_path is None:
            raise RuntimeError("libopus not found — install it (apt install libopus0)")
        self._lib = ctypes.cdll.LoadLibrary(lib_path)

        # opus_decoder_create
        self._lib.opus_decoder_create.argtypes = [ctypes.c_int32, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self._lib.opus_decoder_create.restype = ctypes.POINTER(ctypes.c_char)

        # opus_decode
        self._lib.opus_decode.argtypes = [
            ctypes.POINTER(ctypes.c_char),  # decoder
            ctypes.c_char_p,  # data
            ctypes.c_int32,  # len
            ctypes.POINTER(ctypes.c_int16),  # pcm
            ctypes.c_int,  # frame_size
            ctypes.c_int,  # decode_fec
        ]
        self._lib.opus_decode.restype = ctypes.c_int

        # opus_decoder_destroy
        self._lib.opus_decoder_destroy.argtypes = [ctypes.POINTER(ctypes.c_char)]
        self._lib.opus_decoder_destroy.restype = None

        err = ctypes.c_int()
        self._decoder = self._lib.opus_decoder_create(self.SAMPLE_RATE, self.CHANNELS, ctypes.byref(err))
        if err.value != 0:
            raise RuntimeError(f"opus_decoder_create failed: {err.value}")

    def decode(self, data: bytes) -> bytes:
        """Decode an Opus packet to stereo 16-bit PCM."""
        data = bytes(data)  # ensure bytes for ctypes
        max_samples = self.FRAME_SIZE
        pcm_buf = (ctypes.c_int16 * (max_samples * self.CHANNELS))()
        ret = self._lib.opus_decode(
            self._decoder,
            data,
            len(data),
            pcm_buf,
            max_samples,
            0,
        )
        if ret < 0:
            raise RuntimeError(f"opus_decode failed: {ret}")
        # Return raw PCM bytes from the int16 buffer
        out_samples = ret * self.CHANNELS
        return bytes(ctypes.cast(pcm_buf, ctypes.POINTER(ctypes.c_char * (out_samples * 2))).contents)

    def __del__(self):
        if hasattr(self, "_decoder") and self._decoder:
            self._lib.opus_decoder_destroy(self._decoder)


# ---------------------------------------------------------------------------
# Per-SSRC decoder state
# ---------------------------------------------------------------------------
class _SSRCState:
    __slots__ = ("opus_decoder", "segmenter", "user_id")

    def __init__(self, user_id: int):
        self.user_id = user_id
        self.opus_decoder = OpusDecoder()
        self.segmenter = SpeechSegmenter()


# ---------------------------------------------------------------------------
# Main audio sink
# ---------------------------------------------------------------------------
class UserAudioSink:
    """Receives UDP packets from the voice socket, decrypts, decodes, VADs, saves."""

    def __init__(
        self,
        bot: discord.Client,
        voice_detector: VoiceCommandDetector | None = None,
    ):
        self.bot = bot
        self.voice_detector = voice_detector

        # SSRC → user_id mapping (populated from SPEAKING events via hook)
        self._ssrc_to_user: dict[int, int] = {}
        # user_id → username
        self._user_names: dict[int, str] = {}
        # SSRC → decoder state
        self._ssrc_states: dict[int, _SSRCState] = {}

        # Accumulated data
        self._raw_buffers: dict[int, bytearray] = {}
        self._speech_segments: dict[int, list[bytes]] = {}
        self._excluded_users: set[int] = set()

        # Reference to voice client (set when start_listening is called)
        self._vc: discord.VoiceClient | None = None
        self._listening = False

    def _resolve_username(self, user_id: int) -> str:
        if user_id in self._user_names:
            return self._user_names[user_id]
        user = self.bot.get_user(user_id)
        name = user.name if user else str(user_id)
        self._user_names[user_id] = name
        return name

    def exclude_user(self, user_id: int) -> None:
        self._excluded_users.add(user_id)
        log.info("User %d excluded from recording", user_id)

    def include_user(self, user_id: int) -> None:
        self._excluded_users.discard(user_id)
        log.info("User %d included in recording", user_id)

    def register_speaking(self, user_id: int, ssrc: int) -> None:
        """Called when we receive a SPEAKING event mapping SSRC → user."""
        self._ssrc_to_user[ssrc] = user_id
        if ssrc not in self._ssrc_states:
            self._ssrc_states[ssrc] = _SSRCState(user_id)
            log.info("New speaker: SSRC %d → user %d (%s)", ssrc, user_id, self._resolve_username(user_id))

    # ------------------------------------------------------------------
    # Packet decryption
    # ------------------------------------------------------------------
    def _decrypt_packet(self, data: bytes) -> tuple[int, bytes] | None:
        """Decrypt an incoming RTP packet. Returns (ssrc, opus_payload) or None."""
        if len(data) < 12:
            return None

        # RTP header: V=2, PT=0x78, seq, timestamp, ssrc
        # byte 0: version/flags, byte 1: payload type
        if data[0] >> 6 != 2:  # RTP version 2
            return None

        ssrc = struct.unpack_from(">I", data, 8)[0]

        # Skip our own SSRC
        if self._vc and ssrc == self._vc.ssrc:
            return None

        header = data[:12]

        # Check for RTP header extensions (bit 4 of byte 0)
        has_extension = bool(data[0] & 0x10)
        payload_offset = 12
        if has_extension and len(data) > 16:
            # Extension header: 2 bytes profile, 2 bytes length (in 32-bit words)
            ext_length = struct.unpack_from(">H", data, 14)[0]
            payload_offset = 16 + ext_length * 4

        if payload_offset >= len(data):
            return None

        encrypted_payload = data[payload_offset:]

        # Transport decryption
        vc = self._vc
        if vc is None:
            log.debug("decrypt: no vc")
            return None

        mode = vc.mode
        secret_key = vc.secret_key
        if not secret_key:
            log.debug("decrypt: no secret_key")
            return None

        if self._pkt_count <= 3:
            log.info("decrypt: mode=%s, secret_key_len=%d, header=%s", mode, len(secret_key), header.hex())

        try:
            opus_data = self._transport_decrypt(mode, header, data[12:], secret_key)
        except Exception as e:
            if self._pkt_count <= 5:
                log.warning("Transport decrypt exception: %s", e)
            return None

        if opus_data is None:
            if self._pkt_count <= 5:
                log.warning("Transport decrypt returned None, mode=%s, payload_len=%d", mode, len(data[12:]))
            return None

        if self._pkt_count <= 10:
            log.info("Transport decrypted OK, opus_len=%d, tail=%s", len(opus_data), opus_data[-10:].hex() if len(opus_data) >= 10 else opus_data.hex())

        # DAVE decryption (if session exists - other users may be DAVE-encrypting even if we can't encrypt)
        conn = vc._connection
        if conn.dave_session:
            user_id = self._ssrc_to_user.get(ssrc)
            if user_id is None:
                # Try to learn SSRC from pending map or voice channel members
                from main import _pending_ssrc_map
                if ssrc in _pending_ssrc_map:
                    uid = _pending_ssrc_map[ssrc]
                    self.register_speaking(uid, ssrc)
                    user_id = uid
                    log.info("Auto-registered SSRC %d → user %d from pending map", ssrc, uid)
                else:
                    if self._pkt_count <= 5 or self._pkt_count % 5000 == 0:
                        log.warning("DAVE: unknown SSRC %d (known SSRCs: %s)", ssrc, list(self._ssrc_to_user.keys()))
                    return None
            if self._pkt_count <= 3:
                log.info("DAVE session state: ready=%s, proto=%d, can_encrypt=%s",
                         conn.dave_session.ready, conn.dave_protocol_version, conn.can_encrypt)
            import davey

            # Keep passthrough mode enabled so unencrypted frames aren't rejected
            if not hasattr(self, '_passthrough_set'):
                self._passthrough_set = False
            if not self._passthrough_set and conn.dave_session:
                try:
                    conn.dave_session.set_passthrough_mode(True)
                    self._passthrough_set = True
                    log.info("DAVE passthrough mode enabled permanently")
                except Exception as e:
                    log.warning("Failed to set passthrough mode: %s", e)

            try:
                result = conn.dave_session.decrypt(user_id, davey.MediaType.audio, bytes(opus_data))
                if self._dave_ok < 3:
                    log.info("decrypt returned type=%s len=%d repr=%s", type(result).__name__, len(result), repr(result[:20]))
                # Force to bytes
                if isinstance(result, (list, tuple)):
                    opus_data = bytes(result)
                elif isinstance(result, bytes):
                    opus_data = result
                else:
                    opus_data = bytes(result)
            except Exception as e:
                self._dave_fail += 1
                if self._dave_fail <= 20 or self._dave_fail % 5000 == 0:
                    known_users = conn.dave_session.get_user_ids()
                    log.warning("DAVE decrypt fail #%d (pkt=%d) ssrc=%d→user=%d type=%s (epoch=%s known=%s): %s",
                                self._dave_fail, self._pkt_count, ssrc, user_id, type(user_id).__name__,
                                conn.dave_session.epoch, known_users, e)
                return None
            self._dave_ok += 1
            if self._dave_ok <= 5 or self._dave_ok % 5000 == 0:
                raw = bytes(opus_data)
                log.info("DAVE OK #%d (pkt=%d, fails=%d) opus_len=%d type=%s head=%s tail=%s user=%d",
                         self._dave_ok, self._pkt_count, self._dave_fail, len(raw), type(opus_data).__name__,
                         raw[:8].hex(), raw[-8:].hex(), user_id)

        return ssrc, opus_data

    def _transport_decrypt(self, mode: str, header: bytes, rest: bytes, secret_key: list[int]) -> bytes | None:
        """Reverse the transport encryption. rest = everything after the 12-byte RTP header."""
        key = bytes(secret_key)

        if mode == "aead_xchacha20_poly1305_rtpsize":
            # Format: [rtp_header (12)] [extension?] [encrypted_payload] [4-byte nonce]
            # The extension (if present) is unencrypted and part of the AAD
            if len(rest) < 4:
                return None

            nonce_bytes = rest[-4:]
            between = rest[:-4]  # extension (if any) + encrypted payload
            nonce = bytearray(24)
            nonce[:4] = nonce_bytes

            has_extension = bool(header[0] & 0x10)
            ext_header = b""
            ext_data_len = 0
            ct_start = 0
            if has_extension and len(between) > 4:
                # Only the 4-byte extension header (profile + length) is unencrypted/AAD
                # The extension DATA is part of the encrypted payload
                ext_header = between[:4]
                ext_data_len = struct.unpack_from(">H", ext_header, 2)[0] * 4
                ct_start = 4

            actual_ct = between[ct_start:]
            aad = bytes(header) + ext_header

            box = nacl.secret.Aead(key)
            try:
                plaintext = box.decrypt(actual_ct, aad, bytes(nonce))
                # Skip the encrypted extension data to get just the opus payload
                return plaintext[ext_data_len:]
            except Exception as e:
                if self._pkt_count <= 3:
                    log.warning("AEAD decrypt fail: ext=%d ext_total=%d ct_len=%d aad_len=%d nonce=%s err=%s",
                                has_extension, ct_start, len(actual_ct), len(aad), nonce_bytes.hex(), e)
                    log.warning("  full_pkt hex: %s", (bytes(header) + rest).hex())
                    log.warning("  key hex: %s", key.hex())
                return None

        elif mode == "xsalsa20_poly1305":
            nonce = bytearray(24)
            nonce[:12] = header
            box = nacl.secret.SecretBox(key)
            try:
                return box.decrypt(rest, bytes(nonce))
            except Exception:
                return None

        elif mode == "xsalsa20_poly1305_suffix":
            if len(rest) < 24:
                return None
            nonce = rest[-24:]
            ciphertext = rest[:-24]
            box = nacl.secret.SecretBox(key)
            try:
                return box.decrypt(ciphertext, bytes(nonce))
            except Exception:
                return None

        elif mode == "xsalsa20_poly1305_lite":
            if len(rest) < 4:
                return None
            nonce = bytearray(24)
            nonce[:4] = rest[-4:]
            ciphertext = rest[:-4]
            box = nacl.secret.SecretBox(key)
            try:
                return box.decrypt(ciphertext, bytes(nonce))
            except Exception:
                return None

        return None

    # ------------------------------------------------------------------
    # Socket callback (called from reader thread)
    # ------------------------------------------------------------------
    _pkt_count: int = 0
    _pkt_fail: int = 0
    _dave_ok: int = 0
    _dave_fail: int = 0

    def _on_packet(self, data: bytes) -> None:
        """Called by SocketReader for every UDP packet."""
        self._pkt_count += 1
        if self._pkt_count <= 5 or self._pkt_count % 1000 == 0:
            log.info("UDP packet #%d received, len=%d, first4=%s", self._pkt_count, len(data), data[:4].hex())
        result = self._decrypt_packet(data)
        if result is None:
            self._pkt_fail += 1
            if self._pkt_fail <= 10 or self._pkt_fail % 1000 == 0:
                log.debug("Decrypt failed for packet #%d (total fails: %d)", self._pkt_count, self._pkt_fail)
            return

        ssrc, opus_data = result
        state = self._ssrc_states.get(ssrc)
        if state is None:
            return

        user_id = state.user_id
        if user_id in self._excluded_users:
            return

        # Decode Opus → stereo PCM
        try:
            pcm_stereo = state.opus_decoder.decode(opus_data)
        except Exception as e:
            if self._dave_ok <= 20 or self._dave_ok % 5000 == 0:
                log.warning("Opus decode failed (dave_ok=%d): %s (data_len=%d)", self._dave_ok, e, len(opus_data))
            return

        if self._dave_ok <= 10 or self._dave_ok % 5000 == 0:
            log.info("Opus decoded: dave_ok=%d pkt=%d pcm_len=%d bytes user=%d", self._dave_ok, self._pkt_count, len(pcm_stereo), user_id)

        # Debug: log audio amplitude periodically
        if self._dave_ok <= 10 or self._dave_ok % 2000 == 0:
            samples = struct.unpack_from(f"<{min(100, len(pcm_stereo)//2)}h", pcm_stereo)
            peak = max(abs(s) for s in samples) if samples else 0
            log.info("Audio amplitude user=%d peak=%d (dave_ok=%d)", user_id, peak, self._dave_ok)

        username = self._resolve_username(user_id)

        # Accumulate raw audio
        if user_id not in self._raw_buffers:
            self._raw_buffers[user_id] = bytearray()
        self._raw_buffers[user_id].extend(pcm_stereo)

        # VAD
        segments = state.segmenter.process_chunk(pcm_stereo)
        for segment in segments:
            duration = len(segment) / (SAMPLE_RATE * PCM_SAMPLE_WIDTH)
            if duration < MIN_SPEECH_DURATION:
                continue

            if user_id not in self._speech_segments:
                self._speech_segments[user_id] = []
            self._speech_segments[user_id].append(segment)

            save_pcm_as_wav(
                segment,
                username,
                label="speech",
                sample_rate=SAMPLE_RATE,
                channels=1,
                sample_width=PCM_SAMPLE_WIDTH,
            )
            log.info("Speech segment from %s: %.1fs", username, duration)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start_listening(self, vc: discord.VoiceClient) -> None:
        """Register our callback on the voice client's socket reader."""
        self._vc = vc
        if not self._listening:
            vc._connection.add_socket_listener(self._on_packet)
            self._listening = True
            log.info("Audio sink started listening on voice socket")

    def stop_listening(self) -> None:
        """Unregister our callback."""
        if self._vc and self._listening:
            try:
                self._vc._connection.remove_socket_listener(self._on_packet)
            except Exception:
                pass
            self._listening = False

    def flush_raw_audio(self) -> dict[str, list]:
        results: dict[str, list] = {}
        for user_id, buf in self._raw_buffers.items():
            if not buf:
                continue
            username = self._user_names.get(user_id, str(user_id))
            path = save_pcm_as_wav(bytes(buf), username, label="raw")
            results.setdefault(username, []).append(path)
        self._raw_buffers.clear()
        return results

    def get_speech_segments(self, user_id: int | None = None) -> dict[int, list[bytes]]:
        if user_id is not None:
            return {user_id: self._speech_segments.get(user_id, [])}
        return dict(self._speech_segments)

    def clear_speech_segments(self, user_id: int | None = None) -> None:
        if user_id is not None:
            self._speech_segments.pop(user_id, None)
        else:
            self._speech_segments.clear()

    def cleanup(self) -> None:
        self.stop_listening()

        for ssrc, state in self._ssrc_states.items():
            remaining = state.segmenter.flush()
            if remaining:
                user_id = state.user_id
                username = self._user_names.get(user_id, str(user_id))
                save_pcm_as_wav(
                    remaining,
                    username,
                    label="speech_final",
                    sample_rate=SAMPLE_RATE,
                    channels=1,
                    sample_width=PCM_SAMPLE_WIDTH,
                )

        self.flush_raw_audio()
        self._ssrc_states.clear()
        self._ssrc_to_user.clear()
        self._speech_segments.clear()
        self._vc = None
        log.info("Audio sink cleaned up")
