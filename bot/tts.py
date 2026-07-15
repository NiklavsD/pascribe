"""GCP Cloud Text-to-Speech for Benjamin voice responses.

Synthesizes text to audio files that can be played in Discord VC
via FFmpegPCMAudio.
"""

from __future__ import annotations

import hashlib
import logging
import re
import tempfile
from pathlib import Path

from google.cloud import texttospeech

log = logging.getLogger(__name__)

# Cache directory for TTS audio
CACHE_DIR = Path(tempfile.gettempdir()) / "benjamin_tts"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Max text length for TTS (prevents huge API calls)
MAX_TEXT_LENGTH = 500

# Compiled regex for stripping markdown/mentions
_STRIP_RE = re.compile(
    r"<@!?\d+>|"         # Discord mentions
    r"<#\d+>|"           # Channel mentions
    r"<@&\d+>|"          # Role mentions
    r"\*{1,3}|"          # Bold/italic markers
    r"~~|"               # Strikethrough
    r"__?|"              # Underline
    r"`{1,3}|"           # Code markers
    r"#{1,6}\s|"         # Headers
    r"^\s*[-*+]\s|"      # List markers
    r"^\s*\d+\.\s|"      # Numbered lists
    r"\[([^\]]+)\]\([^)]+\)|"  # Links → keep text
    r"https?://\S+"      # Bare URLs
    , re.MULTILINE
)

_client: texttospeech.TextToSpeechClient | None = None


def _get_client() -> texttospeech.TextToSpeechClient:
    global _client
    if _client is None:
        _client = texttospeech.TextToSpeechClient()
    return _client


def strip_markdown(text: str) -> str:
    """Remove Discord markdown, mentions, and URLs from text."""
    # Replace link markdown with just the text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove everything else
    text = _STRIP_RE.sub("", text)
    # Clean up extra whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"  +", " ", text)
    return text.strip()


def synthesize(
    text: str,
    voice_name: str = "en-US-Chirp3-HD-Charon",
    speaking_rate: float = 1.05,
    pitch: float = -1.0,
    volume_gain_db: float = -8.0,
) -> Path | None:
    """Synthesize text to a WAV file. Returns path to the audio file, or None on error.

    Uses caching — identical text+voice combos return the cached file.
    """
    # Clean and truncate
    clean = strip_markdown(text)
    if not clean:
        return None

    if len(clean) > MAX_TEXT_LENGTH:
        # Truncate at sentence boundary
        truncated = clean[:MAX_TEXT_LENGTH]
        last_period = truncated.rfind(".")
        last_question = truncated.rfind("?")
        last_exclaim = truncated.rfind("!")
        best_cut = max(last_period, last_question, last_exclaim)
        if best_cut > MAX_TEXT_LENGTH // 2:
            clean = truncated[:best_cut + 1]
        else:
            clean = truncated.rstrip() + "..."

    # Check cache
    cache_key = hashlib.md5(f"{clean}:{voice_name}:{speaking_rate}:{pitch}:{volume_gain_db}".encode()).hexdigest()
    cache_path = CACHE_DIR / f"{cache_key}.wav"
    if cache_path.exists() and cache_path.stat().st_size > 100:
        log.debug("TTS cache hit: %s", cache_key[:8])
        return cache_path

    try:
        client = _get_client()

        synthesis_input = texttospeech.SynthesisInput(text=clean)

        voice = texttospeech.VoiceSelectionParams(
            language_code=voice_name[:5],  # e.g. "en-US"
            name=voice_name,
        )

        # Chirp 3 HD voices don't support pitch — only Neural2/WaveNet do
        is_chirp = "Chirp" in voice_name
        audio_kwargs = {
            "audio_encoding": texttospeech.AudioEncoding.LINEAR16,
            "sample_rate_hertz": 48000,  # Match Discord's native rate
            "speaking_rate": speaking_rate,
            "volume_gain_db": volume_gain_db,  # Negative dB lowers volume at synthesis
        }
        if not is_chirp:
            audio_kwargs["pitch"] = pitch
        audio_config = texttospeech.AudioConfig(**audio_kwargs)

        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
        )

        cache_path.write_bytes(response.audio_content)
        log.info("TTS synthesized %d chars → %s (%d bytes)", len(clean), cache_key[:8], len(response.audio_content))
        return cache_path

    except Exception:
        log.exception("TTS synthesis failed")
        return None


def cleanup_cache(max_files: int = 200) -> None:
    """Remove old cached TTS files if cache grows too large."""
    try:
        files = sorted(CACHE_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime)
        if len(files) > max_files:
            for f in files[: len(files) - max_files]:
                f.unlink(missing_ok=True)
            log.info("TTS cache cleanup: removed %d old files", len(files) - max_files)
    except Exception:
        pass
