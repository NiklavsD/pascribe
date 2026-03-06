"""Configuration loaded from environment / .env file."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DISCORD_TOKEN: str = os.environ["DISCORD_TOKEN"]
ASSEMBLYAI_API_KEY: str = os.environ["ASSEMBLYAI_API_KEY"]
# Separate key for streaming (optional — falls back to main key)
ASSEMBLYAI_STREAMING_KEY: str = os.getenv("ASSEMBLYAI_STREAMING_KEY", ASSEMBLYAI_API_KEY)

PASCRIBE_URL: str = os.getenv("PASCRIBE_URL", "http://localhost:3089")
PASCRIBE_TOKEN: str = os.getenv("PASCRIBE_TOKEN", "")

GUILD_ID: int = int(os.getenv("GUILD_ID", "1224716298584592414"))
BLACKLIST_CHANNELS: set[int] = {
    int(c) for c in os.getenv("BLACKLIST_CHANNELS", "1236733405698330635").split(",") if c.strip()
}

RECORDINGS_DIR: Path = Path(os.getenv("RECORDINGS_DIR", "./recordings")).resolve()
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

# Report channel and inactivity trigger
REPORT_CHANNEL_ID: int = int(os.getenv("REPORT_CHANNEL_ID", "1478214341298880583"))
INACTIVITY_THRESHOLD_S: int = int(os.getenv("INACTIVITY_THRESHOLD_H", "3")) * 3600

# OpenClaw gateway for instant wake word response
OPENCLAW_GATEWAY_URL: str = os.getenv("OPENCLAW_GATEWAY_URL", "http://127.0.0.1:18789")
OPENCLAW_GATEWAY_TOKEN: str = os.getenv("OPENCLAW_GATEWAY_TOKEN", "")

# Gap that splits separate "discussions" (seconds)
DISCUSSION_GAP_S: int = int(os.getenv("DISCUSSION_GAP_MIN", "30")) * 60

# OpenRouter API for transcript analysis
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")

# VAD settings
VAD_AGGRESSIVENESS: int = int(os.getenv("VAD_AGGRESSIVENESS", "2"))  # 0-3
SAMPLE_RATE: int = 48000  # Discord sends 48kHz
CHANNELS: int = 2  # Discord sends stereo
PCM_SAMPLE_WIDTH: int = 2  # 16-bit

# How many seconds of silence before we consider a speech segment ended
VAD_SILENCE_TIMEOUT: float = float(os.getenv("VAD_SILENCE_TIMEOUT", "1.5"))

# Minimum speech duration (seconds) to bother transcribing
MIN_SPEECH_DURATION: float = float(os.getenv("MIN_SPEECH_DURATION", "1.0"))
