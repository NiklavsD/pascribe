<p align="center">
  <img src="assets/banner.png" alt="Benjamin Banner" width="100%">
</p>

<p align="center">
  <b>Always-on Discord voice bot — listens, transcribes, and responds when called.</b>
</p>

---

## What is Benjamin?

Benjamin sits in your Discord voice channel, captures per-user audio streams, transcribes them via AssemblyAI, and responds intelligently when someone says his name. He uses a triple-detection system (local Vosk STT + AssemblyAI streaming + batch transcription) to catch wake words with ~95% reliability.

## Features

- 🎙️ **Per-user audio capture** — separate streams per speaker with VAD filtering
- 🔐 **DAVE E2EE support** — full Discord voice encryption (MLS protocol)
- 🗣️ **Wake word detection** — say "Benjamin" and he responds via AI
- ⚡ **Instant response** — Vosk local STT for ~2s detection, AI response in ~15s
- 🔔 **Audio feedback** — ascending chime on trigger, descending beep on cooldown
- 📝 **Dual transcription** — streaming (real-time) + batch (archive quality)
- 🤖 **AI responses** — Claude Opus generates contextual replies posted to Discord
- 📊 **Daily reports** — auto-generated conversation summaries
- 🔄 **Self-improving** — nightly cron reviews response quality and tunes prompts
- 🛡️ **Privacy controls** — voice opt-out keywords, channel blacklisting

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Discord Voice                      │
│  DAVE E2EE → Transport Decrypt → DAVE Decrypt → Opus │
└──────────────────────┬──────────────────────────────┘
                       │ PCM Audio (48kHz stereo)
                       ▼
              ┌────────┴────────┐
              │  UserAudioSink  │
              └────┬───┬───┬───┘
                   │   │   │
          ┌────────┘   │   └────────┐
          ▼            ▼            ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │   Vosk   │ │ Streaming│ │   VAD    │
    │  (local) │ │(Assembly)│ │ + Batch  │
    │ Wake Word│ │ Wake Word│ │Transcribe│
    └────┬─────┘ └────┬─────┘ └────┬─────┘
         │            │            │
         └──────┬─────┘            │
                ▼                  ▼
         ┌────────────┐    ┌────────────┐
         │  Trigger    │    │ Transcript │
         │  System     │    │  Archive   │
         └──────┬──────┘    └────────────┘
                │
                ▼
         ┌────────────┐
         │  OpenClaw   │
         │  /hooks/    │
         │  agent      │
         └──────┬──────┘
                │
                ▼
         ┌────────────┐
         │  Claude     │
         │  Response   │
         │  → Discord  │
         └─────────────┘
```

## Files

```
main.py                     Bot entrypoint, voice management, wake word routing
config.py                   Environment config (.env)
triggers.py                 Unified trigger system (cooldown, prompt, webhook)
wakeword.py                 Local Vosk wake word detector (offline, CPU-only)
analysis.py                 Transcript analysis via OpenRouter (optional)
self_improve.md             Self-improvement log (auto-updated by cron)
audio/
  capture.py                Per-user AudioSink: DAVE decrypt → Opus → PCM → VAD
  vad.py                    WebRTC VAD speech segmenter
  storage.py                WAV file storage + segment tracking
transcription/
  assemblyai.py             AssemblyAI batch API client
  pipeline.py               Per-segment transcription orchestration
  streaming.py              Per-user AssemblyAI real-time streaming
commands/
  slash.py                  /pascribe process & /pascribe status
assets/
  trigger_chime.wav         Ascending two-tone (trigger accepted)
  cooldown_chime.wav        Descending two-tone (cooldown active)
```

## Setup

### Prerequisites

- Python 3.11+
- FFmpeg (`sudo apt install ffmpeg`)
- libopus (`sudo apt install libopus0`)
- Vosk model (auto-downloaded: `vosk-model-small-en-us-0.15`, ~67MB)

### Install

```bash
cd bot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env with your tokens
```

### Environment Variables

```env
DISCORD_TOKEN=           # Discord bot token
ASSEMBLYAI_API_KEY=      # AssemblyAI API key (batch transcription)
GUILD_ID=                # Target Discord server ID
REPORT_CHANNEL_ID=       # Channel for voice reports
OPENCLAW_GATEWAY_URL=    # OpenClaw gateway (for AI responses)
OPENCLAW_GATEWAY_TOKEN=  # OpenClaw auth token
OPENROUTER_API_KEY=      # Optional: transcript analysis
```

### Run

```bash
# Direct
python main.py

# Systemd service (recommended)
systemctl --user enable --now benjamin
```

## How It Works

### Wake Word Detection (Triple Pipeline)

1. **Vosk (instant, local)** — Offline STT on CPU, checks partial results for "benjamin". ~2s latency, no API cost.
2. **AssemblyAI Streaming** — Per-user WebSocket streams, catches what Vosk misses. ~5s latency.
3. **Batch fallback** — Every 30s, new speech segments are transcribed and checked. Catches anything the other two missed.

All three feed into a unified trigger system with a **30-second global cooldown** — only one response per mention regardless of which detector fires first.

### Response Flow

1. Wake word detected → ascending chime plays in VC
2. ⏳ "Processing" placeholder posted to #voice-reports
3. Trigger fires via OpenClaw `/hooks/agent` webhook
4. Claude Opus reads the trigger context + recent conversation
5. Response posted to #voice-reports, placeholder auto-deletes

### Audio Pipeline

- Discord DAVE E2EE (protocol v1, `aead_xchacha20_poly1305_rtpsize`)
- Per-user Opus decoding via custom ctypes wrapper
- 48kHz stereo → mono downsampling for Vosk (16kHz) and streaming
- WebRTC VAD (aggressiveness=3) filters noise into speech segments
- Segments saved as WAV files, batch-transcribed individually for accurate speaker attribution

## Cost

| Component | Cost | Notes |
|-----------|------|-------|
| Vosk wake word | Free | Local, CPU-only |
| AssemblyAI batch | ~$0.21/hr | Per-segment transcription |
| AssemblyAI streaming | ~$0.60/hr | Per-user real-time (optional) |
| Claude Opus response | ~$0.05/trigger | Via OpenClaw |
| **Typical daily cost** | **$5-15** | Depends on VC activity |

## License

Private — part of the Pascribe project.
