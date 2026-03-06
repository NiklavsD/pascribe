# Benjamin — Pascribe Discord Voice Bot

Always-on Discord bot that joins voice channels, captures per-user audio, runs VAD, transcribes via AssemblyAI, and feeds into the Pascribe analysis pipeline.

## Setup

```bash
cd /home/nik/clawd/projects/pascribe/bot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# System dependency
sudo apt install ffmpeg

cp .env.example .env
# Edit .env with your tokens
```

## Run

```bash
python main.py
```

## Systemd Service

```bash
# Copy or symlink the service file
sudo cp benjamin.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now benjamin
```

## Features

- **Auto-join** — joins the VC with the most people (checks every 30s)
- **Blacklist** — configurable channel blacklist (default: Game channel)
- **Per-user recording** — separate WAV files in `recordings/YYYY-MM-DD/username/`
- **VAD** — webrtcvad filters silence; only speech segments get transcribed
- **Voice commands** — say "Benjamin" or "Ben, process this" to trigger analysis
- **Privacy** — say "pause recording" / "stop listening" to opt out
- **Slash commands** — `/pascribe process` and `/pascribe status`
- **Daily report** — auto-runs at 23:55 UTC (configurable)
- **AssemblyAI** — cloud transcription with speaker labels

## Architecture

```
main.py                  Bot entrypoint, voice channel management
config.py                Environment config
audio/capture.py         Per-user AudioSink with VAD integration
audio/vad.py             webrtcvad speech segmenter
audio/storage.py         WAV file storage
transcription/assemblyai.py   AssemblyAI API client
transcription/pipeline.py     Orchestration: transcribe → Pascribe server
commands/slash.py        /pascribe slash commands
commands/voice.py        Voice keyword detection
```
