# Pascribe

**Turn voice conversations into actionable context for AI agents.**

Pascribe started from a simple idea: every evening, friends hop on Discord to hang out, discuss projects, brainstorm ideas, and sometimes actually get work done. Those conversations are full of decisions, action items, and context that disappears the moment someone disconnects.

What if your AI assistant could listen in, understand what was discussed, and pick up where the conversation left off? What if you could say "Benjamin, what did we decide about the API?" and get an instant answer? What if the full context of a 3-hour brainstorming session could be fed directly into your dev environment the next morning?

That's Pascribe — a bridge between voice conversations and AI-powered workflows.

---

## Components

| Component | What it does | You need this if... |
|-----------|-------------|---------------------|
| [**Desktop App**](#desktop-app) | Captures mic/system audio, hotkey → instant transcription | You want local transcription on Windows |
| [**Benjamin**](bot/) | Discord voice bot — joins VC, transcribes, responds by name | You want always-on Discord voice capture |
| [**Analysis Server**](server/) | Processes transcripts into topics, knowledge, action items | You want AI analysis of conversations |

Each component works independently. Clone the repo and use only what you need:

```bash
git clone https://github.com/NiklavsD/pascribe.git

# Desktop only? Just need the root files
cd pascribe && install.bat

# Benjamin only? Just need the bot/ folder
cd pascribe/bot && pip install -r requirements.txt

# Analysis only? Just need server/
cd pascribe/server && pip install -r requirements.txt
```

## The Vision

```
Voice Chat (Discord / IRL)
    │
    ▼
┌──────────────┐     ┌──────────────┐
│   Benjamin   │     │   Desktop    │
│  (Discord)   │     │   (Local)    │
└──────┬───────┘     └──────┬───────┘
       │                    │
       ▼                    ▼
┌─────────────────────────────────────┐
│         Transcript Archive          │
│   Speaker-labeled, timestamped      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│        AI Agent (OpenClaw)          │
│  • Summarize discussions            │
│  • Extract action items             │
│  • Answer questions about context   │
│  • Continue work based on decisions │
│  • Self-improve response quality    │
└─────────────────────────────────────┘
```

The goal: **voice in → structured context out → AI acts on it.**

Your evening Discord session where you discussed refactoring the auth system becomes tomorrow's system prompt. The brainstorm about a new feature becomes a structured spec. The decision to use Postgres over Mongo gets captured forever so nobody asks the same question twice.

> 💡 **AssemblyAI offers $50 free credits** on signup — enough for ~230 hours of transcription. [Sign up here.](https://www.assemblyai.com)

---

## Desktop App

Rolling audio buffer → hotkey → Whisper transcription → clipboard.

Captures mic and system audio into a rolling RAM buffer. Press a hotkey to transcribe the last N minutes to clipboard. Optionally records full-day audio for end-of-day speaker-labeled transcription via AssemblyAI.

**Platform:** Windows 10/11 • **Requirements:** Python 3.11+, NVIDIA GPU recommended

### Setup

```bash
git clone https://github.com/NiklavsD/pascribe.git
cd pascribe
install.bat
```

### Usage

```bash
run.bat
```

On first launch, Pascribe starts in the system tray. **Left-click** the icon to open the control panel, or **right-click** for the quick menu. Select your audio devices in Settings.

### Control Panel

| Tab | Contents |
|-----|----------|
| **Dashboard** | Live status, quick-transcribe buttons, Pause/Resume, last transcript preview |
| **History** | Past quick-transcriptions — click to copy |
| **Daily Transcripts** | Browse speaker-labeled transcripts by date; search, filter, copy |
| **Settings** | Devices, Whisper model, hotkeys, API key, homelab URL |

### Hotkeys (Ctrl+Alt + number)

| Key | Duration | | Key | Duration |
|-----|----------|-|-----|----------|
| Ctrl+Alt+9 | 1 min | | Ctrl+Alt+1 | 10 min |
| Ctrl+Alt+8 | 3 min | | Ctrl+Alt+2 | 20 min |
| Ctrl+Alt+7 | 5 min | | Ctrl+Alt+3 | 30 min |

### Daily Recording + AssemblyAI

Enable **daily recording** in Settings. At end of day, click **Transcribe Today**:

1. Strips silence with VAD to reduce billable audio
2. Uploads to AssemblyAI for transcription
3. Assigns speaker labels ("you" vs "discord")
4. Saves transcript with timestamps
5. Optionally POSTs to your AI agent's webhook

**Cost:** ~$12/month for 8h/day recording after VAD reduces to ~2–3h billable.

### OpenClaw Integration

Set `homelab_url` in Settings to automatically push daily transcripts to your OpenClaw instance. The transcript arrives as structured JSON with speaker labels and timestamps — ready for your agent to process.

```json
{
  "date": "2026-02-28",
  "speech_minutes": 87.3,
  "segments": [
    {"time": "09:14:22", "speaker": "you", "text": "Let's use Postgres for this."},
    {"time": "09:14:35", "speaker": "discord", "text": "Yeah, makes more sense than Mongo."}
  ]
}
```

### Voicemeeter Support

If you use Voicemeeter to separate Discord audio from your mic, set each device independently for accurate per-speaker labeling.

### Config

All options editable in the Settings panel. Key fields:

| Key | Default | Description |
|-----|---------|-------------|
| `whisper_model` | `large-v3` | faster-whisper model |
| `whisper_device` | `cuda` | `cuda` or `cpu` |
| `buffer_minutes` | `60` | Rolling RAM buffer |
| `daily_recording` | `false` | Enable disk recording |
| `assemblyai_key` | `""` | AssemblyAI API key |
| `homelab_url` | `null` | Webhook for transcripts |

### Safety

- **VRAM check** — won't load Whisper if insufficient GPU memory
- **Hallucination filter** — discards known phantom Whisper outputs
- **Single-instance lock** — prevents duplicate tray instances
- **Pause hotkeys** — toggle via Dashboard or tray menu

**Requirements:** Windows 10/11 • Python 3.11+ • NVIDIA GPU with CUDA (recommended) • ~3.5 GB VRAM for large-v3

---

## Benjamin (Discord Voice Bot)

→ **[Full documentation](bot/)**

Always-on Discord bot with DAVE E2EE support, triple wake word detection (Vosk + streaming + batch), and AI responses via OpenClaw.

**Platform:** Linux • **Requirements:** Python 3.11+, FFmpeg, libopus

---

## Analysis Server

→ **[Server documentation](server/)**

Flask server that receives transcripts and runs AI analysis — extracting topics, knowledge items, and research questions. Feeds into the Pascribe analysis pipeline.

---

## License

Private project.
