# Pascribe

Rolling audio buffer → hotkey → Whisper transcription → clipboard.

Captures mic and system audio into a rolling RAM buffer. Press a hotkey to transcribe the last N minutes to clipboard. Optionally records full-day audio for speaker-labeled transcription via AssemblyAI, with a dark-themed control panel, real-time audio visualizer, and homelab integration.

## Setup

```bash
git clone https://github.com/NiklavsD/pascribe.git
cd pascribe
install.bat
```

## Usage

```bash
run.bat
```

On first launch, Pascribe starts in the system tray. **Left-click** the icon to open the control panel, or **right-click** for the quick menu. Select your audio devices in Settings.

## Control Panel

Left-click the tray icon to open the unified control panel (dark theme, Tokyo Night palette) with four tabs:

| Tab | Contents |
|-----|----------|
| **Dashboard** | Live status indicator, full-day audio visualizer (MIC/SYS rows), quick-transcribe buttons, Pause/Resume, Transcribe Today, Stop Recording, Run on Startup toggle, last transcript preview |
| **History** | Scrollable list of past quick-transcriptions — click a row to copy |
| **Daily Transcripts** | Browse speaker-labeled transcripts by date; search, filter by speaker, click-to-copy line, copy buttons, **Send to Homelab** button, transcribe past dates |
| **Settings** | All configuration: devices, Whisper model, hotkeys, API key, homelab URL, daily recording options |

### Audio Visualizer

The dashboard includes a two-row waveform visualizer showing MIC (green, top) and SYS (blue, bottom) audio levels across the full recording day. Pre-loads historical data from raw files on startup and appends live levels in real-time. Shows recording start time, hour markers, and a "now" indicator.

### Stop Recording

Click **Stop Recording** in the Dashboard to end today's recording session. Choose to delete raw audio files or keep them for later transcription.

## Hotkeys (Ctrl+Alt + number)

| Key | Duration |
|-----|----------|
| Ctrl+Alt+9 | 1 min |
| Ctrl+Alt+8 | 3 min |
| Ctrl+Alt+7 | 5 min |
| Ctrl+Alt+1 | 10 min |
| Ctrl+Alt+2 | 20 min |
| Ctrl+Alt+3 | 30 min |
| Ctrl+Alt+4 | 40 min |
| Ctrl+Alt+5 | 50 min |
| Ctrl+Alt+6 | 60 min |

Customize prefix and key mappings in the Settings tab or `config.json`.

## Daily Recording + AssemblyAI Transcription

Enable **daily recording** in Settings. Pascribe writes mic and system audio to separate raw files under `recordings/YYYY-MM-DD/`.

Click **Transcribe Today** (Dashboard tab) or select any past date in the **Daily Transcripts** tab and click the **Transcribe** button:

1. Strips silence with energy VAD to reduce billable audio
2. Uploads to AssemblyAI (Universal-3-Pro model) for transcription
3. Assigns speaker labels ("you" vs "discord") by comparing mic vs system audio levels
4. Saves `recordings/YYYY-MM-DD/transcript.json`
5. Optionally POSTs the transcript to your homelab URL

Past dates with raw audio but no transcript show a "Transcribe" button with file size and duration info.

**Cost:** ~$12/month for 8h/day recording with AssemblyAI ($0.0035/min) after VAD reduces ~8h → ~2–3h billable audio.

## Voicemeeter

If you use Voicemeeter to separate Discord audio from your mic:

- **Mic device** → your physical mic or Voicemeeter output carrying your voice
- **System device** → Voicemeeter output carrying Discord/game audio

This lets Pascribe capture both tracks separately: mixed for Whisper hotkey transcription, and separated for per-segment speaker labeling in daily transcripts.

## OpenClaw Homelab Integration

Set `homelab_url` in Settings to enable sending transcripts to your homelab server (e.g. an OpenClaw instance).

Transcripts can be sent in two ways:
- **Automatically** after each daily transcription run
- **Manually** via the **Send to Homelab** button in the Daily Transcripts tab (works for any date)

### Transcript payload format

```json
{
  "date": "2026-02-28",
  "recorded_from": "2026-02-28T09:12:00",
  "duration_minutes": 480.0,
  "speech_minutes": 87.3,
  "word_count": 12400,
  "segments": [
    {"time": "09:14:22", "speaker": "you",     "text": "Hey, what did you think of that?"},
    {"time": "09:14:35", "speaker": "discord", "text": "Honestly pretty good."}
  ]
}
```

The payload is HTTP POSTed as JSON with `Content-Type: application/json`.

Each `segment` has:
- `time` — wall-clock timestamp (`HH:MM:SS`)
- `speaker` — `"you"` (mic) or `"discord"` (system audio)
- `text` — transcribed text for that utterance

### OpenClaw receiver setup

To receive Pascribe transcripts in [OpenClaw](https://github.com/NiklavsD/openclaw), add an HTTP endpoint that accepts the JSON payload. Example controller:

```python
# openclaw/routes/transcripts.py
from flask import Blueprint, request, jsonify
import json, pathlib

bp = Blueprint("transcripts", __name__)

@bp.route("/api/pascribe/transcript", methods=["POST"])
def receive_transcript():
    data = request.get_json()
    date = data["date"]

    # Store the transcript
    out = pathlib.Path(f"data/transcripts/{date}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    word_count = data.get("word_count", 0)
    speech_min = data.get("speech_minutes", 0)
    segments   = data.get("segments", [])
    print(f"[Pascribe] {date}: {word_count} words, {speech_min:.0f} min, {len(segments)} segments")

    return jsonify({"ok": True, "date": date, "segments": len(segments)})
```

Register the blueprint in your OpenClaw app:

```python
from routes.transcripts import bp as transcripts_bp
app.register_blueprint(transcripts_bp)
```

Then set `homelab_url` in Pascribe to `http://your-openclaw-server:port/api/pascribe/transcript`.

### Standalone Flask receiver

If you don't use OpenClaw, a minimal standalone receiver:

```python
from flask import Flask, request, jsonify
import json, pathlib

app = Flask(__name__)

@app.route("/api/transcript", methods=["POST"])
def receive_transcript():
    data = request.get_json()
    date = data["date"]
    out = pathlib.Path(f"transcripts/{date}.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"[{date}] {data['word_count']} words, {data['speech_minutes']:.0f} min speech")
    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
```

## Config

All options are editable in the Settings panel. Key `config.json` fields:

| Key | Default | Description |
|-----|---------|-------------|
| `hotkey_prefix` | `"ctrl+alt"` | Modifier for transcription hotkeys |
| `hotkeys` | `{"9":1,"8":3,...}` | Key → minutes mapping |
| `whisper_model` | `"large-v3"` | faster-whisper model size |
| `whisper_device` | `"cuda"` | `"cuda"` or `"cpu"` |
| `buffer_minutes` | `60` | Rolling RAM buffer size |
| `daily_recording` | `false` | Enable disk recording |
| `recording_path` | `"recordings"` | Where daily audio is stored |
| `assemblyai_key` | `""` | AssemblyAI API key |
| `homelab_url` | `null` | HTTP endpoint for transcripts |
| `delete_after_transcribe` | `false` | Delete raw audio after transcription |

Set `mic_device` or `system_device` to `null` to re-trigger device selection on next launch.

## Safety Features

- **VRAM check** — refuses to load Whisper if insufficient GPU memory (prevents OOM crashes while gaming)
- **Hallucination filter** — discards known phantom Whisper outputs on near-silence audio
- **Single-instance lock** — prevents duplicate tray instances from accumulating
- **Pause hotkeys** — toggle via the Dashboard or tray menu to disable all hotkeys temporarily

## Requirements

- Windows 10/11
- Python 3.11+
- NVIDIA GPU with CUDA (recommended) — or set `whisper_device: "cpu"`
- ~3.5 GB VRAM for large-v3 model
- ~600 MB RAM for 60-min audio buffer
