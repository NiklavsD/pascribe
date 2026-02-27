# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pascribe is a Windows-only system tray application that:
1. **Quick transcription** — continuously captures mic and system audio into a 60-minute rolling RAM buffer; a global hotkey transcribes the last N minutes to clipboard
2. **Daily recording** — optionally writes full-day audio to disk; "Transcribe today..." uploads VAD-stripped audio to AssemblyAI and saves a speaker-labeled JSON transcript

## Setup & Run

```bash
# Install (creates venv, installs deps + CUDA libraries)
install.bat

# Run
run.bat

# Or manually:
venv/Scripts/python pascribe.py
```

No test suite exists. The app requires audio hardware to run.

## Architecture

The entire application is a single file: `pascribe.py` (~1500 lines).

**Data flow (quick hotkey):**
Audio devices → sounddevice callbacks → AudioBuffer deques → mix on hotkey → faster-whisper → pyperclip → clipboard

**Data flow (daily recording):**
Audio devices → sounddevice callbacks → DailyAudioRecorder → mic.raw / sys.raw on disk → VAD strip → AssemblyAI API → speaker assignment → transcript.json → HTTP POST to homelab

**Key components within pascribe.py:**

- **Config** (~line 64): Loads/saves `config.json`. New keys: `daily_recording`, `recording_path`, `assemblyai_key`, `homelab_url`, `delete_after_transcribe`.
- **AudioBuffer** (~line 200): Thread-safe deque of 1-second numpy chunks. Two globals: `mic_buffer` and `loopback_buffer`.
- **Audio Capture** (~line 255): Two `sounddevice.InputStream` callbacks write mono float32 chunks into RAM buffers AND (if enabled) to `DailyAudioRecorder`.
- **Transcription** (~line 330): Lazy-loads `faster-whisper` `WhisperModel` on first use. Includes VRAM check before loading, hallucination filter on output.
- **Resampling** (~line 420): Linear interpolation via numpy for devices that don't support 16 kHz natively.
- **DailyAudioRecorder** (~line 430): Writes float32 raw PCM to `recordings/YYYY-MM-DD/mic.raw` and `sys.raw` at 16 kHz. Creates `meta.json` with session start time. Resumes across restarts by appending.
- **Energy VAD** (`_energy_vad`, ~line 490): Adaptive energy-based silence detection. Used to pre-strip audio before AssemblyAI upload to reduce billable minutes.
- **AssemblyAI** (~line 580): `transcribe_with_assemblyai()` — upload WAV bytes, poll for completion, group words into utterances by pause threshold.
- **Speaker Assignment** (`_assign_speaker`, ~line 560): Compares per-segment RMS energy of mic vs system stream to label each utterance as "you" or "discord" — no diarization needed.
- **Daily Transcription** (`_daily_transcription_worker`, ~line 625): Orchestrates load → VAD → transcribe → remap timestamps → assign speakers → save JSON → POST to homelab.
- **Hotkey Handling** (~line 710): `keyboard` library registers `{prefix}+{key}` combos. Each press spawns a daemon thread. Includes pause guard and single-transcription-at-a-time lock.
- **Tray Icon** (~line 800): `pystray` icon. Menu: Settings, Transcription History, Daily Transcripts viewer, Transcribe today, Pause hotkeys, Run on startup, Quit.
- **Settings Window** (~line 840): tkinter GUI for device selection and daily recording toggle.
- **History Window** (~line 900): Shows quick-transcription history; click to copy.
- **Daily Transcripts Viewer** (~line 970): Split-pane tkinter window. Left: date list. Right: speaker-colored transcript with search, filter by speaker (you/discord), click-to-copy line, copy buttons.
- **Single-instance lock** (`_acquire_instance_lock`, ~line 1130): Binds port 47823 locally to prevent duplicate tray instances.

**Threading model:** Main thread runs pystray event loop. Audio capture runs in sounddevice callbacks. Hotkey transcriptions and daily transcription run in daemon threads. Whisper is loaded lazily and unloaded after each use.

## Key Dependencies

- `faster-whisper` — Whisper transcription (CTranslate2 backend)
- `sounddevice` — Audio capture via PortAudio/WASAPI
- `keyboard` — Global hotkey registration (requires admin on some systems)
- `pystray` / `Pillow` — System tray icon
- `pyperclip` — Clipboard access
- CUDA libraries (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`) installed separately by `install.bat`
- AssemblyAI REST API (`urllib.request`, stdlib only) — for daily batch transcription

## Config

`config.json` is auto-generated on first run. Important keys:

| Key | Default | Description |
|-----|---------|-------------|
| `hotkey_prefix` | `"ctrl+alt"` | Modifier for transcription hotkeys |
| `hotkeys` | `{"9":1,"8":3,...}` | Key → minutes mapping |
| `whisper_model` | `"large-v3"` | faster-whisper model |
| `whisper_device` | `"cuda"` | `"cuda"` or `"cpu"` |
| `daily_recording` | `false` | Enable disk recording |
| `recording_path` | `"recordings"` | Where to store daily audio |
| `assemblyai_key` | `""` | AssemblyAI API key |
| `homelab_url` | `null` | HTTP endpoint for daily transcripts |
| `delete_after_transcribe` | `false` | Delete raw audio after transcription |

## Output Format (daily transcripts)

```json
{
  "date": "2026-02-28",
  "recorded_from": "2026-02-28T09:12:00",
  "duration_minutes": 480.0,
  "speech_minutes": 87.3,
  "word_count": 12400,
  "segments": [
    {"time": "09:14:22", "speaker": "you", "text": "..."},
    {"time": "09:14:35", "speaker": "discord", "text": "..."}
  ]
}
```

Saved to `recordings/YYYY-MM-DD/transcript.json` and optionally POSTed to `homelab_url`.
