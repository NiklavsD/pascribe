# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pascribe is a Windows-only system tray application that:
1. **Quick transcription** — continuously captures mic and system audio into a 60-minute rolling RAM buffer; a global hotkey transcribes the last N minutes to clipboard
2. **Daily recording** — optionally writes full-day audio to disk; "Transcribe today..." uploads VAD-stripped audio to AssemblyAI and saves a speaker-labeled JSON transcript

## Setup & Run

```bash
# Install GPU or CPU mode
install.bat gpu
install.bat cpu

# Run
run.bat

# Or manually:
venv/Scripts/python pascribe.py
```

Core tests do not require audio hardware or a GPU:

```bash
test.bat
# or: venv/Scripts/python -m unittest discover -s tests -v
```

## Architecture

`pascribe.py` remains the Windows entry point and UI shell. Testable services are
split into `pascribe_app/` modules:

- `audio_processing.py` — streaming VAD, resampling, timestamp maps, bounded WAV writing
- `recordings.py` — fixed-size, disk-backed raw-audio snapshots and speaker comparison
- `jobs.py` — latest-request and exclusive cancellable background-job runners
- `storage.py` — atomic JSON writes and corrupt-file recovery
- `paths.py` — `%LOCALAPPDATA%\Pascribe` paths and legacy-data migration
- `validation.py` — config normalization and settings validation
- `secrets.py` — Windows DPAPI protection for the AssemblyAI key
- `diagnostics.py` — environment, audio, storage, dependency, and CUDA checks
- `transcription.py` — cloud word-response validation and utterance grouping

**Data flow (quick hotkey):**
Audio devices → sounddevice callbacks → AudioBuffer deques → latest-request queue → serialized faster-whisper lifecycle → clipboard

**Data flow (daily recording):**
Audio devices → DailyAudioRecorder → fixed disk snapshot → streaming VAD → speech-only WAV → AssemblyAI/local Whisper → disk-backed speaker assignment → atomic transcript.json → optional webhook

**Key components within pascribe.py:**

- **Config**: Stored under `%LOCALAPPDATA%\Pascribe`, validated on load, written atomically, and migrated from legacy root files.
- **AudioBuffer** (~line 200): Thread-safe deque of 1-second numpy chunks. Two globals: `mic_buffer` and `loopback_buffer`.
- **Audio Capture** (~line 255): Two `sounddevice.InputStream` callbacks write mono float32 chunks into RAM buffers AND (if enabled) to `DailyAudioRecorder`.
- **Transcription** (~line 330): Lazy-loads `faster-whisper` `WhisperModel` on first use. Includes VRAM check before loading, hallucination filter on output.
- **Resampling** (~line 420): Linear interpolation via numpy for devices that don't support 16 kHz natively.
- **DailyAudioRecorder**: Writes int16 raw PCM and exposes flushed sample limits for stable snapshots.
- **Daily Transcription**: Uses bounded reads; it does not construct full mic, system, mixed, or stripped arrays in memory.
- **Hotkey Handling**: Inference is serialized and only the newest pending request is retained.
- **Tray Icon** (~line 800): `pystray` icon. Menu: Settings, Transcription History, Daily Transcripts viewer, Transcribe today, Pause hotkeys, Run on startup, Quit.
- **Settings Window** (~line 840): tkinter GUI for device selection and daily recording toggle.
- **History Window** (~line 900): Shows quick-transcription history; click to copy.
- **Daily Transcripts Viewer** (~line 970): Split-pane tkinter window. Left: date list. Right: speaker-colored transcript with search, filter by speaker (you/discord), click-to-copy line, copy buttons.
- **Single-instance lock** (`_acquire_instance_lock`, ~line 1130): Binds port 47823 locally to prevent duplicate tray instances.

**Threading model:** Main thread runs pystray. `LatestJobRunner` owns quick work, `ExclusiveJobRunner` owns daily work, and `_whisper_inference_lock` protects model load/inference/unload.

## Key Dependencies

- `faster-whisper` — Whisper transcription (CTranslate2 backend)
- `sounddevice` — Audio capture via PortAudio/WASAPI
- `keyboard` — Global hotkey registration (requires admin on some systems)
- `pystray` / `Pillow` — System tray icon
- `pyperclip` — Clipboard access
- CUDA libraries (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`) installed separately by `install.bat`
- AssemblyAI REST API (`urllib.request`, stdlib only) — for daily batch transcription

## Config

`%LOCALAPPDATA%\Pascribe\config.json` is auto-generated on first run. Important keys:

| Key | Default | Description |
|-----|---------|-------------|
| `hotkey_prefix` | `"ctrl+alt"` | Modifier for transcription hotkeys |
| `hotkeys` | `{"9":1,"8":3,...}` | Key → minutes mapping |
| `whisper_model` | `"large-v3"` | faster-whisper model |
| `whisper_device` | `"cuda"` | `"cuda"` or `"cpu"` |
| `daily_recording` | `false` | Enable disk recording |
| `recording_path` | `%LOCALAPPDATA%\Pascribe\recordings` | Where to store daily audio |
| `assemblyai_key` | `""` | In-memory API key; DPAPI-protected on disk |
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

Saved under the configured recording directory as `YYYY-MM-DD/transcript.json` and optionally POSTed to `homelab_url`.
