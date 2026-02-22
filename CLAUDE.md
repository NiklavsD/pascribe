# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pascribe is a Windows-only system tray application that continuously captures microphone and system audio into a 60-minute rolling buffer. A global hotkey triggers faster-whisper transcription of the last N minutes, copying the result to the clipboard.

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

The entire application is a single file: `pascribe.py` (~400 lines).

**Data flow:** Audio devices → sounddevice callbacks → AudioBuffer deques → mix on hotkey → faster-whisper transcription → pyperclip → clipboard

**Key components within pascribe.py:**

- **Config** (lines 22-52): Loads/saves `config.json` with device indices, Whisper model/device, hotkey mappings. Missing keys are merged from `DEFAULT_CONFIG`.
- **Device Selection** (lines 56-114): Interactive console wizard for picking mic and system audio devices. Runs automatically when `config.json` has `null` device values.
- **AudioBuffer** (lines 118-144): Thread-safe dataclass wrapping a `deque` of 1-second numpy chunks with a lock. Two global instances: `mic_buffer` and `loopback_buffer`.
- **Audio Capture** (lines 158-218): Two `sounddevice.InputStream` callbacks write mono float32 chunks into their respective buffers. System audio is downmixed from stereo.
- **Transcription** (lines 222-253): Lazy-loads `faster-whisper` `WhisperModel` on first use. Uses VAD filtering and beam search. Auto-detects language.
- **Hotkey Handling** (lines 257-319): `keyboard` library registers `{prefix}+{key}` combos from config. Each hotkey press spawns a daemon thread that mixes both buffers, normalizes, transcribes, and copies to clipboard.
- **Tray Icon** (lines 323-368): `pystray` icon with status updates. Menu shows config info, reconfigure option, and quit.

**Threading model:** Main thread runs the pystray event loop (`icon.run()`). Audio capture runs in sounddevice's callback threads. Hotkey transcriptions spawn daemon threads. Whisper model is pre-loaded in a background thread at startup.

## Key Dependencies

- `faster-whisper` — Whisper transcription (CTranslate2 backend)
- `sounddevice` — Audio capture via PortAudio/WASAPI
- `keyboard` — Global hotkey registration (requires admin on some systems)
- `pystray` / `Pillow` — System tray icon
- `pyperclip` — Clipboard access
- CUDA libraries (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`) installed separately by `install.bat`

## Config

`config.json` is auto-generated on first run. Device values are sounddevice integer indices. Set a device to `null` to re-trigger the setup wizard. Hotkeys map digit strings to minute durations, prefixed by `hotkey_prefix` (default: `"right shift"`).
