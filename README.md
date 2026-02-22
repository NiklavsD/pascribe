# Pascribe

Rolling audio buffer → hotkey → Whisper transcription → clipboard.

Captures mic + system audio (Voicemeeter, WASAPI loopback, Stereo Mix) into a 60-minute rolling buffer. Press a hotkey to transcribe the last N minutes and copy to clipboard.

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

On first launch, you'll be asked to select your audio devices (mic + system audio). Picks are saved to `config.json`.

### Hotkeys (Right Shift + number)

| Key | Duration |
|-----|----------|
| RShift + 9 | 1 min |
| RShift + 8 | 3 min |
| RShift + 7 | 5 min |
| RShift + 1 | 10 min |
| RShift + 2 | 20 min |
| RShift + 3 | 30 min |
| RShift + 4 | 40 min |
| RShift + 5 | 50 min |
| RShift + 6 | 60 min |

Customize hotkeys in `config.json`.

## Config

Edit `config.json` to change devices, model, hotkeys:

```json
{
  "mic_device": 3,
  "system_device": 5,
  "whisper_model": "large-v3",
  "whisper_device": "cuda",
  "hotkeys": { "1": 10, "2": 20, "9": 1 },
  "hotkey_prefix": "right shift"
}
```

Set device to `null` to re-trigger the setup wizard on next launch.

## Voicemeeter

If you use Voicemeeter to separate Discord audio from your mic:
- **Mic device** → your physical mic or Voicemeeter output with your voice
- **System device** → Voicemeeter output carrying Discord audio

This lets Pascribe capture both tracks separately and mix them for transcription.

## Requirements

- Windows 10/11
- Python 3.11+
- NVIDIA GPU with CUDA (recommended) — or set `whisper_device: "cpu"`
- ~1.5GB VRAM for large-v3 model
- ~600MB RAM for 60-min audio buffer
