# Branch: freddie/windows-local

Personal hardening pass on the **Desktop App only** (`pascribe.py`). Nothing in
`bot/` or `server/` was touched. Base: `main`.

> ⚠️ Written and byte-compiled on Linux — **not** run on Windows/CUDA/audio here.
> Please smoke-test on your machine before relying on it (checklist at the bottom).

## What changed & why

1. **Serialized transcription worker** (`request_transcription` → single
   `_transcription_worker`). Before, each hotkey press spawned its own thread;
   a finishing/cancelled press could call `unload_whisper()` (null the shared
   model) or flip the busy flag out from under a press still running, and two
   `WhisperModel.transcribe()` calls could hit the same CTranslate2 model at
   once. Now one worker runs jobs latest-wins; a new press cancels the in-flight
   one cleanly. Model is only ever loaded/unloaded on that one thread.

2. **Atomic JSON writes** (`_atomic_write_json`, temp+fsync+`os.replace`) for
   `config.json`, `history.json`, daily `transcript.json`, and `meta.json`, plus
   a lock around history append. A kill mid-write used to corrupt the file, and
   `load_history()` silently returns `[]` on corrupt JSON — i.e. it wiped history.

3. **Rotating log** — `pascribe.log` was append-forever (this is a run-at-startup
   daemon). Now `RotatingFileHandler` (2 MB × 3). Stray stdout/stderr is routed
   through the logger instead of a second unbounded file handle.

4. **AssemblyAI can't hang forever** — added per-request timeouts and a 30-min
   poll deadline. A wedged job used to spin the daily worker thread indefinitely.

5. **Resilient hotkey registration** — a bad `hotkey_prefix`/combo used to throw
   inside `register_hotkeys` (which runs *before* the tray), bricking the app
   with no UI. Now each key is wrapped; a bad one is skipped and logged.

6. **VRAM check actually runs** — it imported `torch`, which isn't a dependency
   (faster-whisper uses CTranslate2), so it always `ImportError`d and skipped.
   Rewritten to query `nvidia-smi` (ships with the driver). No nvidia-smi → it
   just doesn't block the load.

7. **Transcribe a past date** — new "Transcribe this date" button in the Daily
   Transcripts tab. Before you could only ever "Transcribe Today", so a failed
   daily run or audio recorded past midnight could never be transcribed from the
   UI. (After it finishes, re-click the date to load the new transcript.)

## Deliberately left for you (minor)
- Hotkey **key→minutes map** still isn't editable in Settings (only the prefix).
- Dashboard "Copy to Clipboard" copies `last_transcript`, which is truncated to
  200 chars (`transcript[:200]`) — the full text does land on the clipboard at
  transcribe time via `pyperclip.copy`. Drop the `[:200]` if you want the button
  to copy everything.

## Smoke-test checklist (Windows)
- [ ] `run.bat` starts, tray icon appears.
- [ ] A hotkey transcribes to clipboard; mash two hotkeys fast → no crash, the
      later one wins, tray returns to green.
- [ ] Settings save → `config.json` intact.
- [ ] Daily recording on → "Transcribe Today" works; then "Transcribe this date"
      on a past day works.
- [ ] Temporarily set a bogus Hotkey Prefix → app still launches to tray (check
      `pascribe.log` for the skip message).
