# Benjamin Bot — Troubleshooting Guide

## Quick Commands
```bash
# Status
systemctl --user is-active benjamin
systemctl --user status benjamin

# Restart (graceful stop hangs — always force kill)
systemctl --user kill -s KILL benjamin; sleep 1; systemctl --user start benjamin

# Logs (filtered)
journalctl --user -u benjamin --since "10 min ago" --no-pager | grep -v "DAVE stats\|UDP packet\|amplitude\|decrypt fail"

# Check if trigger file exists
ls -la /home/nik/clawd/projects/pascribe/bot/_pending_trigger.txt

# Latest transcript
tail -30 /home/nik/clawd/projects/pascribe/bot/recordings/$(date -u +%Y-%m-%d)/_transcript.txt

# Wake word count
cat /home/nik/clawd/projects/pascribe/bot/recordings/$(date -u +%Y-%m-%d)/_benjamin_trigger_count.txt
```

## Common Issues & Fixes

### 1. "Benjamin" said but no trigger fires
**Symptoms:** Someone says "Benjamin" in VC, 5+ min pass, no response in #voice-reports

**Root cause hierarchy (check in order):**
1. **DAVE audio decryption** — The speaker's DAVE key may not be working. Check amplitude:
   ```
   journalctl --user -u benjamin --since "5 min ago" | grep "amplitude.*peak"
   ```
   If peak < 100 for the speaker → their key isn't working. Nothing we can do except wait for key rotation or restart.

2. **Segment threshold not met** — Auto-transcribe needs ≥3 speech segments. Check:
   ```
   journalctl --user -u benjamin --since "5 min ago" | grep "Auto-transcribing"
   ```
   If no "Auto-transcribing" line → not enough speech captured. Lower threshold or wait.

3. **Transcription returned empty** — AssemblyAI got silence audio. Check:
   ```
   journalctl --user -u benjamin --since "5 min ago" | grep "Report result"
   ```
   If `empty (report=None)` → audio was silence.

4. **"benjamin" not in transcribed text** — Check:
   ```
   journalctl --user -u benjamin --since "5 min ago" | grep "Wake word:"
   ```
   If `batch=False full=False` → the word wasn't transcribed. May have been in a different user's audio stream.

5. **Cooldown active** — Check:
   ```
   journalctl --user -u benjamin --since "5 min ago" | grep "cooldown"
   ```
   5-min cooldown between triggers.

6. **Cron not picking up trigger file** — Check cron runs:
   ```
   # Via OpenClaw cron tool, check job c4f57db4-a89a-4261-8d99-9f23aa5b5d26
   ```
   Known issue: `wakeMode: "next-heartbeat"` doesn't fire exactly every 60s.

**Quick manual fix:** Read trigger file, respond in #voice-reports, delete file.

### 2. Bot crashes / gets stuck in stop-sigterm
**Symptoms:** `systemctl --user status benjamin` shows `stop-sigterm` or memory > 900MB

**Fix:** Always use `kill -s KILL`:
```bash
systemctl --user kill -s KILL benjamin; sleep 1; systemctl --user start benjamin
```

**Root cause:** asyncio event loop doesn't clean up voice connections gracefully. Memory leak from raw audio buffers (`_raw_buffers` dict grows indefinitely).

### 3. DAVE decrypt failures after restart
**Symptoms:** Flood of "DAVE decrypt fail" warnings, no speech segments captured

**This is normal.** After every restart/reconnect, DAVE MLS keys need 1-2 min to stabilize. The bot will start capturing audio once keys are exchanged.

**If it persists (>5 min):** Check if the bot is in the right channel:
```
journalctl --user -u benjamin --since "5 min ago" | grep "Joining"
```

### 4. Transcript is speaker-mixed / wrong attribution
**Root cause:** DAVE only decrypts 1-3 users' audio per session. All other users' audio comes through as silence or gets attributed to working SSRCs.

**No fix available.** This is a fundamental `davey` library limitation. The bot "hears" through whichever users' DAVE keys work.

### 5. Pascribe server 404 errors
**Symptoms:** `ERROR transcription.pipeline — Pascribe returned 404` or `TimeoutError`

**This is expected.** The Pascribe server endpoint `/api/transcription` doesn't exist. These errors are harmless — caught and logged, don't affect transcription pipeline.

**To silence:** Remove `send_to_pascribe` calls from `transcription/pipeline.py` or fix the Pascribe server endpoint.

### 6. Voice WebSocket closes (code 1006)
**Symptoms:** `ConnectionClosed: Shard ID None WebSocket closed with 1006`

**This is normal.** Discord periodically drops voice connections. The bot auto-reconnects. DAVE keys reset on reconnect (see issue #3).

### 7. Bot not joining VC
**Symptoms:** Bot shows online but isn't in voice channel

**Check:** Is anyone in a non-blacklisted VC?
- Blacklisted: Game channel (`1236733405698330635`)
- Bot joins the most populated non-blacklisted VC on startup

**Fix:** Restart the bot while people are in VC.

## Architecture Overview
```
Voice Chat → Discord DAVE E2EE → Bot captures UDP packets
  → Transport decrypt (XChaCha20-Poly1305)
  → DAVE per-user decrypt (davey MLS)
  → Opus decode → PCM → VAD filter → Speech segments
  → Every 3 min: AssemblyAI batch transcribe
  → Append to _transcript.txt
  → Wake word check ("benjamin")
    → If found: post snippet to #voice-reports + write _pending_trigger.txt
    → Cron agent reads trigger file + full transcript → posts response
```

## Key Files
- `main.py` — Bot entry, event loop, auto-transcribe, wake word
- `audio/capture.py` — UDP listener, DAVE decrypt, Opus decode, VAD
- `audio/vad.py` — Voice Activity Detection (SpeechSegmenter)
- `audio/storage.py` — Speech segment file management
- `transcription/pipeline.py` — AssemblyAI batch processing
- `transcription/assemblyai.py` — API client
- `recordings/YYYY-MM-DD/` — Daily audio + transcripts
- `_pending_trigger.txt` — Wake word trigger for cron agent
- `_benjamin_trigger_count.txt` — Per-day mention count (prevents refire)

## Lessons Learned
1. **Never use `systemctl --user restart`** — hangs. Always `kill -s KILL` then `start`.
2. **Cron `systemEvent` in main session is unreliable** — use `isolated agentTurn` for trigger response.
3. **Cron `wakeMode: "next-heartbeat"` ≠ exact interval** — fires on heartbeat, not wall clock.
4. **Check `convo` not just the file** — wake word must be in the current batch's text, OR detected via count comparison on full file.
5. **Module-level cooldown > file-based cooldown** — parsing timestamps from files is fragile.
6. **`_build_trigger_snippet` must be a top-level function** — defining inside `if` block caused import issues on restart.
7. **Transcript blob ≠ useful response** — always extract focused snippet (trigger sentence + 1 speaker above/below).
8. **Don't ping everyone in VC** — only ping people relevant to the response.
9. **DAVE keys are per-session** — each restart randomizes which users' audio works.
10. **`send_to_pascribe` timeout (60s) blocks the pipeline** — it runs before transcript append, delaying wake word check. Consider making it fire-and-forget.
