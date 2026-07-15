"""Realtime duplex voice mode for Benjamin (Azure OpenAI Realtime API).

While active, Benjamin holds a live full-duplex conversation in the VC through
gpt-realtime-mini: project updates are spoken by the model, the user replies /
interrupts naturally, and a compiled instruction is submitted back via tool call.

Plumbing:
  - audio in: UserAudioSink._on_packet tees decoded 48k stereo PCM per user into
    feed_pcm() (thread callback) -> batched -> WS input_audio_buffer.append.
    AssemblyAI / GCP / wake-word consumers are untouched and keep running.
  - audio out: response.audio.delta (24k mono) -> 48k stereo -> vc.play source.
  - while active, main.py suppresses the wake-word cascade and routes Ben's
    report-channel responses here (spoken by the same realtime voice).

Env (bot .env): AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, AZURE_REALTIME_DEPLOYMENT.
"""
from __future__ import annotations

import asyncio
import audioop
import base64
import collections
import contextlib
import json
import logging
import math
import os
import random
import struct
import threading
import time

import aiohttp
import discord
import websockets


def _tone(freqs: list[int], ms: int = 150, vol: float = 0.10) -> bytes:
    """Generate a soft 48k-stereo s16 chime that glides through `freqs`, with a
    20ms raised-cosine fade in/out so there are no clicks. Quiet by design
    (vol ~0.10) — a gentle 'nice to the ears' notification, not an alert."""
    sr = 48000
    n = int(sr * ms / 1000)
    fade = int(sr * 0.02)
    mono = bytearray()
    for i in range(n):
        f = freqs[min(len(freqs) - 1, i * len(freqs) // n)]
        env = min(1.0, i / fade, (n - i) / fade)
        s = int(vol * env * 32767 * math.sin(2 * math.pi * f * i / sr))
        mono += struct.pack("<h", s)
    return audioop.tostereo(bytes(mono), 2, 1, 1)


# Rising two-note = waking/on; falling = going quiet/off. Precomputed once.
CHIME_ON = _tone([587, 880], 150)   # D5 → A5, gentle rise
CHIME_OFF = _tone([880, 587], 150)  # A5 → D5, gentle fall

log = logging.getLogger("duplex")

API_VERSION = "2025-04-01-preview"
VOICE = os.getenv("REALTIME_VOICE", "marin")
# Voices supported by gpt-realtime(-mini): alloy, ash, ballad, coral, echo,
# sage, shimmer, verse, marin, cedar. marin/cedar are the newest & most natural.
VOICES = {"alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer",
          "verse", "marin", "cedar"}
# Order used to auto-assign a DISTINCT voice per project (newest/most natural
# first). Each project gets its own voice so switching focus is audibly obvious.
VOICE_ROTATION = ["marin", "cedar", "sage", "ballad", "verse", "coral",
                  "alloy", "ash", "echo", "shimmer"]
# Deployment tiers — the Azure deployment NAMES for each. Override via env if
# your deployments are named differently. "full" won't work until you create a
# gpt-realtime deployment in Azure (only mini exists by default).
DEPLOY = {
    "mini": os.getenv("DUPLEX_DEPLOY_MINI",
                      os.getenv("AZURE_REALTIME_DEPLOYMENT", "gpt-realtime-mini")),
    "full": os.getenv("DUPLEX_DEPLOY_FULL", "gpt-realtime"),
}
# Audio-token pricing per 1M (in, out): mini $10/$20, full $32/$64.
RATES = {"mini": (10.0, 20.0), "full": (32.0, 64.0)}
IDLE_CLOSE_S = int(os.getenv("DUPLEX_IDLE_CLOSE_S", "90"))
# The owner(s) — used only when restricting mic access to owner-only. Audio is
# open to ALL VC members by default now (see DuplexSession.allowed_users).
OWNER_IDS = {int(x) for x in os.getenv("DUPLEX_OWNER_IDS", "300756892571926530")
             .replace(",", " ").split()}
# In standby (after "over") we wait for the project session to reply, which can
# take minutes — don't idle-close nearly as aggressively.
STANDBY_CLOSE_S = int(os.getenv("DUPLEX_STANDBY_CLOSE_S", "1800"))
SIDE_MODEL = os.getenv("DUPLEX_SIDE_MODEL", "google/gemini-2.5-flash")
DISCORD_FRAME = 3840  # 20ms @ 48k stereo s16
SEND_INTERVAL = 0.06  # batch mic audio every 60ms

INSTRUCTIONS = """You are Benjamin, the voice of the user's coding-agent team.
You brief the user on project sessions out loud and take instructions back.
Rules:
- Match reply length to the question. For chit-chat or a quick yes/no, ONE sentence.
  But when the user asks what's going on, for a status update, or "tell me about the
  project", GIVE A REAL ANSWER — two to four sentences that actually convey the recent
  work and current state from the context below. Be informative, not terse. Don't make
  the user drag it out of you.
- Never re-list the same options or repeat yourself across turns. Say a thing once.
- THE STATUS RECAP IS GIVEN AT MOST ONCE per conversation. Rewording it does not make
  it new — a paraphrase of the update IS a repeat. If the user asks another
  status-shaped question after you've briefed them ("what's happening", "what's the
  update"), give ONLY new information; if you have none, say plainly "that's all I
  know — want me to ask the session?" and stop.
- BANNED OPENERS/CLOSERS — never say these or variants: "Here's the latest", "Here's
  the update", "Sure, here's…", "That's the latest", "That's the latest detail/response",
  "Would you like to dive deeper", "Let me know if you'd like…", "Is there anything
  else". Open with the actual content. End when the content ends — no offer, no filler.
- Be direct. Cut hedges and throat-clearing ("I think", "it seems", "basically",
  "actually", "just", "so,"). Lead with the noun/verb that matters.
- A complaint or bug report ("it bugged out", "it's broken", "you're repeating") is
  NOT a status question. Respond to the specific complaint. If the user says you're
  repeating or asks you to stop: apologize in a few words, do NOT summarize anything,
  and ask one short question about what they want instead.
- NEVER read aloud code, file paths, URLs, or identifiers; describe them in plain words.
- SENDING AN INSTRUCTION: whenever the user finishes giving you a task — they almost
  always end with the word "over" — you MUST call submit_prompt. Compile their ACTUAL
  request into a clear, complete instruction (use their real intent and specifics, never
  a vague placeholder). Also fill the `context` argument with the few points from THIS
  voice conversation the coding agent needs beyond the instruction itself (decisions
  made, constraints, preferences mentioned) — leave it empty if there are none; never
  dump chit-chat. Then confirm in ONE short sentence and say you're standing by.
  "over" means send, not end.
- STANDBY: after you send an instruction, your microphone goes to standby — you stop
  hearing the user until they say your wake word ("Benjamin") or the project session
  replies. When you come back from standby, acknowledge in a couple of words, don't
  monologue.
- SESSION REPLIES: when a message arrives marked as a session response, that is the NEW
  main topic. Brief the user on its substance in two to four sentences, then discuss it.
- MULTIPLE PROJECTS: you may supervise several projects at once. You talk about ONE at a
  time (the focused one). submit_prompt always goes to the focused project. If another
  project finishes, you'll get a one-line "project X is ready" notice — pass it on, don't
  discuss its content until the user switches. When the user says "switch to X", "go to
  X", or asks about a different project, call switch_project. Call list_pending when they
  ask what's waiting or which projects finished. After switching, only discuss the new
  project — the earlier one's details no longer apply.
- DEPTH: when the user asks something that needs real analysis you can't answer well
  from the context you have (technical why/how, tradeoffs, judging an idea), call
  deep_answer with their question instead of guessing — then relay its answer naturally.
- Only call end_conversation if the user clearly says they're done, goodbye, or to stop.
  Never end just because they said "over" — that submits an instruction and continues.
"""

TOOLS = [
    {
        "type": "function",
        "name": "submit_prompt",
        "description": "Submit the compiled instruction back to the project session. "
                       "Call when the user finishes an instruction (e.g. says 'over').",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Complete instruction for the coding agent"},
                "context": {"type": "string",
                            "description": "Relevant points from the voice conversation the agent "
                                           "needs beyond the instruction (decisions, constraints, "
                                           "preferences). Empty string if none. No chit-chat."},
            },
            "required": ["prompt"],
        },
    },
    {
        "type": "function",
        "name": "deep_answer",
        "description": "Ask a fast expert side-model for an in-depth answer when the user's "
                       "question needs analysis beyond your context. Returns text to relay aloud.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The user's question, self-contained"},
            },
            "required": ["question"],
        },
    },
    {
        "type": "function",
        "name": "switch_project",
        "description": "Change which project is focused (spoken instructions and briefings "
                       "apply to it). Call when the user asks to switch to / go to / talk "
                       "about a different project.",
        "parameters": {
            "type": "object",
            "properties": {
                "project": {"type": "string", "description": "Name of the project to focus, "
                                                             "as the user said it (fuzzy-matched)"},
            },
            "required": ["project"],
        },
    },
    {
        "type": "function",
        "name": "list_pending",
        "description": "List projects supervised right now and which have finished replies "
                       "waiting to be heard. Call when the user asks what's pending / waiting / "
                       "which projects are running.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "end_conversation",
        "description": "End the duplex voice conversation.",
        "parameters": {"type": "object", "properties": {}},
    },
]


class _Playback(discord.AudioSource):
    """Pull-based playback fed by WS deltas; emits silence when empty."""

    def __init__(self) -> None:
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._read_calls = 0
        self._real_frames = 0

    def feed(self, pcm: bytes) -> None:
        with self._lock:
            self._buf.extend(pcm)

    def clear(self) -> int:
        with self._lock:
            n = len(self._buf)
            self._buf.clear()
            return n

    def read(self) -> bytes:
        with self._lock:
            self._read_calls += 1
            if self._read_calls == 1:
                log.info("playback: discord AudioPlayer started pulling frames")
            if len(self._buf) >= DISCORD_FRAME:
                frame = bytes(self._buf[:DISCORD_FRAME])
                del self._buf[:DISCORD_FRAME]
                self._real_frames += 1
                if self._real_frames == 1 or self._real_frames % 50 == 0:
                    log.info("playback: served %d real frames (%.1fs audio)",
                             self._real_frames, self._real_frames * 0.02)
                return frame
        return b"\x00" * DISCORD_FRAME

    def is_opus(self) -> bool:
        return False


class DuplexSession:
    def __init__(self, bot, vc: discord.VoiceClient, text_channel: discord.abc.Messageable,
                 project: str | None = None, context: str | None = None,
                 channel_id: int | None = None, voice: str | None = None,
                 tier: str | None = None):
        self.bot = bot
        self.vc = vc
        self.text_channel = text_channel
        self.project = project
        self.context = context
        # Explicit voice wins; otherwise pick a RANDOM voice so different
        # sessions / users don't always get the same one.
        self.voice = voice if voice in VOICES else random.choice(VOICE_ROTATION)
        # Deployment tier: "mini" (default) or "full" (gpt-realtime). Drives the
        # Azure deployment used and the cost rates.
        self.tier = tier if tier in DEPLOY else "mini"
        self.deployment = DEPLOY[self.tier]
        self.rate_in, self.rate_out = RATES[self.tier]
        # Standby: mic muted to Azure after submit_prompt until the wake word
        # ("Benjamin", via the local Vosk detector) or a session response wakes
        # us. Output/playback stays attached the whole time.
        self.standby = False
        self._standby_arm = False
        self.transcript: list[tuple[str, str]] = []
        self.deep_context: str | None = None  # larger blob for deep_answer
        self.started_at = time.time()
        self.wakes = 0
        self.dispatches = 0
        self.deep_answers = 0
        # The real project channel (in the bridge's guild). We usually can't
        # post to it from here, so submit_prompt routes to the bridge over
        # localhost using this id instead of the resolved text_channel.
        self.project_channel_id = channel_id
        # Parallel projects: one voice session multiplexed over N coding-agent
        # channels. `projects` maps channel_id -> slot; `active_cid` is the
        # focused one; `pending` holds finished-but-unheard replies from
        # non-focused projects. self.project/context/deep_context/
        # project_channel_id always MIRROR the active slot (the rest of the code
        # reads those, so multi-project stays backward-compatible).
        self.projects: dict[int, dict] = {}
        self.active_cid: int | None = channel_id
        self.pending: collections.deque = collections.deque(maxlen=20)
        # Whose audio reaches Azure. Empty set = ALL VC members (the default now,
        # so any project collaborator can talk to Benjamin). Set DUPLEX_ALLOWED_IDS
        # (or `!duplex owner`) to restrict — worth doing only if heavy cross-talk
        # is wrecking turn-detection, since simultaneous speakers mix into one
        # mono stream. Turn-taking across several people works fine.
        allow_env = os.getenv("DUPLEX_ALLOWED_IDS", "")
        self.allowed_users = {int(x) for x in allow_env.replace(",", " ").split()}
        self.ws = None
        self.playback = _Playback()
        self.closed = False
        self.last_activity = time.monotonic()
        self.assistant_item: str | None = None
        self.queued_ms = 0
        self.in_tokens = 0
        self.out_tokens = 0
        self._resample_state = None
        self._mic_bufs: dict[int, bytearray] = {}
        self._mic_lock = threading.Lock()
        self._response_active = False
        self._t_speech_stop = None
        self._t_resp_created = None
        self._t_first_delta = None
        self._delta_ms = 0
        self._fed_once = False
        self._sent_ms = 0
        self._last_sent_log = 0
        self._seen_events: set = set()
        self._tasks: list[asyncio.Task] = []
        # Set whenever the WS is up; used to await a voice-switch reconnect (the
        # API forbids changing voice mid-session, so switching a project's voice
        # tears down + rebuilds the socket). _fast_reconnect skips the backoff.
        self._ws_ready = asyncio.Event()
        self._fast_reconnect = False

    def set_audio_access(self, mode: str) -> str:
        """Runtime toggle for whose voice reaches Benjamin. 'everyone' (default)
        opens the mic to all VC members; 'owner' restricts to OWNER_IDS."""
        if mode == "owner":
            self.allowed_users = set(OWNER_IDS)
            return "owner-only"
        self.allowed_users = set()
        return "everyone in the voice channel"

    # ---------- lifecycle ----------

    async def _ws_connect(self) -> None:
        endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].removeprefix("https://").rstrip("/")
        url = (f"wss://{endpoint}/openai/realtime?api-version={API_VERSION}"
               f"&deployment={self.deployment}")
        self.ws = await websockets.connect(
            url, additional_headers={"api-key": os.environ["AZURE_OPENAI_KEY"]},
            max_size=16 * 1024 * 1024,
        )

    def _build_instructions(self) -> str:
        """Assemble the system prompt for the ACTIVE project: base rules + user
        style profile + the focused project's name/context + a roster of the
        other supervised projects (so the model knows what it can switch to)."""
        instructions = INSTRUCTIONS
        # User speaking-style profile (generated by a sub-agent from months of
        # VC transcripts; regenerate by re-running the style analysis).
        profile_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "assets", "user_style_profile.md")
        if os.path.isfile(profile_path):
            with contextlib.suppress(OSError):
                with open(profile_path, encoding="utf-8") as f:
                    instructions += "\n\n" + f.read().strip()[:1600]
        if self.project:
            instructions += f"\nCurrent focused project: {self.project}."
        others = [s["project"] for c, s in self.projects.items()
                  if c != self.active_cid and s.get("project")]
        if others:
            instructions += ("\nOther projects you also supervise (switch with "
                             "switch_project): " + ", ".join(others) + ".")
        if self.context:
            instructions += (
                "\n\nHere is the RECENT session work on this project — ground your "
                "answers in this specific latest state, not just the general idea. "
                "When asked what's going on, lead with the most recent items:\n"
                + self.context)
        return instructions

    async def _update_instructions(self, switch: bool = False) -> None:
        """Re-send just the instructions (used on project switch). When switching,
        drop a hard boundary item so the prior project's conversation history
        doesn't bleed into the new project's discussion."""
        await self._send({"type": "session.update",
                          "session": {"instructions": self._build_instructions()}})
        if switch:
            await self._send({
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user",
                         "content": [{"type": "input_text",
                                      "text": f"=== Now focused on project {self.project}. "
                                              "Disregard details of any previously discussed "
                                              "project; only discuss this one from here on. ==="}]},
            })

    async def _configure(self) -> None:
        """(Re)send full session.update — instructions, tools, turn detection."""
        instructions = self._build_instructions()
        await self._send({
            "type": "session.update",
            "session": {
                "modalities": ["audio", "text"],
                "voice": self.voice,
                "instructions": instructions,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                # Transcribe both sides so conversations are auditable
                # (logged to logs/duplex_transcript.log). Pin to English —
                # without a language hint whisper auto-detects on tiny
                # fragments and emits random Latvian/Russian/Polish words.
                "input_audio_transcription": {"model": "whisper-1", "language": "en"},
                "input_audio_noise_reduction": {"type": "far_field"},
                "turn_detection": {
                    "type": "server_vad",
                    # Higher threshold ignores low-level room noise / speaker
                    "threshold": 0.7,      # bleed (echo) so turn boundaries stay clean.
                    "prefix_padding_ms": 250,
                    # 250ms chopped the user mid-sentence into fragments ("Can
                    # you?" / "Can you stop?"), each committing as its own turn
                    # and re-triggering answers — worse than +100ms latency.
                    "silence_duration_ms": 350,
                },
                "tools": TOOLS,
                "tool_choice": "auto",
                # Was 300 — but that truncated long/detailed answers mid-sentence
                # (pricing/status explanations run 250-400 tokens). Length is now
                # controlled by the instructions (be direct, match the question),
                # so this is just a runaway backstop, not the length limiter.
                "max_response_output_tokens": int(os.getenv("DUPLEX_MAX_RESP_TOKENS", "1200")),
            },
        })

    async def connect(self) -> None:
        await self._ws_connect()
        await self._configure()
        self._ws_ready.set()
        self._tasks = [
            asyncio.create_task(self._recv_loop(), name="duplex-recv"),
            asyncio.create_task(self._mic_sender(), name="duplex-mic"),
            asyncio.create_task(self._idle_watch(), name="duplex-idle"),
        ]
        if self.vc.is_playing():
            self.vc.stop()
            await asyncio.sleep(0.1)
        self.vc.play(self.playback)
        log.info("duplex session connected (tier=%s deployment=%s voice=%s)",
                 self.tier, self.deployment, self.voice)

    async def close(self, reason: str = "") -> None:
        if self.closed:
            return
        self.closed = True
        log.info("duplex closing: %s", reason)
        _log_usage(self, "session")  # session-level summary line for the tracker
        # Forget the session ONLY on user intent, so it isn't resumed on boot.
        # For idle/ws closes, keep the state file: a fresh one lets the wake
        # word resurrect the session (wake_from_state) and lets resume_if_active
        # pick it up after a restart. (A process kill never runs close() at all.)
        if reason.startswith(("requested", "end_conversation", "replaced", "stop")):
            _clear_state()
        for t in self._tasks:
            t.cancel()
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
        try:
            # Stop playback but DO NOT disconnect — Benjamin lives in this VC.
            if self.vc.is_playing():
                self.vc.stop()
        except Exception:
            pass
        cost = self.in_tokens / 1e6 * self.rate_in + self.out_tokens / 1e6 * self.rate_out
        try:
            await self.text_channel.send(
                f"🔇 duplex ended{': ' + reason if reason else ''} "
                f"(`{self.tier}` ~${cost:.3f}, {self.in_tokens}/{self.out_tokens} audio tok in/out)")
        except Exception:
            pass
        global _active
        if _active is self:
            _active = None

    # ---------- mic in (called from capture thread) ----------

    def feed_pcm(self, user_id: int, pcm_stereo_48k: bytes) -> None:
        if self.closed or self.standby:
            return  # standby: nothing reaches Azure until wake word / session reply
        if self.allowed_users and user_id not in self.allowed_users:
            return  # ignore everyone but the owner (clean turn-detection)
        if not self._fed_once:
            self._fed_once = True
            log.info("mic: first feed_pcm from user=%s (%d bytes)",
                     user_id, len(pcm_stereo_48k))
        # Per-user buffers: concatenating simultaneous speakers' 20ms frames
        # into one stream interleaves their timelines and turns speech into
        # garbage at the model end — streams must be mixed sample-wise.
        with self._mic_lock:
            self._mic_bufs.setdefault(user_id, bytearray()).extend(pcm_stereo_48k)

    # 60ms of 24kHz mono s16 silence — sent during gaps so Azure's server VAD
    # sees a continuous stream and can actually clock the 500ms of silence that
    # ends a turn. Discord stops delivering packets when the user is quiet, so
    # without this the VAD stalls and turns commit late (or never).
    _SILENCE_24K = b"\x00" * (int(24000 * SEND_INTERVAL) * 2)

    async def _mic_sender(self) -> None:
        while not self.closed:
            await asyncio.sleep(SEND_INTERVAL)
            # Only start streaming once the user has spoken at least once — no
            # point paying for silence tokens on an idle open mic.
            if not self._fed_once:
                continue
            if self.standby:
                # Mic is off: send nothing at all (also saves silence tokens).
                with self._mic_lock:
                    self._mic_bufs.clear()
                continue
            with self._mic_lock:
                streams = [bytes(b) for b in self._mic_bufs.values() if b]
                self._mic_bufs.clear()
            if streams:
                chunk = streams[0]
                for s in streams[1:]:
                    if len(s) < len(chunk):
                        s = s.ljust(len(chunk), b"\x00")
                    elif len(s) > len(chunk):
                        chunk = chunk.ljust(len(s), b"\x00")
                    chunk = audioop.add(chunk, s, 2)
                mono = audioop.tomono(chunk, 2, 0.5, 0.5)
                pcm24, self._resample_state = audioop.ratecv(mono, 2, 1, 48000, 24000, self._resample_state)
            else:
                pcm24 = self._SILENCE_24K
            self._sent_ms += len(pcm24) // 48
            if self._sent_ms // 1000 != self._last_sent_log:
                self._last_sent_log = self._sent_ms // 1000
                log.info("mic: sent %ds of audio to Azure", self._last_sent_log)
            await self._send({"type": "input_audio_buffer.append",
                              "audio": base64.b64encode(pcm24).decode()})

    # ---------- ws events ----------

    async def _recv_loop(self) -> None:
        while not self.closed:
            try:
                async for raw in self.ws:
                    await self._handle(json.loads(raw))
            except asyncio.CancelledError:
                raise
            except websockets.ConnectionClosed:
                pass
            except Exception:
                log.exception("duplex recv error")
            if self.closed:
                return
            if not await self._reconnect():
                await self.close("ws closed")
                return

    async def _reconnect(self) -> bool:
        """Bounded auto-reconnect after a WS drop — rebuild the socket, re-send
        session.update (with the CURRENT self.voice, so a voice switch takes
        effect here), keep standby + vc playback intact. _fast_reconnect skips
        the backoff for an intentional voice-switch reconnect."""
        for attempt in range(1, 4):
            if self.closed:
                return False
            if not (attempt == 1 and self._fast_reconnect):
                await asyncio.sleep(min(2 * attempt, 6))
            self._fast_reconnect = False
            try:
                await self._ws_connect()
                await self._configure()
            except Exception as e:  # noqa: BLE001
                log.warning("duplex reconnect attempt %d failed: %s", attempt, e)
                continue
            self.assistant_item = None
            self.queued_ms = 0
            self._response_active = False
            self._ws_ready.set()
            log.info("duplex reconnected (attempt %d, voice=%s)", attempt, self.voice)
            return True
        return False

    async def _handle(self, ev: dict) -> None:
        t = ev.get("type")
        if t and t not in self._seen_events:
            self._seen_events.add(t)
            log.info("azure event (first): %s", t)
        if t == "response.audio.delta":
            self.last_activity = time.monotonic()
            pcm24 = base64.b64decode(ev["delta"])
            if self._t_first_delta is None:
                self._t_first_delta = time.monotonic()
                if self._t_speech_stop:
                    log.info("latency: speech_stop→first_audio %.0fms "
                             "(→resp.created %.0fms)",
                             (self._t_first_delta - self._t_speech_stop) * 1000,
                             ((self._t_resp_created or self._t_first_delta)
                              - self._t_speech_stop) * 1000)
            self._delta_ms += len(pcm24) // 48
            self.queued_ms += len(pcm24) // 48
            pcm48, _ = audioop.ratecv(pcm24, 2, 1, 24000, 48000, None)
            self.playback.feed(audioop.tostereo(pcm48, 2, 1, 1))
            # Re-attach if something else (chime, legacy TTS interrupt)
            # stopped the voice client — otherwise replies play to nobody.
            if not self.vc.is_playing():
                try:
                    self.vc.play(self.playback)
                    log.info("playback re-attached (player was stopped)")
                except discord.ClientException:
                    pass
        elif t == "conversation.item.input_audio_transcription.completed":
            self._tlog("user", (ev.get("transcript") or "").strip())
        elif t == "response.audio_transcript.done":
            self._tlog("ben", (ev.get("transcript") or "").strip())
        elif t == "input_audio_buffer.speech_stopped":
            self._t_speech_stop = time.monotonic()
            self._t_resp_created = None
            self._t_first_delta = None
            self._delta_ms = 0
        elif t == "response.created":
            self._response_active = True
            self._t_resp_created = time.monotonic()
        elif t == "response.output_item.added":
            item = ev.get("item", {})
            if item.get("type") == "message":
                self.assistant_item = item.get("id")
                self.queued_ms = 0
        elif t == "input_audio_buffer.speech_started":
            self.last_activity = time.monotonic()
            await self._barge_in()
        elif t == "response.function_call_arguments.done":
            await self._on_tool(ev.get("name"), ev.get("arguments"), ev.get("call_id"))
        elif t == "response.done":
            self._response_active = False
            # Surface a truncated reply: status "incomplete" with reason
            # "max_output_tokens" = the answer was cut mid-sentence by the cap.
            resp = ev.get("response") or {}
            if resp.get("status") == "incomplete":
                reason = ((resp.get("status_details") or {}).get("reason")
                          or (resp.get("status_details") or {}).get("type") or "?")
                log.warning("response incomplete (cut off): reason=%s", reason)
            if self._t_first_delta and self._delta_ms:
                wall = (time.monotonic() - self._t_first_delta) * 1000
                log.info("latency: %dms audio streamed in %.0fms wall%s",
                         self._delta_ms, wall,
                         "  ⚠️ SLOWER THAN REALTIME (throttled?)"
                         if wall > self._delta_ms * 1.15 else "")
            usage = (ev.get("response") or {}).get("usage") or {}
            din = (usage.get("input_token_details") or {}).get("audio_tokens", 0)
            dout = (usage.get("output_token_details") or {}).get("audio_tokens", 0)
            self.in_tokens += din
            self.out_tokens += dout
            # Persist per-turn usage immediately — crash-safe (survives a process
            # kill that never runs close()), and gives a time-series to monitor.
            if din or dout:
                _log_usage(self, "turn", din, dout)
            self.last_activity = time.monotonic()
            # Enter standby once the spoken submit-confirmation finishes (the
            # tool-call-only response has no message item, so we wait for the
            # response that actually says "sent, standing by").
            if self._standby_arm:
                out = (ev.get("response") or {}).get("output") or []
                if any(i.get("type") == "message" for i in out):
                    self._enter_standby("after submit")
        elif t == "error":
            err = ev.get("error") or {}
            # Benign barge-in race: we cancel just as the response finishes.
            if err.get("code") == "response_cancel_not_active":
                log.debug("realtime: %s", err.get("code"))
            else:
                log.error("realtime error: %s", err)

    _TRANSCRIPT_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "logs", "duplex_transcript.log")

    def _tlog(self, who: str, text: str) -> None:
        if not text:
            return
        self.transcript.append((who, text))
        log.info("transcript %s: %s", who, text)
        with contextlib.suppress(OSError):
            with open(self._TRANSCRIPT_LOG, "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%F %T')} [{self.project or '-'}] {who}: {text}\n")

    def _chime(self, kind: str) -> None:
        """Feed a soft on/off notification tone into the playback stream."""
        with contextlib.suppress(Exception):
            self.playback.feed(CHIME_ON if kind == "on" else CHIME_OFF)
            if not self.vc.is_playing():
                with contextlib.suppress(discord.ClientException):
                    self.vc.play(self.playback)

    def _enter_standby(self, reason: str) -> None:
        """Go quiet-but-wakeable: mute the mic to Azure, keep the session (and
        Vosk wake word) alive. Reached both after 'over' and after idle."""
        if self.standby:
            return
        self.standby = True
        self._standby_arm = False
        self.last_activity = time.monotonic()
        with self._mic_lock:
            self._mic_bufs.clear()
        self._chime("off")
        _save_state(self)  # keep resumable / wakeable
        log.info("standby: %s — mic off, wakeable via wake word / session reply", reason)

    async def _barge_in(self) -> None:
        dropped = self.playback.clear()
        if self._response_active:
            await self._send({"type": "response.cancel"})
        if self.assistant_item and dropped:
            played = max(0, self.queued_ms - dropped // 192)
            await self._send({
                "type": "conversation.item.truncate",
                "item_id": self.assistant_item,
                "content_index": 0,
                "audio_end_ms": played,
            })

    async def _on_tool(self, name: str, arguments: str, call_id: str) -> None:
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        log.info("duplex tool: %s %s", name, args)
        if name == "submit_prompt":
            prompt = (args.get("prompt") or "").strip()
            context = (args.get("context") or "").strip()
            routed = await self._route_prompt(prompt, context)
            if routed:
                self._standby_arm = True  # standby after the spoken confirmation
                self.dispatches += 1
            await self._tool_result(call_id, {
                "status": "submitted" if routed else "failed",
                "detail": ("instruction routed to the project session; you are now "
                           "going to standby" if routed
                           else "could not reach the project session")})
        elif name == "deep_answer":
            question = (args.get("question") or "").strip()
            self.deep_answers += 1
            # Speak one brief filler immediately, then run the (slow) side model
            # in a BACKGROUND TASK. Critical: _on_tool runs inside _recv_loop, so
            # awaiting the ~10-40s side-model call here would freeze the receive
            # loop — the session stops reading Azure events, drops the WS, and
            # ignores the user ("answered then went dead"). The task returns
            # control to the recv loop instantly.
            await self._send({
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user",
                         "content": [{"type": "input_text",
                                      "text": "[you're looking that up for the user — "
                                              "say ONE short, natural filler in your own "
                                              "words (that you're thinking it over), then "
                                              "stop; the answer follows shortly]"}]},
            })
            await self._send({"type": "response.create"})
            asyncio.create_task(self._deliver_deep_answer(question, call_id))
        elif name == "switch_project":
            target = (args.get("project") or "").strip()
            cid = self._match_project(target)
            if cid and cid != self.active_cid:
                # Resolve the tool call WITHOUT a response.create — switch_to may
                # rebuild the WS on a new voice, and that reconnect (or the
                # post-switch brief) does the speaking. Announcing here would talk
                # on the OLD voice and collide with the reconnect.
                await self._send({
                    "type": "conversation.item.create",
                    "item": {"type": "function_call_output", "call_id": call_id,
                             "output": json.dumps({"status": "switched",
                                                   "project": self.projects[cid].get("project")})}})
                await self.switch_to(cid, announce=True)
            elif cid:
                await self._tool_result(call_id, {"status": "already_focused",
                                                  "project": self.projects[cid].get("project")})
            else:
                known = [s.get("project") for s in self.projects.values() if s.get("project")]
                await self._tool_result(call_id, {"status": "not_found", "known": known})
        elif name == "list_pending":
            pend = {}
            for c, nm, _ in self.pending:
                pend[nm] = pend.get(nm, 0) + 1
            await self._tool_result(call_id, {
                "focused": self.project,
                "projects": [s.get("project") for s in self.projects.values() if s.get("project")],
                "waiting": [{"project": k, "replies": v} for k, v in pend.items()]})
        elif name == "end_conversation":
            await self._tool_result(call_id, {"status": "bye"})
            await asyncio.sleep(4)
            await self.close("end_conversation")

    async def _route_prompt(self, prompt: str, context: str = "") -> bool:
        """Deliver a compiled instruction to the project's Claude session.
        Primary path: POST to the bridge over localhost (it's in the project's
        guild; we're not). Fallback: post a 📤 message to whatever channel we
        resolved, in case the bridge intake is down."""
        if not prompt:
            return False
        cid = self.project_channel_id
        if cid:
            url = os.getenv("BRIDGE_VOICE_API", "http://127.0.0.1:3193/voice-prompt")
            try:
                timeout = aiohttp.ClientTimeout(total=8)
                async with aiohttp.ClientSession(timeout=timeout) as s:
                    async with s.post(url, json={"channel_id": cid,
                                                 "prompt": prompt,
                                                 "context": context}) as r:
                        d = await r.json()
                if d.get("ok"):
                    log.info("submit_prompt → bridge session (cid=%s): %r", cid, prompt)
                    return True
                log.warning("bridge rejected voice-prompt (cid=%s): %s", cid, d)
            except Exception as e:  # noqa: BLE001
                log.warning("bridge voice-prompt POST failed: %s", e)
        # Fallback: Discord marker message.
        with contextlib.suppress(Exception):
            log.info("submit_prompt fallback → channel #%s (id=%s): %r",
                     getattr(self.text_channel, "name", "?"),
                     getattr(self.text_channel, "id", "?"), prompt)
            body = f"📤 {prompt}"
            if context:
                body += f"\n📎 {context}"
            await self.text_channel.send(body)
            return True
        return False

    async def _deliver_deep_answer(self, question: str, call_id: str) -> None:
        """Run the side model off the recv loop, then submit the tool result once
        the filler response has finished (overlapping response.create is rejected)."""
        answer = await self._deep_answer(question)
        if self.closed:
            return
        deadline = time.monotonic() + 6
        while self._response_active and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        await self._tool_result(call_id, {
            "answer": answer or "the side model did not answer — say so briefly"})

    async def _deep_answer(self, question: str) -> str:
        """Side-model call for questions needing more depth than the realtime
        model gives. Prefers Azure (the user's own funded subscription — a
        deployed chat model like gpt-5-mini); falls back to OpenRouter. Set
        DUPLEX_SIDE_AZURE_DEPLOYMENT='' to force OpenRouter."""
        if not question:
            return ""
        sys_prompt = (
            "You are the expert brain behind a voice assistant. Answer the user's "
            "question with real insight, but format it to be READ ALOUD: at most "
            "120 words, plain conversational sentences, no code, no file paths, "
            "no URLs, no markdown, no lists.")
        if self.project:
            sys_prompt += f"\nProject: {self.project}."
        state = self.deep_context or self.context
        if state:
            sys_prompt += "\nRecent project state:\n" + state[:10000]
        if self.transcript:
            convo = "\n".join(f"{w}: {t}" for w, t in self.transcript[-16:])
            sys_prompt += "\nThe voice conversation so far (answer in its context):\n" + convo[:3000]
        messages = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": question}]
        azure_dep = os.getenv("DUPLEX_SIDE_AZURE_DEPLOYMENT", "gpt-5-mini")
        for backend in ("azure", "openrouter"):
            if backend == "azure":
                if not (azure_dep and os.getenv("AZURE_OPENAI_KEY")):
                    continue
                ep = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
                url = (f"{ep}/openai/deployments/{azure_dep}/chat/completions"
                       f"?api-version=2025-04-01-preview")
                headers = {"api-key": os.environ["AZURE_OPENAI_KEY"]}
                # gpt-5 reasoning models use max_completion_tokens (not max_tokens)
                # and need headroom for hidden reasoning before the visible answer.
                # Accuracy-first: reasoning_effort=high (think hard) + verbosity=low
                # (keep the spoken answer short). The filler covers the extra time.
                # medium (~10s) not high (~25-40s): high is too slow for a live
                # voice turn even off the recv loop. DUPLEX_SIDE_REASONING=high to
                # override when you want max accuracy and can wait.
                effort = os.getenv("DUPLEX_SIDE_REASONING", "medium")
                payload = {"messages": messages, "max_completion_tokens": 4000,
                           "reasoning_effort": effort, "verbosity": "low"}
                label = f"azure:{azure_dep}/{effort}"
            else:
                key = os.getenv("OPENROUTER_API_KEY", "")
                if not key:
                    continue
                url = os.getenv("DUPLEX_SIDE_URL",
                                "https://openrouter.ai/api/v1/chat/completions")
                headers = {"Authorization": f"Bearer {key}"}
                payload = {"model": SIDE_MODEL, "max_tokens": 400, "messages": messages}
                label = f"openrouter:{SIDE_MODEL}"
            try:
                # High reasoning takes longer (~25s) — give it room (the spoken
                # filler keeps the user from hearing dead air meanwhile).
                timeout = aiohttp.ClientTimeout(total=35)
                async with aiohttp.ClientSession(timeout=timeout) as s:
                    async with s.post(url, headers=headers, json=payload) as r:
                        d = await r.json()
                answer = ((d.get("choices") or [{}])[0].get("message") or {}).get("content", "")
                if answer and answer.strip():
                    log.info("deep_answer via %s: %d chars for %r", label, len(answer), question[:80])
                    return answer.strip()
                log.warning("deep_answer %s empty/err: %s", label, str(d.get("error") or d)[:200])
            except Exception as e:  # noqa: BLE001
                log.warning("deep_answer %s failed: %s", label, e)
        return ""

    async def wake(self, reason: str = "wake word") -> None:
        """Leave standby (wake word heard or a session reply arrived)."""
        if self.closed or not self.standby:
            return
        self.standby = False
        self._standby_arm = False
        self.last_activity = time.monotonic()
        with self._mic_lock:
            self._mic_bufs.clear()  # drop anything buffered while muted
        self._chime("on")
        self.wakes += 1
        log.info("standby: woke (%s)", reason)
        if reason == "wake word":
            await self._send({
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user",
                         "content": [{"type": "input_text",
                                      "text": "[the user said your wake word — you're "
                                              "listening again; acknowledge in a couple "
                                              "of words, nothing more]"}]},
            })
            await self._send({"type": "response.create"})

    async def _tool_result(self, call_id: str, result: dict) -> None:
        await self._send({
            "type": "conversation.item.create",
            "item": {"type": "function_call_output", "call_id": call_id,
                     "output": json.dumps(result)},
        })
        await self._send({"type": "response.create"})

    # ---------- multi-project focus / routing ----------

    def add_project(self, cid: int | None, project: str | None,
                    context: str | None, deep_context: str | None,
                    text_channel=None, voice: str | None = None) -> None:
        """Register (or refresh) a supervised project slot. Each project keeps its
        OWN voice — given explicitly, or auto-assigned to a distinct one from the
        rotation so switching projects is audibly obvious. Re-adding keeps the
        existing voice."""
        if not cid:
            return
        existing = self.projects.get(cid, {})
        if voice not in VOICES:
            voice = existing.get("voice")
        if not voice:
            # Pick a RANDOM voice among those not already in use by another
            # project (or the session) — random tone per project, never a repeat
            # across the parallel fleet. Falls back to reuse only if all taken.
            used = {s.get("voice") for s in self.projects.values() if s is not existing}
            used.add(self.voice)
            pool = [v for v in VOICE_ROTATION if v not in used]
            voice = random.choice(pool) if pool else self.voice
        self.projects[cid] = {"project": project, "context": context,
                              "deep_context": deep_context, "voice": voice,
                              "text_channel": text_channel or self.text_channel}

    def _match_project(self, name: str) -> int | None:
        """Fuzzy spoken-name → channel_id. Exact, then substring either way."""
        name = (name or "").strip().lower()
        if not name:
            return None
        slots = [(c, (s.get("project") or "").lower()) for c, s in self.projects.items()]
        for c, p in slots:
            if p == name:
                return c
        for c, p in slots:
            if p and (name in p or p in name):
                return c
        return None

    async def switch_to(self, cid: int, announce: bool = False,
                        deliver_pending: bool = True) -> bool:
        """Focus a project: mirror its slot into the active fields. If the
        project uses a DIFFERENT voice, the WS is rebuilt (the API forbids a
        live voice change) — which also gives a clean history, so no bleed
        boundary is needed. Same voice → in-place instruction swap + boundary."""
        slot = self.projects.get(cid)
        if not slot:
            return False
        new_voice = slot.get("voice") or self.voice
        voice_changed = new_voice != self.voice
        self.active_cid = cid
        self.project_channel_id = cid
        self.project = slot.get("project")
        self.context = slot.get("context")
        self.deep_context = slot.get("deep_context")
        self.voice = new_voice
        self.text_channel = slot.get("text_channel") or self.text_channel
        _save_state(self)
        if voice_changed:
            # Rebuild the WS with the new voice. Can't await it inline — this may
            # run inside the recv loop, which is the one that performs the
            # reconnect — so hand the brief-after-reconnect to a separate task.
            if self.standby:
                self.standby = False
                self._chime("on")
                self.wakes += 1
            self._ws_ready.clear()
            self._fast_reconnect = True
            asyncio.create_task(self._brief_after_reconnect(cid, announce, deliver_pending))
            with contextlib.suppress(Exception):
                await self.ws.close()  # recv loop → _reconnect() with the new voice
            return True
        # Same voice: keep the live session, swap instructions + a bleed boundary.
        if self.standby:
            await self.wake(f"switch to {self.project}")
        await self._update_instructions(switch=True)
        if deliver_pending:
            txt = self._pop_pending(cid)
            if txt is not None:
                await self._brief(txt, "session response")
                return True
        if announce:
            await self._brief(f"You're now on {self.project}.", "focus switch")
        return True

    async def _brief_after_reconnect(self, cid: int, announce: bool,
                                     deliver_pending: bool) -> None:
        """After a voice-switch WS rebuild completes, deliver the pending reply
        or announce the switch (on the new voice, in the fresh session)."""
        try:
            await asyncio.wait_for(self._ws_ready.wait(), timeout=8)
        except asyncio.TimeoutError:
            log.warning("voice-switch reconnect timed out for project %s", self.project)
            return
        if self.closed:
            return
        if deliver_pending:
            txt = self._pop_pending(cid)
            if txt is not None:
                await self._brief(txt, "session response")
                return
        if announce:
            await self._brief(f"You're now on {self.project}.", "focus switch")

    def _pop_pending(self, cid: int) -> str | None:
        """Remove and return the latest pending reply for a project (if any)."""
        hits = [item for item in self.pending if item[0] == cid]
        if not hits:
            return None
        for item in hits:
            self.pending.remove(item)
        return hits[-1][2]

    async def _brief(self, text: str, source: str) -> None:
        """Speak an update about the FOCUSED project (the original speak_update)."""
        self.last_activity = time.monotonic()
        if source == "session response":
            framing = ("[the project session finished and replied — this is the NEW "
                       "main topic. Brief the user on its substance in two to four "
                       "sentences, then discuss it with them]")
        else:
            framing = f"[{source} — announce briefly to the user]"
        await self._send({
            "type": "conversation.item.create",
            "item": {"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": f"{framing}\n{text}"}]},
        })
        await self._send({"type": "response.create"})

    async def _notify_pending(self, name: str) -> None:
        """One-liner that a non-focused project finished — no content, no bleed."""
        if self.standby:
            await self.wake(f"pending from {name}")
        self.last_activity = time.monotonic()
        await self._send({
            "type": "conversation.item.create",
            "item": {"type": "message", "role": "user",
                     "content": [{"type": "input_text",
                                  "text": f"[project {name} just finished and has a reply "
                                          f"waiting. Tell the user in ONE sentence that {name} "
                                          "is ready and they can say 'switch to it' to hear it. "
                                          "Do NOT discuss its content — you're still focused on "
                                          "the current project.]"}]},
        })
        await self._send({"type": "response.create"})

    # ---------- update injection ----------

    async def speak_update(self, text: str, source: str = "project",
                           cid: int | None = None) -> None:
        """A project reply arrived. If it's the focused project (or single-project
        mode), brief it now. Otherwise queue it and notify without stealing focus."""
        cid = cid or self.active_cid
        if source == "session response" and cid and cid != self.active_cid \
                and cid in self.projects:
            name = self.projects[cid].get("project") or "another project"
            self.pending.append((cid, name, text))
            await self._notify_pending(name)
            return
        if self.standby:
            await self.wake(f"update from {source}")
        await self._brief(text, source)

    async def _idle_watch(self) -> None:
        while not self.closed:
            await asyncio.sleep(5)
            idle = time.monotonic() - self.last_activity
            if not self.standby:
                # Live idle → drop to standby (quiet + wakeable), don't end the
                # session. This is the fix for "Benjamin won't wake after it goes
                # offline": it no longer goes offline, it goes to standby.
                if idle > IDLE_CLOSE_S:
                    self._enter_standby(f"idle {IDLE_CLOSE_S}s")
            elif idle > STANDBY_CLOSE_S:
                await self.close(f"idle {STANDBY_CLOSE_S}s")
                return

    async def _send(self, obj: dict) -> None:
        if self.ws and not self.closed:
            try:
                await self.ws.send(json.dumps(obj))
            except websockets.ConnectionClosed:
                pass


# ---------------------------------------------------------------------------
# Module-level singleton + helpers used by main.py / capture.py
# ---------------------------------------------------------------------------
_active: DuplexSession | None = None

# Persisted so a Benjamin restart (e.g. deploying a code fix) mid-conversation
# doesn't permanently drop the voice session and lose the pending response
# callback. Written on start, deleted on user-initiated stop/idle. On boot,
# resume_if_active() re-establishes a standby session if the file is fresh.
_STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "logs", "duplex_state.json")
_STATE_MAX_AGE_S = 900  # don't resume a stale session older than 15 min


def _save_state(s: "DuplexSession") -> None:
    with contextlib.suppress(OSError):
        # Persist all supervised project slots (minus the non-serializable
        # text_channel) plus which one is focused, so resume restores the fleet.
        projects = {str(c): {"project": sl.get("project"), "context": sl.get("context"),
                             "deep_context": sl.get("deep_context"), "voice": sl.get("voice")}
                    for c, sl in s.projects.items()}
        with open(_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({"channel_id": s.project_channel_id, "project": s.project,
                       "context": s.context, "deep_context": s.deep_context,
                       "voice": s.voice, "tier": s.tier,
                       "projects": projects, "active_cid": s.active_cid,
                       "ts": time.time()}, f)


def _clear_state() -> None:
    with contextlib.suppress(OSError):
        os.remove(_STATE_FILE)


# ---------------------------------------------------------------------------
# Usage tracking — append-only JSONL time-series in logs/duplex_usage.jsonl.
# One "turn" line per model response (crash-safe), one "session" line on close.
# Read/aggregate via usage_summary(); exposed over HTTP + the !duplex usage cmd.
# ---------------------------------------------------------------------------
_USAGE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "logs", "duplex_usage.jsonl")


def _log_usage(s: "DuplexSession", kind: str, din: int = 0, dout: int = 0) -> None:
    rec = {"ts": round(time.time(), 1), "kind": kind, "project": s.project,
           "tier": s.tier, "in": din, "out": dout,
           "cost": round(din / 1e6 * s.rate_in + dout / 1e6 * s.rate_out, 6)}
    if kind == "session":
        rec.update(duration_s=round(time.time() - s.started_at, 1),
                   in_total=s.in_tokens, out_total=s.out_tokens,
                   cost_total=round(s.in_tokens / 1e6 * s.rate_in
                                    + s.out_tokens / 1e6 * s.rate_out, 6),
                   wakes=s.wakes, dispatches=s.dispatches, deep_answers=s.deep_answers)
    with contextlib.suppress(OSError):
        with open(_USAGE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


def usage_summary(hours: float | None = None) -> dict:
    """Aggregate the usage log. `hours` limits to the recent window (None = all
    time). Returns totals + per-tier / per-project / per-day breakdowns."""
    cutoff = (time.time() - hours * 3600) if hours else 0
    turns = 0
    tot = {"in": 0, "out": 0, "cost": 0.0}
    sessions = 0
    wakes = dispatches = deep = 0
    by_tier: dict[str, dict] = {}
    by_project: dict[str, dict] = {}
    by_day: dict[str, float] = {}
    with contextlib.suppress(OSError):
        with open(_USAGE_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("ts", 0) < cutoff:
                    continue
                if r.get("kind") == "turn":
                    turns += 1
                    tot["in"] += r.get("in", 0)
                    tot["out"] += r.get("out", 0)
                    tot["cost"] += r.get("cost", 0.0)
                    tier = r.get("tier", "?")
                    bt = by_tier.setdefault(tier, {"in": 0, "out": 0, "cost": 0.0})
                    bt["in"] += r.get("in", 0); bt["out"] += r.get("out", 0)
                    bt["cost"] += r.get("cost", 0.0)
                    proj = r.get("project") or "?"
                    bp = by_project.setdefault(proj, {"cost": 0.0})
                    bp["cost"] += r.get("cost", 0.0)
                    day = time.strftime("%Y-%m-%d", time.localtime(r.get("ts", 0)))
                    by_day[day] = by_day.get(day, 0.0) + r.get("cost", 0.0)
                elif r.get("kind") == "session":
                    sessions += 1
                    wakes += r.get("wakes", 0)
                    dispatches += r.get("dispatches", 0)
                    deep += r.get("deep_answers", 0)
    return {"window_hours": hours, "turns": turns, "sessions": sessions,
            "in_tokens": tot["in"], "out_tokens": tot["out"],
            "cost_usd": round(tot["cost"], 4),
            "wakes": wakes, "dispatches": dispatches, "deep_answers": deep,
            "by_tier": {k: {**v, "cost": round(v["cost"], 4)} for k, v in by_tier.items()},
            "by_project": {k: round(v["cost"], 4) for k, v in by_project.items()},
            "by_day": {k: round(v, 4) for k, v in sorted(by_day.items())}}


def is_active() -> bool:
    return _active is not None and not _active.closed


def get() -> DuplexSession | None:
    return _active if is_active() else None


async def start(bot, text_channel, project: str | None = None,
                context: str | None = None,
                channel_id: int | None = None,
                voice: str | None = None,
                deep_context: str | None = None,
                tier: str | None = None) -> DuplexSession:
    """Start (or extend) a duplex session. If one is already live, this REGISTERS
    the project as another supervised slot and switches focus to it, rather than
    replacing the session — that's how multiple projects run in parallel over the
    one voice channel. Only the first start opens the realtime connection."""
    global _active
    if is_active():
        # New project gets its own voice: the one given, else auto-assigned.
        _active.add_project(channel_id, project, context, deep_context,
                            text_channel, voice=voice)
        if channel_id:
            await _active.switch_to(channel_id, announce=True)
        return _active
    guild = bot.get_guild(int(os.environ.get("GUILD_ID", "1224716298584592414")))
    vc = guild.voice_client if guild else None
    if vc is None or not vc.is_connected():
        raise RuntimeError("Benjamin is not in a voice channel")
    session = DuplexSession(bot, vc, text_channel, project=project,
                            context=context, channel_id=channel_id, voice=voice,
                            tier=tier)
    session.deep_context = deep_context
    # First project's slot uses the session's own voice.
    session.add_project(channel_id, project, context, deep_context, text_channel,
                        voice=session.voice)
    _active = session
    try:
        await session.connect()
    except Exception:
        _active = None
        raise
    _save_state(session)
    return session


async def resume_if_active(bot) -> bool:
    """On startup, re-establish a duplex session (in standby) if a fresh state
    file survives from before a restart, so a pending session-response callback
    still reaches the user by voice. Returns True if resumed."""
    if is_active():
        return False
    try:
        with open(_STATE_FILE, encoding="utf-8") as f:
            st = json.load(f)
    except (OSError, ValueError):
        return False
    if time.time() - float(st.get("ts", 0)) > _STATE_MAX_AGE_S:
        _clear_state()
        return False
    try:
        active_cid = st.get("active_cid") or int(st.get("channel_id") or 0) or None
        s = await start(bot, bot.get_channel(active_cid or 0),
                        project=st.get("project"), context=st.get("context"),
                        channel_id=active_cid, voice=st.get("voice"),
                        deep_context=st.get("deep_context"), tier=st.get("tier"))
        # Restore the rest of the supervised fleet (start() registered only the
        # active one). Slots are keyed by str(cid) in the saved state.
        for cid_s, sl in (st.get("projects") or {}).items():
            cid = int(cid_s)
            if cid != s.active_cid:
                s.add_project(cid, sl.get("project"), sl.get("context"),
                              sl.get("deep_context"), bot.get_channel(cid),
                              voice=sl.get("voice"))
        s.standby = True  # resume silent; the response callback / wake word drives speech
        log.info("duplex resumed in standby after restart (focus=%s, %d project(s))",
                 st.get("project"), len(s.projects))
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("duplex resume failed: %s", e)
        _clear_state()  # don't let a session we couldn't resume linger for retry
        return False


async def wake_from_state(bot) -> bool:
    """Wake word heard with no active session — resurrect the most recent one
    (if the state file is fresh) and bring it live listening. Lets 'Benjamin'
    revive a session that fully idle-closed, not just one in standby."""
    if is_active():
        return False
    if not await resume_if_active(bot):
        return False
    if is_active():
        await _active.wake("wake word")  # chime on + acknowledge + listen
        return True
    return False


async def stop(reason: str = "requested") -> bool:
    if is_active():
        await _active.close(reason)
        return True
    _clear_state()  # authoritative: also clear a stale file if resume never completed
    return False


# ---------------------------------------------------------------------------
# Local HTTP control API (used by the Ben bridge) — 127.0.0.1 only
# ---------------------------------------------------------------------------
_http_started = False


async def setup_http(bot, port: int = 3190) -> None:
    global _http_started
    if _http_started:
        return
    from aiohttp import web

    async def _channel(data: dict):
        cid = 0
        try:
            cid = int(data.get("channel_id") or 0)
        except (TypeError, ValueError):
            cid = 0
        ch = bot.get_channel(cid) if cid else None
        # The project channel lives in a DIFFERENT guild than Benjamin's voice
        # home, so it's usually not in the local cache — fetch it over the API
        # so submit_prompt's 📤 lands in the real project channel (where the
        # bridge is listening), not the voice-reports fallback.
        if ch is None and cid:
            with contextlib.suppress(Exception):
                ch = await bot.fetch_channel(cid)
        if ch is None:
            from config import REPORT_CHANNEL_ID
            ch = bot.get_channel(REPORT_CHANNEL_ID)
        return ch

    async def h_start(req):
        data = await req.json()
        try:
            cid = None
            with contextlib.suppress(TypeError, ValueError):
                cid = int(data.get("channel_id") or 0) or None
            s = await start(bot, await _channel(data), project=data.get("project"),
                            context=data.get("context"), channel_id=cid,
                            voice=data.get("voice"),
                            deep_context=data.get("deep_context"),
                            tier=data.get("tier"))
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=400)
        return web.json_response({"ok": True, "voice": s.voice,
                                  "tier": s.tier, "deployment": s.deployment})

    async def h_stop(req):
        stopped = await stop()
        return web.json_response({"ok": True, "stopped": stopped})

    async def h_update(req):
        data = await req.json()
        s = get()
        if not s:
            return web.json_response({"ok": False, "error": "no active session"}, status=400)
        cid = None
        with contextlib.suppress(TypeError, ValueError):
            cid = int(data.get("channel_id") or 0) or None
        await s.speak_update(data.get("text", ""), source=data.get("source", "project"),
                             cid=cid)
        return web.json_response({"ok": True})

    async def h_status(req):
        s = get()
        if not s:
            return web.json_response({"active": False})
        cost = s.in_tokens / 1e6 * s.rate_in + s.out_tokens / 1e6 * s.rate_out
        return web.json_response({"active": True, "project": s.project,
                                  "standby": s.standby, "voice": s.voice,
                                  "tier": s.tier, "deployment": s.deployment,
                                  "in_tokens": s.in_tokens, "out_tokens": s.out_tokens,
                                  "cost_usd": round(cost, 4),
                                  "idle_s": int(time.monotonic() - s.last_activity)})

    async def h_usage(req):
        hours = None
        with contextlib.suppress(TypeError, ValueError):
            h = req.query.get("hours")
            hours = float(h) if h else None
        summ = usage_summary(hours)
        # Fold in the live session's not-yet-flushed session-line counters.
        s = get()
        if s:
            summ["live"] = {"project": s.project, "tier": s.tier,
                            "in_tokens": s.in_tokens, "out_tokens": s.out_tokens,
                            "cost_usd": round(s.in_tokens / 1e6 * s.rate_in
                                              + s.out_tokens / 1e6 * s.rate_out, 4),
                            "wakes": s.wakes, "dispatches": s.dispatches}
        return web.json_response(summ)

    async def h_access(req):
        data = await req.json()
        s = get()
        if not s:
            return web.json_response({"ok": False, "error": "no active session"}, status=400)
        mode = s.set_audio_access(data.get("mode", "everyone"))
        return web.json_response({"ok": True, "access": mode})

    app = web.Application()
    app.router.add_post("/duplex/start", h_start)
    app.router.add_post("/duplex/stop", h_stop)
    app.router.add_post("/duplex/update", h_update)
    app.router.add_get("/duplex/status", h_status)
    app.router.add_get("/duplex/usage", h_usage)
    app.router.add_post("/duplex/access", h_access)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    _http_started = True
    log.info("duplex control API on 127.0.0.1:%d", port)
